from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import cv2
import numpy as np
import torch
from src.depth_engine import DepthEngine
from src.pose_tracker import (
    LAST_STAGE2_SPARSE_POINTS,
    CameraPose,
    _fallback_simple_trajectory,
    align_selected_depths,
    estimate_sfm_poses,
)
from src.presets import Preset
from src.scene_builder import Scene3D, _refine_multiview_depths_gpu, build_3d_scene, select_diverse_views
from src.video_ingest import VideoData, VideoFrame, extract_keyframes


class BuildCancelled(Exception):
    pass


@dataclass
class ReconstructionConfig:
    max_frames: int = 24
    max_dim: int = 512
    point_stride: int = 2
    mesh_stride: int = 4
    edge_thresh: float = 0.12
    depth_model: str = "base"
    device: Optional[str] = None
    scene_type: str = "indoor"
    max_anchors: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    recon_mode: str = "tsdf"
    preset_name: str = ""
    out_dir: str = "outputs"

    @classmethod
    def from_preset(cls, preset: Preset) -> "ReconstructionConfig":
        dev = None if preset.device == "auto" else preset.device
        return cls(
            max_frames=preset.max_frames,
            max_dim=preset.max_dim,
            point_stride=preset.point_stride,
            mesh_stride=preset.mesh_stride,
            edge_thresh=preset.edge_thresh,
            depth_model=preset.depth_model,
            device=dev,
            scene_type=preset.scene_type,
            max_anchors=preset.max_anchors,
            recon_mode=getattr(preset, "recon_mode", "tsdf"),
            preset_name=preset.name,
            out_dir=preset.out_dir,
        )


@dataclass
class ReconstructionResult:
    video_data: VideoData
    scene: Scene3D
    splat_scene: Optional[Any] = None
    splat_ply_path: Optional[Path] = None


class VideoTo3DPipeline:
    def __init__(self, device: Optional[str] = None):
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_engine = DepthEngine(device=target)

    def _build_3dgs_depth_priors(
        self,
        frames: list[VideoFrame],
        valid_idx: list[int],
        poses: list[CameraPose],
        k_mat: np.ndarray,
        k1_cal: float,
        xyz: np.ndarray,
        edge_thresh: float,
        scene_type: str,
        target_device: str,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict[int, tuple[np.ndarray, np.ndarray]]]]:
        """Estimate SfM-aligned DA3 depths and dense multi-view init points without mutating Stage-2 poses."""
        if len(valid_idx) < 8 or len(xyz) < 16 or len(frames) == 0:
            return None, None, None

        sel_frames = [frames[i] for i in valid_idx]
        sel_poses = [poses[i] for i in valid_idx]
        depths_all, masks_all = self.depth_engine.estimate_depths(
            frames=sel_frames,
            edge_thresh=edge_thresh,
            model_size="base",
            scene_type=scene_type,
            poses=sel_poses,
            intrinsics=k_mat,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            progress_range=(0.56, 0.62),
        )
        if not depths_all or len(depths_all) != len(valid_idx):
            return None, None, None

        xyz_sfm = np.asarray(xyz, dtype=np.float64)
        h_d, w_d = depths_all[0].shape
        fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
        cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])

        per_scales: list[Optional[float]] = []
        for m_k, f_idx in enumerate(valid_idx):
            p = poses[f_idx]
            r_cw = np.asarray(p.r_wc, dtype=np.float64).T
            t_cw = -r_cw @ np.asarray(p.t_wc, dtype=np.float64)
            xyz_c = xyz_sfm @ r_cw.T + t_cw[None, :]
            z_s = xyz_c[:, 2]
            in_f = z_s > 0.08
            uv_x = fx * (xyz_c[in_f, 0] / z_s[in_f]) + cx
            uv_y = fy * (xyz_c[in_f, 1] / z_s[in_f]) + cy
            z_s = z_s[in_f]
            ok_proj = (uv_x >= 2) & (uv_x < w_d - 2) & (uv_y >= 2) & (uv_y < h_d - 2)
            sc_f: Optional[float] = None
            if int(np.sum(ok_proj)) >= 18:
                ui = np.round(uv_x[ok_proj]).astype(int)
                vi = np.round(uv_y[ok_proj]).astype(int)
                dm = depths_all[m_k][vi, ui]
                mk = masks_all[m_k][vi, ui] & (dm > 0.05)
                if int(np.sum(mk)) >= 12:
                    sc_f = float(np.median(z_s[ok_proj][mk] / dm[mk]))
            per_scales.append(sc_f)

        valid_sc = [s for s in per_scales if s is not None]
        med_sc = float(np.median(valid_sc)) if valid_sc else 1.0
        aligned_d = [
            (
                depths_all[m_k]
                * float(np.clip(s_f, 0.55 * med_sc, 1.8 * med_sc) if s_f is not None else med_sc)
            ).astype(np.float32)
            for m_k, s_f in enumerate(per_scales)
        ]
        aligned_m = [np.asarray(m, dtype=bool) for m in masks_all]

        rgb_for_pts: dict[int, np.ndarray] = {}
        if abs(k1_cal) > 1e-4:
            dist = np.array([k1_cal, 0.0, 0.0, 0.0], dtype=np.float64)
            k64 = np.asarray(k_mat, dtype=np.float64)
            map1, map2 = cv2.initUndistortRectifyMap(k64, dist, None, k64, (w_d, h_d), cv2.CV_32FC1)
            for m_k in range(len(valid_idx)):
                d_lin = cv2.remap(aligned_d[m_k], map1, map2, interpolation=cv2.INTER_LINEAR)
                d_nn = cv2.remap(aligned_d[m_k], map1, map2, interpolation=cv2.INTER_NEAREST)
                aligned_d[m_k] = np.where(np.abs(d_lin - d_nn) < 0.06, d_lin, d_nn).astype(np.float32)
                aligned_m[m_k] = (
                    cv2.remap(aligned_m[m_k].astype(np.uint8), map1, map2, interpolation=cv2.INTER_NEAREST) > 0
                )

        sub_pick = np.linspace(0, len(valid_idx) - 1, min(32, len(valid_idx)), dtype=int).tolist()
        sub_orig = [valid_idx[k] for k in sub_pick]
        for f_idx in sub_orig:
            rgb_src = frames[f_idx].rgb
            if abs(k1_cal) > 1e-4:
                rgb_src = cv2.remap(
                    rgb_src, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
            rgb_for_pts[f_idx] = rgb_src

        fused_sub, clean_sub = _refine_multiview_depths_gpu(
            [aligned_d[k] for k in sub_pick],
            [aligned_m[k] for k in sub_pick],
            poses,
            k_mat,
            sub_orig,
            device=target_device,
            scene_type=scene_type,
        )

        uu, vv = np.meshgrid(np.arange(w_d, dtype=np.float32), np.arange(h_d, dtype=np.float32))
        rx = (uu - cx) / max(fx, 1e-4)
        ry = (vv - cy) / max(fy, 1e-4)
        pts_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        per_frame_cap = max(800, 65_000 // max(1, len(sub_orig)))
        is_outdoor = str(scene_type).lower() == "outdoor"
        z_max_dense = 7.0 if is_outdoor else 8.0
        erode_kern = np.ones((5, 5), dtype=np.uint8)
        rng = np.random.default_rng(7)
        for k_sub, f_idx in enumerate(sub_orig):
            dm = fused_sub[k_sub]
            mk = cv2.erode(clean_sub[k_sub].astype(np.uint8), erode_kern) > 0
            p = poses[f_idx]
            r_wc = np.asarray(p.r_wc, dtype=np.float32)
            if is_outdoor:
                # Filter sky/horizon boundary leaks so DA3 dense points stay on solid near/mid surfaces
                rc_y = r_wc[1, 0] * rx + r_wc[1, 1] * ry + r_wc[1, 2]
                im01 = rgb_for_pts[f_idx].astype(np.float32) * (1.0 / 255.0)
                sky_like = (rc_y < -0.02) & (im01[..., 2] >= im01[..., 0] - 0.03) & (im01.mean(axis=-1) > 0.42) & (dm > 3.5)
                mk = mk & (~sky_like)
            mk = mk & (dm > 0.12) & (dm < z_max_dense)
            ys_v, xs_v = np.where(mk)
            if len(ys_v) == 0:
                continue
            if len(ys_v) > per_frame_cap:
                sel = rng.choice(len(ys_v), size=per_frame_cap, replace=False)
                ys_v, xs_v = ys_v[sel], xs_v[sel]
            z_v = dm[ys_v, xs_v]
            pts_c = np.stack([rx[ys_v, xs_v] * z_v, ry[ys_v, xs_v] * z_v, z_v], axis=-1)
            pts_list.append(((pts_c @ r_wc.T) + np.asarray(p.t_wc, dtype=np.float32)[None, :]).astype(np.float32))
            cols_list.append(rgb_for_pts[f_idx][ys_v, xs_v].astype(np.uint8))

        view_depths: dict[int, tuple[np.ndarray, np.ndarray]] = {
            f_idx: (aligned_d[m_k], aligned_m[m_k]) for m_k, f_idx in enumerate(valid_idx)
        }
        for k_sub, f_idx in enumerate(sub_orig):
            view_depths[f_idx] = (fused_sub[k_sub], clean_sub[k_sub])

        if not pts_list:
            return None, None, view_depths
        return np.concatenate(pts_list, axis=0), np.concatenate(cols_list, axis=0), view_depths

    def run(
        self,
        video_path: str | Path,
        config: Optional[ReconstructionConfig] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> ReconstructionResult:
        cfg = config or ReconstructionConfig()
        target_device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if target_device != self.depth_engine.device:
            self.depth_engine.offload_to_cpu()
            self.depth_engine = DepthEngine(device=target_device)

        def _check_cancel():
            if cancel_cb and cancel_cb():
                raise BuildCancelled("build cancelled by user")

        # Stage 1: Keyframe scan & motion scoring (0.00 -> 0.28)
        _check_cancel()
        highres_max_dim = 0
        if str(getattr(cfg, "recon_mode", "tsdf")).strip().lower() == "3dgs":
            from src.gsplat_engine import resolve_gsplat_preset

            highres_max_dim = resolve_gsplat_preset(cfg.preset_name).max_dim

        video_data = extract_keyframes(
            video_path=video_path,
            max_frames=cfg.max_frames,
            max_dim=cfg.max_dim,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            highres_max_dim=highres_max_dim,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

        # Stage 2: SfM & Chained 3D-PnP Pose Estimation (0.28 -> 0.56)
        _check_cancel()
        ok, k_mat, k1_cal, poses, sparse_obs, _, anchor_valid_indices = estimate_sfm_poses(
            frames=video_data.frames,
            device=target_device,
            max_anchors=cfg.max_anchors,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

        _check_cancel()
        if str(getattr(cfg, "recon_mode", "tsdf")).strip().lower() == "3dgs":
            if not ok:
                raise RuntimeError("3DGS requires Stage-2 SfM camera poses, which failed on this video")

            from src.gsplat_engine import (
                export_splats_ply,
                is_gsplat_available,
                load_training_views_pyav,
                resolve_gsplat_preset,
                train_3dgs_scene,
            )

            if not is_gsplat_available():
                raise RuntimeError("3DGS mode requires the gsplat CUDA backend")
            if not str(target_device).startswith("cuda"):
                raise RuntimeError(
                    "3DGS training requires a CUDA device; set Compute device to Auto or CUDA"
                )

            xyz = LAST_STAGE2_SPARSE_POINTS.get("xyz", np.zeros((0, 3), dtype=np.float32))
            rgb = LAST_STAGE2_SPARSE_POINTS.get("rgb", np.zeros((0, 3), dtype=np.uint8))
            gs_cfg = resolve_gsplat_preset(cfg.preset_name)

            dense_xyz: Optional[np.ndarray] = None
            dense_rgb: Optional[np.ndarray] = None
            view_depths: Optional[dict[int, tuple[np.ndarray, np.ndarray]]] = None
            valid_idx = [i for i, p in enumerate(poses) if p.valid]
            if len(valid_idx) >= 8 and len(xyz) >= 16:
                dense_xyz, dense_rgb, view_depths = self._build_3dgs_depth_priors(
                    frames=video_data.frames,
                    valid_idx=valid_idx,
                    poses=poses,
                    k_mat=k_mat,
                    k1_cal=k1_cal,
                    xyz=xyz,
                    edge_thresh=cfg.edge_thresh,
                    scene_type=cfg.scene_type,
                    target_device=target_device,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
            _check_cancel()

            self.depth_engine.offload_to_cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def _load_prog(frac: float, msg: str):
                if progress_cb:
                    t_norm = max(0.0, min(1.0, (frac - 0.05) / 0.15))
                    progress_cb(0.62 + 0.05 * t_norm, msg)

            def _train_prog(frac: float, msg: str):
                if progress_cb:
                    t_norm = max(0.0, min(1.0, (frac - 0.20) / 0.78))
                    progress_cb(0.67 + 0.29 * t_norm, msg)

            views = load_training_views_pyav(
                video_path=video_data.path,
                frames=video_data.frames,
                poses=poses,
                k_mat=k_mat,
                k1_cal=k1_cal,
                max_dim=gs_cfg.max_dim,
                progress_cb=_load_prog,
                cancel_cb=cancel_cb,
            )
            _check_cancel()

            splat_scene = train_3dgs_scene(
                views=views,
                xyz=xyz,
                rgb=rgb,
                poses=poses,
                steps=gs_cfg.steps,
                warmup_steps=gs_cfg.warmup_steps,
                mid_res_steps=gs_cfg.mid_res_steps,
                cap_max=gs_cfg.cap_max,
                sh_degree=gs_cfg.sh_degree,
                refine_start_iter=gs_cfg.refine_start_iter,
                refine_stop_iter=gs_cfg.refine_stop_iter,
                refine_every=gs_cfg.refine_every,
                reset_every=gs_cfg.reset_every,
                ssim_half_res=True,
                keep_on_gpu=True,
                scene_type=cfg.scene_type,
                device=target_device,
                progress_cb=_train_prog,
                cancel_cb=cancel_cb,
                dense_xyz=dense_xyz,
                dense_rgb=dense_rgb,
                view_depths=view_depths,
            )
            _check_cancel()

            if progress_cb:
                progress_cb(0.97, "Exporting 3DGS splats (scene_splats.ply)...")
            out_dir = Path(cfg.out_dir or "outputs")
            splat_ply_path = export_splats_ply(splat_scene, out_dir / "scene_splats.ply")
            if progress_cb:
                progress_cb(1.0, "3DGS reconstruction complete")

            h_fr, w_fr = video_data.frames[0].rgb.shape[:2]
            scene = Scene3D(
                points=np.ascontiguousarray(xyz, dtype=np.float32),
                colors=np.ascontiguousarray(rgb, dtype=np.uint8),
                normals=np.zeros_like(xyz, dtype=np.float32),
                mesh_vertices=np.zeros((0, 3), dtype=np.float32),
                mesh_faces=np.zeros((0, 3), dtype=np.int32),
                mesh_colors=np.zeros((0, 3), dtype=np.uint8),
                poses=poses,
                intrinsics=k_mat,
                frame_size=(int(w_fr), int(h_fr)),
            )
            return ReconstructionResult(
                video_data=video_data,
                scene=scene,
                splat_scene=splat_scene,
                splat_ply_path=splat_ply_path,
            )

        # Stage 3: Depth Estimation & Consensus Alignment (0.56 -> 0.76)
        if ok:
            if len(anchor_valid_indices) >= 24:
                fuse_indices = list(anchor_valid_indices)
            else:
                valid_indices = [i for i, p in enumerate(poses) if p.valid] or [0]
                fuse_indices = select_diverse_views(video_data.frames, poses, valid_indices, max_views=180)

            selected_frames = [video_data.frames[i] for i in fuse_indices]
            selected_poses = [poses[i] for i in fuse_indices]
            depths_sub, masks_sub = self.depth_engine.estimate_depths(
                frames=selected_frames,
                edge_thresh=cfg.edge_thresh,
                model_size=cfg.depth_model,
                scene_type=cfg.scene_type,
                poses=selected_poses,
                intrinsics=k_mat,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                progress_range=(0.56, 0.68),
            )

            _check_cancel()
            aligned_depths, aligned_masks, keep_indices = align_selected_depths(
                frames=video_data.frames,
                selected_indices=fuse_indices,
                depths_sub=depths_sub,
                masks_sub=masks_sub,
                poses=poses,
                sparse_obs=sparse_obs,
                k_mat=k_mat,
                k1_cal=k1_cal,
                device=target_device,
                scene_type=cfg.scene_type,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                use_grid_warp=(not getattr(self.depth_engine, "last_used_da3", False)),
            )
            if not keep_indices:
                aligned_depths, aligned_masks, keep_indices = depths_sub, masks_sub, list(fuse_indices)
        else:
            # Fallback path if initial SfM registration completely failed
            depths_all, masks_all = self.depth_engine.estimate_depths(
                frames=video_data.frames,
                edge_thresh=cfg.edge_thresh,
                model_size=cfg.depth_model,
                scene_type=cfg.scene_type,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                progress_range=(0.56, 0.68),
            )
            k_mat, poses, aligned_depths = _fallback_simple_trajectory(video_data.frames, depths_all, masks_all)
            aligned_masks = masks_all
            keep_indices = list(range(len(video_data.frames)))

        # Stage 4: 3D TSDF & Mesh Fusion (0.76 -> 1.00)
        _check_cancel()
        self.depth_engine.offload_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scene = build_3d_scene(
            frames=video_data.frames,
            depths=aligned_depths,
            masks=aligned_masks,
            poses=poses,
            intrinsics=k_mat,
            selected_indices=keep_indices,
            device=target_device,
            point_stride=cfg.point_stride,
            mesh_stride=cfg.mesh_stride,
            scene_type=cfg.scene_type,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            video_path=video_data.path,
            k1_cal=k1_cal if ok else 0.0,
        )

        return ReconstructionResult(video_data=video_data, scene=scene)


def save_result_cache(result: ReconstructionResult, cache_path: str | Path) -> Path:
    out = Path(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    vd = result.video_data
    sc = result.scene

    r_arr = np.stack([p.r_wc for p in sc.poses], axis=0)
    t_arr = np.stack([p.t_wc for p in sc.poses], axis=0)
    s_arr = np.array([p.scale for p in sc.poses], dtype=np.float64)
    sh_arr = np.array([p.shift for p in sc.poses], dtype=np.float64)
    v_arr = np.array([p.valid for p in sc.poses], dtype=bool)

    f_indices = np.array([f.index for f in vd.frames], dtype=np.int32)
    f_ts = np.array([f.timestamp for f in vd.frames], dtype=np.float64)
    f_sharp = np.array([f.sharpness for f in vd.frames], dtype=np.float64)
    f_rgbs = np.stack([f.rgb for f in vd.frames], axis=0)

    np.savez(
        out,
        video_path=str(vd.path),
        fps=vd.fps,
        total_frames=vd.total_frames,
        duration=vd.duration,
        orig_size=np.array(vd.orig_size, dtype=np.int32),
        f_indices=f_indices,
        f_ts=f_ts,
        f_sharp=f_sharp,
        f_rgbs=f_rgbs,
        points=sc.points,
        colors=sc.colors,
        normals=sc.normals,
        mesh_vertices=sc.mesh_vertices,
        mesh_faces=sc.mesh_faces,
        mesh_colors=sc.mesh_colors,
        intrinsics=sc.intrinsics,
        frame_size=np.array(sc.frame_size, dtype=np.int32),
        r_arr=r_arr,
        t_arr=t_arr,
        s_arr=s_arr,
        sh_arr=sh_arr,
        v_arr=v_arr,
        **(  # optional high-res UV texture atlas
            {"mesh_uvs": sc.mesh_uvs, "mesh_texture": sc.mesh_texture}
            if sc.mesh_uvs is not None and sc.mesh_texture is not None
            else {}
        ),
    )
    return out


def load_result_cache(cache_path: str | Path) -> ReconstructionResult:
    data = np.load(Path(cache_path), allow_pickle=False)

    f_indices = data["f_indices"]
    f_ts = data["f_ts"]
    f_sharp = data["f_sharp"]
    f_rgbs = data["f_rgbs"]

    frames = [
        VideoFrame(
            index=int(f_indices[i]),
            timestamp=float(f_ts[i]),
            rgb=f_rgbs[i],
            sharpness=float(f_sharp[i]),
        )
        for i in range(len(f_indices))
    ]

    orig_sz = data["orig_size"]
    video_data = VideoData(
        path=Path(str(data["video_path"])),
        fps=float(data["fps"]),
        total_frames=int(data["total_frames"]),
        duration=float(data["duration"]),
        orig_size=(int(orig_sz[0]), int(orig_sz[1])),
        frames=frames,
    )

    r_arr = data["r_arr"]
    t_arr = data["t_arr"]
    s_arr = data["s_arr"]
    sh_arr = data["sh_arr"]
    v_arr = data["v_arr"] if "v_arr" in data else np.ones(len(r_arr), dtype=bool)
    poses = [
        CameraPose(
            r_wc=r_arr[i],
            t_wc=t_arr[i],
            scale=float(s_arr[i]),
            shift=float(sh_arr[i]),
            valid=bool(v_arr[i]),
        )
        for i in range(len(r_arr))
    ]

    fsz = data["frame_size"]
    scene = Scene3D(
        points=data["points"],
        colors=data["colors"],
        normals=data["normals"],
        mesh_vertices=data["mesh_vertices"],
        mesh_faces=data["mesh_faces"],
        mesh_colors=data["mesh_colors"],
        poses=poses,
        intrinsics=data["intrinsics"],
        frame_size=(int(fsz[0]), int(fsz[1])),
        mesh_uvs=data["mesh_uvs"] if "mesh_uvs" in data else None,
        mesh_texture=data["mesh_texture"] if "mesh_texture" in data else None,
    )
    return ReconstructionResult(video_data=video_data, scene=scene)
