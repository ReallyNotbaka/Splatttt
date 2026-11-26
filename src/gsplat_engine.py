"""Optional-import 3D Gaussian Splatting (MCMC) trainer, renderer, and PLY exporter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pose_tracker import CameraPose
from src.video_ingest import VideoFrame, _iter_grid_nvdec, _open_pyav_video

_SH_C0 = 0.28209479177387814
_SSIM_KERNEL_CACHE: Dict[Tuple[int, torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def is_gsplat_available() -> bool:
    """Return True iff CUDA is available and gsplat's compiled _C extension loads."""
    if not torch.cuda.is_available():
        return False
    try:
        import gsplat  # noqa: F401
        from gsplat.cuda._backend import _C

        return _C is not None
    except Exception:
        return False


@dataclass
class TrainingViews:
    """Decoded training images (CPU uint8) paired with camera extrinsics and intrinsics."""

    images: List[np.ndarray]
    c2ws: np.ndarray  # [V, 4, 4] float32
    viewmats: np.ndarray  # [V, 4, 4] float32 (world-to-camera)
    K: np.ndarray  # [3, 3] float32
    width: int
    height: int
    frame_indices: List[int]

    def subset(self, indices: Sequence[int]) -> "TrainingViews":
        idx_list = [int(i) for i in indices]
        return TrainingViews(
            images=[self.images[i] for i in idx_list],
            c2ws=self.c2ws[idx_list],
            viewmats=self.viewmats[idx_list],
            K=self.K.copy(),
            width=self.width,
            height=self.height,
            frame_indices=[self.frame_indices[i] for i in idx_list],
        )


@dataclass(frozen=True)
class GsplatPresetConfig:
    """3DGS training hyperparameters mapped to an app quality preset."""

    name: str
    steps: int
    warmup_steps: int
    cap_max: int
    max_dim: int
    sh_degree: int
    mid_res_steps: int = 0
    refine_start_iter: int = 400
    refine_every: int = 100
    refine_stop_iter: int = 3200
    reset_every: int = 1500


GSPLAT_PRESETS: Dict[str, GsplatPresetConfig] = {
    "fast": GsplatPresetConfig(
        name="Fast",
        steps=3000,
        warmup_steps=800,
        mid_res_steps=700,
        cap_max=200_000,
        max_dim=960,
        sh_degree=2,
        refine_start_iter=400,
        refine_every=100,
        refine_stop_iter=2000,
        reset_every=1500,
    ),
    "balanced": GsplatPresetConfig(
        name="Balanced",
        steps=4500,
        warmup_steps=1000,
        mid_res_steps=1200,
        cap_max=300_000,
        max_dim=1024,
        sh_degree=2,
        refine_start_iter=400,
        refine_every=100,
        refine_stop_iter=3200,
        reset_every=1500,
    ),
    "high": GsplatPresetConfig(
        name="High",
        steps=6000,
        warmup_steps=1500,
        mid_res_steps=1700,
        cap_max=400_000,
        max_dim=1280,
        sh_degree=2,
        refine_start_iter=400,
        refine_every=100,
        refine_stop_iter=4000,
        reset_every=1500,
    ),
}


def resolve_gsplat_preset(preset_name: str = "") -> GsplatPresetConfig:
    """Map a UI preset name (e.g. Fast Draft, Balanced, High) to 3DGS parameters."""
    key = str(preset_name or "").strip().lower()
    if "fast" in key:
        return GSPLAT_PRESETS["fast"]
    if "high" in key or "ultra" in key:
        return GSPLAT_PRESETS["high"]
    return GSPLAT_PRESETS["balanced"]


@dataclass
class SplatScene:
    """Trained 3D Gaussian Splatting scene parameters (kept GPU-ready for live rendering)."""

    means: torch.Tensor  # [N, 3] float32
    quats: torch.Tensor  # [N, 4] float32
    scales: torch.Tensor  # [N, 3] float32 (log-space)
    opacities: torch.Tensor  # [N] float32 (logit-space)
    sh0: torch.Tensor  # [N, 1, 3] float32
    shN: torch.Tensor  # [N, K, 3] float32
    sh_degree: int
    scene_scale: float
    init_splats: int = 0
    final_splats: int = 0
    peak_vram_gb: float = 0.0
    train_s: float = 0.0
    train_psnr_db: float = 0.0
    val_psnr_db: Optional[float] = None
    sky_logits: Optional[torch.Tensor] = None
    _render_cache: Optional[Tuple[Any, ...]] = None

    def release(self) -> None:
        """Release GPU tensors and clear cached render buffers to free VRAM."""
        self._render_cache = None
        was_cuda = (
            self.means.is_cuda
            or self.quats.is_cuda
            or self.scales.is_cuda
            or self.opacities.is_cuda
            or self.sh0.is_cuda
            or self.shN.is_cuda
            or (self.sky_logits is not None and self.sky_logits.is_cuda)
        )
        self.means = self.means.detach().cpu().contiguous()
        self.quats = self.quats.detach().cpu().contiguous()
        self.scales = self.scales.detach().cpu().contiguous()
        self.opacities = self.opacities.detach().cpu().contiguous()
        self.sh0 = self.sh0.detach().cpu().contiguous()
        self.shN = self.shN.detach().cpu().contiguous()
        if self.sky_logits is not None:
            self.sky_logits = self.sky_logits.detach().cpu().contiguous()
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sky_padded_rgb_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert [1, 3, H, W] sky dome logits into azimuth-wrapped [1, 3, H, W+1] RGB in (0, 1)."""
    rgb = torch.sigmoid(logits)
    return torch.cat([rgb, rgb[..., :1]], dim=-1)


class _SkyGridDome(nn.Module):
    """Bounded periodic upper-hemisphere sky dome for outdoor 3DGS backgrounds."""

    def __init__(self, device: str = "cuda", h_bins: int = 12, w_bins: int = 24):
        super().__init__()
        init_rgb = torch.tensor([0.68, 0.74, 0.82], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        init_logits = torch.log(init_rgb / (1.0 - init_rgb)).expand(1, 3, h_bins, w_bins).clone()
        self.logits = nn.Parameter(init_logits)

    def smoothness_loss(self) -> torch.Tensor:
        padded = _sky_padded_rgb_from_logits(self.logits)
        dx = padded[:, :, :, 1:] - padded[:, :, :, :-1]
        dy = padded[:, :, 1:, :] - padded[:, :, :-1, :]
        return dx.pow(2).mean() + dy.pow(2).mean()

    @staticmethod
    def compute_view_uv_and_up(
        c2ws_t: torch.Tensor,
        k_t: torch.Tensor,
        width: int,
        height: int,
        gw: int = 24,
        gh: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute equirectangular sky UV coordinates and upward ray cosine (-y_world)."""
        fx = float(k_t[0, 0]) * (float(gw) / float(max(1, width)))
        fy = float(k_t[1, 1]) * (float(gh) / float(max(1, height)))
        cx = float(k_t[0, 2]) * (float(gw) / float(max(1, width)))
        cy = float(k_t[1, 2]) * (float(gh) / float(max(1, height)))
        xs = (torch.arange(gw, device=c2ws_t.device, dtype=torch.float32) + 0.5 - cx) / max(fx, 1e-4)
        ys = (torch.arange(gh, device=c2ws_t.device, dtype=torch.float32) + 0.5 - cy) / max(fy, 1e-4)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dc = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        dc = dc / dc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dirs_w = torch.einsum("hwj,vij->vhwi", dc, c2ws_t[:, :3, :3])
        x, y, z = dirs_w.unbind(dim=-1)
        az = torch.atan2(x, z) / math.pi
        ru = -y
        el_01 = torch.asin(ru.clamp(0.0, 1.0)) / (0.5 * math.pi)
        el_norm = el_01 * 2.0 - 1.0
        uv = torch.stack([az, el_norm], dim=-1)
        return uv, ru

    def eval_uv(self, uv_v: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        dome = _sky_padded_rgb_from_logits(self.logits)
        sampled = F.grid_sample(
            dome, uv_v.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=True
        )
        return F.interpolate(sampled, size=(out_h, out_w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def _eval_sky_bg_from_padded(
    padded_dome: torch.Tensor,
    c2w_t: torch.Tensor,
    k_t: torch.Tensor,
    width: int,
    height: int,
    gw: int = 32,
    gh: int = 18,
) -> torch.Tensor:
    uv, rup = _SkyGridDome.compute_view_uv_and_up(c2w_t.unsqueeze(0), k_t, width, height, gw=gw, gh=gh)
    sampled = F.grid_sample(padded_dome, uv, mode="bilinear", padding_mode="border", align_corners=True)
    sky = F.interpolate(sampled, size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    ru = F.interpolate(rup.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    hw = torch.clamp((ru + 0.04) / 0.14, 0.0, 1.0).unsqueeze(-1)
    ground_col = torch.tensor([0.24, 0.26, 0.28], dtype=torch.float32, device=c2w_t.device)
    return hw * sky + (1.0 - hw) * ground_col


def _compute_nearest_cam_dist(means_t: torch.Tensor, cam_centers_t: torch.Tensor) -> torch.Tensor:
    out = torch.empty((means_t.shape[0],), device=means_t.device, dtype=torch.float32)
    chunk = 65536
    for s in range(0, means_t.shape[0], chunk):
        e = min(s + chunk, means_t.shape[0])
        out[s:e] = torch.cdist(means_t[s:e], cam_centers_t).min(dim=1).values
    return out


def _pose_to_c2w_and_viewmat(pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
    c2w = np.eye(4, dtype=np.float32)
    r_wc = np.asarray(pose.r_wc, dtype=np.float64)
    t_wc = np.asarray(pose.t_wc, dtype=np.float64)
    c2w[:3, :3] = r_wc.astype(np.float32)
    c2w[:3, 3] = t_wc.astype(np.float32)

    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = r_cw.astype(np.float32)
    viewmat[:3, 3] = t_cw.astype(np.float32)
    return c2w, viewmat


def _decode_targets_pyav_cpu(
    video_path: Path,
    target_frames: List[int],
    fps: float,
    cancel_cb: Optional[Callable[[], bool]],
) -> Dict[int, np.ndarray]:
    """Fallback PyAV software decode for exact target frame indices."""
    decoded: Dict[int, np.ndarray] = {}
    container = None
    try:
        container = _open_pyav_video(video_path, use_hw=False)
        stream = container.streams.video[0]
        tb = float(stream.time_base) if stream.time_base else (1.0 / fps)
        stream_start = float(stream.start_time * tb) if stream.start_time is not None else 0.0
        frame_iter = None
        last_sec: Optional[float] = None
        gop_samples: List[float] = []

        for f_target in target_frames:
            if cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")

            target_sec = stream_start + (f_target / fps)
            target_pts = int(round(target_sec / tb))
            est_gop = float(np.median(gop_samples)) if gop_samples else 0.0
            seek_gap = max(15.0 / fps, 0.55, est_gop * 1.15)
            need_seek = (
                frame_iter is None
                or last_sec is None
                or (target_sec - last_sec) > seek_gap
                or target_sec < last_sec
            )
            if need_seek:
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                frame_iter = container.decode(stream)

            got = None
            sampled_gop = False
            for fr in frame_iter:
                if fr.time is None:
                    continue
                ft = float(fr.time)
                if need_seek and not sampled_gop:
                    gop_samples.append(max(0.0, target_sec - ft))
                    gop_samples = gop_samples[-5:]
                    sampled_gop = True
                if not need_seek and last_sec is not None and ft <= last_sec + (0.25 / fps):
                    continue
                got = fr
                if ft >= target_sec - (0.35 / fps):
                    break

            if got is None:
                return {}
            decoded[f_target] = np.ascontiguousarray(got.to_ndarray(format="bgr24"), dtype=np.uint8)
            last_sec = float(got.time) if got.time is not None else target_sec
        return decoded
    except RuntimeError as exc:
        if (cancel_cb and cancel_cb()) or "cancelled" in str(exc).lower():
            raise
        return {}
    except Exception:
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _decode_targets_opencv(
    video_path: Path,
    target_frames: List[int],
    cancel_cb: Optional[Callable[[], bool]],
) -> Dict[int, np.ndarray]:
    """Final OpenCV fallback for exact target frame indices."""
    decoded: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return decoded
    try:
        for f_target in target_frames:
            if cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_target))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                decoded[f_target] = np.ascontiguousarray(bgr, dtype=np.uint8)
    finally:
        cap.release()
    return decoded


def load_training_views_pyav(
    video_path: str | Path,
    frames: List[VideoFrame],
    poses: List[CameraPose],
    k_mat: np.ndarray,
    k1_cal: float = 0.0,
    max_dim: int = 1280,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> TrainingViews:
    """Decode valid Stage-2 keyframes at ``max_dim`` via PyAV NVDEC and scale intrinsics.

    Decoded RGB images remain in CPU RAM as uint8 arrays; frames are uploaded to GPU
    one at a time during training to stay well within 6 GB VRAM.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"cannot find video at {path}")
    if len(frames) != len(poses):
        raise ValueError(f"frames ({len(frames)}) and poses ({len(poses)}) length mismatch")

    valid_indices = [i for i, p in enumerate(poses) if p.valid]
    if not valid_indices:
        raise ValueError("no valid camera poses to load training views")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to probe video metadata for {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frames[0].rgb.shape[1])
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frames[0].rgb.shape[0])
    cap.release()

    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    out_w = max(2, int(round(orig_w * scale)))
    out_h = max(2, int(round(orig_h * scale)))
    out_w -= out_w % 2
    out_h -= out_h % 2

    h_s2, w_s2 = frames[0].rgb.shape[:2]
    sx = float(out_w) / float(w_s2)
    sy = float(out_h) / float(h_s2)
    k_scaled = np.asarray(k_mat, dtype=np.float32).copy()
    k_scaled[0, 0] *= sx
    k_scaled[0, 2] *= sx
    k_scaled[1, 1] *= sy
    k_scaled[1, 2] *= sy

    has_cached_hr = all(
        getattr(frames[i], "highres_rgb", None) is not None
        and frames[i].highres_rgb.shape == (out_h, out_w, 3)
        for i in valid_indices
    )

    decoded_bgr: Dict[int, np.ndarray] = {}
    if not has_cached_hr:
        target_video_indices = sorted({int(frames[i].index) for i in valid_indices})
        nv_stats = {
            "backend": "NVDEC",
            "seeks": 0,
            "decoded_frames": 0,
            "decode_s": 0.0,
            "completed": False,
        }
        try:
            for nv_i, (f_idx, bgr) in enumerate(
                _iter_grid_nvdec(path, target_video_indices, fps, nv_stats, cancel_cb)
            ):
                decoded_bgr[int(f_idx)] = bgr
                if progress_cb and nv_i % 12 == 0:
                    progress_cb(
                        0.05 + 0.15 * ((nv_i + 1) / max(1, len(target_video_indices))),
                        f"decoding 3DGS views (NVDEC)... ({nv_i + 1}/{len(target_video_indices)})",
                    )
        except RuntimeError as exc:
            if (cancel_cb and cancel_cb()) or "cancelled" in str(exc).lower():
                raise
            decoded_bgr.clear()
        except Exception:
            decoded_bgr.clear()

        if not nv_stats["completed"] or len(decoded_bgr) < len(target_video_indices):
            decoded_bgr = _decode_targets_pyav_cpu(path, target_video_indices, fps, cancel_cb)
        if len(decoded_bgr) < len(target_video_indices):
            missing = [f for f in target_video_indices if f not in decoded_bgr]
            decoded_bgr.update(_decode_targets_opencv(path, missing, cancel_cb))

    use_undistort = abs(float(k1_cal)) > 1e-6
    k64 = k_scaled.astype(np.float64)
    dist = np.array([float(k1_cal), 0.0, 0.0, 0.0], dtype=np.float64)
    map1 = map2 = None
    if use_undistort:
        map1, map2 = cv2.initUndistortRectifyMap(
            k64, dist, None, k64, (out_w, out_h), cv2.CV_32FC1
        )

    images: List[np.ndarray] = []
    c2w_list: List[np.ndarray] = []
    viewmat_list: List[np.ndarray] = []
    kept_frame_indices: List[int] = []

    for i in valid_indices:
        if has_cached_hr:
            rgb = frames[i].highres_rgb
        else:
            v_idx = int(frames[i].index)
            bgr = decoded_bgr.get(v_idx)
            if bgr is None:
                continue
            if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if use_undistort and map1 is not None and map2 is not None:
            rgb = cv2.remap(
                rgb,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        images.append(np.ascontiguousarray(rgb, dtype=np.uint8))

        c2w, viewmat = _pose_to_c2w_and_viewmat(poses[i])
        c2w_list.append(c2w)
        viewmat_list.append(viewmat)
        kept_frame_indices.append(i)

    for fr in frames:
        fr.highres_rgb = None

    if not images:
        raise RuntimeError(f"failed to decode any training frames from {path}")

    return TrainingViews(
        images=images,
        c2ws=np.stack(c2w_list, axis=0).astype(np.float32),
        viewmats=np.stack(viewmat_list, axis=0).astype(np.float32),
        K=k_scaled,
        width=out_w,
        height=out_h,
        frame_indices=kept_frame_indices,
    )


def compute_scene_scale(poses_or_c2ws: Sequence[CameraPose] | np.ndarray) -> float:
    """Compute scene bounding radius from camera centers."""
    if isinstance(poses_or_c2ws, np.ndarray):
        centers = np.asarray(poses_or_c2ws[:, :3, 3], dtype=np.float64)
    else:
        valid_centers = [np.asarray(p.t_wc, dtype=np.float64) for p in poses_or_c2ws if p.valid]
        if not valid_centers:
            valid_centers = [np.asarray(p.t_wc, dtype=np.float64) for p in poses_or_c2ws]
        centers = np.asarray(valid_centers, dtype=np.float64)

    if len(centers) == 0:
        return 1.0
    center_mean = np.mean(centers, axis=0)
    radii = np.linalg.norm(centers - center_mean[None, :], axis=1)
    return max(float(np.max(radii)) * 1.1, 0.5)


def init_gaussians_from_sfm(
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: Sequence[CameraPose] | np.ndarray,
    sh_degree: int = 3,
    scene_type: str = "indoor",
    device: str = "cuda",
) -> Tuple[Dict[str, torch.nn.Parameter], float]:
    """Initialize 3DGS parameters from SfM sparse points and camera poses."""
    xyz_np = np.asarray(xyz, dtype=np.float32)
    rgb_np = np.asarray(rgb, dtype=np.uint8)

    finite_mask = np.isfinite(xyz_np).all(axis=1)
    xyz_np = xyz_np[finite_mask]
    rgb_np = rgb_np[finite_mask]
    m_pts = len(xyz_np)
    if m_pts == 0:
        raise ValueError("cannot initialize Gaussians from empty SfM point cloud")

    if m_pts >= 2:
        k_query = min(4, m_pts)
        tree = cKDTree(xyz_np.astype(np.float64))
        dists, _ = tree.query(xyz_np.astype(np.float64), k=k_query, workers=-1)
        knn_dist = np.mean(dists[:, 1:], axis=1)
    else:
        knn_dist = np.array([0.05], dtype=np.float64)

    clamped_dist = np.clip(knn_dist, 1e-4, 0.5).astype(np.float32)
    log_scales = np.repeat(np.log(clamped_dist)[:, None], 3, axis=1)

    sh_dc = ((rgb_np.astype(np.float32) / 255.0) - 0.5) / _SH_C0
    k_rest = (int(sh_degree) + 1) ** 2 - 1

    quats = np.zeros((m_pts, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # logit(0.5) == 0.0
    opacities = np.zeros((m_pts,), dtype=np.float32)

    params: Dict[str, torch.nn.Parameter] = {
        "means": torch.nn.Parameter(torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)),
        "scales": torch.nn.Parameter(torch.from_numpy(log_scales).to(device=device, dtype=torch.float32)),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).to(device=device, dtype=torch.float32)),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities).to(device=device, dtype=torch.float32)),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh_dc[:, None, :]).to(device=device, dtype=torch.float32)),
        "shN": torch.nn.Parameter(
            torch.zeros((m_pts, k_rest, 3), dtype=torch.float32, device=device)
        ),
    }
    scene_scale = compute_scene_scale(poses)
    return params, scene_scale


def _get_gaussian_kernels_1d(
    channels: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (channels, dtype, device)
    cached = _SSIM_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    win = 11
    sigma = 1.5
    coords = torch.arange(win, dtype=torch.float32, device=device) - (win // 2)
    g1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g1d = g1d / g1d.sum()
    k_h = g1d.view(1, 1, 1, win).expand(channels, 1, 1, win).contiguous().to(dtype=dtype)
    k_v = g1d.view(1, 1, win, 1).expand(channels, 1, win, 1).contiguous().to(dtype=dtype)
    _SSIM_KERNEL_CACHE[key] = (k_h, k_v)
    return k_h, k_v


def _separable_gaussian_blur_11(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    k_h, k_v = _get_gaussian_kernels_1d(channels, x.dtype, x.device)
    return F.conv2d(
        F.conv2d(x, k_h, padding=(0, 5), groups=channels),
        k_v,
        padding=(5, 0),
        groups=channels,
    )


def _precompute_gt_ssim_stats(
    gt_chw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute (mu2, mu2_sq, sigma2_sq) for a [1, C, H, W] GT tensor."""
    with torch.no_grad():
        yy_blur = _separable_gaussian_blur_11(torch.cat([gt_chw, gt_chw * gt_chw], dim=1))
        mu2, y2_blur = yy_blur.split(gt_chw.shape[1], dim=1)
        mu2_sq = mu2.pow(2)
        sigma2_sq = y2_blur - mu2_sq
    return mu2, mu2_sq, sigma2_sq


def gaussian_ssim_11x11(
    img1: torch.Tensor,
    img2: torch.Tensor,
    gt_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute differentiable separable 11x11 Gaussian-windowed SSIM (sigma=1.5)."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    if img1.shape[-1] in (1, 3, 4) and img1.shape[1] not in (1, 3, 4):
        img1 = img1.permute(0, 3, 1, 2)
    if img2.shape[-1] in (1, 3, 4) and img2.shape[1] not in (1, 3, 4):
        img2 = img2.permute(0, 3, 1, 2)

    if gt_stats is not None:
        mu2, mu2_sq, sigma2_sq = gt_stats
        xx_blur = _separable_gaussian_blur_11(
            torch.cat([img1, img1 * img1, img1 * img2], dim=1)
        )
        mu1, x2_blur, xy_blur = xx_blur.split(img1.shape[1], dim=1)
        mu1_sq = mu1.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = x2_blur - mu1_sq
        sigma12 = xy_blur - mu1_mu2
    elif not img2.requires_grad:
        mu2, mu2_sq, sigma2_sq = _precompute_gt_ssim_stats(img2)
        xx_blur = _separable_gaussian_blur_11(
            torch.cat([img1, img1 * img1, img1 * img2], dim=1)
        )
        mu1, x2_blur, xy_blur = xx_blur.split(img1.shape[1], dim=1)
        mu1_sq = mu1.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = x2_blur - mu1_sq
        sigma12 = xy_blur - mu1_mu2
    else:
        all_blur = _separable_gaussian_blur_11(
            torch.cat([img1, img2, img1 * img1, img2 * img2, img1 * img2], dim=1)
        )
        mu1, mu2, x2_blur, y2_blur, xy_blur = all_blur.split(img1.shape[1], dim=1)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = x2_blur - mu1_sq
        sigma2_sq = y2_blur - mu2_sq
        sigma12 = xy_blur - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (num / den).mean()


def compute_psnr_db(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Compute PSNR in dB for float [0, 1] or uint8 [0, 255] images."""
    if isinstance(pred, np.ndarray):
        p = pred.astype(np.float32) / (255.0 if pred.dtype == np.uint8 else 1.0)
        t = target.astype(np.float32) / (255.0 if isinstance(target, np.ndarray) and target.dtype == np.uint8 else 1.0)
        mse = float(np.mean((np.clip(p, 0.0, 1.0) - np.clip(t, 0.0, 1.0)) ** 2))
    else:
        p = pred.detach().clamp(0.0, 1.0)
        t = target.detach().clamp(0.0, 1.0)
        mse = float(F.mse_loss(p, t).item())
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def train_3dgs_scene(
    views: TrainingViews,
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: Optional[Sequence[CameraPose]] = None,
    *,
    steps: int = 6000,
    warmup_steps: Optional[int] = None,
    mid_res_steps: int = 0,
    cap_max: int = 400_000,
    sh_degree: int = 3,
    refine_start_iter: int = 400,
    refine_stop_iter: int = 4000,
    refine_every: int = 100,
    reset_every: int = 1500,
    min_opacity: float = 0.005,
    noise_lr: float = 1e5,
    packed: bool = True,
    ssim_half_res: bool = True,
    keep_on_gpu: bool = True,
    scene_type: str = "indoor",
    device: str = "cuda",
    seed: int = 42,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    dense_xyz: Optional[np.ndarray] = None,
    dense_rgb: Optional[np.ndarray] = None,
    view_depths: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    use_sky_dome: Optional[bool] = None,
) -> SplatScene:
    """Train a 3DGS AbsGS scene from TrainingViews and sparse SfM points."""
    if not is_gsplat_available():
        raise RuntimeError("gsplat CUDA backend is not available")
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "3DGS training requires a CUDA device; set Compute device to Auto or CUDA"
        )
    if len(views.images) == 0:
        raise ValueError("views contains no training images")

    from gsplat import rasterization
    from gsplat.optimizers import SelectiveAdam
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import reset_opa

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    is_cuda = str(device).startswith("cuda")
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    try:
        num_views = len(views.images)
        width = int(views.width)
        height = int(views.height)
        is_full_run = (num_views >= 8) and (int(steps) >= 400)
        pose_ref: Sequence[CameraPose] | np.ndarray = poses if poses is not None else views.c2ws
        cam_centers_np = views.c2ws[:, :3, 3].astype(np.float32)

        xyz_init = np.asarray(xyz, dtype=np.float32)
        rgb_init = np.asarray(rgb, dtype=np.uint8)
        if dense_xyz is not None and dense_rgb is not None and len(dense_xyz) > 0:
            if len(xyz_init) > 0 and num_views >= 8:
                d_cam_sfm, _ = cKDTree(cam_centers_np).query(xyz_init, k=1)
                sfm_keep = d_cam_sfm >= 0.06
                if int(np.sum(sfm_keep)) >= 16:
                    xyz_init = xyz_init[sfm_keep]
                    rgb_init = rgb_init[sfm_keep]
            xyz_init = np.concatenate([xyz_init, np.asarray(dense_xyz, dtype=np.float32)], axis=0)
            rgb_init = np.concatenate([rgb_init, np.asarray(dense_rgb, dtype=np.uint8)], axis=0)

        params, scene_scale = init_gaussians_from_sfm(
            xyz=xyz_init,
            rgb=rgb_init,
            poses=pose_ref,
            sh_degree=sh_degree,
            scene_type=scene_type,
            device=device,
        )
        init_splats = int(params["means"].shape[0])

        inv_scale = min(float(scene_scale), 1.4) if is_full_run else float(scene_scale)
        train_near_plane = (
            max(0.05, min(0.10, 0.08 * inv_scale))
            if is_full_run
            else max(0.05, 0.06 * float(scene_scale))
        )
        min_log_scale = math.log(max(1e-4, 1e-4 * inv_scale))
        pause_refine = (
            max(200, min(max(1, int(reset_every) - 200), num_views + 30))
            if (is_full_run and int(steps) >= 600 and int(reset_every) > 250)
            else 0
        )

        lr_means_init = 1.6e-4 * scene_scale
        lr_means_final = 1.6e-6 * scene_scale
        lr_map = {
            "means": lr_means_init,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20.0,
        }
        optimizers: Dict[str, torch.optim.Optimizer] = {
            name: SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
            for name, param in params.items()
        }
        for name, opt in optimizers.items():
            opt.param_groups[0]["lr"] = lr_map[name]

        strategy = DefaultStrategy(
            prune_opa=float(min_opacity),
            grow_grad2d=0.0008,
            grow_scale3d=0.01,
            prune_scale3d=0.15,
            refine_scale2d_stop_iter=0 if is_full_run else 4000,
            refine_start_iter=int(refine_start_iter),
            refine_stop_iter=int(refine_stop_iter),
            refine_every=int(refine_every),
            reset_every=int(reset_every),
            pause_refine_after_reset=pause_refine,
            absgrad=True,
            revised_opacity=True,
            verbose=False,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        viewmats_gpu = torch.from_numpy(views.viewmats).to(device=device, dtype=torch.float32)
        c2ws_gpu = torch.from_numpy(views.c2ws).to(device=device, dtype=torch.float32)
        cam_centers_gpu = torch.from_numpy(cam_centers_np).to(device=device)
        k_gpu = torch.from_numpy(views.K).unsqueeze(0).to(device=device, dtype=torch.float32)

        use_sky = (
            (str(scene_type).lower() == "outdoor" and is_full_run)
            if use_sky_dome is None
            else bool(use_sky_dome)
        )
        sky_dome = _SkyGridDome(device=device) if use_sky else None
        sky_opt: Optional[torch.optim.Optimizer] = (
            torch.optim.Adam(sky_dome.parameters(), lr=3e-2) if sky_dome is not None else None
        )

        # Progressive 3-stage resolution (0.5x -> 0.75x -> 1.0x when >= 640x360)
        if steps >= 400 and width >= 640 and height >= 360:
            if warmup_steps is None:
                eff_warmup = min(1500, steps // 3)
            else:
                eff_warmup = max(0, min(int(warmup_steps), steps))
            eff_mid = max(0, min(int(mid_res_steps), steps - eff_warmup))
        else:
            eff_warmup = 0
            eff_mid = 0

        if eff_warmup > 0:
            half_w = max(2, (width // 2) - ((width // 2) % 2))
            half_h = max(2, (height // 2) - ((height // 2) % 2))
            half_images = [
                np.ascontiguousarray(cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_AREA))
                for img in views.images
            ]
            k_half_gpu = k_gpu.clone()
            k_half_gpu[:, 0, 0] *= float(half_w) / float(width)
            k_half_gpu[:, 0, 2] *= float(half_w) / float(width)
            k_half_gpu[:, 1, 1] *= float(half_h) / float(height)
            k_half_gpu[:, 1, 2] *= float(half_h) / float(height)
        else:
            half_w, half_h = width, height
            half_images = views.images
            k_half_gpu = k_gpu

        if eff_mid > 0:
            mid_w = max(2, int(round(width * 0.75)) - (int(round(width * 0.75)) % 2))
            mid_h = max(2, int(round(height * 0.75)) - (int(round(height * 0.75)) % 2))
            mid_images = [
                np.ascontiguousarray(cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_AREA))
                for img in views.images
            ]
            k_mid_gpu = k_gpu.clone()
            k_mid_gpu[:, 0, 0] *= float(mid_w) / float(width)
            k_mid_gpu[:, 0, 2] *= float(mid_w) / float(width)
            k_mid_gpu[:, 1, 1] *= float(mid_h) / float(height)
            k_mid_gpu[:, 1, 2] *= float(mid_h) / float(height)
        else:
            mid_w, mid_h = width, height
            mid_images = views.images
            k_mid_gpu = k_gpu

        full_bytes = num_views * height * width * 3
        use_gpu_views = is_cuda and full_bytes <= 1_000_000_000
        if use_gpu_views:
            half_imgs_gpu = (
                torch.from_numpy(np.stack(half_images, axis=0)).to(
                    device=device, dtype=torch.uint8
                )
                if eff_warmup > 0
                else None
            )
            mid_imgs_gpu = (
                torch.from_numpy(np.stack(mid_images, axis=0)).to(
                    device=device, dtype=torch.uint8
                )
                if (eff_warmup == 0 and eff_mid > 0)
                else None
            )
            full_imgs_gpu = (
                torch.from_numpy(np.stack(views.images, axis=0)).to(
                    device=device, dtype=torch.uint8
                )
                if (eff_warmup == 0 and eff_mid == 0)
                else None
            )
            pin_full = pin_half = pin_mid = None
        else:
            full_imgs_gpu = half_imgs_gpu = mid_imgs_gpu = None
            pin_full = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=is_cuda)
            pin_half = (
                torch.empty((half_h, half_w, 3), dtype=torch.uint8, pin_memory=is_cuda)
                if eff_warmup > 0
                else pin_full
            )
            pin_mid = (
                torch.empty((mid_h, mid_w, 3), dtype=torch.uint8, pin_memory=is_cuda)
                if eff_mid > 0
                else pin_full
            )

        need_stage_masks = is_full_run or (sky_dome is not None) or bool(view_depths)
        if need_stage_masks:
            uv_all, rup_all = _SkyGridDome.compute_view_uv_and_up(c2ws_gpu, k_gpu[0], width, height, gw=24, gh=16)
            dd_tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            if view_depths:
                for v_local, orig_idx in enumerate(views.frame_indices):
                    if int(orig_idx) in view_depths:
                        dm_np, mk_np = view_depths[int(orig_idx)]
                        dd_tensors[v_local] = (
                            torch.from_numpy(np.asarray(dm_np, dtype=np.float32)).to(device=device),
                            torch.from_numpy(np.asarray(mk_np, dtype=bool)).to(device=device),
                        )

            def _build_stage_masks(
                cw: int,
                ch: int,
                imgs_gpu_cur: Optional[torch.Tensor],
                imgs_cpu_cur: List[np.ndarray],
            ) -> Tuple[List[torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
                mw, mh = min(cw, 640), min(ch, 360)
                sky_masks: List[torch.Tensor] = []
                disp_gt_s: Dict[int, torch.Tensor] = {}
                mk_s: Dict[int, torch.Tensor] = {}
                rup_full = F.interpolate(
                    rup_all.unsqueeze(1), size=(mh, mw), mode="bilinear", align_corners=False
                ).squeeze(1)
                for vi in range(num_views):
                    if imgs_gpu_cur is not None:
                        gt = imgs_gpu_cur[vi].float() * (1.0 / 255.0)
                    else:
                        gt = torch.from_numpy(imgs_cpu_cur[vi]).to(device=device, dtype=torch.float32) * (1.0 / 255.0)
                    if (cw, ch) != (mw, mh):
                        gt = F.interpolate(
                            gt.permute(2, 0, 1).unsqueeze(0), size=(mh, mw), mode="area"
                        )[0].permute(1, 2, 0)
                    gt_g = gt.mean(dim=-1)
                    gt_bgr = gt[..., 2] >= (gt[..., 0] - 0.03)
                    gx_s = torch.zeros_like(gt_g)
                    gy_s = torch.zeros_like(gt_g)
                    gx_s[:, 1:] = torch.abs(gt_g[:, 1:] - gt_g[:, :-1])
                    gy_s[1:, :] = torch.abs(gt_g[1:, :] - gt_g[:-1, :])
                    grad = gx_s + gy_s
                    smooth = grad < 0.05
                    ru = rup_full[vi]
                    if vi in dd_tensors:
                        dm_r = F.interpolate(
                            dd_tensors[vi][0][None, None], size=(mh, mw), mode="bilinear", align_corners=False
                        )[0, 0]
                        mk_r = (
                            F.interpolate(dd_tensors[vi][1][None, None].float(), size=(mh, mw), mode="nearest")[0, 0]
                            > 0.5
                        )
                        if sky_dome is not None:
                            s_m = (
                                (~mk_r)
                                & (ru > 0.03)
                                & (dm_r > 5.5)
                                & gt_bgr
                                & (gt_g > 0.42)
                                & (smooth | ((ru > 0.14) & (gt_g > 0.78) & (grad < 0.16)))
                            )
                        else:
                            s_m = torch.zeros((mh, mw), dtype=torch.bool, device=device)
                        disp_gt_s[vi] = 1.0 / dm_r.clamp(min=train_near_plane)
                        mk_s[vi] = mk_r & smooth
                    else:
                        if sky_dome is not None:
                            s_m = (ru > 0.12) & gt_bgr & smooth & (gt_g > 0.90)
                        else:
                            s_m = torch.zeros((mh, mw), dtype=torch.bool, device=device)
                    sky_masks.append(s_m)
                return sky_masks, disp_gt_s, mk_s

            if eff_warmup > 0:
                sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(half_w, half_h, half_imgs_gpu, half_images)
            elif eff_mid > 0:
                sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(mid_w, mid_h, mid_imgs_gpu, mid_images)
            else:
                sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(width, height, full_imgs_gpu, views.images)
        else:
            uv_all = rup_all = None
            dd_tensors = {}
            sky_stage = []
            disp_gt_stage = {}
            mk_stage = {}

        d_cam_buf = (
            _compute_nearest_cam_dist(params["means"].detach(), cam_centers_gpu)
            if is_full_run
            else None
        )

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        order = rng.permutation(num_views)
        order_pos = 0
        psnr_window = min(num_views, 40)
        psnr_start_step = max(0, steps - psnr_window)
        mse_accum = torch.zeros((), dtype=torch.float32, device=device)
        mse_count = 0
        inv_255 = 1.0 / 255.0
        gt_ssim_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        gt_ssim_cache_bytes = 0
        gt_ssim_max_bytes = 600 * 1024 * 1024
        t0 = time.perf_counter()

        for step in range(int(steps)):
            if step % 25 == 0 and cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")
            if eff_warmup > 0 and step == eff_warmup:
                if half_imgs_gpu is not None:
                    del half_imgs_gpu
                    half_imgs_gpu = None
                gt_ssim_cache.clear()
                gt_ssim_cache_bytes = 0
                if use_gpu_views:
                    if eff_mid > 0:
                        mid_imgs_gpu = torch.from_numpy(np.stack(mid_images, axis=0)).to(
                            device=device, dtype=torch.uint8
                        )
                    else:
                        full_imgs_gpu = torch.from_numpy(np.stack(views.images, axis=0)).to(
                            device=device, dtype=torch.uint8
                        )
                if need_stage_masks:
                    del sky_stage, disp_gt_stage, mk_stage
                    if eff_mid > 0:
                        sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(mid_w, mid_h, mid_imgs_gpu, mid_images)
                    else:
                        sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(width, height, full_imgs_gpu, views.images)
            if eff_mid > 0 and step == eff_warmup + eff_mid:
                if mid_imgs_gpu is not None:
                    del mid_imgs_gpu
                    mid_imgs_gpu = None
                gt_ssim_cache.clear()
                gt_ssim_cache_bytes = 0
                if use_gpu_views and full_imgs_gpu is None:
                    full_imgs_gpu = torch.from_numpy(np.stack(views.images, axis=0)).to(
                        device=device, dtype=torch.uint8
                    )
                if need_stage_masks:
                    del sky_stage, disp_gt_stage, mk_stage
                    sky_stage, disp_gt_stage, mk_stage = _build_stage_masks(width, height, full_imgs_gpu, views.images)

            if order_pos >= num_views:
                order = rng.permutation(num_views)
                order_pos = 0
            v_idx = int(order[order_pos])
            order_pos += 1

            t_frac = float(step) / float(max(1, steps - 1))
            current_means_lr = lr_means_init * ((lr_means_final / lr_means_init) ** t_frac)
            optimizers["means"].param_groups[0]["lr"] = current_means_lr

            active_sh = min(int(sh_degree), step // 1000)

            if step < eff_warmup:
                cur_w, cur_h = half_w, half_h
                cur_k = k_half_gpu
                if half_imgs_gpu is not None:
                    gt_hwc = half_imgs_gpu[v_idx].float() * inv_255
                else:
                    pin_half.copy_(torch.from_numpy(half_images[v_idx]))
                    gt_hwc = pin_half.to(device=device, non_blocking=is_cuda).float() * inv_255
            elif step < eff_warmup + eff_mid:
                cur_w, cur_h = mid_w, mid_h
                cur_k = k_mid_gpu
                if mid_imgs_gpu is not None:
                    gt_hwc = mid_imgs_gpu[v_idx].float() * inv_255
                else:
                    pin_mid.copy_(torch.from_numpy(mid_images[v_idx]))
                    gt_hwc = pin_mid.to(device=device, non_blocking=is_cuda).float() * inv_255
            else:
                cur_w, cur_h = width, height
                cur_k = k_gpu
                if full_imgs_gpu is not None:
                    gt_hwc = full_imgs_gpu[v_idx].float() * inv_255
                else:
                    pin_full.copy_(torch.from_numpy(views.images[v_idx]))
                    gt_hwc = pin_full.to(device=device, non_blocking=is_cuda).float() * inv_255

            if active_sh == 0:
                sh_coeffs = params["sh0"]
            elif active_sh < int(sh_degree):
                n_rest = (active_sh + 1) ** 2 - 1
                sh_coeffs = torch.cat([params["sh0"], params["shN"][:, :n_rest]], dim=1)
            else:
                sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)

            scales_act = torch.exp(params["scales"])
            opacities_act = torch.sigmoid(params["opacities"])

            need_ed = is_full_run and (step < int(refine_stop_iter)) and (step % 3 == 0)
            renders, alphas, info = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=scales_act,
                opacities=opacities_act,
                colors=sh_coeffs,
                viewmats=viewmats_gpu[v_idx : v_idx + 1],
                Ks=cur_k,
                width=cur_w,
                height=cur_h,
                sh_degree=active_sh,
                packed=packed,
                absgrad=True,
                near_plane=train_near_plane,
                far_plane=1e10,
                render_mode="RGB+ED" if need_ed else "RGB",
                rasterize_mode="antialiased",
            )

            fg_hwc = renders[0, :, :, :3] if need_ed else renders[0]
            ed_hw = renders[0, :, :, 3] if need_ed else None
            al_hw = alphas[0, :, :, 0]

            if sky_dome is not None and uv_all is not None:
                is_sky_low = sky_stage[v_idx]
                if is_sky_low.shape != (cur_h, cur_w):
                    is_sky = (
                        F.interpolate(is_sky_low[None, None].float(), size=(cur_h, cur_w), mode="nearest")[0, 0]
                        > 0.5
                    )
                else:
                    is_sky = is_sky_low
                sky_bg = sky_dome.eval_uv(uv_all[v_idx], cur_h, cur_w)
                grey_bg = float(rng.uniform(0.22, 0.58))
                bg_hwc = torch.where(is_sky.unsqueeze(-1), sky_bg, grey_bg)
                pred_hwc = fg_hwc + (1.0 - al_hw).unsqueeze(-1) * bg_hwc
            elif is_full_run:
                grey_bg = float(rng.uniform(0.22, 0.58))
                pred_hwc = fg_hwc + (1.0 - al_hw).unsqueeze(-1) * grey_bg
                is_sky_low = sky_stage[v_idx] if need_stage_masks else None
            else:
                pred_hwc = fg_hwc
                is_sky_low = None

            l1 = F.l1_loss(pred_hwc, gt_hwc)
            pred_chw = pred_hwc.permute(2, 0, 1).unsqueeze(0)
            gt_chw = gt_hwc.permute(2, 0, 1).unsqueeze(0)
            if ssim_half_res and cur_w >= 900:
                pred_ssim = F.avg_pool2d(pred_chw, kernel_size=2, stride=2)
                gt_ssim = F.avg_pool2d(gt_chw, kernel_size=2, stride=2)
            else:
                pred_ssim = pred_chw
                gt_ssim = gt_chw
            gt_stats = gt_ssim_cache.get(v_idx)
            if gt_stats is None:
                gt_stats = _precompute_gt_ssim_stats(gt_ssim)
                entry_bytes = sum(t.numel() * t.element_size() for t in gt_stats)
                if gt_ssim_cache_bytes + entry_bytes <= gt_ssim_max_bytes:
                    gt_ssim_cache[v_idx] = gt_stats
                    gt_ssim_cache_bytes += entry_bytes
            ssim_val = gaussian_ssim_11x11(pred_ssim, gt_ssim, gt_stats=gt_stats)

            s_max = scales_act.max(dim=-1).values
            s_min = scales_act.min(dim=-1).values
            if is_full_run and d_cam_buf is not None:
                aniso_reg = 0.008 * F.relu(s_max / s_min.clamp(min=1e-5) - 6.0).mean()
                allowed_s = torch.clamp(0.06 * d_cam_buf, min=0.018, max=max(0.65, 0.55 * float(scene_scale)))
                scale_reg = 0.015 * F.relu(s_max - allowed_s).mean()
                opa_reg = 0.0002 * opacities_act.mean()
                sh_reg = 0.001 * params["shN"].pow(2).mean() if params["shN"].numel() > 0 else 0.0
                extra_reg: torch.Tensor | float = opa_reg + sh_reg + scale_reg
                if sky_dome is not None:
                    extra_reg = extra_reg + 0.025 * (al_hw * is_sky.float()).mean() + 0.02 * sky_dome.smoothness_loss()
                if ed_hw is not None and is_sky_low is not None:
                    mh, mw = is_sky_low.shape
                    if (cur_h, cur_w) != (mh, mw):
                        ed_d = F.interpolate(ed_hw[None, None], size=(mh, mw), mode="bilinear", align_corners=False)[0, 0]
                        al_d = F.interpolate(al_hw[None, None], size=(mh, mw), mode="bilinear", align_corners=False)[0, 0]
                        gt_d = F.interpolate(gt_hwc.permute(2, 0, 1).unsqueeze(0), size=(mh, mw), mode="area")[0].permute(1, 2, 0)
                    else:
                        ed_d, al_d, gt_d = ed_hw, al_hw, gt_hwc
                    disp = 1.0 / ed_d.clamp(min=train_near_plane)
                    valid_fg = (~is_sky_low) & (al_d > 0.5)
                    dx_i = torch.abs(gt_d[:, 1:] - gt_d[:, :-1]).mean(dim=-1)
                    dy_i = torch.abs(gt_d[1:, :] - gt_d[:-1, :]).mean(dim=-1)
                    wx = torch.exp(-12.0 * dx_i) * (valid_fg[:, 1:] & valid_fg[:, :-1]).float()
                    wy = torch.exp(-12.0 * dy_i) * (valid_fg[1:, :] & valid_fg[:-1, :]).float()
                    tv_disp = 0.006 * (
                        (torch.abs(disp[:, 1:] - disp[:, :-1]) * wx).mean()
                        + (torch.abs(disp[1:, :] - disp[:-1, :]) * wy).mean()
                    )
                    extra_reg = extra_reg + tv_disp
                    if step < (eff_warmup + eff_mid) and (v_idx in disp_gt_stage):
                        mk_r = mk_stage[v_idx] & (al_d.detach() > 0.4)
                        if mk_r.any():
                            extra_reg = extra_reg + 0.02 * F.smooth_l1_loss(disp[mk_r], disp_gt_stage[v_idx][mk_r], beta=0.05)
                loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val) + aniso_reg + extra_reg
            else:
                aniso_reg = 0.005 * F.relu(s_max / s_min.clamp(min=1e-5) - 10.0).mean()
                loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val) + aniso_reg

            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()

            n_gs = params["means"].shape[0]
            if packed:
                visibility = torch.zeros(n_gs, dtype=torch.bool, device=device)
                g_ids = info.get("gaussian_ids")
                if g_ids is not None and g_ids.numel() > 0:
                    visibility[g_ids] = True
            else:
                visibility = (info["radii"] > 0).all(dim=-1).any(dim=0)

            for p_name, opt in optimizers.items():
                if params[p_name].grad is not None:
                    opt.step(visibility=visibility)
                    opt.zero_grad(set_to_none=True)
            if sky_opt is not None:
                sky_opt.step()
                sky_opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                params["scales"].data.clamp_(min=min_log_scale)

            if step < int(refine_stop_iter) and params["means"].shape[0] < int(cap_max):
                strategy.step_post_backward(
                    params, optimizers, strategy_state, step, info, packed=packed
                )
                if (
                    is_full_run
                    and int(steps) >= 600
                    and step > 0
                    and int(reset_every) > 0
                    and (step % int(reset_every) == 0)
                    and (step + pause_refine + int(refine_every)) < int(refine_stop_iter)
                ):
                    reset_opa(params=params, optimizers=optimizers, state=strategy_state, value=0.025)
                if is_full_run and (step % 100 == 0 or (d_cam_buf is not None and len(d_cam_buf) != params["means"].shape[0])):
                    d_cam_buf = _compute_nearest_cam_dist(params["means"].detach(), cam_centers_gpu)
                with torch.no_grad():
                    params["scales"].data.clamp_(min=min_log_scale)
            elif is_full_run and step % 200 == 0:
                d_cam_buf = _compute_nearest_cam_dist(params["means"].detach(), cam_centers_gpu)

            if step >= psnr_start_step:
                with torch.no_grad():
                    mse_accum += F.mse_loss(pred_hwc.detach().clamp(0.0, 1.0), gt_hwc)
                    mse_count += 1

            if progress_cb and (step % 200 == 0 or step == steps - 1):
                frac = 0.2 + 0.78 * ((step + 1) / max(1, steps))
                progress_cb(
                    frac,
                    f"3DGS training step {step + 1}/{steps} ({params['means'].shape[0]:,} splats)",
                )

        if is_cuda:
            torch.cuda.synchronize(device)
        train_s = time.perf_counter() - t0
        peak_vram_gb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
            if is_cuda
            else 0.0
        )

        if mse_count > 0:
            mean_mse = float(mse_accum.item()) / float(mse_count)
            train_psnr_db = 10.0 * math.log10(1.0 / max(mean_mse, 1e-10))
        else:
            with torch.no_grad():
                gt_hwc = torch.from_numpy(views.images[0]).to(device=device, dtype=torch.float32) * inv_255
                sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)
                renders, _, _ = rasterization(
                    means=params["means"],
                    quats=params["quats"],
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=sh_coeffs,
                    viewmats=viewmats_gpu[0:1],
                    Ks=k_gpu,
                    width=width,
                    height=height,
                    sh_degree=int(sh_degree),
                    packed=packed,
                    near_plane=train_near_plane,
                    far_plane=1e10,
                    render_mode="RGB",
                    rasterize_mode="antialiased",
                )
                train_psnr_db = compute_psnr_db(renders[0], gt_hwc)

        # Post-training floater/haze pruning
        if num_views >= 8 and params["means"].shape[0] > 1000:
            cam_centers = views.c2ws[:, :3, 3].astype(np.float32)
            cam_centroid = np.mean(cam_centers, axis=0)
            means_np = params["means"].detach().cpu().numpy()
            op_mask = (torch.sigmoid(params["opacities"]) >= 0.01).detach().cpu().numpy()
            dist_cent_mask = (
                np.linalg.norm(means_np - cam_centroid[None, :], axis=1)
                <= (40.0 * float(scene_scale))
            )
            cam_tree = cKDTree(cam_centers)
            d_cam, _ = cam_tree.query(means_np, k=1)
            min_cam_d = (train_near_plane * 0.9) if is_full_run else (0.05 * float(scene_scale))
            cam_near_mask = d_cam >= min_cam_d
            keep_np = op_mask & dist_cent_mask & cam_near_mask
            if int(keep_np.sum()) >= max(100, int(0.1 * len(keep_np))):
                keep_t = torch.from_numpy(keep_np).to(device=device)
                for p_name in ("means", "quats", "scales", "opacities", "sh0", "shN"):
                    params[p_name] = params[p_name].detach()[keep_t]

        store_dev = torch.device(device) if (keep_on_gpu and is_cuda) else torch.device("cpu")
        sky_logits_out = (
            sky_dome.logits.detach().to(store_dev).contiguous()
            if sky_dome is not None
            else None
        )
        scene = SplatScene(
            means=params["means"].detach().to(store_dev).contiguous(),
            quats=params["quats"].detach().to(store_dev).contiguous(),
            scales=params["scales"].detach().to(store_dev).contiguous(),
            opacities=params["opacities"].detach().to(store_dev).contiguous(),
            sh0=params["sh0"].detach().to(store_dev).contiguous(),
            shN=params["shN"].detach().to(store_dev).contiguous(),
            sh_degree=int(sh_degree),
            scene_scale=float(scene_scale),
            init_splats=init_splats,
            final_splats=int(params["means"].shape[0]),
            peak_vram_gb=peak_vram_gb,
            train_s=train_s,
            train_psnr_db=train_psnr_db,
            sky_logits=sky_logits_out,
        )

        del optimizers, strategy_state, params, viewmats_gpu, c2ws_gpu, cam_centers_gpu
        del k_gpu, k_half_gpu, k_mid_gpu, pin_full, pin_half, pin_mid
        del full_imgs_gpu, half_imgs_gpu, mid_imgs_gpu, gt_ssim_cache
        del sky_dome, sky_opt, uv_all, rup_all, dd_tensors, sky_stage, disp_gt_stage, mk_stage, d_cam_buf

        return scene
    finally:
        if is_cuda:
            torch.backends.cudnn.benchmark = prev_cudnn_bench
            torch.cuda.empty_cache()


@torch.no_grad()
def render_view(
    splat_scene: SplatScene,
    c2w: np.ndarray | CameraPose | torch.Tensor,
    K: np.ndarray | torch.Tensor,
    width: int,
    height: int,
    device: str = "cuda",
) -> np.ndarray:
    """Render an RGB uint8 image (H, W, 3) from a SplatScene and camera pose."""
    if not is_gsplat_available():
        raise RuntimeError("gsplat CUDA backend is not available")
    from gsplat import rasterization

    if isinstance(c2w, CameraPose):
        c2w_np, viewmat_np = _pose_to_c2w_and_viewmat(c2w)
        c2w_t = torch.from_numpy(c2w_np)
        viewmat_t = torch.from_numpy(viewmat_np)
    else:
        c2w_t = torch.as_tensor(c2w, dtype=torch.float32)
        r_wc = c2w_t[:3, :3]
        t_wc = c2w_t[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        viewmat_t = torch.eye(4, dtype=torch.float32)
        viewmat_t[:3, :3] = r_cw
        viewmat_t[:3, 3] = t_cw

    k_t = torch.as_tensor(K, dtype=torch.float32)
    dev_key = str(torch.device(device))
    cache = splat_scene._render_cache
    if cache is not None and cache[0] == dev_key and len(cache) == 7:
        _, means_d, quats_d, scales_d, opacities_d, sh_coeffs, sky_padded = cache
    else:
        means_d = splat_scene.means.to(device)
        quats_d = splat_scene.quats.to(device)
        scales_d = torch.exp(splat_scene.scales.to(device))
        opacities_d = torch.sigmoid(splat_scene.opacities.to(device))
        sh_coeffs = torch.cat(
            [splat_scene.sh0.to(device), splat_scene.shN.to(device)], dim=1
        )
        sky_padded = (
            _sky_padded_rgb_from_logits(splat_scene.sky_logits.to(device))
            if splat_scene.sky_logits is not None
            else None
        )
        splat_scene._render_cache = (
            dev_key,
            means_d,
            quats_d,
            scales_d,
            opacities_d,
            sh_coeffs,
            sky_padded,
        )

    inv_scale = min(float(splat_scene.scene_scale), 1.4)
    near_plane = (
        max(0.05, min(0.10, 0.08 * inv_scale))
        if sky_padded is not None
        else max(0.04, min(0.10, 0.08 * inv_scale))
    )

    k_dev = k_t.to(device)
    renders, alphas, _info = rasterization(
        means=means_d,
        quats=quats_d,
        scales=scales_d,
        opacities=opacities_d,
        colors=sh_coeffs,
        viewmats=viewmat_t.unsqueeze(0).to(device),
        Ks=k_dev.unsqueeze(0),
        width=int(width),
        height=int(height),
        sh_degree=int(splat_scene.sh_degree),
        packed=False,
        near_plane=near_plane,
        far_plane=1e10,
        render_mode="RGB",
        rasterize_mode="antialiased",
    )

    fg = renders[0]
    if sky_padded is not None:
        bg = _eval_sky_bg_from_padded(
            sky_padded, c2w_t.to(device), k_dev, int(width), int(height)
        )
        fg = fg + (1.0 - alphas[0, :, :, 0]).unsqueeze(-1) * bg

    rgb = fg.clamp(0.0, 1.0) * 255.0
    return np.ascontiguousarray(rgb.round().to(torch.uint8).cpu().numpy())


def export_splats_ply(splat_scene: SplatScene, output_path: str | Path) -> Path:
    """Export SplatScene to a standard binary little-endian 3DGS .ply file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    means = splat_scene.means.detach().cpu().numpy().astype(np.float32, copy=False)
    n_pts = means.shape[0]
    normals = np.zeros((n_pts, 3), dtype=np.float32)
    f_dc = splat_scene.sh0.detach().cpu().numpy().reshape(n_pts, 3).astype(np.float32, copy=False)

    sh_n = splat_scene.shN.detach().cpu().numpy().astype(np.float32, copy=False)
    if sh_n.shape[1] < 15:
        pad = np.zeros((n_pts, 15 - sh_n.shape[1], 3), dtype=np.float32)
        sh_n = np.concatenate([sh_n, pad], axis=1)
    elif sh_n.shape[1] > 15:
        sh_n = sh_n[:, :15, :]
    # Standard 3DGS PLY orders f_rest as (N, 3, K) flattened -> [R0..R14, G0..G14, B0..B14]
    f_rest = np.transpose(sh_n, (0, 2, 1)).reshape(n_pts, 45)

    opacities = splat_scene.opacities.detach().cpu().numpy().reshape(n_pts, 1).astype(np.float32, copy=False)
    scales = splat_scene.scales.detach().cpu().numpy().reshape(n_pts, 3).astype(np.float32, copy=False)

    quats = splat_scene.quats.detach().cpu().numpy().reshape(n_pts, 4).astype(np.float32, copy=False)
    q_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(q_norm > 1e-8, quats / np.maximum(q_norm, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    table = np.concatenate([means, normals, f_dc, f_rest, opacities, scales, quats], axis=1)
    finite_mask = np.isfinite(table).all(axis=1)
    table = np.ascontiguousarray(table[finite_mask], dtype="<f4")
    n_valid = int(table.shape[0])

    prop_names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_valid}",
        *[f"property float {p}" for p in prop_names],
        "end_header\n",
    ]
    header_bytes = "\n".join(header_lines).encode("ascii")

    with open(out, "wb") as f:
        f.write(header_bytes)
        if n_valid > 0:
            f.write(table.tobytes())

    return out
