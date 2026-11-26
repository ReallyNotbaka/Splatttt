from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
import torch
import torch.nn.functional as F
from src.video_ingest import VideoFrame


@dataclass
class CameraPose:
    r_wc: np.ndarray  # 3x3 camera-to-world rotation (COLMAP: +X right, +Y down, +Z forward)
    t_wc: np.ndarray  # 3D camera center in world space
    scale: float = 1.0
    shift: float = 0.0
    valid: bool = True


LAST_STAGE2_TELEMETRY: Dict[str, Any] = {}
LAST_STAGE2_SPARSE_POINTS: Dict[str, np.ndarray] = {}


def _build_sift_anchor_pairs(
    m: int,
    des_gpu: Optional[List[Optional[torch.Tensor]]] = None,
    device: str = "cpu",
) -> List[tuple[int, int]]:
    if m <= 165:
        return [(i, j) for i in range(m) for j in range(i + 1, m)]

    pairs: set[tuple[int, int]] = set()
    for i in range(m):
        for j in range(i + 1, min(m, i + 10)):
            pairs.add((i, j))
        for step in (16, 32, 64):
            if i + step < m:
                pairs.add((i, i + step))

    des_list = des_gpu or []
    d_dim = next((int(d.shape[1]) for d in des_list if d is not None), 128)
    k_sub = 256 if device == "cuda" else 128
    match_dtype = torch.float16 if device == "cuda" else torch.float32
    sub_bank = torch.zeros((m, k_sub, d_dim), dtype=match_dtype, device=device)
    valid_mask = torch.zeros((m, k_sub), dtype=torch.bool, device=device)

    for idx in range(m):
        d = des_list[idx] if idx < len(des_list) else None
        if d is not None and len(d) >= 8:
            n_d = len(d)
            if n_d > k_sub:
                pick = torch.linspace(0, n_d - 1, k_sub, device=device).long()
                sub = d[pick]
            else:
                sub = d
            cnt = len(sub)
            sub_bank[idx, :cnt] = sub
            valid_mask[idx, :cnt] = True

    with torch.inference_mode():
        for i in range(11, m):
            if not torch.any(valid_mask[i]):
                continue
            n_prev = i - 10
            prev_flat = sub_bank[:n_prev].reshape(n_prev * k_sub, d_dim)
            sim = torch.matmul(sub_bank[i], prev_flat.T).view(k_sub, n_prev, k_sub)
            sim.masked_fill_(~valid_mask[:n_prev].unsqueeze(0), -1.0)
            top2 = torch.topk(sim, k=2, dim=2).values
            d1 = (1.0 - top2[:, :, 0].float()).clamp(min=1e-6)
            d2 = (1.0 - top2[:, :, 1].float()).clamp(min=1e-6)
            good = (
                valid_mask[i].unsqueeze(1)
                & (top2[:, :, 0] > 0.72)
                & ((d1 / d2) < 0.75)
            )
            counts = good.sum(dim=0)
            k_pick = min(14, n_prev)
            top_c, top_j = torch.topk(counts, k=k_pick)
            for c_val, j_val in zip(top_c.tolist(), top_j.tolist()):
                if c_val >= 4:
                    pairs.add((int(j_val), i))

    del sub_bank, valid_mask
    return sorted(pairs)


def _project_colmap_point(
    cam: Any,
    xyz_c: np.ndarray,
    f_fallback: Optional[float] = None,
    cx_fallback: Optional[float] = None,
    cy_fallback: Optional[float] = None,
    k1_fallback: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    z_cam = float(xyz_c[2])
    if z_cam <= 1e-6:
        return None
    try:
        uv_proj = cam.img_from_cam(xyz_c)
        if uv_proj is not None:
            return float(uv_proj[0]), float(uv_proj[1])
    except Exception:
        pass
    try:
        uv_proj = cam.img_from_cam(np.array([xyz_c[0] / z_cam, xyz_c[1] / z_cam], dtype=np.float64))
        if uv_proj is not None:
            return float(uv_proj[0]), float(uv_proj[1])
    except Exception:
        pass
    params = getattr(cam, "params", None)
    f_val = f_fallback if f_fallback is not None else (float(params[0]) if params is not None and len(params) >= 1 else 1.0)
    cx_val = cx_fallback if cx_fallback is not None else (float(params[1]) if params is not None and len(params) >= 2 else 0.0)
    cy_val = cy_fallback if cy_fallback is not None else (float(params[2]) if params is not None and len(params) >= 3 else 0.0)
    k1_val = k1_fallback if k1_fallback is not None else (float(params[3]) if params is not None and len(params) >= 4 else 0.0)
    x_n = float(xyz_c[0]) / z_cam
    y_n = float(xyz_c[1]) / z_cam
    r2 = x_n * x_n + y_n * y_n
    dist_val = 1.0 + k1_val * r2
    return f_val * (x_n * dist_val) + cx_val, f_val * (y_n * dist_val) + cy_val


def _count_good_reg_images(
    rec: Optional[pycolmap.Reconstruction],
    min_anchors: int,
    max_med_reproj: float = 2.5,
) -> int:
    if rec is None or rec.num_reg_images() == 0 or len(rec.cameras) == 0:
        return 0
    cam = next(iter(rec.cameras.values()))
    cam_model = str(getattr(cam, "model", ""))
    cam_params = getattr(cam, "params", None)
    fast_radial = (
        "SIMPLE_RADIAL" in cam_model
        and cam_params is not None
        and len(cam_params) == 4
    )
    if fast_radial:
        f_cam, cx_cam, cy_cam, k1_cam = (
            float(cam_params[0]),
            float(cam_params[1]),
            float(cam_params[2]),
            float(cam_params[3]),
        )

    good_imgs = 0
    pts3d = rec.points3D
    for im in rec.images.values():
        if not getattr(im, "has_pose", True):
            continue
        try:
            cfw = im.cam_from_world()
        except Exception:
            continue
        r_cw = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(cfw.translation, dtype=np.float64)

        if fast_radial:
            xyz_list = []
            xy_list = []
            for p2d in im.points2D:
                if p2d.has_point3D() and p2d.point3D_id in pts3d:
                    xyz_list.append(pts3d[p2d.point3D_id].xyz)
                    xy_list.append(p2d.xy)
            if len(xyz_list) < min_anchors:
                continue
            xyz_w = np.asarray(xyz_list, dtype=np.float64)
            xy_obs = np.asarray(xy_list, dtype=np.float64)
            xyz_c = xyz_w @ r_cw.T + t_cw[None, :]
            z_ok = xyz_c[:, 2] > 1e-4
            if int(np.sum(z_ok)) < min_anchors:
                continue
            zc = xyz_c[z_ok, 2]
            un = xyz_c[z_ok, 0] / zc
            vn = xyz_c[z_ok, 1] / zc
            radial = 1.0 + k1_cam * (un * un + vn * vn)
            up = f_cam * un * radial + cx_cam
            vp = f_cam * vn * radial + cy_cam
            errs = np.hypot(up - xy_obs[z_ok, 0], vp - xy_obs[z_ok, 1])
            if float(np.median(errs)) <= max_med_reproj:
                good_imgs += 1
        else:
            reprojs = []
            for p2d in im.points2D:
                if p2d.has_point3D() and p2d.point3D_id in pts3d:
                    p3 = pts3d[p2d.point3D_id]
                    xyz_c = r_cw @ np.asarray(p3.xyz, dtype=np.float64) + t_cw
                    if xyz_c[2] > 1e-4:
                        uv_proj = _project_colmap_point(cam, xyz_c)
                        if uv_proj is not None:
                            reprojs.append(
                                float(np.hypot(uv_proj[0] - p2d.xy[0], uv_proj[1] - p2d.xy[1]))
                            )
            if len(reprojs) >= min_anchors and float(np.median(reprojs)) <= max_med_reproj:
                good_imgs += 1
    return good_imgs


def _trajectory_jump_stats(rec: Optional[pycolmap.Reconstruction]) -> tuple[int, float]:
    """Return (num_teleport_jumps, max_to_median_step_ratio) across time-ordered registered anchors."""
    if rec is None or rec.num_reg_images() < 6:
        return 0, 1.0
    ordered = []
    for im in rec.images.values():
        if not getattr(im, "has_pose", True):
            continue
        try:
            cfw = im.cam_from_world()
            idx = int(Path(im.name).stem)
        except Exception:
            continue
        r_cw = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(cfw.translation, dtype=np.float64)
        t_wc = -r_cw.T @ t_cw
        ordered.append((idx, t_wc))
    if len(ordered) < 6:
        return 0, 1.0
    ordered.sort(key=lambda x: x[0])
    steps = [
        float(np.linalg.norm(ordered[k + 1][1] - ordered[k][1]))
        for k in range(len(ordered) - 1)
    ]
    med_step = float(np.median(steps))
    if med_step < 1e-6:
        return 0, 1.0
    ratios = [s / med_step for s in steps]
    n_jumps = sum(1 for r in ratios if r > 4.0)
    return n_jumps, float(max(ratios))


def _run_gpu_matched_colmap_sfm(
    anchor_frames: List[VideoFrame],
    device: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> tuple[Optional[pycolmap.Reconstruction], Dict[str, Any]]:
    m = len(anchor_frames)
    h, w = anchor_frames[0].rgb.shape[:2]
    meta: Dict[str, Any] = {}

    try:
        pycolmap.logging.minloglevel = pycolmap.logging.ERROR
    except Exception:
        pass

    if progress_cb:
        progress_cb(0.32, f"extracting RootSIFT for {m} anchor keyframes...")

    t0_sift = time.perf_counter()

    def _ext_one(fr: VideoFrame):
        sift = cv2.SIFT_create(nfeatures=2048)
        gray = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY)
        kps, des = sift.detectAndCompute(gray, None)
        if not kps or des is None or len(kps) < 8:
            return np.zeros((0, 2), dtype=np.float32), None
        xy = np.float32([kp.pt for kp in kps])
        des_l1 = des / (np.sum(des, axis=1, keepdims=True) + 1e-7)
        rsift = np.sqrt(des_l1).astype(np.float32)
        return xy, rsift

    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as pool:
        ext_res = list(pool.map(_ext_one, anchor_frames))

    meta["sift_extract_s"] = time.perf_counter() - t0_sift

    if cancel_cb and cancel_cb():
        raise RuntimeError("build cancelled by user")

    kps_list = [r[0] for r in ext_res]
    match_dtype = torch.float16 if device == "cuda" else torch.float32
    des_gpu: List[Optional[torch.Tensor]] = []
    for _, rs in ext_res:
        if rs is not None and len(rs) >= 8:
            t = torch.from_numpy(rs).to(device=device, dtype=match_dtype)
            des_gpu.append(F.normalize(t, dim=-1))
        else:
            des_gpu.append(None)

    pair_list = _build_sift_anchor_pairs(m, des_gpu, device)
    meta["pairs_tested"] = len(pair_list)

    if progress_cb:
        progress_cb(0.36, f"GPU matching {len(pair_list):,} anchor pairs on {device}...")

    t0_match = time.perf_counter()
    gpu_matches: Dict[tuple[int, int], np.ndarray] = {}
    ratio_lim = 0.64 if m >= 24 else 0.68
    min_matches = 18 if m >= 24 else 14

    with torch.inference_mode():
        valid_lens = [len(d) for d in des_gpu if d is not None]
        if device == "cuda" and valid_lens and len(pair_list) > 0:
            max_k = max(valid_lens)
            des_pad = torch.zeros((m, max_k, 128), device=device, dtype=match_dtype)
            lengths = torch.zeros((m,), device=device, dtype=torch.long)
            for idx_m, d_t in enumerate(des_gpu):
                if d_t is not None:
                    des_pad[idx_m, : len(d_t)] = d_t
                    lengths[idx_m] = len(d_t)

            batch_sz = 64
            pairs_arr = np.asarray(pair_list, dtype=np.int64)
            col_ar = torch.arange(max_k, device=device).view(1, 1, -1)
            row_ar = torch.arange(max_k, device=device).view(1, -1)
            for s0 in range(0, len(pairs_arr), batch_sz):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("build cancelled by user")
                chunk = pairs_arr[s0 : s0 + batch_sz]
                ia_t = torch.from_numpy(chunk[:, 0]).to(device)
                ib_t = torch.from_numpy(chunk[:, 1]).to(device)
                da_b = des_pad[ia_t]
                db_b = des_pad[ib_t]
                la = lengths[ia_t]
                lb = lengths[ib_t]
                sim = torch.bmm(da_b, db_b.transpose(1, 2))
                sim = sim.masked_fill(~(col_ar < lb.view(-1, 1, 1)), -2.0)
                top2_v, top2_i = torch.topk(sim, k=2, dim=2)
                d1 = (1.0 - top2_v[:, :, 0].float()).clamp(min=1e-6)
                d2 = (1.0 - top2_v[:, :, 1].float()).clamp(min=1e-6)
                best_a_for_b = torch.argmax(sim, dim=1)
                matched_a = torch.gather(best_a_for_b, 1, top2_i[:, :, 0])
                valid_row = (row_ar < la.view(-1, 1)) & (la.view(-1, 1) > 0) & (lb.view(-1, 1) > 0)
                mask = valid_row & ((d1 / d2) < ratio_lim) & (matched_a == row_ar)
                counts = mask.sum(dim=1).cpu().tolist()
                good_rows = [r for r, c in enumerate(counts) if c >= min_matches]
                if good_rows:
                    gr_t = torch.tensor(good_rows, device=device)
                    nz = mask[gr_t].nonzero(as_tuple=False)
                    kb_ids = top2_i[gr_t, :, 0][nz[:, 0], nz[:, 1]].to(torch.uint32)
                    stacked = torch.stack([nz[:, 1].to(torch.uint32), kb_ids], dim=1).cpu().numpy()
                    splits = np.cumsum([counts[r] for r in good_rows[:-1]])
                    parts = np.split(stacked, splits) if len(good_rows) > 1 else [stacked]
                    for r_i, m_arr in zip(good_rows, parts):
                        gpu_matches[(int(chunk[r_i, 0]), int(chunk[r_i, 1]))] = m_arr
            del des_pad, lengths, col_ar, row_ar
        else:
            for p_idx, (a, b) in enumerate(pair_list):
                if p_idx % 200 == 0 and cancel_cb and cancel_cb():
                    raise RuntimeError("build cancelled by user")
                da = des_gpu[a]
                db = des_gpu[b]
                if da is None or db is None:
                    continue
                sim = torch.matmul(da, db.T)
                top2_v, top2_i = torch.topk(sim, k=2, dim=1)
                d1 = (1.0 - top2_v[:, 0].float()).clamp(min=1e-6)
                d2 = (1.0 - top2_v[:, 1].float()).clamp(min=1e-6)
                best_a_for_b = torch.argmax(sim, dim=0)
                ar = torch.arange(len(da), device=device)
                mutual = best_a_for_b[top2_i[:, 0]] == ar
                keep = mutual & ((d1 / d2) < ratio_lim)
                idx_a = ar[keep]
                if len(idx_a) >= min_matches:
                    idx_b = top2_i[keep, 0]
                    gpu_matches[(a, b)] = (
                        torch.stack([idx_a, idx_b], dim=-1).cpu().numpy().astype(np.uint32)
                    )

    if device == "cuda":
        torch.cuda.synchronize()
    meta["gpu_match_s"] = time.perf_counter() - t0_match
    meta["pairs_written"] = len(gpu_matches)

    del des_gpu
    if device == "cuda":
        torch.cuda.empty_cache()

    if cancel_cb and cancel_cb():
        raise RuntimeError("build cancelled by user")

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        img_dir = td_p / "images"
        img_dir.mkdir()
        for k, fr in enumerate(anchor_frames):
            cv2.imwrite(str(img_dir / f"{k:04d}.jpg"), cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR))

        db_path = td_p / "db.db"
        pairs_path = td_p / "pairs.txt"

        t0_db = time.perf_counter()
        db = pycolmap.Database.open(str(db_path))
        cam = pycolmap.Camera(
            model="SIMPLE_RADIAL",
            width=w,
            height=h,
            params=[0.92 * max(w, h), w * 0.5, h * 0.5, 0.0],
        )
        cam.has_prior_focal_length = True

        with pycolmap.DatabaseTransaction(db):
            cam_id = db.write_camera(cam)
            img_ids: Dict[int, int] = {}
            for k in range(m):
                im = pycolmap.Image(name=f"{k:04d}.jpg", camera_id=cam_id)
                iid = db.write_image(im)
                img_ids[k] = iid
                db.write_keypoints(iid, kps_list[k])

            with open(pairs_path, "w", encoding="utf-8") as pf:
                for (a, b), m_arr in gpu_matches.items():
                    db.write_matches(img_ids[a], img_ids[b], m_arr)
                    pf.write(f"{a:04d}.jpg {b:04d}.jpg\n")
        meta["db_write_s"] = time.perf_counter() - t0_db
        db.close()

        if progress_cb:
            progress_cb(0.42, f"verifying {len(gpu_matches):,} GPU pairs & running GLOMAP ({m} anchors)...")

        t0_ver = time.perf_counter()
        try:
            tvo = pycolmap.TwoViewGeometryOptions()
            tvo.ransac.max_num_trials = 2000
            if hasattr(tvo.ransac, "random_seed"):
                tvo.ransac.random_seed = 42
            pycolmap.verify_matches(db_path, pairs_path, options=tvo)
        except Exception:
            pycolmap.verify_matches(db_path, pairs_path)
        meta["verify_matches_s"] = time.perf_counter() - t0_ver

        best_rec: Optional[pycolmap.Reconstruction] = None
        min_obs_check = 18 if m >= 20 else 8
        glomap_good = 0
        glomap_jumps, glomap_max_ratio = 0, 1.0

        if m > 165:
            out_global = td_p / "sparse_global"
            out_global.mkdir()
            t0_glob = time.perf_counter()
            gpo = pycolmap.GlobalPipelineOptions()
            if hasattr(gpo, "random_seed"):
                gpo.random_seed = 42
            try:
                maps_g = pycolmap.global_mapping(db_path, img_dir, out_global, options=gpo)
                if maps_g:
                    best_rec = max(maps_g.values(), key=lambda r: (r.num_reg_images(), r.num_points3D()))
            except Exception:
                try:
                    maps_g = pycolmap.global_mapping(db_path, img_dir, out_global)
                    if maps_g:
                        best_rec = max(maps_g.values(), key=lambda r: (r.num_reg_images(), r.num_points3D()))
                except Exception:
                    best_rec = None
            meta["global_mapping_s"] = time.perf_counter() - t0_glob
            meta["glomap_reg_anchors"] = best_rec.num_reg_images() if best_rec else 0

            glomap_good = _count_good_reg_images(best_rec, min_obs_check) if best_rec is not None else 0
            meta["glomap_good_anchors"] = glomap_good
            glomap_jumps, glomap_max_ratio = _trajectory_jump_stats(best_rec)
            meta["glomap_jumps"] = glomap_jumps
            meta["glomap_max_step_ratio"] = glomap_max_ratio
        else:
            meta["global_mapping_s"] = 0.0
            meta["glomap_reg_anchors"] = 0
            meta["glomap_good_anchors"] = 0
            meta["glomap_jumps"] = 0
            meta["glomap_max_step_ratio"] = 1.0

        meta["incremental_triggered"] = False
        meta["incremental_s"] = 0.0
        if (
            best_rec is None
            or best_rec.num_reg_images() < max(3, int(0.75 * m))
            or glomap_good < max(3, int(0.75 * m))
            or glomap_jumps >= 2
            or glomap_max_ratio > 6.5
        ):
            meta["incremental_triggered"] = True
            t0_inc = time.perf_counter()
            out_inc = td_p / "sparse_inc"
            out_inc.mkdir()
            inc_options = None
            if hasattr(pycolmap, "IncrementalPipelineOptions"):
                try:
                    _opt = pycolmap.IncrementalPipelineOptions()
                    if hasattr(_opt, "random_seed"):
                        _opt.random_seed = 42
                    if hasattr(_opt, "ba_global_frames_ratio"):
                        _opt.ba_global_frames_ratio = 1.2
                    if hasattr(_opt, "ba_global_max_num_iterations"):
                        _opt.ba_global_max_num_iterations = 25
                    if hasattr(_opt, "mapper") and hasattr(_opt.mapper, "ba_global_ignore_redundant_points3D"):
                        _opt.mapper.ba_global_ignore_redundant_points3D = True
                    inc_options = _opt
                except Exception:
                    inc_options = None
            try:
                if inc_options is not None:
                    maps_i = pycolmap.incremental_mapping(
                        db_path, img_dir, out_inc, options=inc_options
                    )
                else:
                    maps_i = pycolmap.incremental_mapping(db_path, img_dir, out_inc)
                if maps_i:
                    inc_best = max(
                        maps_i.values(),
                        key=lambda r: (
                            _count_good_reg_images(r, min_obs_check),
                            r.num_reg_images(),
                            r.num_points3D(),
                        ),
                    )
                    inc_good = _count_good_reg_images(inc_best, min_obs_check)
                    meta["inc_good_anchors"] = inc_good
                    inc_jumps, inc_max_ratio = _trajectory_jump_stats(inc_best)
                    meta["inc_jumps"] = inc_jumps
                    meta["inc_max_step_ratio"] = inc_max_ratio
                    inc_min_ok = inc_good >= max(3, int(0.70 * m))
                    glomap_teleported = (glomap_jumps >= 2 or glomap_max_ratio > 6.5)
                    if (
                        best_rec is None
                        or (glomap_teleported and inc_min_ok and (inc_jumps, inc_max_ratio) < (glomap_jumps, glomap_max_ratio))
                        or (inc_good, inc_best.num_reg_images(), inc_best.num_points3D()) > (glomap_good, best_rec.num_reg_images(), best_rec.num_points3D())
                    ):
                        best_rec = inc_best
            except Exception:
                pass
            meta["incremental_s"] = time.perf_counter() - t0_inc

        if best_rec is not None and best_rec.num_reg_images() >= 3:
            return best_rec, meta

    return None, meta


def _localize_non_anchor_frames_chained(
    frames: List[VideoFrame],
    reg_anchors: List[int],
    anchor_world_obs: Dict[int, tuple[np.ndarray, np.ndarray]],  # anchor_idx -> (uv_2d [K,2], xyz_w [K,3])
    raw_poses: List[Optional[CameraPose]],
    sparse_obs: Dict[int, tuple[np.ndarray, np.ndarray]],
    k_mat: np.ndarray,
    k1: float,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> int:
    n = len(frames)
    if not reg_anchors:
        return 0

    non_anchors_count = sum(1 for i in range(n) if raw_poses[i] is None)
    if non_anchors_count == 0:
        return 0

    if progress_cb:
        progress_cb(0.48, f"localizing {non_anchors_count:,} frames via chained bidirectional 3D-PnP...")

    k64 = k_mat.astype(np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    grays = [cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY) for fr in frames]

    if len(reg_anchors) >= 2:
        consec_steps = [
            float(np.linalg.norm(raw_poses[reg_anchors[k + 1]].t_wc - raw_poses[reg_anchors[k]].t_wc))
            for k in range(len(reg_anchors) - 1)
        ]
        med_anchor_step = float(np.median(consec_steps)) if consec_steps else 0.10
        avg_frame_gap = max(1.0, float(np.median(np.diff(reg_anchors))))
        med_per_frame_step = med_anchor_step / avg_frame_gap
    else:
        med_anchor_step = 0.10
        med_per_frame_step = 0.02
    frame_step_cap = max(med_per_frame_step * 6.0, med_anchor_step * 0.45, 1e-3)

    def _step_pnp(
        curr_idx: int,
        next_idx: int,
        p_curr: np.ndarray,
        pts_3d: np.ndarray,
        prev_t_wc: np.ndarray,
        prev_r_wc: np.ndarray,
    ):
        p0 = p_curr.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(grays[curr_idx], grays[next_idx], p0, None, winSize=(21, 21), maxLevel=3)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(grays[next_idx], grays[curr_idx], p1, None, winSize=(21, 21), maxLevel=3)
        fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.8)
        if np.sum(good) < 12:
            return None

        pts_3d_good = pts_3d[good].astype(np.float64)
        p_good = p1.reshape(-1, 2)[good].astype(np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_good,
            p_good,
            k64,
            dist,
            iterationsCount=120,
            reprojectionError=6.0,
            confidence=0.995,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 12:
            return None

        inl = inliers.ravel()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(pts_3d_good[inl], p_good[inl], k64, dist, rvec, tvec)
        except Exception:
            pass

        r_cw, _ = cv2.Rodrigues(rvec)
        t_cw = tvec.reshape(3)
        r_wc = r_cw.T
        t_wc = -r_wc @ t_cw

        step_dist = float(np.linalg.norm(t_wc - prev_t_wc))
        if step_dist > frame_step_cap:
            return None

        # Per-frame rotation gate (blocks wild rotational flips > 22 deg)
        rel_r = r_wc.T @ prev_r_wc
        cos_angle = float(np.clip((np.trace(rel_r) - 1.0) * 0.5, -1.0, 1.0))
        rot_deg = float(np.degrees(np.arccos(cos_angle)))
        if rot_deg > 22.0:
            return None

        z_cam = (pts_3d_good[inl] @ r_cw.T + t_cw[None, :])[:, 2]
        ok_z = z_cam > 0.05
        if np.sum(ok_z) < 10:
            return None

        pts_2d_inl = p_good[inl][ok_z].astype(np.float32)
        z_inl = z_cam[ok_z].astype(np.float32)
        pts_3d_inl = pts_3d_good[inl][ok_z]

        return CameraPose(r_wc=r_wc, t_wc=t_wc, valid=True), pts_2d_inl, z_inl, pts_3d_inl

    # Define chains forward and backward from each registered anchor
    chains = []
    m_reg = len(reg_anchors)
    for idx_k, a in enumerate(reg_anchors):
        # Forward chain
        fwd_target = (reg_anchors[idx_k + 1] + a) // 2 if idx_k < m_reg - 1 else n - 1
        if fwd_target > a:
            chains.append((a, fwd_target, +1))
        # Backward chain
        bwd_target = (reg_anchors[idx_k - 1] + a) // 2 + 1 if idx_k > 0 else 0
        if bwd_target < a:
            chains.append((a, bwd_target, -1))

    chain_results: Dict[int, tuple[CameraPose, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _execute_chain(args):
        a_start, target, direction = args
        uv_a, xyz_a = anchor_world_obs[a_start]
        if len(uv_a) < 12:
            return []
        p_curr = uv_a.copy()
        pts_3d = xyz_a.copy()
        prev_t = raw_poses[a_start].t_wc.copy()
        prev_r = raw_poses[a_start].r_wc.copy()

        curr = a_start
        local_res = []
        while curr != target:
            if cancel_cb and cancel_cb():
                break
            nxt = curr + direction
            step_res = _step_pnp(curr, nxt, p_curr, pts_3d, prev_t, prev_r)
            if step_res is None:
                break
            pose_nxt, p_next_2d, z_next, pts_3d_next = step_res
            local_res.append((nxt, pose_nxt, p_next_2d, z_next, pts_3d_next))
            p_curr = p_next_2d
            pts_3d = pts_3d_next
            prev_t = pose_nxt.t_wc.copy()
            prev_r = pose_nxt.r_wc.copy()
            curr = nxt
        return local_res

    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as pool:
        for res_list in pool.map(_execute_chain, chains):
            for nxt, pose_nxt, p_2d, z_val, pts_3d_val in res_list:
                if raw_poses[nxt] is None and nxt not in chain_results:
                    chain_results[nxt] = (pose_nxt, p_2d, z_val, pts_3d_val)

    for f_idx, (pose_f, p_2d, z_val, _) in chain_results.items():
        raw_poses[f_idx] = pose_f
        sparse_obs[f_idx] = (p_2d, z_val)

    # Pass 2: Quick bidirectional extension from neighboring registered frames into remaining gaps
    for sweep_order in (range(n), range(n - 1, -1, -1)):
        for f in sweep_order:
            if raw_poses[f] is not None:
                continue
            if cancel_cb and cancel_cb():
                break
            cands = []
            if f > 0 and raw_poses[f - 1] is not None and ((f - 1) in chain_results or (f - 1) in anchor_world_obs):
                cands.append(f - 1)
            if f < n - 1 and raw_poses[f + 1] is not None and ((f + 1) in chain_results or (f + 1) in anchor_world_obs):
                cands.append(f + 1)

            for neighbor in cands:
                if neighbor in chain_results:
                    _, p_curr, _, pts_3d = chain_results[neighbor]
                else:
                    p_curr, pts_3d = anchor_world_obs[neighbor]
                prev_t = raw_poses[neighbor].t_wc
                prev_r = raw_poses[neighbor].r_wc
                res = _step_pnp(neighbor, f, p_curr, pts_3d, prev_t, prev_r)
                if res is not None:
                    pose_f, p_2d, z_val, pts_3d_val = res
                    raw_poses[f] = pose_f
                    sparse_obs[f] = (p_2d, z_val)
                    chain_results[f] = (pose_f, p_2d, z_val, pts_3d_val)
                    break

    # Pass 3: SIFT / Keypoint + PnP Rescue Pass for frames still unlocalized after optical flow
    unreg_pass3 = [i for i in range(n) if raw_poses[i] is None]
    if unreg_pass3 and len(reg_anchors) >= 2:
        sift_rescuer = cv2.SIFT_create(nfeatures=1200)
        anchor_des_cache = {}
        for a_idx in reg_anchors:
            if a_idx in anchor_world_obs:
                uv_a, xyz_a = anchor_world_obs[a_idx]
                if len(uv_a) >= 12:
                    kps_a, des_a = sift_rescuer.detectAndCompute(grays[a_idx], None)
                    if des_a is not None and len(kps_a) >= 12:
                        xy_a = np.float32([kp.pt for kp in kps_a])
                        tree = cKDTree(uv_a)
                        dists_kd, idxs_kd = tree.query(xy_a, distance_upper_bound=4.0)
                        valid_kd = dists_kd < 4.0
                        if np.sum(valid_kd) >= 12:
                            anchor_des_cache[a_idx] = (
                                des_a[valid_kd],
                                xyz_a[idxs_kd[valid_kd]],
                            )

        consec_steps = [
            float(np.linalg.norm(raw_poses[reg_anchors[k + 1]].t_wc - raw_poses[reg_anchors[k]].t_wc))
            for k in range(len(reg_anchors) - 1)
        ]
        med_anchor_step = float(np.median(consec_steps)) if consec_steps else 0.10
        max_d_near = max(2.5 * med_anchor_step, 0.40)

        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        for f in unreg_pass3:
            if cancel_cb and cancel_cb():
                break
            near_anchors = sorted(anchor_des_cache.keys(), key=lambda a: abs(a - f))[:2]
            if not near_anchors:
                continue

            kps_f, des_f = sift_rescuer.detectAndCompute(grays[f], None)
            if des_f is None or len(kps_f) < 12:
                continue

            xy_f = np.float32([kp.pt for kp in kps_f])
            all_match_2d = []
            all_match_3d = []

            for a_near in near_anchors:
                des_a, xyz_a = anchor_des_cache[a_near]
                matches = bf_matcher.knnMatch(des_f, des_a, k=2)
                for m_pair in matches:
                    if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                        m = m_pair[0]
                        all_match_2d.append(xy_f[m.queryIdx])
                        all_match_3d.append(xyz_a[m.trainIdx])

            if len(all_match_2d) >= 12:
                pts_2d_m = np.array(all_match_2d, dtype=np.float64)
                pts_3d_m = np.array(all_match_3d, dtype=np.float64)
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_m, pts_2d_m, k64, dist,
                    iterationsCount=150, reprojectionError=6.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP
                )
                if ok and inliers is not None and len(inliers) >= 12:
                    inl = inliers.ravel()
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(pts_3d_m[inl], pts_2d_m[inl], k64, dist, rvec, tvec)
                    except Exception:
                        pass
                    r_cw, _ = cv2.Rodrigues(rvec)
                    t_cw = tvec.reshape(3)
                    r_wc = r_cw.T
                    t_wc = -r_wc @ t_cw

                    a_nearest = near_anchors[0]
                    d_near = float(np.linalg.norm(t_wc - raw_poses[a_nearest].t_wc))
                    rel_r = r_wc.T @ raw_poses[a_nearest].r_wc
                    cos_ang = float(np.clip((np.trace(rel_r) - 1.0) * 0.5, -1.0, 1.0))
                    rot_deg = float(np.degrees(np.arccos(cos_ang)))
                    z_cam_p3 = (pts_3d_m[inl] @ r_cw.T + t_cw[None, :])[:, 2]
                    ok_z_p3 = z_cam_p3 > 0.05
                    if d_near <= max_d_near and rot_deg <= 45.0 and int(np.sum(ok_z_p3)) >= 10:
                        raw_poses[f] = CameraPose(r_wc=r_wc, t_wc=t_wc, valid=True)
                        sparse_obs[f] = (
                            pts_2d_m[inl][ok_z_p3].astype(np.float32),
                            z_cam_p3[ok_z_p3].astype(np.float32),
                        )

    localized_count = sum(1 for p in raw_poses if p is not None)
    return localized_count


def _compute_gravity_level_rotation(r_wcs: List[np.ndarray]) -> np.ndarray:
    if len(r_wcs) < 3:
        return np.eye(3, dtype=np.float64)

    cam_ups = np.array([r @ np.array([0.0, -1.0, 0.0]) for r in r_wcs], dtype=np.float64)
    mean_up = np.mean(cam_ups, axis=0)
    norm = float(np.linalg.norm(mean_up))
    if norm < 1e-5:
        return np.eye(3, dtype=np.float64)
    mean_up /= norm

    # Frame-invariant yaw coverage: project viewing headings perpendicular to mean_up
    forwards = np.array([r[:, 2] for r in r_wcs], dtype=np.float64)
    proj = forwards - np.outer(forwards @ mean_up, mean_up)
    proj_norms = np.linalg.norm(proj, axis=1)
    valid_p = proj_norms > 1e-4
    if int(np.sum(valid_p)) < 3:
        circ_var = 0.0
    else:
        proj_unit = proj[valid_p] / proj_norms[valid_p, None]
        circ_var = 1.0 - float(np.linalg.norm(np.mean(proj_unit, axis=0)))

    target_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(mean_up, target_up), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))

    # If camera didn't do a full yaw rotation (< 0.28) and tilt angle > 12 degrees,
    # camera was pitched downward during a forward walkthrough. Do not slant the scene into a ramp!
    if circ_var < 0.28 and angle_deg > 12.0:
        return np.eye(3, dtype=np.float64)

    if dot > 0.9995:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9995:
        return np.diag([1.0, -1.0, -1.0])

    v = np.cross(mean_up, target_up)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - dot) / (s * s))


def filter_trajectory_spikes(
    raw_poses: List[Optional[CameraPose]],
    clean_indices: List[int],
    anchor_valid_indices: List[int],
    med_anchor_step: float,
    avg_frame_gap: float = 1.0,
    min_anchor_thresh: float = 0.0,
) -> tuple[List[int], List[int], int, int]:
    def _rot_deg(r1: np.ndarray, r2: np.ndarray) -> float:
        c = float(np.clip((np.trace(r1.T @ r2) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    anchor_spikes_removed = 0
    pnp_spikes_removed = 0

    # 1. Local Anchor Spike Guard (chord-based interior check + safe endpoint check; prevents endpoint cascade)
    anchor_valid_indices = sorted(
        [i for i in anchor_valid_indices if i in clean_indices and raw_poses[i] is not None]
    )
    if len(anchor_valid_indices) >= 3:
        a_pts_init = np.array([raw_poses[a].t_wc for a in anchor_valid_indices], dtype=np.float64)
        consec_a = np.linalg.norm(np.diff(a_pts_init, axis=0), axis=1)
        p75_step = float(np.percentile(consec_a, 75)) if len(consec_a) > 0 else med_anchor_step
        base_thresh = max(2.8 * p75_step, 2.8 * med_anchor_step, min_anchor_thresh)
        max_removals = max(2, len(anchor_valid_indices) // 4)

        while len(anchor_valid_indices) >= 3 and anchor_spikes_removed < max_removals:
            worst_ratio = 1.0
            worst_idx = -1
            worst_pos = -1

            m_cur = len(anchor_valid_indices)
            for idx_k in range(m_cur):
                a = anchor_valid_indices[idx_k]
                ta = raw_poses[a].t_wc
                if idx_k == 0:
                    a_nbr = anchor_valid_indices[1]
                    d = float(np.linalg.norm(ta - raw_poses[a_nbr].t_wc))
                    lim = max(base_thresh * 1.25, 6.0 * p75_step)
                    ratio = d / max(lim, 1e-6)
                elif idx_k == m_cur - 1:
                    a_nbr = anchor_valid_indices[m_cur - 2]
                    d = float(np.linalg.norm(ta - raw_poses[a_nbr].t_wc))
                    lim = max(base_thresh * 1.25, 6.0 * p75_step)
                    ratio = d / max(lim, 1e-6)
                else:
                    a_l = anchor_valid_indices[idx_k - 1]
                    a_r = anchor_valid_indices[idx_k + 1]
                    alpha = float(a - a_l) / float(a_r - a_l)
                    p_chord = (1.0 - alpha) * raw_poses[a_l].t_wc + alpha * raw_poses[a_r].t_wc
                    local_chord = float(np.linalg.norm(raw_poses[a_r].t_wc - raw_poses[a_l].t_wc))
                    d = float(np.linalg.norm(ta - p_chord))
                    lim = max(base_thresh, 1.25 * local_chord)

                    ratio = d / max(lim, 1e-6)

                    # 2-step chord check catches 2-frame rogue anchor jumps without clipping smooth turns
                    if 2 <= idx_k < m_cur - 2:
                        a_l2 = anchor_valid_indices[idx_k - 2]
                        a_r2 = anchor_valid_indices[idx_k + 2]
                        alpha2 = float(a - a_l2) / float(a_r2 - a_l2)
                        p_chord2 = (1.0 - alpha2) * raw_poses[a_l2].t_wc + alpha2 * raw_poses[a_r2].t_wc
                        chord2 = float(np.linalg.norm(raw_poses[a_r2].t_wc - raw_poses[a_l2].t_wc))
                        d2 = float(np.linalg.norm(ta - p_chord2))
                        lim2 = max(base_thresh * 1.35, 1.35 * chord2)
                        ratio = max(ratio, d2 / max(lim2, 1e-6))

                    # Rotation flip guard: anchor snapped >45 deg away from consistent neighbors
                    rl, rr, ra = raw_poses[a_l].r_wc, raw_poses[a_r].r_wc, raw_poses[a].r_wc
                    if _rot_deg(rl, rr) < 35.0 and min(_rot_deg(ra, rl), _rot_deg(ra, rr)) > 45.0:
                        ratio = max(ratio, 2.0)

                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_idx = a
                    worst_pos = idx_k

            if worst_idx >= 0:
                raw_poses[worst_idx] = None
                anchor_valid_indices.pop(worst_pos)
                if worst_idx in clean_indices:
                    clean_indices.remove(worst_idx)
                anchor_spikes_removed += 1
            else:
                break

    # 2. Anchor-Chord Envelope Guard for ALL Non-Anchor PnP Frames (kills multi-frame drifted PnP chains!)
    if len(anchor_valid_indices) >= 2:
        clean_anchor_set = set(anchor_valid_indices)
        sorted_anchors = sorted(anchor_valid_indices)
        first_anchor = sorted_anchors[0]
        last_anchor = sorted_anchors[-1]

        curr_clean = [i for i in clean_indices if raw_poses[i] is not None]
        to_drop_envelope = set()

        for i in curr_clean:
            if i in clean_anchor_set:
                continue
            if i < first_anchor:
                dist_end = float(np.linalg.norm(raw_poses[i].t_wc - raw_poses[first_anchor].t_wc))
                cap_dist = max(med_anchor_step * 1.5, 0.35 * (abs(i - first_anchor) / max(1.0, avg_frame_gap)) * med_anchor_step)
                if dist_end > cap_dist:
                    to_drop_envelope.add(i)
            elif i > last_anchor:
                dist_end = float(np.linalg.norm(raw_poses[i].t_wc - raw_poses[last_anchor].t_wc))
                cap_dist = max(med_anchor_step * 1.5, 0.35 * (abs(i - last_anchor) / max(1.0, avg_frame_gap)) * med_anchor_step)
                if dist_end > cap_dist:
                    to_drop_envelope.add(i)
            else:
                a_left = max(a for a in sorted_anchors if a < i)
                a_right = min(a for a in sorted_anchors if a > i)
                alpha = float(i - a_left) / float(a_right - a_left)
                p_ref = (1.0 - alpha) * raw_poses[a_left].t_wc + alpha * raw_poses[a_right].t_wc
                anchor_chord = float(np.linalg.norm(raw_poses[a_right].t_wc - raw_poses[a_left].t_wc))
                envelope = max(anchor_chord * 0.75, med_anchor_step * 0.55)
                dist_ref = float(np.linalg.norm(raw_poses[i].t_wc - p_ref))
                if dist_ref > envelope:
                    to_drop_envelope.add(i)

        for d_idx in to_drop_envelope:
            raw_poses[d_idx] = None
        pnp_spikes_removed += len(to_drop_envelope)
        clean_indices = [i for i in clean_indices if i not in to_drop_envelope]

    # 3. 2-pass neighbor spike filter
    clean_anchor_set = set(anchor_valid_indices)
    for _ in range(2):
        curr_clean = [i for i in clean_indices if raw_poses[i] is not None]
        if len(curr_clean) < 3:
            break
        to_drop_neighbor = set()
        for idx_pos in range(1, len(curr_clean) - 1):
            i = curr_clean[idx_pos]
            if i in clean_anchor_set:
                continue
            i_prev = curr_clean[idx_pos - 1]
            i_next = curr_clean[idx_pos + 1]
            alpha = float(i - i_prev) / float(i_next - i_prev)
            p_interp = (1.0 - alpha) * raw_poses[i_prev].t_wc + alpha * raw_poses[i_next].t_wc
            chord = float(np.linalg.norm(raw_poses[i_next].t_wc - raw_poses[i_prev].t_wc))
            dev = float(np.linalg.norm(raw_poses[i].t_wc - p_interp))
            thresh = max(chord * 1.25, med_anchor_step * 0.65)
            if dev > thresh:
                to_drop_neighbor.add(i)
        if not to_drop_neighbor:
            break
        pnp_spikes_removed += len(to_drop_neighbor)
        for d_idx in to_drop_neighbor:
            raw_poses[d_idx] = None
        clean_indices = [i for i in clean_indices if i not in to_drop_neighbor]

    return clean_indices, anchor_valid_indices, anchor_spikes_removed, pnp_spikes_removed


def estimate_sfm_poses(
    frames: List[VideoFrame],
    device: str = "cuda",
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    max_anchors: int = 0,
) -> tuple[bool, np.ndarray, float, List[CameraPose], Dict[int, tuple[np.ndarray, np.ndarray]], int, List[int]]:
    if not frames:
        raise ValueError("no frames passed to estimate_sfm_poses")

    LAST_STAGE2_SPARSE_POINTS.clear()

    n = len(frames)
    if max_anchors > 0:
        anchor_cap = max(12, min(500, int(max_anchors)))
    else:
        dur_sec = max(1.0, float(frames[-1].timestamp - frames[0].timestamp)) if len(frames) >= 2 else 1.0
        anchor_cap = int(np.clip(round(dur_sec * 0.8), 140, 360))

    if n <= anchor_cap:
        anchor_indices = np.arange(n, dtype=int)
    else:
        anchor_indices = np.unique(np.linspace(0, n - 1, anchor_cap, dtype=int))

    anchor_frames = [frames[int(i)] for i in anchor_indices]
    rec, sfm_meta = _run_gpu_matched_colmap_sfm(
        anchor_frames, device=device, progress_cb=progress_cb, cancel_cb=cancel_cb
    )

    if rec is None or rec.num_reg_images() < 3 or len(rec.cameras) == 0:
        h, w = frames[0].rgb.shape[:2]
        f = 0.92 * max(w, h)
        k_mat = np.array([[f, 0.0, w * 0.5], [0.0, f, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        return False, k_mat, 0.0, [], {}, 0, []

    cam = next(iter(rec.cameras.values()))
    f_cal = float(cam.params[0])
    cx_cal = float(cam.params[1])
    cy_cal = float(cam.params[2])
    k1_cal = float(cam.params[3]) if len(cam.params) >= 4 else 0.0

    # Sane FOV bounds and radial distortion clamping
    h_fr, w_fr = frames[0].rgb.shape[:2]
    max_d = max(w_fr, h_fr)
    f_cal = float(np.clip(f_cal, 0.45 * max_d, 2.2 * max_d))
    if abs(k1_cal) > 0.35:
        k1_cal = 0.0
    k_mat = np.array([[f_cal, 0.0, cx_cal], [0.0, f_cal, cy_cal], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Scale normalization: compute median sparse point depth and scale to 1.0
    all_raw_z = []
    for im in rec.images.values():
        if not getattr(im, "has_pose", True):
            continue
        try:
            cfw = im.cam_from_world()
        except Exception:
            continue
        r_cw = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(cfw.translation, dtype=np.float64)
        for p2d in im.points2D:
            if p2d.has_point3D() and p2d.point3D_id in rec.points3D:
                p3 = rec.points3D[p2d.point3D_id]
                z_v = float((r_cw @ np.asarray(p3.xyz, dtype=np.float64) + t_cw)[2])
                if z_v > 1e-4:
                    all_raw_z.append(z_v)
    med_colmap_z = float(np.median(all_raw_z)) if len(all_raw_z) >= 10 else 1.0
    s_col_norm = 1.0 / max(med_colmap_z, 1e-4)

    raw_poses: List[Optional[CameraPose]] = [None] * n
    sparse_obs: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    anchor_world_obs: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    anchor_valid_indices: List[int] = []
    min_anchors = 18 if len(anchor_indices) >= 20 else 8

    for im in rec.images.values():
        if not getattr(im, "has_pose", True):
            continue
        try:
            cfw = im.cam_from_world()
        except Exception:
            continue
        a_local = int(Path(im.name).stem)
        orig_idx = int(anchor_indices[a_local])
        r_cw = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(cfw.translation, dtype=np.float64) * s_col_norm
        r_wc = r_cw.T
        t_wc = -r_wc @ t_cw

        uv_list, z_list, xyz_list, reproj_list = [], [], [], []
        for p2d in im.points2D:
            if p2d.has_point3D() and p2d.point3D_id in rec.points3D:
                p3 = rec.points3D[p2d.point3D_id]
                xyz_w = np.asarray(p3.xyz, dtype=np.float64) * s_col_norm
                xyz_c = r_cw @ xyz_w + t_cw
                z_cam = float(xyz_c[2])
                if z_cam > 0.05:
                    uv_proj = _project_colmap_point(
                        cam,
                        xyz_c / s_col_norm,
                        f_fallback=f_cal,
                        cx_fallback=cx_cal,
                        cy_fallback=cy_cal,
                        k1_fallback=k1_cal,
                    )
                    if uv_proj is not None:
                        reproj_list.append(float(np.hypot(uv_proj[0] - p2d.xy[0], uv_proj[1] - p2d.xy[1])))

                    uv_list.append(np.asarray(p2d.xy, dtype=np.float32))
                    z_list.append(z_cam)
                    xyz_list.append(xyz_w)

        med_reproj = float(np.median(reproj_list)) if reproj_list else 999.0
        if len(uv_list) >= min_anchors and med_reproj <= 2.5:
            raw_poses[orig_idx] = CameraPose(r_wc=r_wc, t_wc=t_wc, valid=True)
            uv_np = np.array(uv_list, dtype=np.float32)
            sparse_obs[orig_idx] = (uv_np, np.array(z_list, dtype=np.float32))
            anchor_world_obs[orig_idx] = (uv_np, np.array(xyz_list, dtype=np.float64))
            anchor_valid_indices.append(orig_idx)

    n_anchors_passed_reproj = len(anchor_valid_indices)

    reg_anchors = sorted(anchor_world_obs.keys())
    if len(reg_anchors) < 3:
        h, w = frames[0].rgb.shape[:2]
        f = 0.92 * max(w, h)
        k_mat = np.array([[f, 0.0, w * 0.5], [0.0, f, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        return False, k_mat, 0.0, [], {}, 0, []

    if len(reg_anchors) >= 2:
        consec_steps = [
            float(np.linalg.norm(raw_poses[reg_anchors[k + 1]].t_wc - raw_poses[reg_anchors[k]].t_wc))
            for k in range(len(reg_anchors) - 1)
        ]
        med_anchor_step = float(np.median(consec_steps)) if consec_steps else 0.10
        avg_frame_gap = max(1.0, float(np.median(np.diff(reg_anchors))))
    else:
        med_anchor_step = 0.10
        avg_frame_gap = 1.0

    # Chained Bidirectional LK + 3D-PnP localization for all remaining non-anchor frames
    t0_pnp = time.perf_counter()
    if len(anchor_indices) < n:
        _localize_non_anchor_frames_chained(
            frames,
            reg_anchors,
            anchor_world_obs,
            raw_poses,
            sparse_obs,
            k_mat,
            k1_cal,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    chained_pnp_s = time.perf_counter() - t0_pnp

    reg_indices = [i for i, p in enumerate(raw_poses) if p is not None]
    if len(reg_indices) < 3:
        h, w = frames[0].rgb.shape[:2]
        f = 0.92 * max(w, h)
        k_mat = np.array([[f, 0.0, w * 0.5], [0.0, f, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        return False, k_mat, 0.0, [], {}, 0, []

    all_centers = np.array([raw_poses[i].t_wc for i in reg_indices], dtype=np.float64)
    med_c = np.median(all_centers, axis=0)
    dists = np.linalg.norm(all_centers - med_c, axis=1)
    q25, q50, q75 = np.percentile(dists, [25.0, 50.0, 75.0])
    iqr = max(float(q75 - q25), float(q50) * 0.25, 1e-3)
    dist_cap = float(q75 + 2.2 * iqr)

    clean_indices = [i for i, d in zip(reg_indices, dists) if d <= dist_cap]
    if len(clean_indices) < 3:
        clean_indices = reg_indices
    n_anchors_dropped_dist_cap = sum(1 for a in anchor_valid_indices if a not in clean_indices)

    r_grav = _compute_gravity_level_rotation([raw_poses[i].r_wc for i in clean_indices])
    origin_t = np.median([raw_poses[i].t_wc for i in clean_indices], axis=0)

    for i in clean_indices:
        p = raw_poses[i]
        p.r_wc = r_grav @ p.r_wc
        p.t_wc = r_grav @ (p.t_wc - origin_t)

    for i in reg_indices:
        if i not in clean_indices:
            raw_poses[i] = None
            sparse_obs.pop(i, None)

    # 3-Layered trajectory spike filter (catches rogue anchors, multi-frame PnP drift, and single-frame jumps)
    clean_indices, anchor_valid_indices, anchor_spikes_removed, pnp_spikes_removed = filter_trajectory_spikes(
        raw_poses=raw_poses,
        clean_indices=clean_indices,
        anchor_valid_indices=anchor_valid_indices,
        med_anchor_step=med_anchor_step,
        avg_frame_gap=avg_frame_gap,
        min_anchor_thresh=0.04,
    )

    for i in range(n):
        if raw_poses[i] is None:
            sparse_obs.pop(i, None)

    if not clean_indices:
        return False, k_mat, 0.0, [], {}, 0, []

    # SLERP + Linear interpolation for remaining unlocalized frames
    clean_indices_set = set(clean_indices)
    poses: List[CameraPose] = []
    for i in range(n):
        if i in clean_indices_set and raw_poses[i] is not None:
            poses.append(raw_poses[i])
        elif i < clean_indices[0]:
            nearest = clean_indices[0]
            poses.append(
                CameraPose(
                    r_wc=raw_poses[nearest].r_wc.copy(),
                    t_wc=raw_poses[nearest].t_wc.copy(),
                    valid=False,
                )
            )
        elif i > clean_indices[-1]:
            nearest = clean_indices[-1]
            poses.append(
                CameraPose(
                    r_wc=raw_poses[nearest].r_wc.copy(),
                    t_wc=raw_poses[nearest].t_wc.copy(),
                    valid=False,
                )
            )
        else:
            prev_v = max(k for k in clean_indices if k < i)
            next_v = min(k for k in clean_indices if k > i)
            alpha = float(i - prev_v) / float(next_v - prev_v)
            t_interp = (1.0 - alpha) * raw_poses[prev_v].t_wc + alpha * raw_poses[next_v].t_wc
            rots = Rotation.from_matrix([raw_poses[prev_v].r_wc, raw_poses[next_v].r_wc])
            slerp = Slerp([0.0, 1.0], rots)
            r_interp = slerp([alpha])[0].as_matrix()
            poses.append(CameraPose(r_wc=r_interp, t_wc=t_interp, valid=False))

    # Apply light 5-frame Gaussian smoothing [0.1, 0.2, 0.4, 0.2, 0.1] to non-anchor t_wc only (pinning anchors fixed)
    # Only smooth valid frames and never blur across unlocalized gaps or large jumps
    w_offsets = [-2, -1, 0, 1, 2]
    w_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    all_ts = np.array([p.t_wc for p in poses], dtype=np.float64)
    clean_anchor_set = set(anchor_valid_indices)
    step_thresh = max(1.8 * med_anchor_step, 0.05)
    for i in range(n):
        if i not in clean_anchor_set and poses[i].valid:
            valid_ts = []
            valid_wts = []
            for offset, wt in zip(w_offsets, w_weights):
                idx = i + offset
                if 0 <= idx < n and poses[idx].valid:
                    if np.linalg.norm(all_ts[idx] - all_ts[i]) < step_thresh:
                        valid_ts.append(all_ts[idx])
                        valid_wts.append(wt)
            if valid_wts:
                valid_wts_arr = np.array(valid_wts, dtype=np.float64)
                valid_wts_arr /= np.sum(valid_wts_arr)
                poses[i].t_wc = np.sum(np.array(valid_ts) * valid_wts_arr[:, None], axis=0)

    n_pts_total = rec.num_points3D()
    if progress_cb:
        n_valid = sum(1 for p in poses if p.valid)
        progress_cb(
            0.56,
            f"GPU-SfM registered {n_valid}/{n} poses ({n_pts_total:,} 3D pts, f={int(round(f_cal))}px)",
        )

    anchor_valid_indices = sorted([i for i in anchor_valid_indices if i in clean_indices_set])

    max_step_m = float(max(np.linalg.norm(poses[k + 1].t_wc - poses[k].t_wc) for k in range(n - 1))) if n >= 2 else 0.0

    LAST_STAGE2_TELEMETRY.clear()
    LAST_STAGE2_TELEMETRY.update({
        "sift_extract_s": sfm_meta.get("sift_extract_s", 0.0),
        "gpu_match_s": sfm_meta.get("gpu_match_s", 0.0),
        "db_write_s": sfm_meta.get("db_write_s", 0.0),
        "pairs_tested": sfm_meta.get("pairs_tested", 0),
        "pairs_written": sfm_meta.get("pairs_written", 0),
        "verify_matches_s": sfm_meta.get("verify_matches_s", 0.0),
        "global_mapping_s": sfm_meta.get("global_mapping_s", 0.0),
        "glomap_jumps": sfm_meta.get("glomap_jumps", 0),
        "glomap_max_step_ratio": sfm_meta.get("glomap_max_step_ratio", 0.0),
        "inc_jumps": sfm_meta.get("inc_jumps", 0),
        "inc_max_step_ratio": sfm_meta.get("inc_max_step_ratio", 0.0),
        "glomap_reg_anchors": sfm_meta.get("glomap_reg_anchors", 0),
        "glomap_good_anchors": sfm_meta.get("glomap_good_anchors", 0),
        "inc_good_anchors": sfm_meta.get("inc_good_anchors", 0),
        "num_points3D": n_pts_total,
        "anchors_rejected_by_reproj": max(0, rec.num_reg_images() - n_anchors_passed_reproj),
        "anchors_rejected_by_dist_cap": n_anchors_dropped_dist_cap,
        "incremental_triggered": sfm_meta.get("incremental_triggered", False),
        "incremental_s": sfm_meta.get("incremental_s", 0.0),
        "chained_pnp_s": chained_pnp_s,
        "anchor_spikes_removed": anchor_spikes_removed,
        "pnp_spikes_removed": pnp_spikes_removed,
        "anchor_cap": anchor_cap,
        "valid_poses": sum(1 for p in poses if p.valid),
        "total_poses": len(poses),
        "max_step_m": max_step_m,
    })

    if rec.points3D:
        raw_xyz = np.array([p3.xyz for p3 in rec.points3D.values()], dtype=np.float64) * s_col_norm
        raw_rgb = np.array([p3.color for p3 in rec.points3D.values()], dtype=np.uint8)
        world_xyz = ((raw_xyz - origin_t[None, :]) @ r_grav.T).astype(np.float32)
        finite_mask = np.isfinite(world_xyz).all(axis=1)
        LAST_STAGE2_SPARSE_POINTS.update({
            "xyz": np.ascontiguousarray(world_xyz[finite_mask], dtype=np.float32),
            "rgb": np.ascontiguousarray(raw_rgb[finite_mask], dtype=np.uint8),
        })
    else:
        LAST_STAGE2_SPARSE_POINTS.update({
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
        })

    return True, k_mat, k1_cal, poses, sparse_obs, n_pts_total, anchor_valid_indices


def align_selected_depths(
    frames: List[VideoFrame],
    selected_indices: List[int],
    depths_sub: List[np.ndarray],
    masks_sub: List[np.ndarray],
    poses: List[CameraPose],
    sparse_obs: Dict[int, tuple[np.ndarray, np.ndarray]],
    k_mat: np.ndarray,
    k1_cal: float,
    device: str = "cuda",
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    scene_type: str = "indoor",
    use_grid_warp: bool = True,
) -> tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    m_sub = len(selected_indices)
    if m_sub == 0 or len(depths_sub) == 0:
        return [], [], []
    h, w = depths_sub[0].shape
    fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
    cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])

    if progress_cb:
        progress_cb(0.68, f"undistorting & aligning {m_sub} depth maps...")

    # Edge-aware undistortion on selected frames only (AFTER depth estimation!)
    if abs(k1_cal) > 1e-4:
        dist = np.array([k1_cal, 0.0, 0.0, 0.0], dtype=np.float64)
        k64 = k_mat.astype(np.float64)
        map1, map2 = cv2.initUndistortRectifyMap(k64, dist, None, k64, (w, h), cv2.CV_32FC1)

        def _und_one(k: int):
            idx = selected_indices[k]
            frames[idx].rgb = cv2.remap(
                frames[idx].rgb,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            d_lin = cv2.remap(depths_sub[k], map1, map2, interpolation=cv2.INTER_LINEAR)
            d_nn = cv2.remap(depths_sub[k], map1, map2, interpolation=cv2.INTER_NEAREST)
            depths_sub[k] = np.where(np.abs(d_lin - d_nn) < 0.06, d_lin, d_nn).astype(np.float32)
            masks_sub[k] = cv2.remap(masks_sub[k].astype(np.uint8), map1, map2, interpolation=cv2.INTER_NEAREST) > 0

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            list(pool.map(_und_one, range(m_sub)))

        for idx in selected_indices:
            if idx in sparse_obs:
                uv, z_arr = sparse_obs[idx]
                if len(uv) > 0:
                    pts_in = uv.reshape(-1, 1, 2).astype(np.float64)
                    pts_und = cv2.undistortPoints(pts_in, k64, dist, P=k64).reshape(-1, 2).astype(np.float32)
                    sparse_obs[idx] = (pts_und, z_arr)

    # Compute global_metric_scale from sparse_obs on selected_indices
    all_mono_z, all_col_z = [], []
    for k, idx in enumerate(selected_indices):
        if idx not in sparse_obs or not poses[idx].valid:
            continue
        uv, z_col = sparse_obs[idx]
        if len(uv) < 6:
            continue
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        z_m = depths_sub[k][v, u]
        ok = masks_sub[k][v, u] & (z_m > 0.20) & (z_col > 0.05) & np.isfinite(z_col)
        if np.sum(ok) >= 6:
            all_mono_z.append(z_m[ok])
            all_col_z.append(z_col[ok])

    if all_mono_z:
        cat_mono = np.concatenate(all_mono_z)
        cat_col = np.concatenate(all_col_z)
        global_metric_scale = float(np.clip(np.median(cat_mono / np.maximum(cat_col, 1e-4)), 0.05, 50.0))
    else:
        global_metric_scale = 1.0

    # Rescale the ENTIRE camera trajectory (all N poses) so it matches the 3D world in meters
    for p in poses:
        p.t_wc = p.t_wc * global_metric_scale

    sp_f_list, sp_uv_list, sp_zm_list, sp_zc_list = [], [], [], []
    init_scale = np.ones(m_sub, dtype=np.float32)

    for k, idx in enumerate(selected_indices):
        if idx not in sparse_obs or not poses[idx].valid:
            continue
        uv, z_col = sparse_obs[idx]
        if len(uv) < 6:
            continue
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        z_m = depths_sub[k][v, u]
        z_c = z_col * global_metric_scale
        z_max_cand = 60.0 if scene_type == "outdoor" else 14.0
        ok = masks_sub[k][v, u] & (z_m > 0.20) & (z_c > 0.20) & (z_c < z_max_cand) & np.isfinite(z_c)
        if np.sum(ok) < 6:
            continue

        med_r = float(np.median(z_c[ok] / z_m[ok]))
        init_scale[k] = float(np.clip(med_r, 0.35, 3.20))

        sp_f_list.append(np.full(int(np.sum(ok)), k, dtype=np.int64))
        sp_uv_list.append(uv[ok])
        sp_zm_list.append(z_m[ok])
        sp_zc_list.append(z_c[ok])

    if not sp_f_list:
        return depths_sub, masks_sub, selected_indices

    sp_f = torch.tensor(np.concatenate(sp_f_list), dtype=torch.long, device=device)
    sp_uv = torch.tensor(np.concatenate(sp_uv_list, axis=0), dtype=torch.float32, device=device)
    sp_zm = torch.tensor(np.concatenate(sp_zm_list), dtype=torch.float32, device=device)
    sp_zc = torch.tensor(np.concatenate(sp_zc_list), dtype=torch.float32, device=device)
    sp_uv_norm = torch.stack(
        [(sp_uv[:, 0] / max(1, w - 1)) * 2.0 - 1.0, (sp_uv[:, 1] / max(1, h - 1)) * 2.0 - 1.0],
        dim=-1,
    )

    # Phase A: global per-frame scalar (scale_i, shift_i)
    is_outdoor = (scene_type == "outdoor")
    shift_cap_a = 0.25 if (not use_grid_warp) else (2.0 if is_outdoor else 0.30)
    shift_cap_b = 2.5 if is_outdoor else 0.35
    d_sh_cap = 0.22 if is_outdoor else 0.18
    b_reg_a = 0.50 if (not use_grid_warp) else (0.02 if is_outdoor else 0.15)

    log_s_scalar = torch.tensor(
        np.log(np.clip(init_scale, 0.35, 3.20)), dtype=torch.float32, device=device, requires_grad=True
    )
    b_scalar = torch.zeros(m_sub, dtype=torch.float32, device=device, requires_grad=True)
    opt_a = torch.optim.Adam([log_s_scalar, b_scalar], lr=0.03)
    for it_a in range(80):
        if it_a % 20 == 0 and cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")
        opt_a.zero_grad()
        z_p = (torch.exp(log_s_scalar[sp_f].clamp(-1.2, 1.2)) * sp_zm + b_scalar[sp_f].clamp(-shift_cap_a, shift_cap_a)).clamp(min=0.20)
        if is_outdoor:
            loss_a = F.smooth_l1_loss(torch.log(z_p), torch.log(sp_zc), beta=0.04) + b_reg_a * b_scalar.pow(2).mean()
        else:
            loss_a = F.smooth_l1_loss(z_p, sp_zc, beta=0.04) + 0.15 * b_scalar.pow(2).mean()
        loss_a.backward()
        opt_a.step()

    with torch.no_grad():
        s_a_np = torch.exp(log_s_scalar.clamp(-1.2, 1.2)).cpu().numpy()
        b_a_np = b_scalar.clamp(-shift_cap_a, shift_cap_a).cpu().numpy()

    if not use_grid_warp:
        has_anchor = np.zeros(m_sub, dtype=bool)
        has_anchor[sp_f.cpu().numpy()] = True
        if np.any(has_anchor) and not np.all(has_anchor):
            anchored_idx = np.where(has_anchor)[0]
            unanchored_idx = np.where(~has_anchor)[0]
            interp_log_s = np.interp(unanchored_idx, anchored_idx, np.log(s_a_np[anchored_idx]))
            interp_b = np.interp(unanchored_idx, anchored_idx, b_a_np[anchored_idx])
            s_a_np[unanchored_idx] = np.exp(interp_log_s)
            b_a_np[unanchored_idx] = interp_b
        s_full = s_a_np[:, None, None]
        b_full = b_a_np[:, None, None]
    else:
        # Build dense 16x10 multi-view ties on selected_indices
        sub_forwards = np.array([poses[idx].r_wc[:, 2] for idx in selected_indices], dtype=np.float32)
        sub_ts = np.array([poses[idx].t_wc for idx in selected_indices], dtype=np.float32)
        if len(selected_indices) >= 2:
            sub_steps = [
                float(np.linalg.norm(sub_ts[k + 1] - sub_ts[k])) for k in range(len(sub_ts) - 1)
            ]
            med_sub_step = float(np.median(sub_steps)) if sub_steps else 0.20
        else:
            med_sub_step = 0.20
        d_sub_scale = max(1.0 if scene_type == "outdoor" else 0.60, 3.0 * med_sub_step)

        gu = np.linspace(int(w * 0.06), int(w * 0.94), 16, dtype=np.int32)
        gv = np.linspace(int(h * 0.06), int(h * 0.94), 10, dtype=np.int32)
        ggu, ggv = np.meshgrid(gu, gv)
        grid_u = ggu.ravel()
        grid_v = ggv.ravel()
        grid_rx = (grid_u.astype(np.float32) - cx) / fx
        grid_ry = (grid_v.astype(np.float32) - cy) / fy

        tie_min = 0.30 if scene_type == "outdoor" else 0.25
        tie_max = 45.0 if scene_type == "outdoor" else 9.0

        mv_i_l, mv_j_l = [], []
        mv_uvi_l, mv_uvj_l = [], []
        mv_zi_l, mv_zj_l = [], []
        max_mv_pairs = 160000
        mv_count = 0

        for k_i in range(m_sub):
            if mv_count >= max_mv_pairs:
                break
            idx_i = selected_indices[k_i]
            if not poses[idx_i].valid:
                continue
            cos_d = sub_forwards @ sub_forwards[k_i]
            d_ti = np.linalg.norm(sub_ts - sub_ts[k_i], axis=-1)
            nbr_scores = cos_d / (1.0 + d_ti / d_sub_scale)
            cands = [
                k for k in np.argsort(nbr_scores)[::-1]
                if k != k_i
                and poses[selected_indices[k]].valid
                and cos_d[k] > 0.65
                and d_ti[k] <= (3.5 * d_sub_scale)
            ]
            if not cands:
                continue
            pick_pos = [0, 2, 5, 9]
            chosen_nbrs = []
            for pp in pick_pos:
                if pp < len(cands):
                    chosen_nbrs.append(cands[pp])

            m_i = masks_sub[k_i][grid_v, grid_u]
            if not np.any(m_i):
                continue
            ui = grid_u[m_i]
            vi = grid_v[m_i]
            zi_raw = depths_sub[k_i][vi, ui]
            zi_a = np.clip(s_a_np[k_i] * zi_raw + b_a_np[k_i], tie_min, tie_max)
            pts_ci = np.stack([grid_rx[m_i] * zi_a, grid_ry[m_i] * zi_a, zi_a], axis=-1)
            pts_w = (pts_ci @ poses[idx_i].r_wc.astype(np.float32).T) + poses[idx_i].t_wc.astype(np.float32)[None, :]

            for k_j in chosen_nbrs:
                idx_j = selected_indices[k_j]
                pts_cj = (pts_w - poses[idx_j].t_wc.astype(np.float32)[None, :]) @ poses[idx_j].r_wc.astype(np.float32)
                zj_proj = pts_cj[:, 2]
                in_front = zj_proj > 0.28
                uj = fx * (pts_cj[:, 0] / np.maximum(zj_proj, 0.1)) + cx
                vj = fy * (pts_cj[:, 1] / np.maximum(zj_proj, 0.1)) + cy
                inside = in_front & (uj >= 3) & (uj < w - 3) & (vj >= 3) & (vj < h - 3)
                if not np.any(inside):
                    continue

                uj_i = np.round(uj[inside]).astype(int)
                vj_i = np.round(vj[inside]).astype(int)
                m_j = masks_sub[k_j][vj_i, uj_i]
                if not np.any(m_j):
                    continue

                zj_raw = depths_sub[k_j][vj_i[m_j], uj_i[m_j]]
                zj_a = np.clip(s_a_np[k_j] * zj_raw + b_a_np[k_j], tie_min, tie_max)
                if scene_type == "outdoor":
                    agree = np.abs(zj_proj[inside][m_j] - zj_a) < np.maximum(0.30, 0.045 * zj_a)
                else:
                    agree = np.abs(zj_proj[inside][m_j] - zj_a) < 0.28
                if np.sum(agree) < 4:
                    continue

                idx_ok = np.flatnonzero(inside)[m_j][agree]
                cnt = len(idx_ok)
                mv_i_l.append(np.full(cnt, k_i, dtype=np.int64))
                mv_j_l.append(np.full(cnt, k_j, dtype=np.int64))
                mv_uvi_l.append(np.stack([ui[idx_ok], vi[idx_ok]], axis=-1).astype(np.float32))
                mv_uvj_l.append(np.stack([uj[idx_ok], vj[idx_ok]], axis=-1).astype(np.float32))
                mv_zi_l.append(zi_raw[idx_ok].astype(np.float32))
                mv_zj_l.append(zj_raw[agree].astype(np.float32))
                mv_count += cnt

        has_mv = len(mv_i_l) > 0
        if has_mv:
            mv_i = torch.tensor(np.concatenate(mv_i_l), dtype=torch.long, device=device)
            mv_j = torch.tensor(np.concatenate(mv_j_l), dtype=torch.long, device=device)
            mv_uvi = torch.tensor(np.concatenate(mv_uvi_l, axis=0), dtype=torch.float32, device=device)
            mv_uvj = torch.tensor(np.concatenate(mv_uvj_l, axis=0), dtype=torch.float32, device=device)
            mv_zi = torch.tensor(np.concatenate(mv_zi_l), dtype=torch.float32, device=device)
            mv_zj = torch.tensor(np.concatenate(mv_zj_l), dtype=torch.float32, device=device)

            mv_uvi_n = torch.stack([(mv_uvi[:, 0] / max(1, w - 1)) * 2.0 - 1.0, (mv_uvi[:, 1] / max(1, h - 1)) * 2.0 - 1.0], dim=-1)
            mv_uvj_n = torch.stack([(mv_uvj[:, 0] / max(1, w - 1)) * 2.0 - 1.0, (mv_uvj[:, 1] / max(1, h - 1)) * 2.0 - 1.0], dim=-1)
            ones_mv = torch.ones_like(mv_zi)
            ray_mvi = torch.stack([(mv_uvi[:, 0] - cx) / fx, (mv_uvi[:, 1] - cy) / fy, ones_mv], dim=-1)
            ray_mvj = torch.stack([(mv_uvj[:, 0] - cx) / fx, (mv_uvj[:, 1] - cy) / fy, ones_mv], dim=-1)

            r_sub = torch.tensor(np.stack([poses[idx].r_wc for idx in selected_indices]), dtype=torch.float32, device=device)
            t_sub = torch.tensor(np.stack([poses[idx].t_wc for idx in selected_indices]), dtype=torch.float32, device=device)

        base_log_s_t = torch.tensor(np.log(np.clip(s_a_np, 0.35, 3.20))[:, None, None, None], dtype=torch.float32, device=device)
        base_b_t = torch.tensor(b_a_np[:, None, None, None], dtype=torch.float32, device=device)

        # Residual grids tightly clamped around Phase-A base scalars (prevents rubber-sheet warping)
        d_log_s_grid = torch.zeros((m_sub, 1, 4, 4), dtype=torch.float32, device=device, requires_grad=True)
        d_shift_grid = torch.zeros((m_sub, 1, 4, 4), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([d_log_s_grid, d_shift_grid], lr=0.02)

        def _sample(grid: torch.Tensor, f_ids: torch.Tensor, uv_n: torch.Tensor) -> torch.Tensor:
            return F.grid_sample(
                grid[f_ids], uv_n.view(-1, 1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True
            ).view(-1)

        for it_b in range(140):
            if it_b % 20 == 0 and cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")
            opt.zero_grad()

            d_ls_clamped = d_log_s_grid.clamp(-0.18, 0.18)
            d_sh_clamped = d_shift_grid.clamp(-d_sh_cap, d_sh_cap)
            log_s_grid = (base_log_s_t + d_ls_clamped).clamp(-1.25, 1.25)
            shift_grid = (base_b_t + d_sh_clamped).clamp(-shift_cap_b, shift_cap_b)

            s_sp = torch.exp(_sample(log_s_grid, sp_f, sp_uv_norm))
            b_sp = _sample(shift_grid, sp_f, sp_uv_norm)
            z_pred = (s_sp * sp_zm + b_sp).clamp(min=0.20)
            if is_outdoor:
                loss_colmap = F.smooth_l1_loss(torch.log(z_pred), torch.log(sp_zc), beta=0.04)
            else:
                loss_colmap = F.smooth_l1_loss(z_pred, sp_zc, beta=0.04)

            loss_mv = torch.tensor(0.0, device=device)
            if has_mv:
                si = torch.exp(_sample(log_s_grid, mv_i, mv_uvi_n))
                bi = _sample(shift_grid, mv_i, mv_uvi_n)
                sj = torch.exp(_sample(log_s_grid, mv_j, mv_uvj_n))
                bj = _sample(shift_grid, mv_j, mv_uvj_n)
                zi = (si * mv_zi + bi).clamp(min=0.25)
                zj = (sj * mv_zj + bj).clamp(min=0.25)
                wi = torch.bmm(r_sub[mv_i], (ray_mvi * zi.unsqueeze(-1)).unsqueeze(-1)).squeeze(-1) + t_sub[mv_i]
                wj = torch.bmm(r_sub[mv_j], (ray_mvj * zj.unsqueeze(-1)).unsqueeze(-1)).squeeze(-1) + t_sub[mv_j]
                if is_outdoor:
                    z_pair = torch.clamp(0.5 * (zi + zj), min=1.0).unsqueeze(-1)
                    loss_mv = F.smooth_l1_loss(wi / z_pair, wj / z_pair, beta=0.03)
                else:
                    loss_mv = F.smooth_l1_loss(wi, wj, beta=0.04)

            ds_x = d_ls_clamped[:, :, :, 1:] - d_ls_clamped[:, :, :, :-1]
            ds_y = d_ls_clamped[:, :, 1:, :] - d_ls_clamped[:, :, :-1, :]
            db_x = d_sh_clamped[:, :, :, 1:] - d_sh_clamped[:, :, :, :-1]
            db_y = d_sh_clamped[:, :, 1:, :] - d_sh_clamped[:, :, :-1, :]
            loss_smooth = 1.5 * (ds_x.pow(2).mean() + ds_y.pow(2).mean()) + 1.5 * (
                db_x.pow(2).mean() + db_y.pow(2).mean()
            )
            loss_base = 0.8 * (d_ls_clamped.pow(2).mean() + d_sh_clamped.pow(2).mean())

            total = loss_colmap + 0.65 * loss_mv + loss_smooth + loss_base
            total.backward()
            opt.step()

        with torch.no_grad():
            final_log_s = (base_log_s_t + d_log_s_grid.clamp(-0.18, 0.18)).clamp(-1.25, 1.25)
            final_shift = (base_b_t + d_shift_grid.clamp(-d_sh_cap, d_sh_cap)).clamp(-shift_cap_b, shift_cap_b)
            s_eval = torch.exp(final_log_s)
            b_eval = final_shift
            s_full_list, b_full_list = [], []
            for st in range(0, m_sub, 64):
                s_full_list.append(
                    F.interpolate(s_eval[st : st + 64], size=(h, w), mode="bilinear", align_corners=True)
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                b_full_list.append(
                    F.interpolate(b_eval[st : st + 64], size=(h, w), mode="bilinear", align_corners=True)
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            s_full = np.concatenate(s_full_list, axis=0)
            b_full = np.concatenate(b_full_list, axis=0)

    all_good_zc = sp_zc.cpu().numpy()
    max_col_z = float(np.percentile(all_good_zc, 98.0)) if len(all_good_zc) > 10 else 6.0
    if scene_type == "outdoor":
        depth_ceil = float(np.clip(max_col_z * 1.35, 10.0, 40.0))
    else:
        depth_ceil = float(np.clip(max_col_z * 1.40, 4.0, 9.2))

    world_sparse_pts = []
    for idx in selected_indices:
        if idx not in sparse_obs or not poses[idx].valid:
            continue
        uv, z_col = sparse_obs[idx]
        if len(uv) < 4:
            continue
        zc_m = z_col * global_metric_scale
        ok_z = (zc_m > 0.20) & (zc_m < (60.0 if scene_type == "outdoor" else 14.0))
        if not np.any(ok_z):
            continue
        rx_s = (uv[ok_z, 0] - cx) / fx
        ry_s = (uv[ok_z, 1] - cy) / fy
        pc_s = np.stack([rx_s * zc_m[ok_z], ry_s * zc_m[ok_z], zc_m[ok_z]], axis=-1).astype(np.float32)
        pw_s = (pc_s @ poses[idx].r_wc.astype(np.float32).T) + poses[idx].t_wc.astype(np.float32)[None, :]
        world_sparse_pts.append(pw_s)

    has_hull = False
    if world_sparse_pts:
        all_w_sp = np.concatenate(world_sparse_pts, axis=0)
        if len(all_w_sp) >= 150:
            cam_pts = np.array([poses[idx].t_wc for idx in selected_indices if poses[idx].valid], dtype=np.float32)
            comb = np.concatenate([all_w_sp, cam_pts], axis=0)
            hull_pad = 12.0 if scene_type == "outdoor" else 2.50
            hull_min = np.percentile(comb, 0.2, axis=0) - hull_pad
            hull_max = np.percentile(comb, 99.8, axis=0) + hull_pad
            has_hull = True

    uu_h, vv_h = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rx_h = (uu_h - cx) / fx
    ry_h = (vv_h - cy) / fy

    aligned_depths_sub: List[np.ndarray] = [None] * m_sub  # type: ignore

    def _warp_one(k: int):
        idx = selected_indices[k]
        d_warped = s_full[k] * depths_sub[k] + b_full[k]
        d_clamped = np.clip(d_warped, 0.25, depth_ceil).astype(np.float32)
        m_k = masks_sub[k] & (d_clamped < depth_ceil * 0.98)
        if has_hull and poses[idx].valid:
            pc_k = np.stack([rx_h * d_clamped, ry_h * d_clamped, d_clamped], axis=-1)
            r_k = poses[idx].r_wc.astype(np.float32)
            t_k = poses[idx].t_wc.astype(np.float32)
            pw_k = pc_k @ r_k.T + t_k[None, None, :]
            in_hull = np.all((pw_k >= hull_min[None, None, :]) & (pw_k <= hull_max[None, None, :]), axis=-1)
            m_k = m_k & in_hull
        aligned_depths_sub[k] = d_clamped
        masks_sub[k] = m_k

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        list(pool.map(_warp_one, range(m_sub)))

    # Error gating on selected_indices: drops severe outlier depth frames from keep_sub_indices WITHOUT invalidating pose
    frame_errs: Dict[int, float] = {}
    for k, idx in enumerate(selected_indices):
        if idx not in sparse_obs or not poses[idx].valid:
            continue
        uv, z_col = sparse_obs[idx]
        if len(uv) < 8:
            continue
        u_i = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v_i = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        z_pred_k = aligned_depths_sub[k][v_i, u_i]
        z_true_k = z_col * global_metric_scale
        ok = (z_true_k > 0.25) & (z_true_k < depth_ceil)
        if np.sum(ok) >= 8:
            frame_errs[idx] = float(np.median(np.abs(z_pred_k[ok] - z_true_k[ok])))

    keep_sub_indices = []
    final_depths = []
    final_masks = []
    if len(frame_errs) >= 12:
        glob_med_err = float(np.median(list(frame_errs.values())))
        err_cap = max(0.35, glob_med_err * 2.8) if scene_type == "outdoor" else max(0.18, glob_med_err * 2.5)
        for k, idx in enumerate(selected_indices):
            if frame_errs.get(idx, 0.0) <= err_cap:
                keep_sub_indices.append(idx)
                final_depths.append(aligned_depths_sub[k])
                final_masks.append(masks_sub[k])
    else:
        keep_sub_indices = list(selected_indices)
        final_depths = aligned_depths_sub
        final_masks = masks_sub

    if progress_cb:
        progress_cb(0.76, f"depth aligned on {len(keep_sub_indices)}/{m_sub} views")

    return final_depths, final_masks, keep_sub_indices


def _fallback_simple_trajectory(
    frames: List[VideoFrame],
    depths: List[np.ndarray],
    masks: List[np.ndarray],
) -> tuple[np.ndarray, List[CameraPose], List[np.ndarray]]:
    n = len(frames)
    h, w = depths[0].shape
    f = 0.92 * max(w, h)
    cx, cy = w * 0.5, h * 0.5
    k_mat = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    sift = cv2.SIFT_create(nfeatures=800)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    poses = [CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64), valid=True)]
    aligned = [depths[0].copy()]

    for i in range(1, n):
        g1 = cv2.cvtColor(frames[i - 1].rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i].rgb, cv2.COLOR_RGB2GRAY)
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)
        r_rel = np.eye(3, dtype=np.float64)
        t_rel = np.array([0.04, 0.0, 0.04], dtype=np.float64)

        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            good_m = [
                p[0] for p in bf.knnMatch(des1, des2, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
            ]
            if len(good_m) >= 8:
                uv1 = np.float32([kp1[m.queryIdx].pt for m in good_m])
                uv2 = np.float32([kp2[m.trainIdx].pt for m in good_m])
                u1, v1 = np.clip(uv1[:, 0].astype(int), 0, w - 1), np.clip(uv1[:, 1].astype(int), 0, h - 1)
                u2, v2 = np.clip(uv2[:, 0].astype(int), 0, w - 1), np.clip(uv2[:, 1].astype(int), 0, h - 1)
                valid_m = masks[i - 1][v1, u1] & masks[i][v2, u2]
                z1 = depths[i - 1][v1, u1]
                z2 = depths[i][v2, u2]
                valid_pts = valid_m & (z1 > 0.25) & (z2 > 0.25) & np.isfinite(z1) & np.isfinite(z2)
                if np.sum(valid_pts) >= 8:
                    uv1_v = uv1[valid_pts]
                    uv2_v = uv2[valid_pts]
                    z1_v = z1[valid_pts]
                    p1 = np.stack([(uv1_v[:, 0] - cx) * z1_v / f, (uv1_v[:, 1] - cy) * z1_v / f, z1_v], axis=-1)
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        p1.astype(np.float64), uv2_v.astype(np.float64), k_mat.astype(np.float64), None
                    )
                    if ok and inliers is not None and len(inliers) >= 8:
                        r_cp, _ = cv2.Rodrigues(rvec)
                        cand_r = r_cp.T
                        cand_t = -r_cp.T @ tvec.reshape(3)
                        if float(np.linalg.norm(cand_t)) < 1.5:
                            r_rel = cand_r
                            t_rel = cand_t

        prev = poses[-1]
        poses.append(CameraPose(r_wc=prev.r_wc @ r_rel, t_wc=prev.r_wc @ t_rel + prev.t_wc, scale=1.0, valid=True))
        aligned.append(depths[i].copy())

    return k_mat, poses, aligned
