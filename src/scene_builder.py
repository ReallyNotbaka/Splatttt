from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import trimesh
from src.pose_tracker import CameraPose
from src.video_ingest import VideoFrame


def _chunked_index_add(
    target: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    chunk_size: int = 2000000,
) -> torch.Tensor:
    """Chunked torch.index_add_ on device to eliminate slow CPU np.add.at."""
    n = index.shape[0]
    if n <= chunk_size:
        target.index_add_(dim, index, source)
    else:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            target.index_add_(dim, index[start:end], source[start:end])
    return target


WORLD_FLIP = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)

LAST_STAGE4_TELEMETRY: Dict[str, Any] = {}


@dataclass
class Scene3D:
    points: np.ndarray          # (N, 3) float32
    colors: np.ndarray          # (N, 3) uint8
    normals: np.ndarray         # (N, 3) float32
    mesh_vertices: np.ndarray   # (M, 3) float32
    mesh_faces: np.ndarray      # (F, 3) int32
    mesh_colors: np.ndarray     # (M, 3) uint8
    poses: List[CameraPose]
    intrinsics: np.ndarray
    frame_size: tuple[int, int] # (w, h)
    mesh_uvs: Optional[np.ndarray] = None      # (M, 2) float32 OpenGL UV (v=0 bottom, v=1 top)
    mesh_texture: Optional[np.ndarray] = None  # (S, S, 3) uint8 RGB top-down image


def pose_to_world_camera(
    pose: CameraPose, look_dist: float = 2.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_world = WORLD_FLIP @ pose.r_wc.astype(np.float32)
    cam_pos = WORLD_FLIP @ pose.t_wc.astype(np.float32)
    forward = r_world @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = r_world @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    focal = cam_pos + forward * look_dist
    return cam_pos, focal, up


def build_frustum_lines(
    pose: CameraPose,
    frame_size: tuple[int, int],
    intrinsics: np.ndarray,
    scale: float = 0.14,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = frame_size
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    corners_uv = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32
    )
    x_c = (corners_uv[:, 0] - cx) * scale / fx
    y_c = (corners_uv[:, 1] - cy) * scale / fy
    z_c = np.full(4, scale, dtype=np.float32)
    local_pts = np.vstack(
        [[[0.0, 0.0, 0.0]], np.stack([x_c, y_c, z_c], axis=-1)]
    )

    r_world = WORLD_FLIP @ pose.r_wc.astype(np.float32)
    t_world = WORLD_FLIP @ pose.t_wc.astype(np.float32)
    world_pts = (local_pts @ r_world.T) + t_world[None, :]

    edges = [
        [2, 0, 1],
        [2, 0, 2],
        [2, 0, 3],
        [2, 0, 4],
        [2, 1, 2],
        [2, 2, 3],
        [2, 3, 4],
        [2, 4, 1],
    ]
    lines = np.array(edges, dtype=np.int64).ravel()
    return world_pts, lines


def _refine_multiview_depths_gpu(
    depths: List[np.ndarray],
    masks: List[np.ndarray],
    poses: List[CameraPose],
    intrinsics: np.ndarray,
    valid_indices: List[int],
    device: str = "cuda",
    cancel_cb: Optional[Callable[[], bool]] = None,
    scene_type: str = "indoor",
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    h, w = depths[0].shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    vy = np.cos(np.linspace(-np.pi * 0.45, np.pi * 0.45, h, dtype=np.float32))[:, None]
    vx = np.cos(np.linspace(-np.pi * 0.45, np.pi * 0.45, w, dtype=np.float32))[None, :]
    center_w = np.clip((vy * vx) ** 1.2, 0.08, 1.0)

    forwards = np.array([p.r_wc[:, 2] for p in poses], dtype=np.float32)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rx_t = torch.from_numpy((uu - cx) / fx).to(dev)
    ry_t = torch.from_numpy((vv - cy) / fy).to(dev)
    cw_t = torch.from_numpy(center_w).to(dev)

    is_sub = len(depths) == len(valid_indices)
    idx_to_sub = {idx: k for k, idx in enumerate(valid_indices)} if is_sub else {i: i for i in range(len(depths))}
    get_d = lambda idx: depths[idx_to_sub[idx]]
    get_m = lambda idx: masks[idx_to_sub[idx]]

    fused_depths = [d.copy() for d in depths]
    clean_masks = [m.copy() for m in masks]

    # Pre-upload depths & masks once to GPU if <= 420 views (~150MB total)
    gpu_depths = {}
    gpu_masks = {}
    if dev == "cuda" and len(valid_indices) <= 420:
        for idx in valid_indices:
            gpu_depths[idx] = torch.from_numpy(get_d(idx)).to(dev)
            gpu_masks[idx] = torch.from_numpy(get_m(idx)).to(dev)

    def _fetch_d(idx: int):
        return gpu_depths[idx] if idx in gpu_depths else torch.from_numpy(get_d(idx)).to(dev)

    def _fetch_m(idx: int):
        return gpu_masks[idx] if idx in gpu_masks else torch.from_numpy(get_m(idx)).to(dev)

    raw_fused_list = []
    valid_ts = np.array([poses[k].t_wc for k in valid_indices], dtype=np.float32)
    if len(valid_indices) >= 2:
        consec_steps = [
            float(np.linalg.norm(poses[valid_indices[k + 1]].t_wc - poses[valid_indices[k]].t_wc))
            for k in range(len(valid_indices) - 1)
        ]
        med_step = float(np.median(consec_steps)) if consec_steps else 0.20
    else:
        med_step = 0.20
    d_scale = max(1.0 if scene_type == "outdoor" else 0.60, 3.0 * med_step)

    with torch.no_grad():
        for count_idx, i in enumerate(valid_indices):
            if cancel_cb and (count_idx % 16 == 0) and cancel_cb():
                raise RuntimeError("build cancelled by user")

            cos_dots = forwards[valid_indices] @ forwards[i]
            d_trans = np.linalg.norm(valid_ts - poses[i].t_wc.astype(np.float32), axis=-1)
            spatial_scores = cos_dots / (1.0 + d_trans / d_scale)
            nbr_candidates = [
                valid_indices[k]
                for k in np.argsort(spatial_scores)[::-1]
                if valid_indices[k] != i and cos_dots[k] > 0.45 and d_trans[k] <= (4.5 * d_scale)
            ]
            if len(valid_indices) > 40 and len(nbr_candidates) > 12:
                nbr_candidates = nbr_candidates[:6] + nbr_candidates[6::2][:6]
            else:
                nbr_candidates = nbr_candidates[:12]
            nbrs = [j for j in nbr_candidates if (j in idx_to_sub or not is_sub)]
            if not nbrs:
                continue

            d_i = _fetch_d(i)
            m_i = _fetch_m(i)
            r_i = torch.from_numpy(poses[i].r_wc.astype(np.float32)).to(dev)
            t_i = torch.from_numpy(poses[i].t_wc.astype(np.float32)).to(dev)

            pts_ci = torch.stack([rx_t * d_i, ry_t * d_i, d_i], dim=-1).view(-1, 3)
            pts_w = (pts_ci @ r_i.T) + t_i.unsqueeze(0)

            ptscj_list = []
            rji_list = []
            grid_list = []
            zproj_list = []
            inside_list = []
            dj_list = []
            mj_list = []

            for j in nbrs:
                d_j = _fetch_d(j)
                m_j = _fetch_m(j)
                r_j = torch.from_numpy(poses[j].r_wc.astype(np.float32)).to(dev)
                t_j = torch.from_numpy(poses[j].t_wc.astype(np.float32)).to(dev)

                pts_cj = (pts_w - t_j.unsqueeze(0)) @ r_j
                zj_proj = pts_cj[:, 2].view(h, w)
                in_front = zj_proj > 0.25

                uj = fx * (pts_cj[:, 0] / torch.clamp(pts_cj[:, 2], min=0.1)) + cx
                vj = fy * (pts_cj[:, 1] / torch.clamp(pts_cj[:, 2], min=0.1)) + cy

                grid = torch.stack(
                    [((uj / (w - 1)) * 2.0 - 1.0).view(1, h, w), ((vj / (h - 1)) * 2.0 - 1.0).view(1, h, w)],
                    dim=-1,
                )
                inside = (
                    in_front
                    & (uj.view(h, w) >= 2)
                    & (uj.view(h, w) < w - 2)
                    & (vj.view(h, w) >= 2)
                    & (vj.view(h, w) < h - 2)
                )

                ptscj_list.append(pts_cj)
                rji_list.append(r_j.T @ r_i)
                grid_list.append(grid)
                zproj_list.append(zj_proj)
                inside_list.append(inside)
                dj_list.append(d_j)
                mj_list.append(m_j.float())

            if not grid_list:
                continue

            # Batch neighbor sampling into single [K, 1, H, W] operations
            k_nbrs = len(grid_list)
            stacked_grids = torch.cat(grid_list, dim=0)
            stacked_dj = torch.stack(dj_list, dim=0).unsqueeze(1)
            stacked_mj = torch.stack(mj_list, dim=0).unsqueeze(1)
            stacked_cw = cw_t.view(1, 1, h, w).expand(k_nbrs, -1, -1, -1)

            zj_samps = F.grid_sample(
                stacked_dj, stacked_grids, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze(1)
            mj_samps = (
                F.grid_sample(
                    stacked_mj, stacked_grids, mode="nearest", padding_mode="zeros", align_corners=True
                ).squeeze(1)
                > 0.5
            )
            cwj_samps = F.grid_sample(
                stacked_cw, stacked_grids, mode="bilinear", padding_mode="zeros", align_corners=True
            ).squeeze(1)

            stacked_zproj = torch.stack(zproj_list, dim=0)
            stacked_inside = torch.stack(inside_list, dim=0)

            dz = torch.abs(stacked_zproj - zj_samps)
            if scene_type == "outdoor":
                tol_agree = torch.clamp(0.035 * stacked_zproj, min=0.20, max=1.20)
                tol_cov = torch.clamp(0.045 * stacked_zproj, min=0.25, max=1.50)
                tol_viol = tol_cov
                sig_blend = torch.clamp(0.018 * stacked_zproj, min=0.09, max=0.55)
                is_covering = stacked_inside & mj_samps & (stacked_zproj <= (zj_samps + tol_cov))
                agree = stacked_inside & m_i.unsqueeze(0) & mj_samps & (dz < tol_agree)
                free_space_viol = stacked_inside & mj_samps & (stacked_zproj < (zj_samps - tol_viol))
                w_blend = cwj_samps * torch.exp(-((dz / sig_blend) ** 2))
            else:
                is_covering = stacked_inside & mj_samps & (stacked_zproj <= (zj_samps + 0.25))
                agree = stacked_inside & m_i.unsqueeze(0) & mj_samps & (dz < 0.20)
                free_space_viol = stacked_inside & mj_samps & (stacked_zproj < (zj_samps - 0.22))
                w_blend = cwj_samps * torch.exp(-((dz / 0.09) ** 2))

            cover_count = torch.sum(is_covering.int(), dim=0)
            agree_count = torch.sum(agree.int(), dim=0)
            viol_count = torch.sum(free_space_viol.int(), dim=0)

            w_agree = torch.where(agree, w_blend, torch.zeros_like(w_blend))

            # Exact 3D reprojection: transforms neighbor measurement back to camera i ray
            dcand_list = []
            for k in range(k_nbrs):
                scale_k = zj_samps[k].view(-1, 1) / torch.clamp(zproj_list[k].view(-1, 1), min=0.15)
                disp_j = ptscj_list[k] * (scale_k - 1.0)
                disp_i = disp_j @ rji_list[k]
                d_cand = torch.clamp(d_i + disp_i[:, 2].view(h, w), min=0.20)
                dcand_list.append(d_cand)

            stacked_dcand = torch.stack(dcand_list, dim=0)
            z_num = d_i * cw_t + torch.sum(stacked_dcand * w_agree, dim=0)
            z_den = cw_t + torch.sum(w_agree, dim=0)

            d_fused = (z_num / torch.clamp(z_den, min=1e-4)).cpu().numpy().astype(np.float32)
            raw_fused_list.append((i, d_fused))

            cov_np = cover_count.cpu().numpy()
            agr_np = agree_count.cpu().numpy()
            viol_np = viol_count.cpu().numpy()
            if len(valid_indices) >= 16:
                if scene_type == "outdoor":
                    mv_ok = (
                        ((cov_np < 2) | ((cov_np < 5) & (agr_np >= 1)) | (agr_np >= 2))
                        & (~((viol_np >= 5) & (viol_np > (agr_np * 2.5))))
                    )
                else:
                    mv_ok = (
                        ((cov_np < 2) | ((cov_np < 5) & (agr_np >= 1)) | (agr_np >= 2))
                        & (~((viol_np >= 3) & (viol_np > (agr_np * 1.5))))
                    )
            else:
                mv_ok = np.ones((h, w), dtype=bool)

            out_k = idx_to_sub[i] if is_sub else i
            clean_masks[out_k] = get_m(i) & mv_ok

    gpu_depths.clear()
    gpu_masks.clear()
    del rx_t, ry_t, cw_t
    if dev == "cuda":
        torch.cuda.empty_cache()

    # Multithread cv2.bilateralFilter across CPU threads
    if raw_fused_list:
        sigma_c = 0.05

        def _filter_job(item):
            idx_item, arr_item = item
            return idx_item, cv2.bilateralFilter(arr_item, d=5, sigmaColor=sigma_c, sigmaSpace=2.0)

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            filtered_results = list(pool.map(_filter_job, raw_fused_list))

        for idx_res, arr_filtered in filtered_results:
            out_k = idx_to_sub[idx_res] if is_sub else idx_res
            fused_depths[out_k] = arr_filtered

    return fused_depths, clean_masks


def _recolor_and_orient_vertices_gpu(
    verts_colmap: np.ndarray,
    norms_colmap: np.ndarray,
    fallback_cols: np.ndarray,
    frames: List[VideoFrame],
    depths: List[np.ndarray],
    masks: List[np.ndarray],
    poses: List[CameraPose],
    intrinsics: np.ndarray,
    valid_indices: List[int],
    device: str = "cuda",
    cancel_cb: Optional[Callable[[], bool]] = None,
    scene_type: str = "indoor",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(verts_colmap) == 0:
        return fallback_cols, np.zeros_like(verts_colmap), np.zeros(0, dtype=np.float32)

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    h, w = depths[0].shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    # if video has 500-1500+ frames, pick up to 240 evenly spaced views for fast crisp recoloring
    if len(valid_indices) > 240:
        pick = np.linspace(0, len(valid_indices) - 1, 240, dtype=int)
        recolor_ids = [valid_indices[k] for k in pick]
    else:
        recolor_ids = list(valid_indices)

    is_sub = len(depths) == len(valid_indices)
    idx_to_sub = {idx: k for k, idx in enumerate(valid_indices)} if is_sub else {i: i for i in range(len(depths))}
    get_d = lambda idx: depths[idx_to_sub[idx]]
    get_m = lambda idx: masks[idx_to_sub[idx]]

    m = len(verts_colmap)
    verts_t = torch.from_numpy(verts_colmap.astype(np.float32)).to(dev)
    norms_t = F.normalize(torch.from_numpy(norms_colmap.astype(np.float32)).to(dev), dim=-1)
    tsdf_cols_t = torch.from_numpy(fallback_cols.astype(np.float32)).to(dev)

    sharp_vals = np.array([frames[i].sharpness for i in recolor_ids], dtype=np.float32)
    med_sharp = max(1.0, float(np.median(sharp_vals)))

    col_acc = torch.zeros((m, 3), dtype=torch.float32, device=dev)
    dir_acc = torch.zeros((m, 3), dtype=torch.float32, device=dev)
    w_acc = torch.zeros((m, 1), dtype=torch.float32, device=dev)

    with torch.no_grad():
        for count_recolor, i in enumerate(recolor_ids):
            if cancel_cb and (count_recolor % 16 == 0) and cancel_cb():
                raise RuntimeError("build cancelled by user")

            r_wc = torch.from_numpy(poses[i].r_wc.astype(np.float32)).to(dev)
            t_wc = torch.from_numpy(poses[i].t_wc.astype(np.float32)).to(dev)

            pts_c = (verts_t - t_wc.unsqueeze(0)) @ r_wc
            z = pts_c[:, 2]
            in_front = z > 0.25

            u = fx * (pts_c[:, 0] / torch.clamp(z, min=0.1)) + cx
            v = fy * (pts_c[:, 1] / torch.clamp(z, min=0.1)) + cy

            u_n = (u / max(1, w - 1)) * 2.0 - 1.0
            v_n = (v / max(1, h - 1)) * 2.0 - 1.0
            inside = in_front & (u >= 2.0) & (u < w - 2.0) & (v >= 2.0) & (v < h - 2.0)

            if not torch.any(inside):
                continue

            vis_idx = torch.nonzero(inside, as_tuple=False).squeeze(-1)
            grid = torch.stack([u_n[vis_idx], v_n[vis_idx]], dim=-1).view(1, 1, -1, 2)

            d_img = torch.from_numpy(get_d(i)).to(dev).view(1, 1, h, w)
            m_img = torch.from_numpy(get_m(i).astype(np.float32)).to(dev).view(1, 1, h, w)

            z_obs = F.grid_sample(d_img, grid, mode="bilinear", padding_mode="border", align_corners=True).view(-1)
            m_obs = (
                F.grid_sample(m_img, grid, mode="nearest", padding_mode="zeros", align_corners=True).view(-1)
                > 0.5
            )

            dz = torch.abs(z[vis_idx] - z_obs)
            if scene_type == "outdoor":
                vis_tol = torch.clamp(0.035 * z[vis_idx], min=0.25, max=1.20)
                sig_d = torch.clamp(0.015 * z[vis_idx], min=0.08, max=0.45)
                vis_ok = (dz < vis_tol) & m_obs
            else:
                vis_ok = (dz < 0.25) & m_obs
                sig_d = 0.08
            if not torch.any(vis_ok):
                continue

            ok_idx = vis_idx[vis_ok]
            grid_ok = grid[:, :, vis_ok, :]

            rgb_t = (
                torch.from_numpy(np.ascontiguousarray(frames[i].rgb))
                .to(dev)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            rgb_samp = F.grid_sample(rgb_t, grid_ok, mode="bilinear", padding_mode="border", align_corners=True)
            rgb_vals = rgb_samp.squeeze(0).squeeze(1).T

            if len(ok_idx) >= 64:
                ref_rgb = tsdf_cols_t[ok_idx]
                bright = (rgb_vals.mean(dim=-1) > 25.0) & (rgb_vals.mean(dim=-1) < 235.0)
                if torch.sum(bright) >= 32:
                    ratio = (ref_rgb[bright] + 8.0) / (rgb_vals[bright] + 8.0)
                    gain = torch.median(ratio, dim=0).values.clamp(0.70, 1.42)
                    rgb_vals = torch.clamp(rgb_vals * gain.unsqueeze(0), 0.0, 255.0)

            view_dir = F.normalize(t_wc.unsqueeze(0) - verts_t[ok_idx], dim=-1)
            cos_abs = torch.abs((norms_t[ok_idx] * view_dir).sum(dim=-1)).clamp(min=0.05)

            vig = torch.clamp(1.0 - 0.55 * (u_n[ok_idx] ** 2 + v_n[ok_idx] ** 2) * 0.5, min=0.12)
            s_bonus = float(np.clip(frames[i].sharpness / med_sharp, 0.75, 1.35))
            dist_w = 1.0 / (z[ok_idx] + 0.55)
            sig_d_ok = sig_d[vis_ok] if isinstance(sig_d, torch.Tensor) else sig_d
            depth_agree = torch.exp(-((dz[vis_ok] / sig_d_ok) ** 2))

            score = (cos_abs * vig * s_bonus * dist_w * depth_agree).pow(2.5).unsqueeze(-1)

            col_acc[ok_idx] += rgb_vals * score
            dir_acc[ok_idx] += view_dir * score
            w_acc[ok_idx] += score

        w_flat = w_acc.squeeze(-1).cpu().numpy().astype(np.float32)
        has_col = w_flat > 1e-6
        recolored = fallback_cols.copy()
        if np.any(has_col):
            blended = (col_acc / torch.clamp(w_acc, min=1e-6)).cpu().numpy()
            recolored[has_col] = np.clip(np.round(blended[has_col]), 0, 255).astype(np.uint8)

        vis_dirs = F.normalize(dir_acc, dim=-1).cpu().numpy().astype(np.float32)

    del verts_t, norms_t, tsdf_cols_t, col_acc, dir_acc, w_acc
    if dev == "cuda":
        torch.cuda.empty_cache()

    return recolored, vis_dirs, w_flat


def _fill_floor_grid_holes(
    verts: np.ndarray,
    faces: np.ndarray,
    cols: np.ndarray,
    norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # fills enclosed unfilmed floor holes using distance-transform color/height copy (zero black bleed!)
    if len(verts) < 1000 or len(faces) < 1000:
        return verts, faces, cols, norms

    up_mask = norms[:, 1] > 0.78
    if np.sum(up_mask) < 300:
        return verts, faces, cols, norms

    y_floor = float(np.percentile(verts[up_mask, 1], 14.0))
    floor_mask = up_mask & (np.abs(verts[:, 1] - y_floor) < 0.14)
    if np.sum(floor_mask) < 250:
        return verts, faces, cols, norms

    cell = 0.025
    xmin, zmin = verts[:, [0, 2]].min(axis=0) - 0.40
    xmax, zmax = verts[:, [0, 2]].max(axis=0) + 0.40
    gw = int(np.ceil((xmax - xmin) / cell)) + 2
    gh = int(np.ceil((zmax - zmin) / cell)) + 2
    if gw < 16 or gh < 16 or gw > 800 or gh > 800:
        return verts, faces, cols, norms

    def to_grid(pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.clip(np.round((pts_3d[:, 0] - xmin) / cell).astype(np.int32), 0, gw - 1)
        gz = np.clip(np.round((pts_3d[:, 2] - zmin) / cell).astype(np.int32), 0, gh - 1)
        return gx, gz

    f_pts = verts[floor_mask]
    f_cols = cols[floor_mask].astype(np.float32)
    fx_i, fz_i = to_grid(f_pts)

    f_cnt = np.zeros((gh, gw), dtype=np.float32)
    f_y_sum = np.zeros((gh, gw), dtype=np.float32)
    f_c_sum = np.zeros((gh, gw, 3), dtype=np.float32)
    np.add.at(f_cnt, (fz_i, fx_i), 1.0)
    np.add.at(f_y_sum, (fz_i, fx_i), f_pts[:, 1])
    np.add.at(f_c_sum, (fz_i, fx_i), f_cols)

    has_floor = f_cnt > 0
    if not np.any(has_floor):
        return verts, faces, cols, norms

    # Build full room footprint (floor + walls + furniture) to seal interior against small furniture-edge floor gaps
    all_gx, all_gz = to_grid(verts)
    room_footprint = np.zeros((gh, gw), dtype=bool)
    room_footprint[all_gz, all_gx] = True
    # Only apply small doorway/gap closing when wall/furniture vertices exist above the floor
    has_above_floor = (verts[:, 1] > (y_floor + 0.14)) & (verts[:, 1] < (y_floor + 1.40))
    if np.sum(has_above_floor) >= 200:
        k_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        room_closed = cv2.morphologyEx(room_footprint.astype(np.uint8), cv2.MORPH_CLOSE, k_seal) > 0
        obs_gx, obs_gz = to_grid(verts[has_above_floor])
        obstacle_mask = np.zeros((gh, gw), dtype=bool)
        obstacle_mask[obs_gz, obs_gx] = True
        obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    else:
        room_closed = has_floor.copy()
        obstacle_mask = np.zeros((gh, gw), dtype=bool)

    # Guarantee a continuous 1-px exterior ring so floodFill from (0, 0) always marks all exterior pockets
    room_closed[0, :] = False
    room_closed[-1, :] = False
    room_closed[:, 0] = False
    room_closed[:, -1] = False
    ff_img = (~room_closed).astype(np.uint8)
    exterior_mask = np.zeros((gh + 2, gw + 2), dtype=np.uint8)
    cv2.floodFill(ff_img, exterior_mask, (0, 0), 2)
    is_exterior = (ff_img == 2)

    # Bridge unfilmed standing spot using morphological closing strictly inside non-exterior, non-obstacle room interior
    k_floor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    floor_envelope = cv2.morphologyEx(has_floor.astype(np.uint8), cv2.MORPH_CLOSE, k_floor) > 0
    floor_hole = floor_envelope & (~has_floor) & (~is_exterior) & (~obstacle_mask)
    if not np.any(floor_hole):
        return verts, faces, cols, norms

    floor_y_grid = np.full((gh, gw), y_floor, dtype=np.float32)
    floor_y_grid[has_floor] = f_y_sum[has_floor] / f_cnt[has_floor]
    floor_rgb_grid = np.zeros((gh, gw, 3), dtype=np.float32)
    floor_rgb_grid[has_floor] = f_c_sum[has_floor] / f_cnt[has_floor, None]

    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(floor_hole.astype(np.uint8), connectivity=8)
    floor_dil = cv2.dilate(has_floor.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    support_dil = floor_dil | cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0

    patch_mask = np.zeros((gh, gw), dtype=bool)
    for lbl in range(1, num_lbl):
        comp = labels == lbl
        touches_border = comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any()
        if touches_border:
            continue
        area_cells = int(stats[lbl, cv2.CC_STAT_AREA])
        area_m2 = area_cells * (cell * cell)
        if area_cells < 4 or area_m2 > 8.0:
            continue
        rim = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0) & (~comp)
        if not np.any(rim) or np.any(rim & is_exterior):
            continue
        floor_rim_frac = float(np.mean(floor_dil[rim]))
        total_rim_frac = float(np.mean(support_dil[rim]))
        if floor_rim_frac >= 0.50 and total_rim_frac >= 0.75:
            patch_mask |= comp

    if not np.any(patch_mask):
        return verts, faces, cols, norms

    # use nearest-valid-floor-pixel EDT so empty [0,0,0] cells can NEVER bleed black into the floor!
    _, (nn_y, nn_x) = distance_transform_edt(~has_floor, return_indices=True)
    filled_rgb = floor_rgb_grid[nn_y, nn_x]
    filled_y = floor_y_grid[nn_y, nn_x]

    # Inpaint the masked hole RGB using cv2.inpaint (TELEA, radius 5)
    inpaint_mask = cv2.dilate(patch_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) * 255
    inpaint_src = np.clip(np.round(filled_rgb), 0, 255).astype(np.uint8)
    try:
        inpainted_rgb = cv2.inpaint(inpaint_src, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA).astype(np.float32)
    except Exception:
        inpainted_rgb = filled_rgb.copy()

    # Wood grain synthesis: blur void-free filled_rgb so black empty cells never bias high-pass grain
    low_freq = cv2.GaussianBlur(filled_rgb, (21, 21), 0)
    grain = np.where(has_floor[:, :, None], filled_rgb - low_freq, np.zeros_like(filled_rgb))

    ys_f, xs_f = np.nonzero(has_floor)
    if len(ys_f) > 0:
        min_y, max_y = int(ys_f.min()), int(ys_f.max())
        min_x, max_x = int(xs_f.min()), int(xs_f.max())
        tw = min(32, max(8, (max_y - min_y) // 4, (max_x - min_x) // 4))

        best_var = -1.0
        best_quad = None
        stride_s = max(4, tw // 2)
        for sy in range(min_y, max(min_y + 1, max_y - tw + 1), stride_s):
            for sx in range(min_x, max(min_x + 1, max_x - tw + 1), stride_s):
                block_has = has_floor[sy : sy + tw, sx : sx + tw]
                if np.mean(block_has) > 0.80:
                    q = grain[sy : sy + tw, sx : sx + tw]
                    v = float(np.var(q))
                    if v > best_var:
                        best_var = v
                        best_quad = q

        if best_quad is not None and best_quad.shape[0] >= 4 and best_quad.shape[1] >= 4 and best_var > 1e-4:
            mirror_h = np.concatenate([best_quad, best_quad[:, ::-1]], axis=1)
            mirrored_tile = np.concatenate([mirror_h, mirror_h[::-1]], axis=0)
            reps_y = int(np.ceil(gh / mirrored_tile.shape[0]))
            reps_x = int(np.ceil(gw / mirrored_tile.shape[1]))
            tiled_grain = np.tile(mirrored_tile, (reps_y, reps_x, 1))[:gh, :gw]
        else:
            tiled_grain = np.zeros((gh, gw, 3), dtype=np.float32)
    else:
        tiled_grain = np.zeros((gh, gw, 3), dtype=np.float32)

    dist_inside = distance_transform_edt(~has_floor).astype(np.float32)
    feather = np.clip(dist_inside / 4.0, 0.0, 1.0)
    smooth_y = cv2.GaussianBlur(filled_y, (9, 9), 0)
    final_rgb = np.clip(inpainted_rgb + 0.75 * feather[:, :, None] * tiled_grain, 0.0, 255.0)

    # expand patch_ext by 9x9 with 0mm -> -5mm underlap, strictly clipped inside room perimeter
    patch_ext = (cv2.dilate(patch_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)) > 0) & (~is_exterior)
    dist_outside = distance_transform_edt(~patch_mask).astype(np.float32)
    underlap = np.clip(dist_outside / 4.0, 0.0, 1.0) * 0.005
    final_y = smooth_y - underlap
    ys, xs = np.nonzero(patch_ext)
    if len(xs) < 4:
        return verts, faces, cols, norms

    grid_vid = np.full((gh, gw), -1, dtype=np.int32)
    v_offset = len(verts)
    grid_vid[ys, xs] = np.arange(v_offset, v_offset + len(xs), dtype=np.int32)

    px = xmin + xs.astype(np.float32) * cell
    pz = zmin + ys.astype(np.float32) * cell
    py = final_y[ys, xs].astype(np.float32)

    p_verts = np.stack([px, py, pz], axis=-1).astype(np.float32)
    p_cols = np.clip(np.round(final_rgb[ys, xs]), 0, 255).astype(np.uint8)
    p_norms = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(xs), 1))

    v00 = grid_vid[:-1, :-1]
    v10 = grid_vid[1:, :-1]
    v01 = grid_vid[:-1, 1:]
    v11 = grid_vid[1:, 1:]
    q_ok = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) & (v11 >= 0)
    if not np.any(q_ok):
        return verts, faces, cols, norms

    t1 = np.stack([v00[q_ok], v10[q_ok], v01[q_ok]], axis=-1)
    t2 = np.stack([v10[q_ok], v11[q_ok], v01[q_ok]], axis=-1)
    p_faces = np.concatenate([t1, t2], axis=0).astype(np.int32)

    return (
        np.concatenate([verts, p_verts], axis=0),
        np.concatenate([faces, p_faces], axis=0),
        np.concatenate([cols, p_cols], axis=0),
        np.concatenate([norms, p_norms], axis=0),
    )


def _bilateral_polish_flat_surfaces(
    verts: np.ndarray,
    faces: np.ndarray,
    cols: np.ndarray,
    norms: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    # 6 iterations of Bilateral Face-Normal Filtering (sigma_r = 18 deg)
    # + 4 passes of Normal-Guided Vertex Projection with +-0.025m clamp
    # + 2 iterations of exposure-seam color blending on flat same-tone surfaces
    if len(verts) < 200 or len(faces) < 200:
        return verts, cols

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    n_v = len(verts)
    n_f = len(faces)
    chunk_e = 2_000_000

    with torch.no_grad():
        faces_t = torch.from_numpy(faces).long().to(dev)

        # 1. Extract edge-adjacent face pairs (dual graph)
        edges = torch.cat([faces_t[:, [0, 1]], faces_t[:, [1, 2]], faces_t[:, [2, 0]]], dim=0)
        face_ids = torch.arange(n_f, dtype=torch.long, device=dev).repeat_interleave(3)
        sorted_edges, _ = torch.sort(edges, dim=1)
        del edges
        keys = sorted_edges[:, 0] * 1000000000 + sorted_edges[:, 1]
        del sorted_edges
        sorted_keys, sort_idx = torch.sort(keys)
        del keys
        sorted_fids = face_ids[sort_idx]
        del face_ids, sort_idx

        mask = sorted_keys[:-1] == sorted_keys[1:]
        del sorted_keys
        fa = sorted_fids[:-1][mask]
        fb = sorted_fids[1:][mask]
        del sorted_fids, mask
        valid_pair = fa != fb
        src_t = torch.cat([fa[valid_pair], fb[valid_pair]], dim=0)
        dst_t = torch.cat([fb[valid_pair], fa[valid_pair]], dim=0)
        del fa, fb, valid_pair

        new_verts = verts.copy()
        if len(src_t) > 0:
            verts_t = torch.from_numpy(verts).float().to(dev)
            v0 = verts_t[faces_t[:, 0]]
            v1 = verts_t[faces_t[:, 1]]
            v2 = verts_t[faces_t[:, 2]]
            centroids = (v0 + v1 + v2) / 3.0
            cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
            del v0, v1, v2
            cross_norm = torch.norm(cross, dim=-1, keepdim=True)
            areas = 0.5 * cross_norm
            normals = cross / torch.clamp(cross_norm, min=1e-8)
            del cross, cross_norm

            n_dual = len(src_t)
            d_s = torch.empty(n_dual, dtype=torch.float32, device=dev)
            for st in range(0, n_dual, chunk_e):
                en = min(st + chunk_e, n_dual)
                d_s[st:en] = torch.norm(centroids[src_t[st:en]] - centroids[dst_t[st:en]], dim=-1)
            sigma_s = torch.clamp(torch.mean(d_s), min=0.005)
            w_s = torch.exp(-(d_s ** 2) / (2.0 * sigma_s ** 2))
            del d_s
            sigma_r = float(np.radians(22.0))

            # 6 iterations of Bilateral Face-Normal Filtering (chunked)
            for _ in range(6):
                acc_n = normals * areas
                for st in range(0, n_dual, chunk_e):
                    en = min(st + chunk_e, n_dual)
                    s_c = src_t[st:en]
                    d_c = dst_t[st:en]
                    n_src = normals[s_c]
                    n_dst = normals[d_c]
                    cos_ang = torch.clamp((n_src * n_dst).sum(dim=-1), -1.0, 1.0)
                    ang = torch.acos(cos_ang)
                    w_r = torch.exp(-(ang ** 2) / (2.0 * sigma_r ** 2))
                    w_tot = areas[d_c].squeeze(-1) * w_s[st:en] * w_r
                    acc_n.index_add_(0, s_c, n_dst * w_tot.unsqueeze(-1))
                normals = acc_n / torch.clamp(torch.norm(acc_n, dim=-1, keepdim=True), min=1e-8)
                del acc_n
            del src_t, dst_t, w_s, areas

            # 4 passes of Normal-Guided Vertex Projection (chunked, refreshed centroids)
            v_orig = verts_t.clone()
            v_curr = verts_t
            v_incident = faces_t.view(-1)
            f_incident = torch.arange(n_f, dtype=torch.long, device=dev).repeat_interleave(3)
            n_inc = len(v_incident)

            cnt = torch.zeros((n_v, 1), dtype=torch.float32, device=dev)
            for st in range(0, n_inc, chunk_e):
                en = min(st + chunk_e, n_inc)
                cnt.index_add_(0, v_incident[st:en], torch.ones((en - st, 1), dtype=torch.float32, device=dev))
            cnt.clamp_(min=1.0)

            for pass_i in range(4):
                if pass_i > 0:
                    centroids = (v_curr[faces_t[:, 0]] + v_curr[faces_t[:, 1]] + v_curr[faces_t[:, 2]]) / 3.0
                sum_disp = torch.zeros_like(v_curr)
                for st in range(0, n_inc, chunk_e):
                    en = min(st + chunk_e, n_inc)
                    vi_c = v_incident[st:en]
                    fi_c = f_incident[st:en]
                    nf_c = normals[fi_c]
                    dot_val = ((centroids[fi_c] - v_curr[vi_c]) * nf_c).sum(dim=-1, keepdim=True)
                    sum_disp.index_add_(0, vi_c, nf_c * dot_val)
                v_curr = v_curr + 0.5 * (sum_disp / cnt)
                del sum_disp

                cum_disp = v_curr - v_orig
                disp_norm = torch.norm(cum_disp, dim=-1, keepdim=True)
                scale = torch.clamp(0.025 / torch.clamp(disp_norm, min=1e-6), max=1.0)
                v_curr = v_orig + cum_disp * scale
                del cum_disp, disp_norm, scale

            new_verts = v_curr.cpu().numpy().astype(np.float32)
            del v_orig, v_curr, v_incident, f_incident, cnt, centroids, normals
        else:
            del src_t, dst_t

        if dev == "cuda":
            torch.cuda.empty_cache()

        # 2. Exposure-seam color blending on flat same-tone walls/doors (2 iterations, chunked)
        cols_t = torch.from_numpy(cols.astype(np.float32)).to(dev)
        norms_t = torch.from_numpy(norms.astype(np.float32)).to(dev)

        u_t = torch.cat([faces_t[:, 0], faces_t[:, 1], faces_t[:, 2]], dim=0)
        v_t = torch.cat([faces_t[:, 1], faces_t[:, 2], faces_t[:, 0]], dim=0)
        a_t = torch.cat([u_t, v_t], dim=0)
        b_t = torch.cat([v_t, u_t], dim=0)
        del u_t, v_t, faces_t

        at_chunks, bt_chunks, wc_chunks = [], [], []
        n_edges = a_t.shape[0]
        for st in range(0, n_edges, chunk_e):
            en = min(st + chunk_e, n_edges)
            ac = a_t[st:en]
            bc = b_t[st:en]
            n_dot = (norms_t[ac] * norms_t[bc]).sum(dim=-1)
            coplanar = n_dot > 0.86
            if not torch.any(coplanar):
                continue
            ac = ac[coplanar]
            bc = bc[coplanar]
            c_diff = torch.abs(cols_t[ac] - cols_t[bc]).max(dim=-1).values
            same_tone = c_diff < 32.0
            if torch.any(same_tone):
                at_chunks.append(ac[same_tone])
                bt_chunks.append(bc[same_tone])
                wc_chunks.append(torch.exp(-((c_diff[same_tone] / 16.0) ** 2)))
        del a_t, b_t, norms_t

        if at_chunks:
            at_t = torch.cat(at_chunks, dim=0)
            bt_t = torch.cat(bt_chunks, dim=0)
            w_col_t = torch.cat(wc_chunks, dim=0)
            del at_chunks, bt_chunks, wc_chunks

            sum_wc = torch.zeros(n_v, dtype=torch.float32, device=dev)
            _chunked_index_add(sum_wc, 0, at_t, w_col_t)
            has_c = sum_wc > 1e-4

            n_st = at_t.shape[0]
            for _ in range(2):
                acc_c = torch.zeros((n_v, 3), dtype=torch.float32, device=dev)
                for st in range(0, n_st, chunk_e):
                    en = min(st + chunk_e, n_st)
                    acc_c.index_add_(0, at_t[st:en], cols_t[bt_t[st:en]] * w_col_t[st:en].unsqueeze(-1))
                nbr_avg = torch.zeros_like(cols_t)
                nbr_avg[has_c] = acc_c[has_c] / sum_wc[has_c].unsqueeze(-1)
                cols_t[has_c] = 0.60 * cols_t[has_c] + 0.40 * nbr_avg[has_c]
                del acc_c, nbr_avg
            del at_t, bt_t, w_col_t, sum_wc, has_c

        final_cols = torch.clamp(torch.round(cols_t), 0, 255).cpu().numpy().astype(np.uint8)
        del cols_t

    if dev == "cuda":
        torch.cuda.empty_cache()

    return new_verts, final_cols


def select_diverse_views(
    frames: List[VideoFrame],
    poses: List[CameraPose],
    valid_indices: List[int],
    max_views: int = 260,
) -> List[int]:
    if len(valid_indices) <= max_views:
        return list(valid_indices)
    # divide valid trajectory into max_views buckets and pick the sharpest frame per bucket
    buckets = np.array_split(np.array(valid_indices, dtype=int), max_views)
    picked = []
    for b in buckets:
        if len(b) == 0:
            continue
        best_i = int(max(b, key=lambda idx: frames[int(idx)].sharpness))
        picked.append(best_i)
    return picked


def _extract_outdoor_context_mesh(
    fuse_indices: List[int],
    idx_to_sub: Dict[int, int],
    fused_depths: List[np.ndarray],
    clean_masks: List[np.ndarray],
    frames: List[VideoFrame],
    poses: List[CameraPose],
    intrinsics: np.ndarray,
    tsdf_depth_max: float,
    tsdf_voxel: float,
    hero_verts: np.ndarray,
    o3d_dev_str: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds a coarse, grounded context ring around the outdoor hero mesh.

    Uses a second, 4x coarser VoxelBlockGrid with a strict multi-view gate, then
    strips everything that overlaps the hero mesh, floats in mid-air, sits too
    far from every camera, or forms a tiny detached island.
    """
    empty = (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.uint8),
    )
    if len(fuse_indices) == 0 or len(hero_verts) == 0:
        return empty

    outer_depth_max = float(np.clip(tsdf_depth_max * 2.1, 16.0, 26.0))
    outer_voxel = float(np.clip(tsdf_voxel * 4.0, 0.08, 0.14))
    outer_block_count = 60000

    dev = o3c.Device(o3d_dev_str)

    def _make_vbg(device: o3c.Device) -> "o3d.t.geometry.VoxelBlockGrid":
        return o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=outer_voxel,
            block_resolution=8,
            block_count=outer_block_count,
            device=device,
        )

    try:
        vbg_outer = _make_vbg(dev)
    except Exception as e:
        if o3d_dev_str == "CUDA:0":
            print(f"[WARNING] coarse context VBG CUDA alloc failed ({e}); using CPU:0")
            dev = o3c.Device("CPU:0")
            o3d_dev_str = "CPU:0"
            vbg_outer = _make_vbg(dev)
        else:
            raise

    k_t = o3c.Tensor(intrinsics.astype(np.float64), device=dev)
    err_count = 0
    for count_i, idx in enumerate(fuse_indices):
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")

        sub_i = idx_to_sub[idx]
        depth = fused_depths[sub_i].copy()
        depth[~clean_masks[sub_i]] = 0.0
        if np.count_nonzero(depth) < 120:
            continue

        depth_u16 = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_u16, device=dev))
        color_t = o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(frames[idx].rgb), device=dev)
        )

        t_wc = np.eye(4, dtype=np.float64)
        t_wc[:3, :3] = poses[idx].r_wc
        t_wc[:3, 3] = poses[idx].t_wc
        ext_t = o3c.Tensor(np.linalg.inv(t_wc), device=dev)

        try:
            coords = vbg_outer.compute_unique_block_coordinates(
                depth_t, k_t, ext_t, depth_scale=1000.0,
                depth_max=outer_depth_max, trunc_voxel_multiplier=3.5,
            )
            if coords.shape[0] > 0:
                vbg_outer.integrate(
                    coords, depth_t, color_t, k_t, k_t, ext_t,
                    depth_scale=1000.0, depth_max=outer_depth_max,
                    trunc_voxel_multiplier=3.5,
                )
        except Exception as e:
            err_count += 1
            if err_count <= 3:
                print(f"[WARNING] coarse context integrate error on view {idx}: {e}")

        if progress_cb and (count_i % 8 == 0 or count_i == len(fuse_indices) - 1):
            progress_cb(
                0.925,
                f"fusing coarse surrounding context ({count_i + 1}/{len(fuse_indices)})",
            )

    w_outer = 2.4 if len(fuse_indices) >= 60 else 1.6
    mesh_o = vbg_outer.extract_triangle_mesh(weight_threshold=w_outer).to_legacy()
    del vbg_outer, k_t
    if o3d_dev_str == "CUDA:0":
        try:
            o3c.cuda.release_cache()
        except Exception:
            pass

    mesh_o.remove_degenerate_triangles()
    mesh_o.remove_unreferenced_vertices()
    if len(mesh_o.triangles) < 100:
        return empty

    # --- overlap removal: never duplicate geometry the hero mesh already owns
    hero_sample = np.asarray(hero_verts, dtype=np.float32)
    if len(hero_sample) > 120_000:
        sel = np.random.default_rng(0).choice(
            len(hero_sample), 120_000, replace=False
        )
        hero_sample = hero_sample[sel]
    o_verts = np.asarray(mesh_o.vertices, dtype=np.float32)
    o_faces = np.asarray(mesh_o.triangles, dtype=np.int32)
    tree = cKDTree(hero_sample)
    tri_c = (o_verts[o_faces[:, 0]] + o_verts[o_faces[:, 1]] + o_verts[o_faces[:, 2]]) / 3.0
    d_hero, _ = tree.query(tri_c)
    keep = d_hero >= max(0.22, outer_voxel * 2.0)
    if not np.any(keep):
        return empty
    mesh_o.remove_triangles_by_mask(~keep)
    mesh_o.remove_unreferenced_vertices()
    if len(mesh_o.triangles) < 100:
        return empty

    # --- anti-garbage pruning: tiny islands, floating shreds, far outliers ---
    ground_y_colmap = float(np.percentile(np.asarray(hero_verts, np.float32)[:, 1], 92.0))
    cam_pts = np.array(
        [poses[i].t_wc for i in fuse_indices], dtype=np.float32
    )
    tri_clusters, cluster_n_tris, _ = mesh_o.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    v_np = np.asarray(mesh_o.vertices, dtype=np.float32)
    f_np = np.asarray(mesh_o.triangles, dtype=np.int32)
    tri_c2 = (v_np[f_np[:, 0]] + v_np[f_np[:, 1]] + v_np[f_np[:, 2]]) / 3.0

    # Label space can be sparse after the earlier removals; count from labels.
    num_c = int(tri_clusters.max()) + 1 if len(tri_clusters) > 0 else 0
    cluster_sizes = np.bincount(tri_clusters, minlength=num_c)
    drop = np.zeros(num_c, dtype=bool)
    for cid in range(num_c):
        mask = tri_clusters == cid
        if cluster_sizes[cid] < 250:
            drop[cid] = True
            continue
        cl_verts = v_np[np.unique(f_np[mask])]
        if len(cl_verts) > 0 and float(cl_verts[:, 1].max()) < ground_y_colmap - 1.60:
            drop[cid] = True
            continue
        c_med = np.median(tri_c2[mask], axis=0)
        d_cam = float(np.min(np.linalg.norm(cam_pts - c_med[None, :], axis=1)))
        if d_cam > outer_depth_max * 0.92:
            drop[cid] = True

    rm_mask = drop[tri_clusters]
    if np.any(rm_mask):
        mesh_o.remove_triangles_by_mask(rm_mask)
        mesh_o.remove_unreferenced_vertices()
    if len(mesh_o.triangles) < 100:
        return empty

    mesh_o = mesh_o.filter_smooth_taubin(number_of_iterations=1)
    mesh_o.compute_vertex_normals()

    o_verts = np.asarray(mesh_o.vertices, dtype=np.float32)
    o_faces = np.asarray(mesh_o.triangles, dtype=np.int32)
    o_norms = np.asarray(mesh_o.vertex_normals, dtype=np.float32)
    if len(o_norms) != len(o_verts):
        o_norms = np.zeros_like(o_verts)
    raw_c = np.asarray(mesh_o.vertex_colors)
    if len(raw_c) == len(o_verts) and len(o_verts) > 0:
        o_cols = np.clip(np.round(raw_c * 255.0), 0, 255).astype(np.uint8)
    else:
        o_cols = np.full((len(o_verts), 3), 180, dtype=np.uint8)

    return o_verts, o_faces, o_norms, o_cols


def build_3d_scene(
    frames: List[VideoFrame],
    depths: List[np.ndarray],
    masks: List[np.ndarray],
    poses: List[CameraPose],
    intrinsics: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    device: str = "cuda",
    point_stride: int = 2,
    mesh_stride: int = 4,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    scene_type: str = "indoor",
    video_path: Optional[Path] = None,
    k1_cal: float = 0.0,
    bake_texture: bool = True,
) -> Scene3D:
    t0_stage4 = time.perf_counter()
    if not depths or not frames:
        empty_f3 = np.zeros((0, 3), dtype=np.float32)
        empty_u3 = np.zeros((0, 3), dtype=np.uint8)
        empty_i3 = np.zeros((0, 3), dtype=np.int32)
        fh, fw = frames[0].rgb.shape[:2] if frames else (0, 0)
        return Scene3D(
            points=empty_f3,
            colors=empty_u3,
            normals=empty_f3,
            mesh_vertices=empty_f3,
            mesh_faces=empty_i3,
            mesh_colors=empty_u3,
            poses=poses,
            intrinsics=intrinsics,
            frame_size=(fw, fh),
        )
    h, w = depths[0].shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    torch_dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    if selected_indices is not None and len(selected_indices) > 0:
        fuse_indices = [idx for idx in selected_indices if idx < len(frames)]
    else:
        valid_indices = [i for i, p in enumerate(poses) if p.valid]
        if not valid_indices:
            valid_indices = [0]
        fuse_indices = select_diverse_views(frames, poses, valid_indices, max_views=260)

    if cancel_cb and cancel_cb():
        raise RuntimeError("build cancelled by user")

    if progress_cb:
        progress_cb(0.77, "blending multi-view depth consensus on GPU...")

    t0_mv = time.perf_counter()
    fused_depths, clean_masks = _refine_multiview_depths_gpu(
        depths, masks, poses, intrinsics, fuse_indices, device=torch_dev, cancel_cb=cancel_cb, scene_type=scene_type
    )
    multiview_s = time.perf_counter() - t0_mv

    if cancel_cb and cancel_cb():
        raise RuntimeError("build cancelled by user")

    is_sub = len(depths) == len(fuse_indices)
    idx_to_sub = {idx: k for k, idx in enumerate(fuse_indices)} if is_sub else {i: i for i in range(len(depths))}

    t0_tsdf = time.perf_counter()
    sample_pts = []
    uu_s, vv_s = np.meshgrid(np.arange(0, w, 16, dtype=np.float32), np.arange(0, h, 16, dtype=np.float32))
    for idx in fuse_indices[:: max(1, len(fuse_indices) // 12)]:
        sub_i = idx_to_sub[idx]
        sm = clean_masks[sub_i][::16, ::16]
        if not np.any(sm):
            continue
        sz = fused_depths[sub_i][::16, ::16][sm]
        sx = (uu_s[sm] - cx) * sz / fx
        sy = (vv_s[sm] - cy) * sz / fy
        pc = np.stack([sx, sy, sz], axis=-1)
        pw = (pc @ poses[idx].r_wc.astype(np.float32).T) + poses[idx].t_wc.astype(np.float32)[None, :]
        sample_pts.append(pw)

    if sample_pts:
        cat_s = np.concatenate(sample_pts, axis=0)
        span = float(np.linalg.norm(np.percentile(cat_s, 95, axis=0) - np.percentile(cat_s, 5, axis=0)))
    else:
        span = 4.0

    # Wire mesh_stride (2..10) to finer voxel resolution (stride 2: 0.0014/0.0075, stride 3: 0.0017/0.0085, stride 4: 0.0019/0.0095, stride 6: 0.0032/0.0160)
    mesh_stride_clamped = max(2, min(10, int(mesh_stride)))
    stride_pts = [2, 3, 4, 6, 10]
    mult_pts = [0.0014, 0.0017, 0.0019, 0.0032, 0.0055]
    min_pts = [0.0075, 0.0085, 0.0095, 0.0160, 0.0280]
    voxel_mult = float(np.interp(mesh_stride_clamped, stride_pts, mult_pts))
    min_vox = float(np.interp(mesh_stride_clamped, stride_pts, min_pts))

    cam_pts_f = np.array([poses[idx].t_wc for idx in fuse_indices if poses[idx].valid], dtype=np.float32)
    if scene_type == "outdoor":
        if len(cam_pts_f) >= 6:
            orbit_span = float(np.linalg.norm(np.percentile(cam_pts_f, 95, axis=0) - np.percentile(cam_pts_f, 5, axis=0)))
            ref_span = max(3.5, min(span, orbit_span * 1.45)) if orbit_span > 0.5 else min(span, 10.0)
            orbit_depth_cap = float(np.clip(max(7.5, orbit_span * 1.45), 7.5, 13.0)) if orbit_span > 0.5 else 11.0
        else:
            orbit_span = 0.0
            ref_span = min(span, 10.0)
            orbit_depth_cap = 11.0

        cat_d_list = [fused_depths[idx_to_sub[idx]][clean_masks[idx_to_sub[idx]]] for idx in fuse_indices if np.any(clean_masks[idx_to_sub[idx]])]
        if cat_d_list:
            cat_d = np.concatenate(cat_d_list)
            tsdf_depth_max = float(np.clip(np.percentile(cat_d, 85.0) * 1.10, 7.5, orbit_depth_cap))
        else:
            tsdf_depth_max = orbit_depth_cap

        min_outdoor_vox = max(min_vox, tsdf_depth_max / 520.0, 0.018)
        tsdf_voxel = float(np.clip(ref_span * voxel_mult, min_outdoor_vox, 0.032))
        trunc_mult = 2.8
        block_count = 120000
    else:
        tsdf_depth_max = 10.5
        tsdf_voxel = float(np.clip(span * voxel_mult, min_vox, 0.038))
        trunc_mult = 4.0
        block_count = 120000

    # Dynamic Open3D device: CUDA:0 if available else CPU:0
    o3d_dev_str = "CUDA:0" if (device == "cuda" and o3c.cuda.is_available()) else "CPU:0"
    dev = o3c.Device(o3d_dev_str)
    try:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=tsdf_voxel,
            block_resolution=8,
            block_count=block_count,
            device=dev,
        )
    except Exception as e:
        if o3d_dev_str == "CUDA:0":
            print(f"[WARNING] VoxelBlockGrid CUDA alloc failed ({e}); falling back to CPU:0")
            o3d_dev_str = "CPU:0"
            dev = o3c.Device(o3d_dev_str)
            vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=tsdf_voxel,
                block_resolution=8,
                block_count=block_count,
                device=dev,
            )
        else:
            raise e
    k_t = o3c.Tensor(intrinsics.astype(np.float64), device=dev)
    tsdf_err_count = 0
    max_active_blocks = 110000 if o3d_dev_str == "CUDA:0" else 180000
    block_cap_hit = False

    for count_i, idx in enumerate(fuse_indices):
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")

        sub_i = idx_to_sub[idx]
        depth = fused_depths[sub_i].copy()
        depth[~clean_masks[sub_i]] = 0.0
        if np.count_nonzero(depth) < 120:
            continue

        depth_u16 = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)

        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_u16, device=dev))
        color_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(frames[idx].rgb), device=dev))

        t_wc = np.eye(4, dtype=np.float64)
        t_wc[:3, :3] = poses[idx].r_wc
        t_wc[:3, 3] = poses[idx].t_wc
        ext_t = o3c.Tensor(np.linalg.inv(t_wc), device=dev)

        try:
            coords = vbg.compute_unique_block_coordinates(
                depth_t, k_t, ext_t, depth_scale=1000.0, depth_max=tsdf_depth_max, trunc_voxel_multiplier=trunc_mult
            )
            if coords.shape[0] > 0:
                cur_blocks = int(vbg.hashmap().size())
                if cur_blocks >= max_active_blocks:
                    if not block_cap_hit:
                        print(f"[WARNING] TSDF active block cap ({max_active_blocks}) reached at view {idx}; updating existing blocks only")
                        block_cap_hit = True
                    _, found_mask = vbg.hashmap().find(coords)
                    coords = coords[found_mask]
                if coords.shape[0] > 0:
                    vbg.integrate(
                        coords,
                        depth_t,
                        color_t,
                        k_t,
                        k_t,
                        ext_t,
                        depth_scale=1000.0,
                        depth_max=tsdf_depth_max,
                        trunc_voxel_multiplier=trunc_mult,
                    )
        except Exception as e:
            tsdf_err_count += 1
            if tsdf_err_count <= 3:
                print(f"[WARNING] TSDF integrate error on view {idx}: {e}")

        if progress_cb and (count_i % 6 == 0 or count_i == len(fuse_indices) - 1):
            progress_cb(
                0.80 + 0.12 * ((count_i + 1) / len(fuse_indices)),
                f"integrating high-res TSDF surface ({count_i + 1}/{len(fuse_indices)})",
            )

    if cancel_cb and cancel_cb():
        raise RuntimeError("build cancelled by user")

    if scene_type == "outdoor":
        if len(fuse_indices) >= 60:
            w_thresh = 2.5
        elif len(fuse_indices) >= 20:
            w_thresh = 1.6
        else:
            w_thresh = 0.45
    else:
        if len(fuse_indices) >= 40:
            w_thresh = 1.15
        elif len(fuse_indices) >= 16:
            w_thresh = 1.05
        else:
            w_thresh = 0.35

    if o3d_dev_str == "CUDA:0" and int(vbg.hashmap().size()) > 55000:
        vbg = vbg.cpu()
        try:
            o3c.cuda.release_cache()
        except Exception:
            pass

    mesh_o3d = vbg.extract_triangle_mesh(weight_threshold=w_thresh).to_legacy()
    if len(mesh_o3d.triangles) < 100 and w_thresh > 0.35:
        mesh_o3d = vbg.extract_triangle_mesh(weight_threshold=0.35).to_legacy()

    # Release ~1.2GB VoxelBlockGrid VRAM immediately before GPU recoloring & bilateral polishing
    del vbg, k_t
    if o3d_dev_str == "CUDA:0":
        try:
            o3c.cuda.release_cache()
        except Exception:
            pass
    if torch_dev == "cuda":
        torch.cuda.empty_cache()

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    if len(mesh_o3d.triangles) > 400:
        tri_clusters, cluster_n_tris, _ = mesh_o3d.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        n_total_tris = len(mesh_o3d.triangles)
        max_c_id = int(np.argmax(cluster_n_tris)) if len(cluster_n_tris) > 0 else 0

        if scene_type == "outdoor":
            largest_tris = int(cluster_n_tris[max_c_id]) if len(cluster_n_tris) > 0 else 0
            min_cluster = max(2500, min(25000, int(largest_tris * 0.025)))
            drop_cluster = (cluster_n_tris < min_cluster)
            drop_cluster[max_c_id] = False

            if len(cam_pts_f) > 0 and len(cluster_n_tris) > 1:
                v_np = np.asarray(mesh_o3d.vertices, dtype=np.float32)
                f_np = np.asarray(mesh_o3d.triangles, dtype=np.int32)
                tri_c = (v_np[f_np[:, 0]] + v_np[f_np[:, 1]] + v_np[f_np[:, 2]]) / 3.0
                far_lim = min(tsdf_depth_max * 0.70, 5.8)
                for cid in range(len(cluster_n_tris)):
                    if cid == max_c_id or drop_cluster[cid]:
                        continue
                    c_pts = tri_c[tri_clusters == cid]
                    if len(c_pts) == 0:
                        continue
                    c_med = np.median(c_pts, axis=0)
                    d_cam = float(np.min(np.linalg.norm(cam_pts_f - c_med[None, :], axis=1)))
                    if d_cam > far_lim:
                        drop_cluster[cid] = True

            rm_mask = drop_cluster[tri_clusters]
        else:
            if n_total_tris > 50000:
                min_cluster = max(450, min(1800, int(n_total_tris * 0.002)))
            else:
                min_cluster = 150
            rm_mask = (cluster_n_tris[tri_clusters] < min_cluster) & (tri_clusters != max_c_id)

        if np.sum(~rm_mask) > 100:
            mesh_o3d.remove_triangles_by_mask(rm_mask)
            mesh_o3d.remove_unreferenced_vertices()

    if len(mesh_o3d.vertices) > 100:
        taubin_iters = 1 if scene_type == "outdoor" else 2
        mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=taubin_iters)
    if len(mesh_o3d.vertices) > 0:
        mesh_o3d.compute_vertex_normals()

    tsdf_s = time.perf_counter() - t0_tsdf

    m_verts_raw = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    m_faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)
    m_norms_raw = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)
    if len(m_norms_raw) != len(m_verts_raw):
        m_norms_raw = np.zeros_like(m_verts_raw)
    raw_vc = np.asarray(mesh_o3d.vertex_colors)
    if len(raw_vc) == len(m_verts_raw) and len(m_verts_raw) > 0:
        tsdf_cols = np.clip(np.round(raw_vc * 255.0), 0, 255).astype(np.uint8)
    else:
        tsdf_cols = np.full((len(m_verts_raw), 3), 180, dtype=np.uint8)

    # --- Tier 2: coarse outdoor context ring (outdoor only) -----------------
    outer_context_tris = 0
    hero_face_count: Optional[int] = None
    if (
        scene_type == "outdoor"
        and len(fuse_indices) >= 16
        and len(m_faces) > 200
    ):
        if progress_cb:
            progress_cb(0.925, "fusing coarse surrounding context...")
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")
        try:
            o_v, o_f, o_n, o_c = _extract_outdoor_context_mesh(
                fuse_indices=fuse_indices,
                idx_to_sub=idx_to_sub,
                fused_depths=fused_depths,
                clean_masks=clean_masks,
                frames=frames,
                poses=poses,
                intrinsics=intrinsics,
                tsdf_depth_max=tsdf_depth_max,
                tsdf_voxel=tsdf_voxel,
                hero_verts=m_verts_raw,
                o3d_dev_str=o3d_dev_str,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            if len(o_f) > 0:
                hero_face_count = int(len(m_faces))
                base_n = len(m_verts_raw)
                m_verts_raw = np.vstack([m_verts_raw, o_v]).astype(np.float32)
                m_faces = np.vstack([m_faces, o_f + base_n]).astype(np.int32)
                m_norms_raw = np.vstack([m_norms_raw, o_n]).astype(np.float32)
                tsdf_cols = np.vstack([tsdf_cols, o_c]).astype(np.uint8)
                outer_context_tris = int(len(o_f))
        except Exception as e:
            if "cancel" in str(e).lower():
                raise
            print(f"[WARNING] coarse outdoor context extraction failed ({e})")
            hero_face_count = None
            outer_context_tris = 0

    if progress_cb:
        progress_cb(0.94, "projecting exposure-matched colors & orienting dollhouse faces...")

    recolor_fuse_indices = fuse_indices
    recolor_depths = fused_depths
    recolor_masks = clean_masks
    if bake_texture and len(m_faces) >= 50 and len(fuse_indices) > 32:
        pick = np.linspace(0, len(fuse_indices) - 1, 32, dtype=int)
        recolor_fuse_indices = [fuse_indices[k] for k in pick]
        recolor_depths = [fused_depths[idx_to_sub[idx]] for idx in recolor_fuse_indices]
        recolor_masks = [clean_masks[idx_to_sub[idx]] for idx in recolor_fuse_indices]

    t0_recolor = time.perf_counter()
    m_cols, vis_dirs_colmap, vis_weights = _recolor_and_orient_vertices_gpu(
        m_verts_raw,
        m_norms_raw,
        tsdf_cols,
        frames,
        recolor_depths,
        recolor_masks,
        poses,
        intrinsics,
        recolor_fuse_indices,
        device=torch_dev,
        cancel_cb=cancel_cb,
        scene_type=scene_type,
    )
    recolor_s = time.perf_counter() - t0_recolor

    mesh_vertices = m_verts_raw @ WORLD_FLIP.T if len(m_verts_raw) > 0 else np.zeros((0, 3), dtype=np.float32)
    mesh_norms = m_norms_raw @ WORLD_FLIP.T if len(m_norms_raw) > 0 else np.zeros((0, 3), dtype=np.float32)
    vis_dirs_world = vis_dirs_colmap @ WORLD_FLIP.T if len(vis_dirs_colmap) > 0 else np.zeros((0, 3), dtype=np.float32)

    t0_polish = time.perf_counter()
    # orient EVERY triangle individually toward cameras WITHOUT deleting corner/occluded geometry
    if len(mesh_vertices) > 0 and len(m_faces) > 0:
        # smooth vis_dirs_world across connected coplanar mesh edges using chunked GPU index_add_
        if len(vis_dirs_world) > 0:
            u_e = np.concatenate([m_faces[:, 0], m_faces[:, 1], m_faces[:, 2]])
            v_e = np.concatenate([m_faces[:, 1], m_faces[:, 2], m_faces[:, 0]])
            a_e = np.concatenate([u_e, v_e])
            b_e = np.concatenate([v_e, u_e])
            cop_e = np.sum(mesh_norms[a_e] * mesh_norms[b_e], axis=1) > 0.80
            if np.any(cop_e):
                ac_e = torch.from_numpy(a_e[cop_e]).long().to(torch_dev)
                bc_e = torch.from_numpy(b_e[cop_e]).long().to(torch_dev)
                v_dirs_t = torch.from_numpy(vis_dirs_world).float().to(torch_dev)
                ones_e = torch.ones((len(ac_e), 1), dtype=torch.float32, device=torch_dev)

                for _ in range(2):
                    acc_d = torch.zeros_like(v_dirs_t)
                    cnt_d = torch.zeros((len(vis_dirs_world), 1), dtype=torch.float32, device=torch_dev)
                    _chunked_index_add(acc_d, 0, ac_e, v_dirs_t[bc_e])
                    _chunked_index_add(cnt_d, 0, ac_e, ones_e)
                    has_d = (cnt_d[:, 0] > 0.5)
                    v_dirs_t[has_d] = 0.50 * v_dirs_t[has_d] + 0.50 * (acc_d[has_d] / cnt_d[has_d])
                    mag_d = torch.norm(v_dirs_t, dim=-1, keepdim=True)
                    v_dirs_t = torch.where(mag_d > 1e-4, v_dirs_t / torch.clamp(mag_d, min=1e-6), v_dirs_t)
                vis_dirs_world = v_dirs_t.cpu().numpy().astype(np.float32)

        v0 = mesh_vertices[m_faces[:, 0]]
        v1 = mesh_vertices[m_faces[:, 1]]
        v2 = mesh_vertices[m_faces[:, 2]]
        tri_n = np.cross(v1 - v0, v2 - v0)

        tri_cam_dir = (
            vis_dirs_world[m_faces[:, 0]]
            + vis_dirs_world[m_faces[:, 1]]
            + vis_dirs_world[m_faces[:, 2]]
        )
        cam_center_w = np.mean([WORLD_FLIP @ poses[i].t_wc.astype(np.float32) for i in fuse_indices], axis=0)
        dir_mag = np.linalg.norm(tri_cam_dir, axis=1)
        weak = dir_mag < 0.10
        if np.any(weak):
            tri_cam_dir[weak] = cam_center_w[None, :] - v0[weak]

        # Component-level majority orientation: flips entire manifold islands uniformly, never individual triangles
        temp_mesh = o3d.geometry.TriangleMesh()
        temp_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices.astype(np.float64))
        temp_mesh.triangles = o3d.utility.Vector3iVector(m_faces.astype(np.int32))
        tri_clusters, _, _ = temp_mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)

        dot_cam = np.sum(tri_n * tri_cam_dir, axis=1)
        strong_cam = ~weak

        num_clusters = int(np.max(tri_clusters)) + 1 if len(tri_clusters) > 0 else 0
        if num_clusters > 0:
            strong_cam_f = strong_cam.astype(np.float64)
            pos_strong = ((dot_cam > 0.0) & strong_cam).astype(np.float64)
            count_strong = np.bincount(tri_clusters, weights=strong_cam_f, minlength=num_clusters)
            pos_strong_count = np.bincount(tri_clusters, weights=pos_strong, minlength=num_clusters)

            total_count = np.bincount(tri_clusters, minlength=num_clusters)
            pos_total_count = np.bincount(tri_clusters, weights=(dot_cam > 0.0).astype(np.float64), minlength=num_clusters)

            has_strong = count_strong > 0
            has_total = total_count > 0

            flip_comp_arr = np.zeros(num_clusters, dtype=bool)
            if np.any(has_strong):
                frac_strong = pos_strong_count[has_strong] / count_strong[has_strong]
                flip_comp_arr[has_strong] = frac_strong < 0.50

            fallback = (~has_strong) & has_total
            if np.any(fallback):
                frac_fallback = pos_total_count[fallback] / total_count[fallback]
                flip_comp_arr[fallback] = frac_fallback < 0.50

            flip_faces = flip_comp_arr[tri_clusters]
            if np.any(flip_faces):
                m_faces[flip_faces] = m_faces[flip_faces][:, [0, 2, 1]]

        if scene_type != "outdoor":
            mesh_vertices, m_cols = _bilateral_polish_flat_surfaces(
                mesh_vertices, m_faces, m_cols, mesh_norms, device=torch_dev
            )

        # Fast chunked recomputation of smooth vertex normals directly from wound faces
        v0_t = torch.from_numpy(mesh_vertices[m_faces[:, 0]]).float().to(torch_dev)
        v1_t = torch.from_numpy(mesh_vertices[m_faces[:, 1]]).float().to(torch_dev)
        v2_t = torch.from_numpy(mesh_vertices[m_faces[:, 2]]).float().to(torch_dev)
        fn_t = torch.cross(v1_t - v0_t, v2_t - v0_t, dim=-1)

        vn_acc = torch.zeros((len(mesh_vertices), 3), dtype=torch.float32, device=torch_dev)
        f0_t = torch.from_numpy(m_faces[:, 0].astype(np.int64)).to(torch_dev)
        f1_t = torch.from_numpy(m_faces[:, 1].astype(np.int64)).to(torch_dev)
        f2_t = torch.from_numpy(m_faces[:, 2].astype(np.int64)).to(torch_dev)
        _chunked_index_add(vn_acc, 0, f0_t, fn_t)
        _chunked_index_add(vn_acc, 0, f1_t, fn_t)
        _chunked_index_add(vn_acc, 0, f2_t, fn_t)
        mesh_norms = F.normalize(vn_acc, dim=-1).cpu().numpy().astype(np.float32)
        del v0_t, v1_t, v2_t, fn_t, vn_acc, f0_t, f1_t, f2_t
        if torch_dev == "cuda":
            torch.cuda.empty_cache()

        if scene_type != "outdoor":
            mesh_vertices, m_faces, m_cols, mesh_norms = _fill_floor_grid_holes(
                mesh_vertices, m_faces, m_cols, mesh_norms
            )
    polish_and_floor_s = time.perf_counter() - t0_polish

    bake_texture_s = 0.0
    mesh_uvs: Optional[np.ndarray] = None
    mesh_texture: Optional[np.ndarray] = None
    if bake_texture and len(m_faces) >= 50:
        if progress_cb:
            progress_cb(0.96, "unwrapping UVs & baking high-res texture atlas...")
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")

        t0_bake = time.perf_counter()
        try:
            from src.texture_baker import bake_highres_uv_texture

            v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_rgb = bake_highres_uv_texture(
                verts_colmap=(mesh_vertices @ WORLD_FLIP.T).astype(np.float32),
                faces=m_faces,
                norms_colmap=(mesh_norms @ WORLD_FLIP.T).astype(np.float32),
                fallback_cols=m_cols,
                frames=frames,
                depths=fused_depths,
                masks=clean_masks,
                poses=poses,
                intrinsics=intrinsics,
                fuse_indices=fuse_indices,
                video_path=video_path,
                k1_cal=k1_cal,
                device=torch_dev,
                scene_type=scene_type,
                hero_face_count=hero_face_count,
                cancel_cb=cancel_cb,
            )
            if uvs_gl is not None and tex_rgb is not None:
                mesh_vertices = (v_uv @ WORLD_FLIP.T).astype(np.float32)
                mesh_norms = (n_uv @ WORLD_FLIP.T).astype(np.float32)
                m_faces = f_uv.astype(np.int32)
                m_cols = c_uv
                mesh_uvs = uvs_gl.astype(np.float32)
                mesh_texture = tex_rgb
        except Exception as e:
            if "cancel" in str(e).lower():
                raise
            print(
                f"[WARNING] high-res UV texture bake failed ({e}); "
                "falling back to vertex colors"
            )
            mesh_uvs = None
            mesh_texture = None
        bake_texture_s = time.perf_counter() - t0_bake

    t0_sample = time.perf_counter()
    if len(m_faces) > 0:
        clean_o3d = o3d.geometry.TriangleMesh()
        clean_o3d.vertices = o3d.utility.Vector3dVector(mesh_vertices.astype(np.float64))
        clean_o3d.triangles = o3d.utility.Vector3iVector(m_faces.astype(np.int32))
        clean_o3d.vertex_colors = o3d.utility.Vector3dVector(m_cols.astype(np.float64) / 255.0)
        clean_o3d.vertex_normals = o3d.utility.Vector3dVector(mesh_norms.astype(np.float64))

        # Wire point_stride (1..6) to uniform sampling cap (600k..45k)
        point_stride_clamped = max(1, min(6, int(point_stride)))
        cap_map = {1: 600000, 2: 450000, 3: 220000, 4: 120000, 5: 70000, 6: 45000}
        max_sample_pts = cap_map.get(point_stride_clamped, 450000)
        n_sample_pts = int(np.clip(len(m_faces) * 1.0, min(len(m_faces), 25000), max_sample_pts))

        pcd_uniform = clean_o3d.sample_points_uniformly(number_of_points=n_sample_pts)
        fused_pts = np.asarray(pcd_uniform.points, dtype=np.float32)
        fused_cols = np.clip(np.round(np.asarray(pcd_uniform.colors) * 255.0), 0, 255).astype(np.uint8)
        fused_norms = (
            np.asarray(pcd_uniform.normals, dtype=np.float32)
            if pcd_uniform.has_normals()
            else np.zeros_like(fused_pts)
        )
    else:
        fused_pts = mesh_vertices
        fused_cols = m_cols
        fused_norms = mesh_norms
    point_sample_s = time.perf_counter() - t0_sample

    stage4_total_s = time.perf_counter() - t0_stage4

    LAST_STAGE4_TELEMETRY.clear()
    LAST_STAGE4_TELEMETRY.update({
        "multiview_s": multiview_s,
        "tsdf_s": tsdf_s,
        "tsdf_err_count": tsdf_err_count,
        "recolor_s": recolor_s,
        "polish_and_floor_s": polish_and_floor_s,
        "bake_texture_s": bake_texture_s,
        "outer_context_tris": int(outer_context_tris),
        "point_sample_s": point_sample_s,
        "stage4_total_s": stage4_total_s,
    })

    if progress_cb:
        progress_cb(1.0, f"done! {len(fused_pts):,} points & {len(m_faces):,} triangles")

    return Scene3D(
        points=fused_pts,
        colors=fused_cols,
        normals=fused_norms,
        mesh_vertices=mesh_vertices,
        mesh_faces=m_faces,
        mesh_colors=m_cols,
        poses=poses,
        intrinsics=intrinsics,
        frame_size=(w, h),
        mesh_uvs=mesh_uvs,
        mesh_texture=mesh_texture,
    )


def export_scene(scene: Scene3D, output_path: str | Path, export_mesh: bool = True) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()

    if export_mesh and len(scene.mesh_faces) > 0 and ext in (".obj", ".glb", ".gltf", ".ply"):
        if (
            scene.mesh_uvs is not None
            and scene.mesh_texture is not None
            and ext in (".obj", ".glb", ".gltf")
        ):
            from PIL import Image

            pil_img = Image.fromarray(scene.mesh_texture)
            if ext in (".glb", ".gltf"):
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=pil_img,
                    metallicFactor=0.0,
                    roughnessFactor=0.95,
                )
            else:
                mat = trimesh.visual.material.SimpleMaterial(image=pil_img)

            mesh = trimesh.Trimesh(
                vertices=scene.mesh_vertices,
                faces=scene.mesh_faces,
                visual=trimesh.visual.TextureVisuals(uv=scene.mesh_uvs, material=mat),
                process=False,
            )
            mesh.export(str(out))
            return out

        mesh = trimesh.Trimesh(
            vertices=scene.mesh_vertices,
            faces=scene.mesh_faces,
            vertex_colors=scene.mesh_colors,
            process=False,
        )
        mesh.export(str(out))
        return out

    cloud = trimesh.PointCloud(vertices=scene.points, colors=scene.colors)
    cloud.export(str(out))
    return out
