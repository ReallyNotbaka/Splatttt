"""High-resolution UV texture baking for the TSDF mesh.

The TSDF stage produces geometry at a fixed voxel size; vertex colours are then
the only colour signal, which softens texture detail.  This module decouples the
two: decimate the mesh, unwrap it with xatlas, rasterise 3D position / normal /
fallback-colour into UV space with Open3D's C++ baker, then stream a handful of
sharp source frames at up to ``max_src_dim`` pixels and project them onto the
atlas with a running-max sharpness blend.

UV convention: ``uvs_gl`` is OpenGL style (v = 0 bottom, v = 1 top) which is
what Trimesh, PyVista and the OBJ/glTF exporters expect.  The returned texture
``tex_rgb`` is a top-down image (row 0 == v == 1), so sampling at
``row = (1 - v) * (size - 1)`` recovers the baked colour.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
import xatlas

from src.pose_tracker import CameraPose
from src.video_ingest import VideoFrame


TEXEL_CHUNK = 2_000_000
GPU_ACC_MAX_TEXELS = 12_000_000


def _select_sharpest_views(
    fuse_indices: List[int], frames: List[VideoFrame], max_views: int
) -> List[int]:
    """Buckets the timeline and keeps the sharpest frame in each bucket."""
    if max_views <= 0 or len(fuse_indices) <= max_views:
        return list(fuse_indices)

    picked: List[int] = []
    for bucket in np.array_split(np.arange(len(fuse_indices)), max_views):
        if len(bucket) == 0:
            continue
        best_pos = max(bucket, key=lambda k: frames[fuse_indices[int(k)]].sharpness)
        picked.append(fuse_indices[int(best_pos)])

    seen: set[int] = set()
    unique: List[int] = []
    for idx in picked:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def _decimate_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    cols: np.ndarray,
    target_tris: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(faces) <= int(target_tris):
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        m.compute_vertex_normals()
        n = np.asarray(m.vertex_normals, dtype=np.float32)
        if len(n) != len(verts):
            n = np.zeros_like(verts, dtype=np.float32)
        return verts.astype(np.float32), faces.astype(np.int32), n, cols.astype(np.uint8)

    try:
        import fast_simplification
        from scipy.spatial import cKDTree

        v_dec, f_dec = fast_simplification.simplify(
            verts.astype(np.float32),
            faces.astype(np.int32),
            target_count=int(target_tris),
            agg=7,
        )
        if len(f_dec) > 0 and len(v_dec) > 0:
            m_dec = o3d.geometry.TriangleMesh()
            m_dec.vertices = o3d.utility.Vector3dVector(np.asarray(v_dec, dtype=np.float64))
            m_dec.triangles = o3d.utility.Vector3iVector(np.asarray(f_dec, dtype=np.int32))
            m_dec.remove_degenerate_triangles()
            m_dec.remove_unreferenced_vertices()
            m_dec.compute_vertex_normals()

            v_dec = np.asarray(m_dec.vertices, dtype=np.float32)
            f_dec = np.asarray(m_dec.triangles, dtype=np.int32)
            n_dec = np.asarray(m_dec.vertex_normals, dtype=np.float32)

            if len(f_dec) > 0 and len(v_dec) > 0:
                if len(n_dec) != len(v_dec):
                    n_dec = np.zeros_like(v_dec, dtype=np.float32)
                tree = cKDTree(verts)
                _, nearest_idx = tree.query(v_dec, workers=-1)
                c_dec = cols[nearest_idx].astype(np.uint8)
                return v_dec, f_dec, n_dec, c_dec
    except Exception as e:
        print(f"[WARNING] fast_simplification failed ({e}); falling back to Open3D")

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(cols.astype(np.float64) / 255.0, 0.0, 1.0)
    )

    mesh_dec = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=int(target_tris)
    )
    mesh_dec.remove_degenerate_triangles()
    mesh_dec.remove_unreferenced_vertices()
    mesh_dec.compute_vertex_normals()

    v = np.asarray(mesh_dec.vertices, dtype=np.float32)
    f = np.asarray(mesh_dec.triangles, dtype=np.int32)
    n = np.asarray(mesh_dec.vertex_normals, dtype=np.float32)
    if len(n) != len(v):
        n = np.zeros_like(v)
    raw_c = np.asarray(mesh_dec.vertex_colors)
    if len(raw_c) == len(v) and len(v) > 0:
        c = np.clip(np.round(raw_c * 255.0), 0, 255).astype(np.uint8)
    else:
        c = np.full((len(v), 3), 180, dtype=np.uint8)
    return v, f, n, c


def _unwrap_uvs(
    verts: np.ndarray, faces: np.ndarray, tex_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v32 = np.ascontiguousarray(verts.astype(np.float32))
    f32 = np.ascontiguousarray(faces.astype(np.uint32))

    vmapping = None
    f_uv = None
    uvs = None
    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(v32, f32)
        pack_opts = xatlas.PackOptions()
        pack_opts.resolution = int(tex_size)
        pack_opts.padding = 4
        chart_opts = xatlas.ChartOptions()
        # A single chart-fitting iteration costs ~5x the whole unwrap on
        # 160k-triangle scans for a <10% stretch improvement, so keep the
        # initial LSCM parametrisation only.
        chart_opts.max_iterations = 0
        atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
        if atlas.atlas_count > 0:
            vmapping, f_uv, uvs = atlas[0]
    except Exception:
        vmapping, f_uv, uvs = None, None, None

    if vmapping is None or f_uv is None or uvs is None:
        vmapping, f_uv, uvs = xatlas.parametrize(v32, f32)

    vmapping = np.asarray(vmapping, dtype=np.int64)
    f_uv = np.asarray(f_uv, dtype=np.int32)
    uvs = np.clip(np.asarray(uvs, dtype=np.float32), 0.0, 1.0)
    return vmapping, f_uv, uvs


def _compact_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    norms: np.ndarray,
    cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drops unreferenced vertices and reindexes ``faces`` accordingly."""
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    v = np.asarray(verts[used], dtype=np.float32)
    f = remap[faces].astype(np.int32)
    n = (
        np.asarray(norms[used], dtype=np.float32)
        if len(norms) == len(verts)
        else np.zeros((len(used), 3), dtype=np.float32)
    )
    c = (
        np.asarray(cols[used], dtype=np.uint8)
        if len(cols) == len(verts)
        else np.full((len(used), 3), 180, dtype=np.uint8)
    )
    return v, f, n, c


def _unwrap_uvs_dual(
    v_hero: np.ndarray,
    f_hero: np.ndarray,
    v_outer: np.ndarray,
    f_outer: np.ndarray,
    tex_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwraps two meshes into one atlas with Hero getting priority.

    The outer mesh is only scaled down (0.40x around its centroid) for the
    xatlas call, so it consumes ~16% of the atlas texel area while the true 3D
    coordinates are still used for baking and rendering.  Returned ``vmapping``
    indexes the concatenated ``[hero, outer]`` vertex arrays.
    """
    v_hero32 = np.ascontiguousarray(v_hero.astype(np.float32))
    f_hero32 = np.ascontiguousarray(f_hero.astype(np.uint32))
    v_outer32 = np.ascontiguousarray(v_outer.astype(np.float32))
    f_outer32 = np.ascontiguousarray(f_outer.astype(np.uint32))
    n_hero = int(len(v_hero32))

    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_hero32, f_hero32)
        c = v_outer32.mean(axis=0, keepdims=True)
        atlas.add_mesh(
            np.ascontiguousarray((v_outer32 - c) * 0.40, dtype=np.float32),
            f_outer32,
        )
        pack_opts = xatlas.PackOptions()
        pack_opts.resolution = int(tex_size)
        pack_opts.padding = 4
        chart_opts = xatlas.ChartOptions()
        chart_opts.max_iterations = 0
        atlas.generate(chart_options=chart_opts, pack_options=pack_opts)

        if atlas.mesh_count >= 2:
            vm0, f0, uv0 = atlas[0]
            vm1, f1, uv1 = atlas[1]
            vm0 = np.asarray(vm0, dtype=np.int64)
            vm1 = np.asarray(vm1, dtype=np.int64)
            f0 = np.asarray(f0, dtype=np.int32)
            f1 = np.asarray(f1, dtype=np.int32)
            uv0 = np.asarray(uv0, dtype=np.float32)
            uv1 = np.asarray(uv1, dtype=np.float32)
            vmapping = np.concatenate([vm0, vm1 + n_hero])
            f_uv = np.vstack([f0, f1 + len(vm0)]).astype(np.int32)
            uvs = np.clip(np.vstack([uv0, uv1]), 0.0, 1.0).astype(np.float32)
            return vmapping, f_uv, uvs
    except Exception:
        pass

    # Fallback: single unwrap over the concatenated, unscaled meshes.
    v_all = np.vstack([v_hero32, v_outer32])
    f_all = np.vstack([f_hero32, f_outer32 + n_hero]).astype(np.uint32)
    return _unwrap_uvs(v_all, f_all, tex_size)


def _sky_pixels(
    rgb_vals: torch.Tensor, d_w_up: torch.Tensor, active: bool
) -> torch.Tensor:
    """Mask of bright sky/overcast pixels seen along upward-pointing rays."""
    if not active or rgb_vals.shape[0] == 0:
        return torch.zeros(
            (int(rgb_vals.shape[0]),), dtype=torch.bool, device=rgb_vals.device
        )
    bright_blue = (
        (d_w_up > 0.02)
        & (rgb_vals[:, 2] > 185.0)
        & (rgb_vals[:, 1] > 140.0)
        & ((rgb_vals[:, 2] - rgb_vals[:, 0]) > 28.0)
    )
    blown = (
        (d_w_up > 0.05)
        & (rgb_vals[:, 2] >= rgb_vals[:, 0] - 10.0)
        & (rgb_vals.mean(dim=-1) > 232.0)
    )
    return bright_blue | blown


def _bake_geometry_textures(
    v_uv: np.ndarray,
    f_uv: np.ndarray,
    n_uv: np.ndarray,
    c_uv: np.ndarray,
    uvs_gl: np.ndarray,
    tex_size: int,
) -> Dict[str, np.ndarray]:
    tmesh = o3d.t.geometry.TriangleMesh(o3c.Device("CPU:0"))
    tmesh.vertex.positions = o3c.Tensor(v_uv.astype(np.float32))
    tmesh.triangle.indices = o3c.Tensor(f_uv.astype(np.int32))
    tmesh.triangle.texture_uvs = o3c.Tensor(
        uvs_gl[f_uv].astype(np.float32), dtype=o3c.float32
    )
    tmesh.vertex["pos"] = o3c.Tensor(v_uv.astype(np.float32), dtype=o3c.float32)
    tmesh.vertex["nrm"] = o3c.Tensor(n_uv.astype(np.float32), dtype=o3c.float32)
    tmesh.vertex["vcol"] = o3c.Tensor(
        (c_uv.astype(np.float32) / 255.0), dtype=o3c.float32
    )
    tmesh.vertex["cov"] = o3c.Tensor(
        np.ones((len(v_uv), 1), dtype=np.float32), dtype=o3c.float32
    )

    baked = tmesh.bake_vertex_attr_textures(
        size=int(tex_size),
        vertex_attr={"pos", "nrm", "vcol", "cov"},
        margin=0.0,
        fill=0.0,
        update_material=False,
    )
    out = {
        "pos": baked["pos"].numpy(),
        "nrm": baked["nrm"].numpy(),
        "vcol": baked["vcol"].numpy(),
        "cov": baked["cov"].numpy(),
    }
    del tmesh, baked
    return out


def _load_highres_frame(
    cap: Optional["cv2.VideoCapture"],
    frame: VideoFrame,
    work_size: Tuple[int, int],
    intrinsics: np.ndarray,
    k1_cal: float,
    max_src_dim: int,
    map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    work_rgb = np.ascontiguousarray(frame.rgb)
    if cap is None:
        return work_rgb

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame.index))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return work_rgb

    orig_h, orig_w = bgr.shape[:2]
    w_work, h_work = work_size
    # Keep the *working* aspect ratio: the extractor rounds frame sizes to
    # multiples of 8, so resizing the source by its own aspect would shift
    # projected texels by a few pixels.  `k` is the exact upscale factor.
    k = min(float(max_src_dim), float(max(orig_w, orig_h))) / float(max(w_work, h_work))
    if k <= 1.001:
        return work_rgb

    w_hi = max(w_work, int(round(w_work * k)))
    h_hi = max(h_work, int(round(h_work * k)))
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (w_hi, h_hi), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    )

    if abs(k1_cal) > 1e-4:
        key = (w_hi, h_hi)
        maps = map_cache.get(key)
        if maps is None:
            sx = w_hi / float(w_work)
            sy = h_hi / float(h_work)
            k_hi = np.array(
                [
                    [float(intrinsics[0, 0]) * sx, 0.0, float(intrinsics[0, 2]) * sx],
                    [0.0, float(intrinsics[1, 1]) * sy, float(intrinsics[1, 2]) * sy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist = np.array([k1_cal, 0.0, 0.0, 0.0], dtype=np.float64)
            maps = cv2.initUndistortRectifyMap(
                k_hi, dist, None, k_hi, (w_hi, h_hi), cv2.CV_32FC1
            )
            map_cache[key] = maps
        rgb = cv2.remap(
            rgb, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return np.ascontiguousarray(rgb)


def _iter_highres_frames_pyav(
    video_path: Path | str,
    targets: List[Tuple[int, int, float]],
    w_hi: int,
    h_hi: int,
):
    """Yield ``(slot_idx, rgb)`` for exact source timestamps via PyAV seeking.

    ``targets`` is a sorted list of ``(bake_slot_idx, frame_index, timestamp_s)``.
    Hardware NVDEC is preferred; if PyAV cannot decode a target the generator
    falls back to software PyAV, and if that fails, stops so the caller
    transparently falls back to OpenCV for that frame.
    """
    from src.video_ingest import _open_pyav_video

    if len(targets) == 0:
        return

    remaining = list(targets)
    for use_hw in (True, False):
        if not remaining:
            break
        try:
            container = _open_pyav_video(video_path, use_hw=use_hw)
        except Exception:
            continue

        try:
            stream = container.streams.video[0]
            tb = float(stream.time_base) if stream.time_base else (1.0 / 30.0)
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            stream_start = (
                float(stream.start_time * tb) if stream.start_time is not None else 0.0
            )
            frame_iter = None
            last_ts: Optional[float] = None
            est_gop_sec: float = 0.0

            while remaining:
                slot_idx, _f_idx, f_ts = remaining[0]
                target_rel = float(f_ts)
                target_abs = stream_start + target_rel
                seek_gap = max(35.0 / fps, 0.8, est_gop_sec * 1.15)
                need_seek = (
                    frame_iter is None
                    or last_ts is None
                    or (target_rel - last_ts) > seek_gap
                    or target_rel < last_ts
                )
                if need_seek:
                    target_pts = int(round(target_abs / tb))
                    container.seek(
                        target_pts, stream=stream, backward=True, any_frame=False
                    )
                    frame_iter = container.decode(stream)

                frame = None
                first_seek_t: Optional[float] = None
                for fr in frame_iter:
                    if fr.time is None:
                        continue
                    if need_seek and first_seek_t is None:
                        first_seek_t = float(fr.time)
                        est_gop_sec = min(12.0, max(est_gop_sec, target_abs - first_seek_t))
                    if last_ts is not None and (fr.time - stream_start) <= last_ts + (0.25 / fps):
                        continue
                    frame = fr
                    # Fresh backward seek lands on an I-frame at/before the target.
                    if need_seek and abs(fr.time - target_abs) <= 0.45 / fps:
                        break
                    if fr.time >= target_abs - (0.35 / fps):
                        break

                if frame is None:
                    break

                rgb = frame.reformat(
                    width=w_hi, height=h_hi, format="rgb24", interpolation="AREA"
                ).to_ndarray()
                remaining.pop(0)
                last_ts = max(0.0, float(frame.time) - stream_start)
                yield slot_idx, np.ascontiguousarray(rgb, dtype=np.uint8)
        except Exception:
            pass
        finally:
            try:
                container.close()
            except Exception:
                pass


def bake_highres_uv_texture(
    verts_colmap: np.ndarray,
    faces: np.ndarray,
    norms_colmap: np.ndarray,
    fallback_cols: np.ndarray,
    frames: List[VideoFrame],
    depths: List[np.ndarray],
    masks: List[np.ndarray],
    poses: List[CameraPose],
    intrinsics: np.ndarray,
    fuse_indices: List[int],
    video_path: Optional[Path] = None,
    k1_cal: float = 0.0,
    device: str = "cuda",
    scene_type: str = "indoor",
    max_bake_views: int = 48,
    max_src_dim: int = 1600,
    target_tris: int = 220000,
    tex_size: Optional[int] = None,
    hero_face_count: Optional[int] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if len(verts_colmap) < 50 or len(faces) < 50:
        return verts_colmap, faces, norms_colmap, fallback_cols, None, None

    n_views_used = len(fuse_indices)
    if tex_size is None:
        if n_views_used < 20:
            tex_size = 1024
        elif n_views_used < 80:
            tex_size = 2048
        else:
            tex_size = 4096
    tex_size = int(tex_size)

    v_work = np.asarray(verts_colmap, dtype=np.float32)
    f_work = np.asarray(faces, dtype=np.int32)
    n_work = np.asarray(norms_colmap, dtype=np.float32)
    c_work = np.asarray(fallback_cols, dtype=np.uint8)

    dual = (
        hero_face_count is not None
        and 0 < int(hero_face_count) < len(f_work)
    )
    if dual:
        hfc = int(hero_face_count)
        v_hero, f_hero, n_hero, c_hero = _compact_mesh(
            v_work, f_work[:hfc], n_work, c_work
        )
        v_outer, f_outer, n_outer, c_outer = _compact_mesh(
            v_work, f_work[hfc:], n_work, c_work
        )
        if len(f_hero) < 4 or len(f_outer) < 4:
            dual = False
        else:
            if len(f_hero) > int(target_tris):
                v_hero, f_hero, n_hero, c_hero = _decimate_mesh(
                    v_hero, f_hero, c_hero, int(target_tris)
                )
            if len(f_outer) > 45000:
                v_outer, f_outer, n_outer, c_outer = _decimate_mesh(
                    v_outer, f_outer, c_outer, 45000
                )
            v_work = np.vstack([v_hero, v_outer]).astype(np.float32)
            n_work = np.vstack([n_hero, n_outer]).astype(np.float32)
            c_work = np.vstack([c_hero, c_outer]).astype(np.uint8)
            f_work = np.vstack([f_hero, f_outer + len(v_hero)]).astype(np.int32)

    if dual:
        vmapping, f_uv, uvs_gl = _unwrap_uvs_dual(
            v_hero, f_hero, v_outer, f_outer, tex_size
        )
    else:
        if len(f_work) > int(target_tris):
            v_work, f_work, n_work, c_work = _decimate_mesh(
                v_work, f_work, c_work, int(target_tris)
            )
        vmapping, f_uv, uvs_gl = _unwrap_uvs(v_work, f_work, tex_size)
    v_uv = np.asarray(v_work[vmapping], dtype=np.float32)
    n_uv = np.asarray(n_work[vmapping], dtype=np.float32)
    c_uv = np.asarray(c_work[vmapping], dtype=np.uint8)

    baked = _bake_geometry_textures(v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_size)

    cov_map = baked["cov"][..., 0]
    valid_mask = cov_map > 0.5
    vcol_img = np.clip(np.round(baked["vcol"] * 255.0), 0, 255).astype(np.uint8)
    tex_rgb = vcol_img.copy()

    ys, xs = np.nonzero(valid_mask)
    n_valid = int(len(ys))
    if n_valid == 0:
        return v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_rgb

    pos_valid = np.asarray(baked["pos"][ys, xs], dtype=np.float32)
    nrm_valid = np.asarray(baked["nrm"][ys, xs], dtype=np.float32)
    fb_valid = np.asarray(baked["vcol"][ys, xs], dtype=np.float32) * 255.0
    del baked

    is_sub = len(depths) == len(fuse_indices)
    idx_to_sub = (
        {idx: k for k, idx in enumerate(fuse_indices)}
        if is_sub
        else {i: i for i in range(len(depths))}
    )

    h_work, w_work = depths[0].shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    bake_views = _select_sharpest_views(fuse_indices, frames, max_bake_views)
    bake_views = [i for i in bake_views if (i in idx_to_sub or not is_sub)]
    bake_views.sort(key=lambda idx: frames[idx].index)
    if not bake_views:
        return v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_rgb

    sharp_vals = np.array([frames[i].sharpness for i in bake_views], dtype=np.float32)
    med_sharp = max(1.0, float(np.median(sharp_vals)))

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    store_dev = "cuda" if (dev == "cuda" and n_valid <= GPU_ACC_MAX_TEXELS) else "cpu"

    pos_t = torch.from_numpy(pos_valid).to(store_dev)
    nrm_t = F.normalize(torch.from_numpy(nrm_valid).to(store_dev), dim=-1)
    fb_t = torch.from_numpy(fb_valid).to(store_dev)

    m_log_s = torch.full((n_valid,), -float("inf"), dtype=torch.float32, device=store_dev)
    acc_w = torch.zeros((n_valid,), dtype=torch.float32, device=store_dev)
    acc_rgb = torch.zeros((n_valid, 3), dtype=torch.float32, device=store_dev)

    gain_idx_np = np.unique(
        np.linspace(0, n_valid - 1, min(n_valid, 80_000), dtype=np.int64)
    )
    gain_idx = torch.from_numpy(gain_idx_np).to(store_dev)

    caps: Optional["cv2.VideoCapture"] = None
    orig_w, orig_h = w_work, h_work
    if video_path is not None:
        vp = Path(video_path)
        if vp.exists():
            cap_try = cv2.VideoCapture(str(vp))
            if cap_try.isOpened():
                caps = cap_try
                orig_w = int(cap_try.get(cv2.CAP_PROP_FRAME_WIDTH) or w_work)
                orig_h = int(cap_try.get(cv2.CAP_PROP_FRAME_HEIGHT) or h_work)
            else:
                cap_try.release()
    map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    k = min(float(max_src_dim), float(max(orig_w, orig_h))) / float(max(w_work, h_work))
    if k > 1.001:
        w_hi = max(w_work, int(round(w_work * k)))
        h_hi = max(h_work, int(round(h_work * k)))
    else:
        w_hi, h_hi = w_work, h_work

    highres_frames_iter = None
    if video_path is not None and Path(video_path).exists() and k > 1.001:
        highres_targets = [
            (vi, int(frames[i].index), float(frames[i].timestamp))
            for vi, i in enumerate(bake_views)
        ]
        highres_frames_iter = _iter_highres_frames_pyav(
            video_path, highres_targets, w_hi, h_hi
        )

    try:
        with torch.no_grad():
            for vi, i in enumerate(bake_views):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("build cancelled by user")

                pose = poses[i]
                depth_i = np.asarray(
                    depths[idx_to_sub[i]] if is_sub else depths[i], dtype=np.float32
                )
                mask_i = np.asarray(
                    masks[idx_to_sub[i]] if is_sub else masks[i]
                )

                rgb_hi = None
                if highres_frames_iter is not None:
                    try:
                        slot_i, f_rgb = next(highres_frames_iter)
                        if slot_i == vi:
                            rgb_hi = f_rgb
                        else:
                            highres_frames_iter = None
                    except Exception:
                        highres_frames_iter = None

                if rgb_hi is None:
                    rgb_hi = _load_highres_frame(
                        caps, frames[i], (w_work, h_work), intrinsics, k1_cal,
                        max_src_dim, map_cache,
                    )
                else:
                    if abs(k1_cal) > 1e-4:
                        key = (w_hi, h_hi)
                        maps = map_cache.get(key)
                        if maps is None:
                            sx = w_hi / float(w_work)
                            sy = h_hi / float(h_work)
                            k_hi = np.array(
                                [
                                    [float(intrinsics[0, 0]) * sx, 0.0, float(intrinsics[0, 2]) * sx],
                                    [0.0, float(intrinsics[1, 1]) * sy, float(intrinsics[1, 2]) * sy],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float64,
                            )
                            dist = np.array([k1_cal, 0.0, 0.0, 0.0], dtype=np.float64)
                            maps = cv2.initUndistortRectifyMap(
                                k_hi, dist, None, k_hi, (w_hi, h_hi), cv2.CV_32FC1
                            )
                            map_cache[key] = maps
                        rgb_hi = cv2.remap(
                            rgb_hi, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                    rgb_hi = np.ascontiguousarray(rgb_hi)

                r_wc = torch.from_numpy(pose.r_wc.astype(np.float32)).to(dev)
                t_wc = torch.from_numpy(pose.t_wc.astype(np.float32)).to(dev)
                rgb_t = (
                    torch.from_numpy(rgb_hi)
                    .to(dev)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                d_t = torch.from_numpy(np.ascontiguousarray(depth_i)).to(dev).view(
                    1, 1, h_work, w_work
                )
                m_t = (
                    torch.from_numpy(np.ascontiguousarray(mask_i.astype(np.float32)))
                    .to(dev)
                    .view(1, 1, h_work, w_work)
                )

                def _project(idx: torch.Tensor):
                    p = pos_t[idx].to(dev)
                    nr = nrm_t[idx].to(dev)
                    pc = (p - t_wc.unsqueeze(0)) @ r_wc
                    z = pc[:, 2]
                    in_front = z > 0.25
                    z_safe = torch.clamp(z, min=1e-3)
                    u = fx * (pc[:, 0] / z_safe) + cx
                    v = fy * (pc[:, 1] / z_safe) + cy
                    inside = (
                        in_front
                        & (u >= 2.0)
                        & (u < w_work - 2.0)
                        & (v >= 2.0)
                        & (v < h_work - 2.0)
                    )
                    u_n = (u / max(1, w_work - 1)) * 2.0 - 1.0
                    v_n = (v / max(1, h_work - 1)) * 2.0 - 1.0
                    grid = torch.stack([u_n, v_n], dim=-1).view(1, 1, -1, 2)
                    z_obs = F.grid_sample(
                        d_t, grid, mode="bilinear", padding_mode="border",
                        align_corners=True,
                    ).view(-1)
                    m_obs = (
                        F.grid_sample(
                            m_t, grid, mode="nearest", padding_mode="zeros",
                            align_corners=True,
                        ).view(-1)
                        > 0.5
                    )
                    dz = torch.abs(z - z_obs)
                    if scene_type == "outdoor":
                        vis_tol = torch.clamp(0.022 * z, min=0.12, max=0.55)
                        sig_d = torch.clamp(0.010 * z, min=0.05, max=0.22)
                    else:
                        vis_tol = torch.full_like(z, 0.22)
                        sig_d = torch.full_like(z, 0.08)
                    view_dir = F.normalize(t_wc.unsqueeze(0) - p, dim=-1)
                    cos_v = (nr * view_dir).sum(dim=-1)
                    # COLMAP +Y points down, so a texel above the camera gives a
                    # positive "looking up" ray component.
                    dist = torch.norm(p - t_wc.unsqueeze(0), dim=-1)
                    d_w_up = -(p[:, 1] - t_wc[1]) / torch.clamp(dist, min=1e-4)
                    ok = inside & m_obs & (dz < vis_tol) & (cos_v > 0.10)
                    return (
                        p, z, u_n, v_n, dz, sig_d, cos_v, ok, grid, d_w_up, u, v,
                    )

                # --- exposure gain from a quick texel subsample -----------------
                p_g, z_g, u_n_g, v_n_g, dz_g, _, cos_g, ok_g, grid_g, up_g, _, _ = _project(gain_idx)
                gain = torch.ones(3, dtype=torch.float32, device=dev)
                if int(ok_g.sum()) >= 32:
                    rgb_g = (
                        F.grid_sample(
                            rgb_t, grid_g[:, :, ok_g, :], mode="bilinear",
                            padding_mode="border", align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(1)
                        .T
                    )
                    fb_g = fb_t[gain_idx][ok_g].to(dev)
                    sky_g = _sky_pixels(rgb_g, up_g[ok_g], scene_type == "outdoor")
                    bright = (
                        (rgb_g.mean(dim=-1) > 25.0)
                        & (rgb_g.mean(dim=-1) < 235.0)
                        & (~sky_g)
                    )
                    if int(bright.sum()) >= 32:
                        ratio = (fb_g[bright] + 8.0) / (rgb_g[bright] + 8.0)
                        gain = torch.median(ratio, dim=0).values.clamp(0.75, 1.35)

                s_bonus = float(np.clip(frames[i].sharpness / med_sharp, 0.75, 1.35))

                # --- full streamed projection ----------------------------------
                for start in range(0, n_valid, TEXEL_CHUNK):
                    end = min(start + TEXEL_CHUNK, n_valid)
                    idx = torch.arange(start, end, dtype=torch.long, device=store_dev)
                    p, z, u_n, v_n, dz, sig_d, cos_v, ok, grid, d_w_up, u_px, v_px = _project(idx)
                    n_ok = int(ok.sum())
                    if n_ok == 0:
                        continue

                    ok_idx = idx[ok]
                    rgb_ok = (
                        F.grid_sample(
                            rgb_t, grid[:, :, ok, :], mode="bilinear",
                            padding_mode="border", align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(1)
                        .T
                    )
                    rgb_ok = torch.clamp(rgb_ok * gain.unsqueeze(0), 0.0, 255.0)

                    # Drop bright sky seen along up-looking rays (outdoor only).
                    sky = _sky_pixels(rgb_ok, d_w_up[ok], scene_type == "outdoor")
                    keep = ~sky
                    if not bool(keep.any()):
                        continue
                    if bool(keep.all()):
                        sel = ok
                    else:
                        sel = torch.zeros_like(ok)
                        sel[ok] = keep
                    ok_idx = idx[sel]
                    rgb_ok = rgb_ok[keep]

                    cos_ok = cos_v[sel]
                    vig = torch.clamp(
                        1.0 - 0.55 * (u_n[sel].pow(2) + v_n[sel].pow(2)) * 0.5, min=0.12
                    )
                    # Smooth 8% frustum-border feather: hard frustum edges are
                    # what leaves straight seam cuts across a surface.
                    border_u = torch.minimum(u_px[sel], (w_work - 1.0) - u_px[sel])
                    border_v = torch.minimum(v_px[sel], (h_work - 1.0) - v_px[sel])
                    feather = max(8.0, 0.08 * min(w_work, h_work))
                    edge_fade = torch.clamp(
                        torch.minimum(border_u, border_v) / feather, 0.05, 1.0
                    )
                    depth_agree = torch.exp(-((dz[sel] / sig_d[sel]) ** 2))
                    base_score = (
                        torch.clamp(cos_ok, min=0.05, max=1.0).pow(2.0)
                        * vig
                        * s_bonus
                        * (1.0 / (z[sel] + 0.5))
                        * depth_agree
                        * edge_fade
                    )
                    log_s = 9.0 * torch.log(base_score.clamp(min=1e-6))

                    cur = ok_idx.to(store_dev)
                    prev_m = m_log_s[cur]
                    new_m = torch.maximum(prev_m, log_s.to(store_dev))
                    decay = torch.exp(prev_m - new_m)
                    w_cur = torch.exp(log_s.to(store_dev) - new_m)
                    acc_w[cur] = acc_w[cur] * decay + w_cur
                    acc_rgb[cur] = (
                        acc_rgb[cur] * decay.unsqueeze(-1)
                        + rgb_ok.to(store_dev) * w_cur.unsqueeze(-1)
                    )
                    m_log_s[cur] = new_m

                del r_wc, t_wc, rgb_t, d_t, m_t
    finally:
        if caps is not None:
            caps.release()
        if highres_frames_iter is not None:
            try:
                highres_frames_iter.close()
            except Exception:
                pass

    if int(n_valid) > 0:
        has_col = acc_w > 1e-6
        if bool(has_col.any()):
            seen = has_col.cpu().numpy()
            blended = (
                acc_rgb / torch.clamp(acc_w, min=1e-6).unsqueeze(-1)
            )[has_col].cpu().numpy()
            rows = ys[seen]
            cols = xs[seen]
            tex_rgb[rows, cols] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    del m_log_s, acc_w, acc_rgb, pos_t, nrm_t, fb_t, gain_idx
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 3px gutter dilation inside 4px island padding so mipmaps never bleed seams
    if np.any(~valid_mask):
        dist, inds = distance_transform_edt(~valid_mask, return_indices=True)
        fill = (~valid_mask) & (dist <= 3.0)
        if np.any(fill):
            tex_rgb[fill] = tex_rgb[inds[0][fill], inds[1][fill]]

    return v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_rgb
