import numpy as np
import pytest
import trimesh

from src.pipeline import (
    ReconstructionResult,
    load_result_cache,
    save_result_cache,
)
from src.pose_tracker import CameraPose
from src.scene_builder import Scene3D, export_scene
from src.texture_baker import bake_highres_uv_texture
from src.video_ingest import VideoData, VideoFrame


def _sample_uv(tex_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """OpenGL UV -> texel colour, row = (1 - v) * (S - 1)."""
    size = tex_rgb.shape[0]
    col = int(np.clip(round(float(uv[0]) * (size - 1)), 0, size - 1))
    row = int(np.clip(round((1.0 - float(uv[1])) * (size - 1)), 0, size - 1))
    return tex_rgb[row, col]


def _two_tone_scene():
    """Planar quad in front of an identity camera: top half red, bottom half blue."""
    res = 10
    xs = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, res, dtype=np.float32)  # COLMAP world +Y is down
    xx, yy = np.meshgrid(xs, ys)
    verts = np.stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, 2.0, dtype=np.float32)], axis=-1
    ).astype(np.float32)

    faces = []
    for r in range(res - 1):
        for c in range(res - 1):
            i0 = r * res + c
            i1 = i0 + 1
            i2 = i0 + res
            i3 = i2 + 1
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])
    faces = np.asarray(faces, dtype=np.int32)

    norms = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (len(verts), 1))
    cols = np.full((len(verts), 3), 128, dtype=np.uint8)

    h = w = 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[: h // 2] = np.array([255, 0, 0], dtype=np.uint8)  # image top = world up (y < 0)
    rgb[h // 2 :] = np.array([0, 0, 255], dtype=np.uint8)

    frame = VideoFrame(index=0, timestamp=0.0, rgb=rgb, sharpness=100.0)
    depths = [np.full((h, w), 2.0, dtype=np.float32)]
    masks = [np.ones((h, w), dtype=bool)]
    poses = [
        CameraPose(
            r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64)
        )
    ]
    intrinsics = np.array(
        [[100.0, 0.0, w * 0.5], [0.0, 100.0, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return verts, faces, norms, cols, ([frame], depths, masks, poses, intrinsics)


def test_uv_orientation_and_color_accuracy():
    verts, faces, norms, cols, (frames, depths, masks, poses, intrinsics) = _two_tone_scene()

    v_uv, f_uv, n_uv, c_uv, uvs_gl, tex_rgb = bake_highres_uv_texture(
        verts,
        faces,
        norms,
        cols,
        frames,
        depths,
        masks,
        poses,
        intrinsics,
        fuse_indices=[0],
        tex_size=256,
        device="cpu",
    )

    assert uvs_gl is not None and tex_rgb is not None
    assert uvs_gl.shape == (len(v_uv), 2)
    assert float(uvs_gl.min()) >= 0.0 and float(uvs_gl.max()) <= 1.0
    assert tex_rgb.shape == (256, 256, 3)
    assert tex_rgb.dtype == np.uint8
    assert len(f_uv) > 0 and f_uv.max() < len(v_uv)

    top_samples = []
    bottom_samples = []
    for pos, uv in zip(v_uv, uvs_gl):
        if abs(float(pos[0])) > 0.85 or abs(float(pos[1])) > 0.85:
            continue
        colour = _sample_uv(tex_rgb, uv)
        if pos[1] < -0.20:  # world up (image top) -> RED
            top_samples.append(colour)
        elif pos[1] > 0.20:  # world down (image bottom) -> BLUE
            bottom_samples.append(colour)

    assert len(top_samples) >= 5 and len(bottom_samples) >= 5
    for colour in top_samples:
        assert colour[0] > 180 and colour[2] < 60, f"top not red: {colour}"
    for colour in bottom_samples:
        assert colour[2] > 180 and colour[0] < 60, f"bottom not blue: {colour}"


def _tiny_scene(uvs, texture):
    verts = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    pose = CameraPose(
        r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64)
    )
    return Scene3D(
        points=verts.copy(),
        colors=np.full((4, 3), 90, dtype=np.uint8),
        normals=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
        mesh_vertices=verts,
        mesh_faces=faces,
        mesh_colors=np.full((4, 3), 90, dtype=np.uint8),
        poses=[pose],
        intrinsics=np.eye(3, dtype=np.float32),
        frame_size=(32, 32),
        mesh_uvs=uvs,
        mesh_texture=texture,
    )


def test_textured_glb_round_trip(tmp_path):
    texture = np.zeros((32, 32, 3), dtype=np.uint8)
    texture[:16] = np.array([255, 40, 40], dtype=np.uint8)
    texture[16:] = np.array([40, 40, 255], dtype=np.uint8)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    out = export_scene(_tiny_scene(uvs, texture), tmp_path / "out.glb")
    assert out.exists() and out.stat().st_size > 1024

    loaded = trimesh.load(str(out))
    if isinstance(loaded, trimesh.Scene):
        assert len(loaded.geometry) > 0
        loaded = next(iter(loaded.geometry.values()))

    assert loaded.visual.uv is not None
    assert len(loaded.visual.uv) == len(loaded.vertices)
    mat = loaded.visual.material
    assert mat is not None
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    assert img is not None


def test_cache_round_trip_preserves_uv_texture(tmp_path):
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    texture = np.random.default_rng(0).integers(
        0, 256, size=(16, 16, 3), dtype=np.uint8
    )
    scene = _tiny_scene(uvs, texture)

    frames = [
        VideoFrame(
            index=0,
            timestamp=0.0,
            rgb=np.full((8, 8, 3), 120, dtype=np.uint8),
            sharpness=42.0,
        )
    ]
    video_data = VideoData(
        path=tmp_path / "clip.mp4",
        fps=24.0,
        total_frames=1,
        duration=1.0 / 24.0,
        orig_size=(8, 8),
        frames=frames,
    )

    cache = save_result_cache(
        ReconstructionResult(video_data=video_data, scene=scene),
        tmp_path / "result.npz",
    )
    loaded = load_result_cache(cache)

    assert loaded.scene.mesh_uvs is not None
    assert loaded.scene.mesh_texture is not None
    assert np.array_equal(loaded.scene.mesh_uvs, uvs)
    assert np.array_equal(loaded.scene.mesh_texture, texture)


def test_cache_round_trip_without_texture(tmp_path):
    scene = _tiny_scene(None, None)
    frames = [
        VideoFrame(
            index=0,
            timestamp=0.0,
            rgb=np.full((8, 8, 3), 120, dtype=np.uint8),
            sharpness=42.0,
        )
    ]
    video_data = VideoData(
        path=tmp_path / "clip.mp4",
        fps=24.0,
        total_frames=1,
        duration=1.0 / 24.0,
        orig_size=(8, 8),
        frames=frames,
    )
    cache = save_result_cache(
        ReconstructionResult(video_data=video_data, scene=scene),
        tmp_path / "plain.npz",
    )
    loaded = load_result_cache(cache)
    assert loaded.scene.mesh_uvs is None
    assert loaded.scene.mesh_texture is None
    assert np.array_equal(loaded.scene.mesh_vertices, scene.mesh_vertices)


def _flat_grid(
    n: int = 6,
    cx: float = 0.0,
    cy: float = 0.0,
    half: float = 0.5,
    z: float = 2.0,
):
    """Planar grid in COLMAP coordinates, facing the identity camera."""
    xs = np.linspace(cx - half, cx + half, n, dtype=np.float32)
    ys = np.linspace(cy - half, cy + half, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, z, dtype=np.float32)], axis=-1
    ).astype(np.float32)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i0 = r * n + c
            faces.append([i0, i0 + 1, i0 + n + 1])
            faces.append([i0, i0 + n + 1, i0 + n])
    return verts, np.asarray(faces, dtype=np.int32)


def _uv_area_density(v: np.ndarray, f: np.ndarray, uv: np.ndarray) -> float:
    """Sum(UV area) / Sum(3D area) for the given faces."""
    V = v[f]
    U = uv[f]
    a3 = 0.5 * np.linalg.norm(
        np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=-1
    )
    e1 = U[:, 1] - U[:, 0]
    e2 = U[:, 2] - U[:, 0]
    a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return float(a2.sum() / max(a3.sum(), 1e-12))


def test_dual_tier_uv_packing_prioritises_hero():
    vh, fh = _flat_grid(n=6, cx=-0.6, half=0.5)
    vo, fo = _flat_grid(n=6, cx=0.6, half=0.5)
    verts = np.vstack([vh, vo]).astype(np.float32)
    faces = np.vstack([fh, fo + len(vh)]).astype(np.int32)
    hero_face_count = len(fh)

    norms = np.tile(
        np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (len(verts), 1)
    )
    cols = np.full((len(verts), 3), 110, dtype=np.uint8)

    h = w = 128
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    frame = VideoFrame(index=0, timestamp=0.0, rgb=rgb, sharpness=100.0)
    depths = [np.full((h, w), 2.0, dtype=np.float32)]
    masks = [np.ones((h, w), dtype=bool)]
    poses = [
        CameraPose(
            r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64)
        )
    ]
    intrinsics = np.array(
        [[100.0, 0.0, w * 0.5], [0.0, 100.0, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    v_uv, f_uv, _, _, uvs_gl, tex_rgb = bake_highres_uv_texture(
        verts,
        faces,
        norms,
        cols,
        [frame],
        depths,
        masks,
        poses,
        intrinsics,
        fuse_indices=[0],
        tex_size=256,
        hero_face_count=hero_face_count,
        device="cpu",
        scene_type="indoor",
    )

    assert uvs_gl is not None and tex_rgb is not None
    assert uvs_gl.shape == (len(v_uv), 2)
    assert float(uvs_gl.min()) >= 0.0 and float(uvs_gl.max()) <= 1.0
    assert len(f_uv) == len(fh) + len(fo)
    assert int(f_uv.max()) < len(v_uv)

    hero_density = _uv_area_density(v_uv, f_uv[: len(fh)], uvs_gl)
    outer_density = _uv_area_density(v_uv, f_uv[len(fh):], uvs_gl)
    assert hero_density > 0.0 and outer_density > 0.0
    # xatlas receives outer at 0.40x scale -> ~1/6.25 the texel density.
    assert hero_density > 2.0 * outer_density


def test_outdoor_sky_rim_is_rejected():
    verts, faces = _flat_grid(n=8, cx=0.0, cy=-0.30, half=0.15, z=2.0)
    norms = np.tile(
        np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (len(verts), 1)
    )
    fallback = np.array([0, 200, 0], dtype=np.uint8)
    cols = np.tile(fallback, (len(verts), 1))

    h = w = 128
    rgb = np.full((h, w, 3), np.array([120, 180, 220], dtype=np.uint8))
    frame = VideoFrame(index=0, timestamp=0.0, rgb=rgb, sharpness=100.0)
    depths = [np.full((h, w), 2.0, dtype=np.float32)]
    masks = [np.ones((h, w), dtype=bool)]
    poses = [
        CameraPose(
            r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64)
        )
    ]
    intrinsics = np.array(
        [[100.0, 0.0, w * 0.5], [0.0, 100.0, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    _, _, _, _, uvs_gl, tex_rgb = bake_highres_uv_texture(
        verts,
        faces,
        norms,
        cols,
        [frame],
        depths,
        masks,
        poses,
        intrinsics,
        fuse_indices=[0],
        tex_size=256,
        device="cpu",
        scene_type="outdoor",
    )

    assert uvs_gl is not None and tex_rgb is not None
    samples = np.array([_sample_uv(tex_rgb, uv) for uv in uvs_gl])
    assert len(samples) >= 8
    # Sky blue must never replace the fallback colour on these up-looking rays.
    assert float(np.mean(samples[:, 1])) > 120.0  # green preserved
    assert float(np.mean(samples[:, 0])) < 90.0
    assert float(np.mean(samples[:, 2])) < 120.0


def test_build_sift_anchor_pairs():
    from src.pose_tracker import _build_sift_anchor_pairs

    # HEAD graph: all pairs for m <= 165.
    pairs_140 = _build_sift_anchor_pairs(140)
    assert len(pairs_140) == 9730
    assert (0, 139) in pairs_140
    assert (15, 75) in pairs_140
    assert all(0 <= i < j < 140 for i, j in pairs_140)

    pairs_24 = _build_sift_anchor_pairs(24)
    assert len(pairs_24) == 276


def test_pyav_decode_targets_accepts_single_target(tmp_path):
    import cv2

    from src.video_ingest import _pyav_decode_targets

    video = tmp_path / "single.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 36)
    )
    for i in range(6):
        writer.write(np.full((36, 64, 3), i * 40, dtype=np.uint8))
    writer.release()

    frames = _pyav_decode_targets(video, [3], 64, 36, 10.0, None, None, use_hw=False)
    assert frames and frames[0][0] == 3


def test_decimate_mesh_fast_simplification_cleanup():
    from src.texture_baker import _decimate_mesh

    res = 30
    xs = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)], axis=-1)

    faces = []
    for r in range(res - 1):
        for c in range(res - 1):
            i0 = r * res + c
            i1 = i0 + 1
            i2 = i0 + res
            i3 = i2 + 1
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])
    # Add an explicit degenerate triangle [0, 0, 1]
    faces.append([0, 0, 1])
    faces = np.asarray(faces, dtype=np.int32)
    cols = np.full((len(verts), 3), 128, dtype=np.uint8)

    v_dec, f_dec, n_dec, c_dec = _decimate_mesh(verts, faces, cols, target_tris=200)

    assert len(f_dec) <= 200
    assert len(v_dec) > 0
    for f in f_dec:
        assert f[0] != f[1] and f[1] != f[2] and f[0] != f[2]
    norm_lens = np.linalg.norm(n_dec, axis=-1)
    assert np.all(norm_lens > 0.9) and np.all(norm_lens < 1.1)
    assert len(c_dec) == len(v_dec)


def test_align_selected_depths_interpolates_unanchored_scales():
    from src.pose_tracker import CameraPose, align_selected_depths
    from src.video_ingest import VideoFrame

    h, w = 64, 64
    k_mat = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    frames = [
        VideoFrame(index=i, timestamp=float(i), rgb=np.zeros((h, w, 3), dtype=np.uint8), sharpness=100.0)
        for i in range(5)
    ]
    # Frame 0, 1, 2 have mono depth 1.0; Frame 3, 4 have mono depth 1.5
    # Frame 0, 2 have colmap 1.5; Frame 3, 4 have colmap 1.0 (balances global_metric_scale ~1.0)
    # Frame 1 is UNANCHORED (no sparse points)
    depths = [
        np.full((h, w), 1.0, dtype=np.float32),
        np.full((h, w), 1.0, dtype=np.float32),
        np.full((h, w), 1.0, dtype=np.float32),
        np.full((h, w), 1.5, dtype=np.float32),
        np.full((h, w), 1.5, dtype=np.float32),
    ]
    masks = [np.ones((h, w), dtype=bool) for _ in range(5)]
    poses = [
        CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([float(i) * 0.1, 0.0, 0.0]), valid=True)
        for i in range(5)
    ]

    uv_pts = np.array(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0], [25.0, 25.0], [35.0, 35.0]],
        dtype=np.float32,
    )

    sparse_obs = {
        0: (uv_pts, np.full(len(uv_pts), 1.5, dtype=np.float32)),
        2: (uv_pts, np.full(len(uv_pts), 1.5, dtype=np.float32)),
        3: (uv_pts, np.full(len(uv_pts), 1.0, dtype=np.float32)),
        4: (uv_pts, np.full(len(uv_pts), 1.0, dtype=np.float32)),
    }

    final_depths, final_masks, keep_idx = align_selected_depths(
        frames=frames,
        selected_indices=[0, 1, 2, 3, 4],
        depths_sub=depths,
        masks_sub=masks,
        poses=poses,
        sparse_obs=sparse_obs,
        k_mat=k_mat,
        k1_cal=0.0,
        device="cpu",
        use_grid_warp=False,
    )

    assert len(final_depths) == 5
    med_d0 = float(np.median(final_depths[0]))
    med_d1 = float(np.median(final_depths[1]))
    med_d2 = float(np.median(final_depths[2]))
    # Frames 0 and 2 are scaled up above 1.4
    assert med_d0 > 1.4
    assert med_d2 > 1.4
    # Frame 1 (unanchored) is interpolated from its neighbors (0 and 2) rather than staying at 1.0
    assert med_d1 > 1.4
    assert abs(med_d1 - med_d0) < 0.1
