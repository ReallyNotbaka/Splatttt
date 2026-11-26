from pathlib import Path
import sys
import numpy as np
from PySide6.QtWidgets import QApplication
import torch
from src.native_viewer import Native3DWindow, configure_hidpi
from src.pose_tracker import (
    CameraPose,
    _compute_gravity_level_rotation,
    _count_good_reg_images,
    _project_colmap_point,
)
from src.scene_builder import (
    _fill_floor_grid_holes,
    _refine_multiview_depths_gpu,
    build_3d_scene,
)
from src.video_ingest import extract_keyframes, generate_demo_video


def test_depth_reprojection_linear_not_ratio():
    # Cam 0 closer (z=1.5, d0=1.50) and Cam 1 further (z=0.0, d1=3.08):
    # Exact 3D reprojection yields d_cand=1.58 -> center fused ~1.5250m
    # (Broken projective ratio multiplication would yield d_cand=1.54 -> ~1.5125m)
    h, w = 64, 64
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    d0 = np.full((h, w), 1.50, dtype=np.float32)
    d1 = np.full((h, w), 3.08, dtype=np.float32)
    m0 = np.ones((h, w), dtype=bool)
    m1 = np.ones((h, w), dtype=bool)
    p0 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.0, 0.0, 1.5]), valid=True)
    p1 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.0, 0.0, 0.0]), valid=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fused, _ = _refine_multiview_depths_gpu([d0, d1], [m0, m1], [p0, p1], intrinsics, [0, 1], device=dev)
    val = float(fused[0][32, 32])
    assert 1.520 <= val <= 1.530, f"Expected ~1.525m linear reprojection, got {val:.4f}m"
    assert float(np.min(fused[0])) >= 0.20


def test_floor_hole_fill_walls_at_edge_and_no_exterior_leak():
    # L-shaped room with walls at perimeter (>200 above-floor verts so k_seal + floodFill run)
    # and a 40cm interior hole; exterior cutout x>0.30 & z>0.30 must NEVER be patched.
    cell = 0.025
    floor_pts = []
    for x in np.arange(-1.0, 1.0, cell):
        for z in np.arange(-1.0, 1.0, cell):
            if x > 0.30 and z > 0.30:
                continue
            if abs(x + 0.30) < 0.20 and abs(z + 0.30) < 0.20:
                continue
            floor_pts.append([x, 0.0, z])

    wall_pts = []
    for y in np.linspace(0.25, 0.95, 5):
        for s in np.arange(-1.0, 1.0, cell):
            wall_pts.append([-1.0, y, s])
            wall_pts.append([s, y, -1.0])
            if s <= 0.30:
                wall_pts.append([1.0, y, s])
                wall_pts.append([s, y, 1.0])

    floor_arr = np.array(floor_pts, dtype=np.float32)
    wall_arr = np.array(wall_pts, dtype=np.float32)
    verts = np.concatenate([floor_arr, wall_arr], axis=0)
    cols = np.full((len(verts), 3), 170, dtype=np.uint8)
    norms = np.concatenate(
        [
            np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(floor_arr), 1)),
            np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(wall_arr), 1)),
        ],
        axis=0,
    )
    faces = np.zeros((1200, 3), dtype=np.int32)
    faces[:, 0] = np.arange(1200) % (len(verts) - 2)
    faces[:, 1] = faces[:, 0] + 1
    faces[:, 2] = faces[:, 0] + 2

    new_v, _, _, _ = _fill_floor_grid_holes(verts, faces, cols, norms)
    assert len(new_v) > len(verts), "Interior floor hole was not patched"
    patched = new_v[len(verts) :]
    leaked_cutout = (patched[:, 0] > 0.35) & (patched[:, 2] > 0.35)
    assert not np.any(leaked_cutout), "Floor patch leaked into L-shaped exterior cutout"
    assert np.all(patched[:, 0] >= -1.05) and np.all(patched[:, 0] <= 1.05)
    assert np.all(patched[:, 2] >= -1.05) and np.all(patched[:, 2] <= 1.05)


def test_gravity_leveling_rotated_world_360():
    # World rotated 90 deg around X so camera vertical axis is (0, 0, -1) instead of (0, -1, 0)
    rx90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    r_wcs = []
    for ang in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        ry = np.array(
            [[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]],
            dtype=np.float64,
        )
        r_wcs.append(rx90 @ ry)

    rot_level = _compute_gravity_level_rotation(r_wcs)
    leveled_ups = np.array([rot_level @ (r @ np.array([0.0, -1.0, 0.0])) for r in r_wcs])
    mean_leveled_up = np.mean(leveled_ups, axis=0)
    assert np.allclose(mean_leveled_up, np.array([0.0, -1.0, 0.0]), atol=1e-3)


def test_project_colmap_point_and_count_good_reg_images_handles_unregistered():
    class _Cam2DOnly:
        params = [500.0, 250.0, 250.0, 0.0]

        def img_from_cam(self, pt):
            if len(pt) == 3:
                raise TypeError("expects 2D normalized coords")
            return np.array([500.0 * pt[0] + 250.0, 500.0 * pt[1] + 250.0], dtype=np.float64)

    class _Rot:
        def matrix(self):
            return np.eye(3, dtype=np.float64)

    class _Pose:
        rotation = _Rot()
        translation = np.zeros(3, dtype=np.float64)

    class _P3D:
        def __init__(self, xyz):
            self.xyz = np.array(xyz, dtype=np.float64)

    class _P2D:
        def __init__(self, xy, pid):
            self.xy = np.array(xy, dtype=np.float64)
            self.point3D_id = pid

        def has_point3D(self):
            return True

    class _UnregIm:
        has_pose = False
        points2D = []

        def cam_from_world(self):
            raise RuntimeError("Image does not have a valid pose")

    class _RegIm:
        has_pose = True

        def __init__(self, pts2d):
            self.points2D = pts2d

        def cam_from_world(self):
            return _Pose()

    class _Rec:
        def __init__(self, cam, images, points3d):
            self.cameras = {1: cam}
            self.images = images
            self.points3D = points3d

        def num_reg_images(self):
            return 1

    cam = _Cam2DOnly()
    uv = _project_colmap_point(cam, np.array([0.2, -0.4, 2.0], dtype=np.float64))
    assert uv is not None and np.allclose(uv, (300.0, 150.0), atol=1e-4)

    p3s = {}
    p2s = []
    for i in range(10):
        xyz = [0.1 * (i - 5), 0.05 * (i - 5), 2.0]
        p3s[i] = _P3D(xyz)
        p2s.append(_P2D([500.0 * (xyz[0] / xyz[2]) + 250.0, 500.0 * (xyz[1] / xyz[2]) + 250.0], i))

    rec = _Rec(cam, {1: _UnregIm(), 2: _RegIm(p2s)}, p3s)
    assert _count_good_reg_images(rec, min_anchors=8) == 1


def test_gui_theme_propagates_to_all_m3_buttons_swatches_and_switches(tmp_path: Path):
    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv)
    win = Native3DWindow(presets_path=tmp_path / "p.json")

    light_idx = win.theme_combo.findData("alabaster_light")
    assert light_idx >= 0
    win.theme_combo.setCurrentIndex(light_idx)

    # Every M3Button & M3SwatchButton must receive the light palette
    assert win.build_btn._pal.key == "alabaster_light"
    assert win.demo_btn._pal.key == "alabaster_light"
    assert win.reset_cam_btn._pal.key == "alabaster_light"
    assert win.fly_btn._pal.key == "alabaster_light"
    assert win.swatch_buttons["alabaster_light"]._pal.key == "alabaster_light"
    assert win.swatch_buttons["alabaster_light"].isChecked() is True
    assert win.swatch_buttons["graphite_cobalt"].isChecked() is False

    # M3Switch must sync visual _pos when setChecked is called with signals blocked
    win.dollhouse_chk.blockSignals(True)
    win.dollhouse_chk.setChecked(False)
    win.dollhouse_chk.blockSignals(False)
    assert win.dollhouse_chk._pos == 0.0

    win.close()
    app.processEvents()


def test_build_3d_scene_empty_depths_guard():
    k_mat = np.eye(3, dtype=np.float32)
    scene = build_3d_scene(frames=[], depths=[], masks=[], poses=[], intrinsics=k_mat, device="cpu")
    assert len(scene.points) == 0
    assert len(scene.mesh_faces) == 0


def test_video_ingest_monotonic_progress(tmp_path: Path):
    demo_mp4 = tmp_path / "prog_test.mp4"
    generate_demo_video(demo_mp4, duration_sec=2.0, fps=12, width=256, height=192)
    fracs = []
    vd = extract_keyframes(demo_mp4, max_frames=8, max_dim=256, progress_cb=lambda f, _m: fracs.append(f))
    assert len(vd.frames) == 8
    assert len(fracs) > 0
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))
    assert fracs[-1] <= 0.28 + 1e-9


def test_video_ingest_trim_window_uses_window_stride_and_open_end_duration(tmp_path: Path):
    demo_mp4 = tmp_path / "long_trim.mp4"
    generate_demo_video(demo_mp4, duration_sec=6.0, fps=20, width=256, height=192)

    # 1.0s trim window (20 frames) out of a 120-frame video with max_frames=12:
    # must extract 12 frames inside [2.0s, 3.0s], not starve from total_frames stride
    vd_win = extract_keyframes(demo_mp4, max_frames=12, max_dim=256, start_sec=2.0, end_sec=3.0)
    assert len(vd_win.frames) == 12
    assert 0.9 <= vd_win.duration <= 1.1

    # Open-ended trim (start_sec=4.0, end_sec=0.0) must report ~2.0s duration, not full 6.0s
    vd_tail = extract_keyframes(demo_mp4, max_frames=10, max_dim=256, start_sec=4.0, end_sec=0.0)
    assert len(vd_tail.frames) == 10
    assert 1.8 <= vd_tail.duration <= 2.2
    assert all(fr.timestamp >= 3.9 for fr in vd_tail.frames)


def test_gui_preserves_max_anchors_and_trim_window(tmp_path: Path):
    from src.presets import Preset

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv)
    init_p = Preset(name="Outdoor High", scene_type="outdoor", max_anchors=250).sanitized()
    win = Native3DWindow(
        initial_preset=init_p,
        presets_path=tmp_path / "p.json",
        initial_start_sec=5.0,
        initial_end_sec=20.0,
    )
    gathered = win.gather_ui_preset()
    assert gathered.max_anchors == 250
    assert gathered.scene_type == "outdoor"
    assert win._initial_start_sec == 5.0
    assert win._initial_end_sec == 20.0
    win.close()
    app.processEvents()


def test_eval_gt_transformation_matrix_and_ply_mesh_sampling(tmp_path: Path):
    import open3d as o3d
    from src.eval_gt import evaluate_against_gt, load_point_cloud

    # 1. Verify load_point_cloud samples faces of a .ply triangle mesh (not just 4 vertices)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    ply_mesh = tmp_path / "quad_mesh.ply"
    o3d.io.write_triangle_mesh(str(ply_mesh), mesh)
    sampled_pts = load_point_cloud(ply_mesh)
    assert len(sampled_pts) >= 10000, f"Expected uniform surface sampling on .ply mesh, got {len(sampled_pts)} pts"

    # 2. Verify returned 4x4 'transformation' directly aligns raw pred_pts -> gt_pts (scale included)
    rng = np.random.default_rng(7)
    box1 = rng.uniform([0, 0, 0], [1.5, 1.5, 1.5], (1000, 3))
    box2 = rng.uniform([1.5, 0, 0], [4.0, 1.2, 0.9], (1400, 3))
    gt_pts = np.vstack([box1, box2]).astype(np.float64)
    pred_pts = (gt_pts * 2.15) + np.array([3.0, -2.0, 1.0])

    res = evaluate_against_gt(pred_pts, gt_pts, thresholds=(0.05, 0.10))
    T_full = np.array(res["transformation"], dtype=np.float64)
    aligned_via_T = (T_full[:3, :3] @ pred_pts.T).T + T_full[:3, 3]
    assert np.allclose(aligned_via_T, res["aligned_pred_pts"], atol=1e-6)
    assert res["median_p2g_cm"] < 2.0


def test_depth_engine_device_flag():
    from src.depth_engine import DepthEngine

    eng_cpu = DepthEngine(device="cpu")
    assert eng_cpu._is_cuda is False
    assert eng_cpu.dtype == torch.float32
    if torch.cuda.is_available():
        eng_c0 = DepthEngine(device="cuda:0")
        assert eng_c0._is_cuda is True
        assert eng_c0.dtype == torch.float16


def test_align_selected_depths_drops_invalid_poses_without_sparse_obs():
    from src.pose_tracker import align_selected_depths
    from src.video_ingest import VideoFrame

    h, w = 32, 32
    k_mat = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    frames = [VideoFrame(index=i, timestamp=float(i) * 0.1, rgb=np.zeros((h, w, 3), dtype=np.uint8), sharpness=10.0) for i in range(14)]
    depths = [np.full((h, w), 2.0, dtype=np.float32) for _ in range(14)]
    masks = [np.ones((h, w), dtype=bool) for _ in range(14)]
    poses = [
        CameraPose(r_wc=np.eye(3), t_wc=np.array([float(i) * 0.1, 0.0, 0.0]), valid=(i != 13))
        for i in range(14)
    ]
    uv = np.stack([np.linspace(4, 28, 12), np.linspace(4, 28, 12)], axis=1).astype(np.float32)
    sparse_obs = {i: (uv, np.full(12, 2.0, dtype=np.float32)) for i in range(13)}

    _, _, keep = align_selected_depths(
        frames=frames,
        selected_indices=list(range(14)),
        depths_sub=depths,
        masks_sub=masks,
        poses=poses,
        sparse_obs=sparse_obs,
        k_mat=k_mat,
        k1_cal=0.0,
        device="cpu",
        use_grid_warp=False,
    )
    assert 13 not in keep
    assert len(keep) == 13

