"""Unit tests for src/gsplat_engine.py (3DGS MCMC engine, SSIM, and PLY export)."""

from pathlib import Path
import struct
import numpy as np
import pytest
import torch

from src.gsplat_engine import (
    SplatScene,
    TrainingViews,
    export_splats_ply,
    gaussian_ssim_11x11,
    is_gsplat_available,
    render_view,
    resolve_gsplat_preset,
    train_3dgs_scene,
)
from src.native_viewer import SplatOrbitCamera
from src.pipeline import ReconstructionConfig
from src.pose_tracker import CameraPose
from src.presets import Preset


def _read_ply_raw(path: Path):
    data = path.read_bytes()
    marker = b"end_header\n"
    idx = data.index(marker) + len(marker)
    header = data[:idx].decode("ascii")
    payload = data[idx:]
    return header, payload


def test_is_gsplat_available_returns_bool():
    avail = is_gsplat_available()
    assert isinstance(avail, bool)


def test_export_splats_ply_binary_layout_and_roundtrip(tmp_path: Path):
    n_pts = 5
    torch.manual_seed(7)
    means = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [float("nan"), 0.0, 0.0],  # should be filtered
            [-0.5, 0.5, 2.5],
            [0.1, float("inf"), 0.2],  # should be filtered
        ],
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scales = torch.full((n_pts, 3), -2.5, dtype=torch.float32)
    opacities = torch.tensor([0.0, 1.2, 0.0, -0.8, 0.0], dtype=torch.float32)
    sh0 = torch.arange(n_pts * 3, dtype=torch.float32).reshape(n_pts, 1, 3) * 0.1
    shN = torch.arange(n_pts * 15 * 3, dtype=torch.float32).reshape(n_pts, 15, 3) * 0.01

    scene = SplatScene(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        sh_degree=3,
        scene_scale=1.0,
    )
    ply_path = export_splats_ply(scene, tmp_path / "test_splats.ply")
    header, payload = _read_ply_raw(ply_path)

    assert "format binary_little_endian 1.0" in header
    assert "element vertex 3" in header
    for prop in (
        "property float x",
        "property float nx",
        "property float f_dc_0",
        "property float f_rest_0",
        "property float f_rest_44",
        "property float opacity",
        "property float scale_0",
        "property float rot_0",
        "property float rot_3",
    ):
        assert prop in header

    # 62 float32 properties * 4 bytes = 248 bytes per vertex
    assert len(payload) == 3 * 62 * 4
    arr = np.frombuffer(payload, dtype="<f4").reshape(3, 62)
    assert np.all(np.isfinite(arr))

    # Row 1 in output corresponds to input index 1
    np.testing.assert_allclose(arr[1, 0:3], [1.0, 2.0, 3.0], atol=1e-6)
    # Normals are zero
    np.testing.assert_allclose(arr[1, 3:6], [0.0, 0.0, 0.0], atol=1e-6)
    # f_dc_0..2
    np.testing.assert_allclose(arr[1, 6:9], sh0[1, 0].numpy(), atol=1e-6)
    # f_rest must be (3, 15) flattened -> R0..14, G0..14, B0..14
    expected_f_rest = shN[1].numpy().T.reshape(45)
    np.testing.assert_allclose(arr[1, 9:54], expected_f_rest, atol=1e-6)
    # Raw logit opacity and log scales
    assert arr[1, 54] == pytest.approx(1.2, abs=1e-6)
    np.testing.assert_allclose(arr[1, 55:58], [-2.5, -2.5, -2.5], atol=1e-6)
    # Normalized quaternion for [1, 1, 0, 0] -> [1/sqrt(2), 1/sqrt(2), 0, 0]
    inv_s2 = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(arr[1, 58:62], [inv_s2, inv_s2, 0.0, 0.0], atol=1e-6)

    # Check raw little-endian struct unpack matches vertex 0 x
    x0 = struct.unpack("<f", payload[:4])[0]
    assert x0 == pytest.approx(0.0, abs=1e-6)


def test_gaussian_ssim_11x11_values_and_gradients():
    torch.manual_seed(0)
    img = torch.rand((1, 3, 64, 64), dtype=torch.float32, requires_grad=True)

    ssim_same = gaussian_ssim_11x11(img, img.detach())
    assert ssim_same.item() == pytest.approx(1.0, abs=1e-5)

    # High-contrast pattern so 1.0 - pat is genuinely anti-correlated with pat
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, 64),
        torch.linspace(0.0, 1.0, 64),
        indexing="ij",
    )
    checker = (((torch.floor(xx * 8) + torch.floor(yy * 8)) % 2) * 0.9 + 0.05).unsqueeze(0).unsqueeze(0)
    pat = checker.expand(1, 3, 64, 64).clone().detach().requires_grad_(True)
    ssim_inv = gaussian_ssim_11x11(pat, 1.0 - pat.detach())
    assert ssim_inv.item() < 0.1

    # Precomputed gt_stats must match standard gaussian_ssim_11x11
    from src.gsplat_engine import _precompute_gt_ssim_stats

    gt_chw = (1.0 - pat).detach()
    gt_stats = _precompute_gt_ssim_stats(gt_chw)
    ssim_cached = gaussian_ssim_11x11(pat, gt_chw, gt_stats=gt_stats)
    assert ssim_cached.item() == pytest.approx(ssim_inv.item(), abs=1e-5)

    loss = 1.0 - gaussian_ssim_11x11(pat, (1.0 - pat).detach())
    loss.backward()
    assert pat.grad is not None
    assert torch.isfinite(pat.grad).all()
    assert pat.grad.abs().sum().item() > 0.0


@pytest.mark.skipif(not is_gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_train_3dgs_scene_mini_synthetic():
    width, height = 64, 64
    k_mat = np.array(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, 3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[16:48, 16:48] = np.array([220, 80, 40], dtype=np.uint8)

    views = TrainingViews(
        images=[img, img],
        c2ws=np.stack([c2w, c2w], axis=0),
        viewmats=np.stack([viewmat, viewmat], axis=0),
        K=k_mat,
        width=width,
        height=height,
        frame_indices=[0, 1],
    )

    rng = np.random.default_rng(123)
    xyz = rng.uniform(-0.3, 0.3, size=(120, 3)).astype(np.float32)
    rgb = np.tile(np.array([[220, 80, 40]], dtype=np.uint8), (120, 1))

    scene = train_3dgs_scene(
        views=views,
        xyz=xyz,
        rgb=rgb,
        steps=10,
        cap_max=200,
        refine_start_iter=2,
        refine_stop_iter=8,
        refine_every=4,
    )
    assert scene.final_splats >= 120
    assert torch.isfinite(scene.means).all()

    rendered = render_view(scene, c2w=c2w, K=k_mat, width=width, height=height)
    assert rendered.shape == (height, width, 3)
    assert rendered.dtype == np.uint8

    scene.release()
    assert not scene.means.is_cuda
    assert scene._render_cache is None


@pytest.mark.skipif(not is_gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_train_3dgs_scene_ssim_half_res_branch():
    width, height = 960, 540
    k_mat = np.array(
        [[900.0, 0.0, width * 0.5], [0.0, 900.0, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, 3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    img = np.full((height, width, 3), 120, dtype=np.uint8)
    views = TrainingViews(
        images=[img],
        c2ws=c2w[None],
        viewmats=viewmat[None],
        K=k_mat,
        width=width,
        height=height,
        frame_indices=[0],
    )
    xyz = np.random.default_rng(1).uniform(-0.2, 0.2, size=(64, 3)).astype(np.float32)
    rgb = np.full((64, 3), 120, dtype=np.uint8)

    scene = train_3dgs_scene(
        views=views,
        xyz=xyz,
        rgb=rgb,
        steps=4,
        cap_max=100,
        ssim_half_res=True,
        refine_start_iter=99,
        refine_stop_iter=99,
    )
    assert scene.final_splats == 64
    assert np.isfinite(scene.train_psnr_db)
    scene.release()


def test_preset_mapping_and_pipeline_defaults():
    cfg_default = ReconstructionConfig()
    assert cfg_default.recon_mode == "tsdf"
    assert Preset(name="DefaultCheck").recon_mode == "tsdf"

    fast = resolve_gsplat_preset("Fast Draft")
    assert fast.steps == 3000
    assert fast.warmup_steps == 800
    assert fast.mid_res_steps == 700
    assert fast.cap_max == 200_000
    assert fast.max_dim == 960
    assert fast.sh_degree == 2
    assert fast.refine_start_iter == 400
    assert fast.refine_every == 100
    assert fast.refine_stop_iter == 2000
    assert fast.reset_every == 1500

    bal = resolve_gsplat_preset("Balanced")
    assert bal.steps == 4500
    assert bal.warmup_steps == 1000
    assert bal.mid_res_steps == 1200
    assert bal.cap_max == 300_000
    assert bal.max_dim == 1024
    assert bal.sh_degree == 2
    assert bal.refine_start_iter == 400
    assert bal.refine_every == 100
    assert bal.refine_stop_iter == 3200
    assert bal.reset_every == 1500

    high = resolve_gsplat_preset("High")
    assert high.steps == 6000
    assert high.warmup_steps == 1500
    assert high.mid_res_steps == 1700
    assert high.cap_max == 400_000
    assert high.max_dim == 1280
    assert high.sh_degree == 2
    assert high.refine_start_iter == 400
    assert high.refine_every == 100
    assert high.refine_stop_iter == 4000
    assert high.reset_every == 1500

    assert resolve_gsplat_preset("High (Custom)").name == "High"
    assert resolve_gsplat_preset("Ultra Sharp").name == "High"
    assert resolve_gsplat_preset("Unknown Custom").name == "Balanced"


def test_splat_orbit_camera_right_side_up_and_transforms():
    cam = SplatOrbitCamera()
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    poses = [
        CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.0, 0.0, -3.0]), valid=True),
        CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([1.0, 0.0, -2.5]), valid=True),
    ]
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, size=(100, 3)).astype(np.float32)
    cam.set_overview(pts, poses)

    def _assert_valid_cam():
        c2w = cam.get_c2w()
        assert c2w.shape == (4, 4)
        assert np.all(np.isfinite(c2w))
        rot = c2w[:3, :3]
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-5)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-5)
        # Camera -Y axis must point upward (+dot with COLMAP world_up [0, -1, 0])
        cam_up = -c2w[:3, 1]
        assert float(np.dot(cam_up, world_up)) > 0.05
        k_mat = cam.get_intrinsics(800, 600)
        assert k_mat.shape == (3, 3)
        assert k_mat[0, 0] > 0.0 and k_mat[1, 1] > 0.0
        assert k_mat[0, 2] == pytest.approx(400.0)
        assert k_mat[1, 2] == pytest.approx(300.0)

    _assert_valid_cam()

    prev_yaw = cam.yaw
    cam.orbit(120.0, 40.0)
    assert cam.yaw != prev_yaw
    _assert_valid_cam()

    # Extreme vertical drag must clamp pitch and never flip upside-down
    cam.orbit(0.0, 10000.0)
    _assert_valid_cam()
    cam.orbit(0.0, -20000.0)
    _assert_valid_cam()

    prev_dist = cam.distance
    cam.zoom(2.0)
    assert cam.distance < prev_dist
    _assert_valid_cam()

    prev_target = cam.target.copy()
    cam.pan(50.0, -30.0)
    assert np.linalg.norm(cam.target - prev_target) > 1e-4
    _assert_valid_cam()

    prev_target2 = cam.target.copy()
    cam.fly(1.0, 0.5, 0.25)
    assert np.linalg.norm(cam.target - prev_target2) > 1e-4
    _assert_valid_cam()

    cam.set_from_pose(poses[0], look_dist=2.5)
    _assert_valid_cam()

    # Orbit direction parity: dx > 0 moves front-of-target right; dy > 0 moves front-of-target down
    cam.set_from_pose(poses[0], look_dist=2.5)
    c2w0 = cam.get_c2w()
    p_front = cam.target - c2w0[:3, 2] * 0.5
    cam.orbit(+100.0, 0.0)
    c2w_h = cam.get_c2w()
    assert float(np.dot(p_front - c2w_h[:3, 3], c2w_h[:3, 0])) > 0.0

    cam.set_from_pose(poses[0], look_dist=2.5)
    c2w0 = cam.get_c2w()
    p_front = cam.target - c2w0[:3, 2] * 0.5
    cam.orbit(0.0, +100.0)
    c2w_v = cam.get_c2w()
    assert float(np.dot(p_front - c2w_v[:3, 3], c2w_v[:3, 1])) > 0.0


@pytest.mark.skipif(not is_gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_train_3dgs_scene_cpu_device_guard():
    k_mat = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    views = TrainingViews(
        images=[img],
        c2ws=c2w[None],
        viewmats=c2w[None],
        K=k_mat,
        width=32,
        height=32,
        frame_indices=[0],
    )
    xyz = np.zeros((16, 3), dtype=np.float32)
    rgb = np.zeros((16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="CUDA"):
        train_3dgs_scene(views=views, xyz=xyz, rgb=rgb, steps=2, device="cpu")


def test_gui_preset_preservation_and_disabled_state(tmp_path: Path):
    from PySide6.QtWidgets import QApplication
    from src.native_viewer import Native3DWindow
    from src.presets import PresetManager

    app = QApplication.instance() or QApplication([])
    assert app is not None
    win = Native3DWindow(presets_path=tmp_path / "p.json")
    try:
        if win._gsplat_avail:
            win.recon_mode_combo.setCurrentIndex(1)
            assert win.recon_mode_combo.currentData() == "3dgs"
            assert not win.depth_model_combo.isEnabled()

            bal_idx = win.preset_combo.findData("Balanced")
            assert bal_idx >= 0
            win.preset_combo.setCurrentIndex(bal_idx)
            assert win.recon_mode_combo.currentData() == "3dgs"

            win.recon_mode_combo.setCurrentIndex(0)
            assert win.depth_model_combo.isEnabled()
        else:
            assert win.recon_mode_combo.currentData() == "tsdf"
            assert win.depth_model_combo.isEnabled()
    finally:
        win.close()


def test_load_training_views_reuses_highres_rgb(tmp_path: Path):
    from src.gsplat_engine import load_training_views_pyav
    from src.video_ingest import VideoFrame

    dummy_vid = tmp_path / "dummy.mp4"
    # Create a tiny valid mp4 just for cv2.VideoCapture metadata probe
    import cv2

    writer = cv2.VideoWriter(
        str(dummy_vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (128, 96)
    )
    for _ in range(3):
        writer.write(np.zeros((96, 128, 3), dtype=np.uint8))
    writer.release()

    hr0 = np.full((96, 128, 3), 77, dtype=np.uint8)
    hr1 = np.full((96, 128, 3), 144, dtype=np.uint8)
    frames = [
        VideoFrame(index=0, timestamp=0.0, rgb=np.zeros((48, 64, 3), dtype=np.uint8), sharpness=10.0, highres_rgb=hr0),
        VideoFrame(index=1, timestamp=0.033, rgb=np.zeros((48, 64, 3), dtype=np.uint8), sharpness=10.0, highres_rgb=hr1),
    ]
    poses = [
        CameraPose(r_wc=np.eye(3), t_wc=np.array([0.0, 0.0, 0.0]), valid=True),
        CameraPose(r_wc=np.eye(3), t_wc=np.array([0.5, 0.0, 0.0]), valid=True),
    ]
    k_mat = np.array([[64.0, 0.0, 32.0], [0.0, 64.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    views = load_training_views_pyav(dummy_vid, frames, poses, k_mat, max_dim=128)
    assert len(views.images) == 2
    assert np.all(views.images[0] == 77)
    assert np.all(views.images[1] == 144)
    assert frames[0].highres_rgb is None
    assert frames[1].highres_rgb is None


@pytest.mark.skipif(not is_gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_train_3dgs_scene_scale_clamp_and_floater_prune():
    width, height = 64, 64
    k_mat = np.array(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    poses = []
    images = []
    c2ws = []
    viewmats = []
    for i in range(10):
        ang = 2.0 * np.pi * i / 10.0
        eye = np.array([2.0 * np.cos(ang), 0.0, 2.0 * np.sin(ang)], dtype=np.float32)
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_wc = np.stack([right, down, fwd], axis=1).astype(np.float64)
        poses.append(CameraPose(r_wc=r_wc, t_wc=eye.astype(np.float64), valid=True))
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r_wc.astype(np.float32)
        c2w[:3, 3] = eye
        c2ws.append(c2w)
        viewmats.append(np.linalg.inv(c2w))
        images.append(np.full((height, width, 3), 128, dtype=np.uint8))

    views = TrainingViews(
        images=images,
        c2ws=np.stack(c2ws, axis=0),
        viewmats=np.stack(viewmats, axis=0),
        K=k_mat,
        width=width,
        height=height,
        frame_indices=list(range(10)),
    )
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-0.4, 0.4, size=(1200, 3)).astype(np.float32)
    rgb = np.full((1200, 3), 128, dtype=np.uint8)

    scene = train_3dgs_scene(
        views=views,
        xyz=xyz,
        rgb=rgb,
        poses=poses,
        steps=12,
        cap_max=1500,
        packed=True,
        refine_start_iter=2,
        refine_stop_iter=10,
        refine_every=4,
    )
    min_log_scale = float(np.log(max(1e-4, 1e-4 * scene.scene_scale)))
    assert float(scene.scales.min().item()) >= min_log_scale - 1e-4
    assert scene.final_splats > 0
    assert torch.isfinite(scene.means).all()
    scene.release()


@pytest.mark.skipif(not is_gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_splat_scene_sky_dome_render_and_dense_depth_inputs():
    width, height = 64, 64
    k_mat = np.array(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    poses = []
    images = []
    c2ws = []
    viewmats = []
    view_depths = {}
    for i in range(10):
        ang = 2.0 * np.pi * i / 10.0
        eye = np.array([2.0 * np.cos(ang), 0.0, 2.0 * np.sin(ang)], dtype=np.float32)
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_wc = np.stack([right, down, fwd], axis=1).astype(np.float64)
        poses.append(CameraPose(r_wc=r_wc, t_wc=eye.astype(np.float64), valid=True))
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r_wc.astype(np.float32)
        c2w[:3, 3] = eye
        c2ws.append(c2w)
        viewmats.append(np.linalg.inv(c2w))
        im = np.full((height, width, 3), 210, dtype=np.uint8)
        im[24:40, 24:40] = np.array([60, 120, 80], dtype=np.uint8)
        images.append(im)
        dm = np.full((height, width), 2.0, dtype=np.float32)
        mk = np.ones((height, width), dtype=bool)
        mk[:10, :] = False
        view_depths[i] = (dm, mk)

    views = TrainingViews(
        images=images,
        c2ws=np.stack(c2ws, axis=0),
        viewmats=np.stack(viewmats, axis=0),
        K=k_mat,
        width=width,
        height=height,
        frame_indices=list(range(10)),
    )
    rng = np.random.default_rng(11)
    xyz = rng.uniform(-0.3, 0.3, size=(120, 3)).astype(np.float32)
    rgb = np.full((120, 3), 100, dtype=np.uint8)
    dense_xyz = rng.uniform(-0.5, 0.5, size=(180, 3)).astype(np.float32)
    dense_rgb = np.full((180, 3), 120, dtype=np.uint8)

    scene = train_3dgs_scene(
        views=views,
        xyz=xyz,
        rgb=rgb,
        poses=poses,
        steps=16,
        cap_max=600,
        scene_type="outdoor",
        dense_xyz=dense_xyz,
        dense_rgb=dense_rgb,
        view_depths=view_depths,
        use_sky_dome=True,
        refine_start_iter=2,
        refine_stop_iter=12,
        refine_every=5,
    )
    assert scene.sky_logits is not None
    assert scene.sky_logits.shape == (1, 3, 12, 24)
    rendered = render_view(scene, c2w=c2ws[0], K=k_mat, width=width, height=height)
    assert rendered.shape == (height, width, 3)
    assert rendered[:4, :, :].mean() > 40.0
    assert scene._render_cache is not None and len(scene._render_cache) == 7
    scene.release()
    assert scene._render_cache is None
    assert scene.sky_logits is not None and not scene.sky_logits.is_cuda


def test_set_overview_frames_inward_orbit_inside_outer_clutter_and_linear_path():
    cam = SplatOrbitCamera()
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    # 360 inward orbit of radius 0.5 with outer trees/walls at r = 2.0
    poses_circle = []
    for i in range(16):
        ang = 2.0 * np.pi * i / 16.0
        eye = np.array([0.5 * np.cos(ang), 0.0, 0.5 * np.sin(ang)], dtype=np.float64)
        fwd = (-eye / np.linalg.norm(eye)).astype(np.float32)
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_wc = np.stack([right, down, fwd], axis=1).astype(np.float64)
        poses_circle.append(CameraPose(r_wc=r_wc, t_wc=eye, valid=True))

    rng = np.random.default_rng(3)
    inner_pts = rng.uniform(-0.2, 0.2, size=(200, 3)).astype(np.float32)
    outer_clutter = rng.uniform(-2.0, 2.0, size=(200, 3)).astype(np.float32)
    pts = np.concatenate([inner_pts, outer_clutter], axis=0)

    cam.set_overview(pts, poses_circle)
    c2w_c = cam.get_c2w()
    eye_dist = float(np.linalg.norm(c2w_c[:3, 3] - cam.target))
    assert eye_dist < 1.1  # Stays inside/near the camera ring instead of spawning behind outer clutter
    assert float(np.dot(-c2w_c[:3, 1], world_up)) > 0.1

    # Linear trackside walkaround (open angular coverage)
    poses_lin = []
    for i in range(12):
        eye = np.array([-2.0 + 0.4 * i, 0.0, -1.5], dtype=np.float64)
        poses_lin.append(CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=eye, valid=True))
    track_pts = rng.uniform([-2.0, -0.5, 0.5], [2.5, 0.5, 2.5], size=(300, 3)).astype(np.float32)
    cam.set_overview(track_pts, poses_lin)
    c2w_l = cam.get_c2w()
    assert c2w_l[2, 3] < cam.target[2]  # Stays on observed side of subject
    assert float(np.dot(-c2w_l[:3, 1], world_up)) > 0.1


def test_build_3dgs_depth_priors_preserves_pose_scale():
    from src.pipeline import VideoTo3DPipeline
    from src.video_ingest import VideoFrame

    pipe = VideoTo3DPipeline(device="cpu")
    k_mat = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    poses = [
        CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.2 * i, 0.0, -2.0]), valid=True)
        for i in range(8)
    ]
    orig_ts = [p.t_wc.copy() for p in poses]
    frames = [
        VideoFrame(index=i, timestamp=float(i), rgb=np.full((48, 64, 3), 128, dtype=np.uint8), sharpness=10.0)
        for i in range(8)
    ]
    xyz = np.random.default_rng(5).uniform([-0.4, -0.3, -0.2], [1.8, 0.3, 0.2], size=(120, 3)).astype(np.float32)

    def _fake_est(frames, **_kw):
        depths = [np.full((48, 64), 1.0, dtype=np.float32) for _ in frames]
        masks = [np.ones((48, 64), dtype=bool) for _ in frames]
        return depths, masks

    pipe.depth_engine.estimate_depths = _fake_est
    dense_xyz, dense_rgb, view_depths = pipe._build_3dgs_depth_priors(
        frames=frames,
        valid_idx=list(range(8)),
        poses=poses,
        k_mat=k_mat,
        k1_cal=0.0,
        xyz=xyz,
        edge_thresh=0.12,
        scene_type="outdoor",
        target_device="cpu",
    )
    for p, t0 in zip(poses, orig_ts):
        np.testing.assert_allclose(p.t_wc, t0, atol=1e-7)
    assert view_depths is not None and len(view_depths) == 8
    assert dense_xyz is not None and dense_rgb is not None
    assert len(dense_xyz) > 0 and np.all(np.isfinite(dense_xyz))


