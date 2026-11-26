from pathlib import Path
import numpy as np
import pytest
import torch

from src.eval_gt import evaluate_against_gt
from src.pose_tracker import CameraPose
from src.presets import BUILTIN_PRESETS, Preset
from src.scene_builder import _refine_multiview_depths_gpu
from src.video_ingest import extract_keyframes, generate_demo_video


def test_preset_scene_type_and_anchors():
    p = Preset(name="Test", scene_type="outdoor", max_anchors=200).sanitized()
    assert p.scene_type == "outdoor"
    assert p.max_anchors == 200

    p_auto = Preset(name="Test", scene_type="indoor", max_anchors=0).sanitized()
    assert p_auto.scene_type == "indoor"
    assert p_auto.max_anchors == 0

    p_clamp = Preset(name="Test", scene_type="invalid_type", max_anchors=999).sanitized()
    assert p_clamp.scene_type == "indoor"
    assert p_clamp.max_anchors == 500

    p_clamp_low = Preset(name="Test", max_anchors=5).sanitized()
    assert p_clamp_low.max_anchors == 12

    assert len(BUILTIN_PRESETS) == 6
    assert any(bp.name == "High" for bp in BUILTIN_PRESETS)
    for bp in BUILTIN_PRESETS:
        assert bp.scene_type == "indoor"
        assert bp.max_anchors == 0


def test_video_ingest_trim_window(tmp_path: Path):
    demo_mp4 = tmp_path / "trim_test.mp4"
    generate_demo_video(demo_mp4, duration_sec=3.0, fps=15, width=320, height=240)
    assert demo_mp4.exists()

    vdata = extract_keyframes(demo_mp4, max_frames=30, start_sec=0.8, end_sec=2.2)
    assert len(vdata.frames) > 0
    for fr in vdata.frames:
        assert 0.7 <= fr.timestamp <= 2.3, f"Timestamp {fr.timestamp} out of bounds [0.7, 2.3]"


def test_outdoor_depth_reprojection_far_range():
    # Setup: 2 cameras looking at a far planar wall at Z=18.0m in outdoor mode
    # Camera 0 is at (0, 0, 0)
    # Camera 1 is at (0, 0, 3.0) (3.0m closer to wall)
    h, w = 64, 64
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # True wall depth for cam0 is 18.0m, for cam1 is 15.0m
    d0 = np.full((h, w), 18.0, dtype=np.float32)
    # Add 0.25m noise to cam1 measurement: 15.25m instead of 15.0m
    d1 = np.full((h, w), 15.25, dtype=np.float32)

    m0 = np.ones((h, w), dtype=bool)
    m1 = np.ones((h, w), dtype=bool)

    p0 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64), valid=True)
    p1 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.0, 0.0, 3.0], dtype=np.float64), valid=True)

    depths = [d0, d1]
    masks = [m0, m1]
    poses = [p0, p1]
    fuse_indices = [0, 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused_depths, clean_masks = _refine_multiview_depths_gpu(
        depths, masks, poses, intrinsics, fuse_indices, device=device, scene_type="outdoor"
    )

    fused_d0 = fused_depths[0]
    center_val = float(fused_d0[int(cy), int(cx)])
    # Depth must stay accurate (17.8m..18.3m) and NOT be clipped at the indoor 8.5m boundary
    assert 17.8 <= center_val <= 18.3, f"Outdoor fused depth {center_val:.3f}m out of bounds [17.8, 18.3]!"


def test_eval_gt_sim3_recovery():
    np.random.seed(42)
    cab = np.random.uniform([0, 0, 0], [1.5, 1.8, 1.6], (1200, 3))
    bed = np.random.uniform([1.5, 0, 0], [4.5, 1.8, 1.0], (1800, 3))
    w1 = np.random.uniform([0.5, -0.2, -0.4], [1.1, 0, 0.2], (250, 3))
    w2 = np.random.uniform([0.5, 1.8, -0.4], [1.1, 2.0, 0.2], (250, 3))
    w3 = np.random.uniform([3.0, -0.2, -0.4], [3.6, 0, 0.2], (250, 3))
    w4 = np.random.uniform([3.0, 1.8, -0.4], [3.6, 2.0, 0.2], (250, 3))
    gt_pts = np.vstack([cab, bed, w1, w2, w3, w4]).astype(np.float64)

    scale_true = 1.37
    theta = 0.45
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    t_true = np.array([2.5, -1.8, 0.7])
    pred_pts = (gt_pts @ R_true.T) * scale_true + t_true
    pred_pts += np.random.normal(0, 0.008, pred_pts.shape)

    res = evaluate_against_gt(pred_pts, gt_pts, thresholds=(0.05, 0.10, 0.20, 0.30))
    expected_scale = 1.0 / scale_true

    assert abs(res["scale"] - expected_scale) / expected_scale < 0.03
    assert res["f_score_0.05"] > 0.95
    assert res["median_p2g_cm"] < 2.0
