import numpy as np
import torch
from src.pose_tracker import CameraPose
from src.scene_builder import _refine_multiview_depths_gpu


def test_depth_reprojection_translation():
    # Setup: 2 cameras looking at a planar wall at Z=3.0m
    # Camera 0 is at (0, 0, 0)
    # Camera 1 is at (0, 0, 1.5) (moved 1.5m toward the wall)
    h, w = 64, 64
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # True wall depth for cam0 is 3.0m, for cam1 is 1.5m
    d0 = np.full((h, w), 3.0, dtype=np.float32)
    # Add slight 5cm noise to cam1 measurement: 1.55m instead of 1.50m
    d1 = np.full((h, w), 1.55, dtype=np.float32)

    m0 = np.ones((h, w), dtype=bool)
    m1 = np.ones((h, w), dtype=bool)

    p0 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64), valid=True)
    p1 = CameraPose(r_wc=np.eye(3, dtype=np.float64), t_wc=np.array([0.0, 0.0, 1.5], dtype=np.float64), valid=True)

    depths = [d0, d1]
    masks = [m0, m1]
    poses = [p0, p1]
    fuse_indices = [0, 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused_depths, clean_masks = _refine_multiview_depths_gpu(
        depths, masks, poses, intrinsics, fuse_indices, device=device
    )

    # The fused depth for cam0 should be close to 3.0m + slight blend toward 3.05m
    # Under old broken ratio multiplication, d0 * (1.55 / 1.50) = 3.0 * 1.033 = 3.10
    # But if camera was closer: cam0 at 1.5, cam1 at 0.0:
    # d1 = 3.0m, cam0 sees at 1.5m. Under old bug: 1.5 * (3.0 / 1.5) = 3.0m!
    # Specifically, fused depth for cam0 should be between 3.0m and 3.06m (linear blending of 3.0m and 3.05m)
    fused_d0 = fused_depths[0]
    center_val = float(fused_d0[cy_int := int(cy), cx_int := int(cx)])
    assert 2.95 <= center_val <= 3.08, f"Fused depth {center_val:.3f}m is warped!"
