import numpy as np
from src.pose_tracker import CameraPose


def test_spatial_neighbor_selection():
    # Camera 0 is at (0, 0, 0) facing +Z
    # Camera 1 is at (0.2, 0, 0.1) facing +Z turned 3 deg (cos = 0.998, distance = 0.22m)
    # Camera 2 is at (0, 0, 8.0) facing +Z exactly (cos = 1.000, distance = 8.0m)
    p0 = CameraPose(r_wc=np.eye(3), t_wc=np.array([0.0, 0.0, 0.0]), valid=True)

    ang = np.radians(3.0)
    r1 = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    p1 = CameraPose(r_wc=r1, t_wc=np.array([0.2, 0.0, 0.1]), valid=True)

    p2 = CameraPose(r_wc=np.eye(3), t_wc=np.array([0.0, 0.0, 8.0]), valid=True)

    poses = [p0, p1, p2]
    forwards = np.array([p.r_wc[:, 2] for p in poses], dtype=np.float32)
    ts = np.array([p.t_wc for p in poses], dtype=np.float32)

    d_scale = 0.60
    cos_d = forwards @ forwards[0]
    d_ti = np.linalg.norm(ts - ts[0], axis=-1)
    nbr_scores = cos_d / (1.0 + d_ti / d_scale)

    # Score of Cam 1 (close, slight angle) MUST be far higher than Cam 2 (8 meters away)
    score_cam1 = nbr_scores[1]
    score_cam2 = nbr_scores[2]
    assert score_cam1 > 5.0 * score_cam2, f"Cam 1 ({score_cam1}) should vastly outscore distant Cam 2 ({score_cam2})"
