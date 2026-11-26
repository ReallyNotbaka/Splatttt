import numpy as np
from src.pose_tracker import _compute_gravity_level_rotation, CameraPose


def test_gravity_leveling_walkthrough_guard():
    # If camera walks straight along +Z with a 25 degree downward tilt (no 360 yaw spin)
    tilt_rad = np.radians(25.0)
    r_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0.0, np.sin(tilt_rad), np.cos(tilt_rad)],
    ])

    # 10 poses all tilted 25 deg down walking straight
    r_wcs = [r_tilt.copy() for _ in range(10)]
    rot_level = _compute_gravity_level_rotation(r_wcs)

    # Should NOT rotate the scene into a slanted ramp! (rot_level should remain identity)
    assert np.allclose(rot_level, np.eye(3), atol=1e-3), "Walkthrough without yaw spin was incorrectly tilted into a ramp!"


def test_gravity_leveling_360_circle():
    # If camera spins 360 degrees around Y axis
    r_wcs = []
    tilt_rad = np.radians(10.0)
    r_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0.0, np.sin(tilt_rad), np.cos(tilt_rad)],
    ])
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        r_wcs.append(ry @ r_tilt)

    rot_level = _compute_gravity_level_rotation(r_wcs)
    # With balanced 360 spin, rot_level should successfully calculate leveling rotation
    assert rot_level.shape == (3, 3)
    assert np.allclose(rot_level.T @ rot_level, np.eye(3), atol=1e-4)
