from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from src.depth_engine import DepthEngine
from src.native_viewer import Native3DWindow, configure_hidpi
from src.pipeline import (
    ReconstructionConfig,
    VideoTo3DPipeline,
    load_result_cache,
    save_result_cache,
)
from src.presets import Preset, PresetManager
from src.video_ingest import generate_demo_video


def test_preset_persistence_and_builtin_protection(tmp_path: Path):
    p_file = tmp_path / "custom_presets.json"
    mgr = PresetManager(p_file)

    initial = mgr.list_presets()
    assert len(initial) == 6
    assert mgr.get("Balanced").max_frames == 32
    assert mgr.get("Balanced").depth_model == "large"
    assert mgr.get("Balanced").dollhouse_cull is True
    assert mgr.get("Balanced").show_frustums is False
    assert mgr.get("High").max_frames == 140
    assert mgr.get("High").max_dim == 640
    assert mgr.get("High").mesh_stride == 3
    assert mgr.get("High").depth_model == "large"
    assert mgr.get("NonExistentPreset").name == "High"

    # trying to save with a built-in name should NOT overwrite the built-in
    attempt_overwrite = Preset(name="Balanced", max_frames=77, max_dim=512)
    saved_safe = mgr.save_preset(attempt_overwrite)
    assert saved_safe.name == "Balanced (Custom)"
    assert mgr.get("Balanced").max_frames == 32
    assert mgr.get("Balanced (Custom)").max_frames == 77

    # create a named custom preset and test clamping
    custom = Preset(
        name="My Drone Preset",
        description="high altitude sweep",
        max_frames=35,
        max_dim=603,  # snaps to 600
        point_stride=1,
        mesh_stride=3,
        edge_thresh=0.09,
        render_mode="hybrid",
        theme="neon_ember",
    )
    saved = mgr.save_preset(custom)
    assert saved.max_dim == 600

    # reload from disk in a fresh manager
    mgr2 = PresetManager(p_file)
    assert mgr2.is_custom("My Drone Preset")
    loaded = mgr2.get("My Drone Preset")
    assert loaded.max_frames == 35
    assert loaded.max_dim == 600
    assert loaded.render_mode == "hybrid"
    assert loaded.theme == "neon_ember"

    # delete custom presets
    assert mgr2.delete_preset("My Drone Preset") is True
    assert mgr2.delete_preset("Balanced") is False


def test_scene_cache_roundtrip(tmp_path: Path):
    demo_mp4 = tmp_path / "cache_test.mp4"
    generate_demo_video(demo_mp4, duration_sec=2.0, fps=10, width=320, height=240)

    pipe = VideoTo3DPipeline()
    cfg = ReconstructionConfig(max_frames=5, max_dim=320, point_stride=3, mesh_stride=4)
    res = pipe.run(demo_mp4, config=cfg)

    cache_file = tmp_path / "scene_cache.npz"
    save_result_cache(res, cache_file)
    assert cache_file.exists()

    loaded = load_result_cache(cache_file)
    assert len(loaded.video_data.frames) == len(res.video_data.frames)
    assert loaded.scene.points.shape == res.scene.points.shape
    assert loaded.scene.mesh_faces.shape == res.scene.mesh_faces.shape
    assert len(loaded.scene.poses) == len(res.scene.poses)


def test_native_gui_presets_themes_and_milestones(tmp_path: Path):
    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv)

    p_file = tmp_path / "gui_presets.json"
    win = Native3DWindow(presets_path=p_file)

    # verify custom preset save & delete inside the native GUI
    win.preset_name_edit.setText("Handheld Room")
    win.preset_desc_edit.setText("indoor walk")
    win.max_frames_spin.setValue(26)
    win.max_dim_spin.setValue(480)
    win.save_custom_preset_from_ui()

    assert win.preset_mgr.is_custom("Handheld Room")
    assert win.preset_mgr.get("Handheld Room").max_frames == 26
    assert win.del_preset_btn.isEnabled()

    # switch themes
    win.theme_combo.setCurrentIndex(1)  # neon_ember
    win.theme_combo.setCurrentIndex(2)  # matrix_green

    # verify in-place progress updates do not spam milestone log on repeated calls in same stage
    lines_before = win.milestone_log.blockCount()
    win._on_worker_progress(0.32, "SfM feature matching...")
    win._on_worker_progress(0.42, "SfM bundle adjustment...")
    win._on_worker_progress(0.50, "Chained 3D-PnP localization...")
    lines_after = win.milestone_log.blockCount()
    assert (lines_after - lines_before) == 1

    # request_cancel null guard safety
    win.worker = None
    win.request_cancel()

    # delete custom preset
    win.delete_current_custom_preset()
    assert not win.preset_mgr.is_custom("Handheld Room")

    win.close()
    app.processEvents()


def test_pipeline_device_switching_and_presets():
    import torch
    pipe = VideoTo3DPipeline()
    initial_device = pipe.depth_engine.device

    # switch to cpu
    cfg_cpu = ReconstructionConfig(device="cpu")
    target_cpu = cfg_cpu.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_cpu != pipe.depth_engine.device:
        pipe.depth_engine = DepthEngine(device=target_cpu)
    assert pipe.depth_engine.device == "cpu"

    # switch back to auto (device=None)
    cfg_auto = ReconstructionConfig(device=None)
    target_auto = cfg_auto.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_auto != pipe.depth_engine.device:
        pipe.depth_engine = DepthEngine(device=target_auto)
    assert pipe.depth_engine.device == ("cuda" if torch.cuda.is_available() else "cpu")

    # verify preset stride bounds
    p = Preset(name="Test", mesh_stride=12, point_stride=0).sanitized()
    assert p.mesh_stride == 10
    assert p.point_stride == 1


def test_connected_component_orientation_and_worker():
    import numpy as np
    import open3d as o3d
    from src.native_viewer import ReconstructWorker

    # Test open3d connected component clustering behavior
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    )
    tri_clusters, num_triangles_per_cluster, _ = mesh.cluster_connected_triangles()
    assert len(num_triangles_per_cluster) >= 1
    assert len(tri_clusters) == 2

    # Check ReconstructWorker export_options attribute
    pipe = VideoTo3DPipeline()
    cfg = ReconstructionConfig()
    worker = ReconstructWorker(pipe, Path("dummy.mp4"), cfg, export_options=(Path("out"), True, True, False))
    assert worker.export_options is not None
    assert worker.export_options[1] is True
    assert worker.export_options[2] is True
    assert worker.export_options[3] is False


def test_trajectory_spike_filter():
    import numpy as np
    from src.pose_tracker import CameraPose, filter_trajectory_spikes

    # Case 1: Single-frame outlier spike at frame 5
    n = 10
    raw_poses = [
        CameraPose(r_wc=np.eye(3), t_wc=np.array([0.0, 0.0, float(i) * 0.1]))
        for i in range(n)
    ]
    raw_poses[5].t_wc = np.array([2.0, 0.0, 0.5])
    clean_indices = list(range(n))
    anchor_valid_indices = [0, 9]

    clean_res, _, a_spikes, pnp_spikes = filter_trajectory_spikes(
        raw_poses=raw_poses,
        clean_indices=clean_indices,
        anchor_valid_indices=anchor_valid_indices,
        med_anchor_step=0.10,
        avg_frame_gap=1.0,
    )
    assert 5 not in clean_res
    assert raw_poses[5] is None
    assert 4 in clean_res and 6 in clean_res
    assert pnp_spikes >= 1

    # Case 2: Multi-frame 4-frame drifted PnP burst (frames 4, 5, 6, 7) between anchors 0 and 15
    n2 = 16
    raw_poses2 = [
        CameraPose(r_wc=np.eye(3), t_wc=np.array([0.0, 0.0, float(i) * 0.1]))
        for i in range(n2)
    ]
    for k in (4, 5, 6, 7):
        raw_poses2[k].t_wc = np.array([1.5, 0.0, float(k) * 0.1])
    clean_indices2 = list(range(n2))
    anchor_valid_indices2 = [0, 15]

    clean_res2, _, _, pnp_spikes2 = filter_trajectory_spikes(
        raw_poses=raw_poses2,
        clean_indices=clean_indices2,
        anchor_valid_indices=anchor_valid_indices2,
        med_anchor_step=0.10,
        avg_frame_gap=1.0,
    )
    for k in (4, 5, 6, 7):
        assert k not in clean_res2
        assert raw_poses2[k] is None
    assert 3 in clean_res2 and 8 in clean_res2
    assert pnp_spikes2 >= 4
