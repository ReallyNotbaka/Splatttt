from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from src.native_viewer import Native3DWindow
from src.pipeline import ReconstructionConfig, VideoTo3DPipeline
from src.video_ingest import generate_demo_video


def test_native_viewer_loads_scene(tmp_path: Path):
    app = QApplication.instance() or QApplication(sys.argv)

    demo_mp4 = tmp_path / "short_test.mp4"
    generate_demo_video(demo_mp4, duration_sec=2.0, fps=12, width=384, height=256)

    pipe = VideoTo3DPipeline()
    cfg = ReconstructionConfig(max_frames=6, max_dim=320, point_stride=3, mesh_stride=4)
    res = pipe.run(demo_mp4, config=cfg)

    win = Native3DWindow()
    win.set_video_path(demo_mp4)
    win.load_result(res)

    # check that 3D actors exist in renderer (Balanced defaults to mesh mode, dollhouse=True, frustums=False)
    actors = win.plotter.renderer.actors
    assert ("surface_mesh" in actors) or ("point_cloud" in actors)
    assert "camera_frustums" not in actors
    assert win.dollhouse_chk.isChecked() is True

    # toggle camera frustums on
    win.frustums_chk.setChecked(True)
    assert "camera_frustums" in win.plotter.renderer.actors

    # switch to point cloud mode
    win.mode_combo.setCurrentIndex(0)
    assert "point_cloud" in win.plotter.renderer.actors

    # switch to surface mesh mode
    win.mode_combo.setCurrentIndex(1)
    assert "surface_mesh" in win.plotter.renderer.actors

    # switch to hybrid mode
    win.mode_combo.setCurrentIndex(2)
    assert "point_cloud" in win.plotter.renderer.actors
    assert "surface_mesh" in win.plotter.renderer.actors

    # test keyframe scrubbing & camera snap
    win.kf_slider.setValue(2)
    win.snap_to_selected_keyframe()
    win._step_flythrough()

    win.close()
    app.processEvents()
