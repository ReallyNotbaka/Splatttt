from pathlib import Path
import numpy as np
import torch
from src.pipeline import ReconstructionConfig, VideoTo3DPipeline
from src.scene_builder import export_scene
from src.video_ingest import generate_demo_video


def test_end_to_end_reconstruction(tmp_path: Path):
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu device: {torch.cuda.get_device_name(0)}")

    demo_mp4 = tmp_path / "demo_room.mp4"
    generate_demo_video(demo_mp4, duration_sec=3.0, fps=16, width=480, height=320)
    assert demo_mp4.exists() and demo_mp4.stat().st_size > 1000

    pipe = VideoTo3DPipeline()
    cfg = ReconstructionConfig(max_frames=10, max_dim=400, point_stride=2, mesh_stride=4)
    res = pipe.run(demo_mp4, config=cfg)

    assert len(res.video_data.frames) == 10
    assert len(res.scene.points) > 2000
    assert res.scene.points.shape[1] == 3
    assert res.scene.colors.shape == res.scene.points.shape
    assert len(res.scene.mesh_faces) > 500
    assert len(res.scene.poses) == 10

    p5 = np.percentile(res.scene.points, 5, axis=0)
    p95 = np.percentile(res.scene.points, 95, axis=0)
    span = float(np.linalg.norm(p95 - p5))
    assert 1.0 < span < 12.0

    ply_out = export_scene(res.scene, tmp_path / "scene_cloud.ply", export_mesh=False)
    glb_out = export_scene(res.scene, tmp_path / "scene_mesh.glb", export_mesh=True)
    assert ply_out.exists() and ply_out.stat().st_size > 1000
    assert glb_out.exists() and glb_out.stat().st_size > 1000
