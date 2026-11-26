import argparse
from pathlib import Path
import signal
import sys
import time
from src.presets import DEFAULT_PRESET_NAME, PresetManager


def main():
    parser = argparse.ArgumentParser(
        description="arg: turn videos of any length n into an interactive 3D environment"
    )
    parser.add_argument("video", nargs="?", help="path to input video file")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="run headless CLI reconstruction and export files without opening the desktop window",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        help="preset name to load (e.g. 'High', 'Fast Draft', 'Balanced', 'Ultra Sharp', or custom)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="list all available built-in and custom presets and exit",
    )
    parser.add_argument(
        "--load-scene",
        type=str,
        default=None,
        help="load a cached .npz, 3DGS .ply, or .glb/.obj scene directly into the native 3D viewer",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="generate a synthetic 3d courtyard video of length --demo-duration",
    )
    parser.add_argument(
        "--demo-duration",
        type=float,
        default=5.0,
        help="duration n in seconds if generating a demo video (default: 5.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="override max keyframes to sample across the video length n",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="override directory for exported 3d files",
    )
    parser.add_argument(
        "--scene-type",
        choices=["indoor", "outdoor"],
        default=None,
        help="scene profile: 'indoor' (default) or 'outdoor' (80m metric depth, sky/horizon culling)",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="override SfM anchor budget (0 for auto adaptive 140..360, or explicit 12..500)",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="trim start offset in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=0.0,
        help="trim end offset in seconds (default: 0.0 = until end of video)",
    )
    args = parser.parse_args()

    preset_mgr = PresetManager()
    if args.list_presets:
        print("Available Presets:")
        for p in preset_mgr.list_presets():
            kind = "custom" if preset_mgr.is_custom(p.name) else "built-in"
            print(
                f"  - {p.name:18s} [{kind:8s}] : frames={p.max_frames}, res={p.max_dim}px, "
                f"stride={p.point_stride} | {p.description}"
            )
        return

    preset = preset_mgr.get(args.preset)
    if args.max_frames is not None:
        preset.max_frames = args.max_frames
    if args.out_dir is not None:
        preset.out_dir = args.out_dir
    if args.scene_type is not None:
        preset.scene_type = args.scene_type
    if args.max_anchors is not None:
        preset.max_anchors = args.max_anchors
    preset = preset.sanitized()

    video_path = Path(args.video) if args.video else None

    if args.demo and video_path is None:
        video_path = Path(preset.out_dir) / "demo_courtyard.mp4"
        print(f"generating {args.demo_duration:.1f}s demo video at {video_path}...")
        from src.video_ingest import generate_demo_video
        generate_demo_video(video_path, duration_sec=args.demo_duration)

    if args.cli:
        if video_path is None:
            video_path = Path(preset.out_dir) / "demo_courtyard.mp4"
            print(f"no video specified, generating {args.demo_duration:.1f}s demo video...")
            from src.video_ingest import generate_demo_video
            generate_demo_video(video_path, duration_sec=args.demo_duration)

        from src.pipeline import ReconstructionConfig, VideoTo3DPipeline
        from src.scene_builder import export_scene

        t0 = time.perf_counter()
        pipe = VideoTo3DPipeline()
        cfg = ReconstructionConfig.from_preset(preset)
        if args.start_sec > 0.0:
            cfg.start_sec = args.start_sec
        if args.end_sec > 0.0:
            cfg.end_sec = args.end_sec

        def _cli_progress(frac: float, msg: str):
            pct = int(round(frac * 100))
            el = time.perf_counter() - t0
            sys.stdout.write(f"\r[{pct:3d}% | {el:5.1f}s] {msg:<58}")
            sys.stdout.flush()
            if frac >= 1.0:
                sys.stdout.write("\n")

        res = pipe.run(video_path, config=cfg, progress_cb=_cli_progress)
        if getattr(res, "splat_scene", None) is not None:
            print(f"saved 3DGS splats -> {res.splat_ply_path}")
            res.splat_scene.release()
            return
        out_dir = Path(preset.out_dir)
        if preset.export_ply:
            ply_path = export_scene(res.scene, out_dir / "scene_cloud.ply", export_mesh=False)
            print(f"saved point cloud -> {ply_path}")
        if preset.export_glb:
            glb_path = export_scene(res.scene, out_dir / "scene_mesh.glb", export_mesh=True)
            print(f"saved 3d mesh     -> {glb_path}")
        if preset.export_obj:
            obj_path = export_scene(res.scene, out_dir / "scene_mesh.obj", export_mesh=True)
            print(f"saved obj mesh    -> {obj_path}")
        return

    from PySide6.QtWidgets import QApplication
    from src.native_viewer import Native3DWindow, configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv)

    cached_res = None
    load_scene_file = None
    if args.load_scene:
        cache_p = Path(args.load_scene)
        if cache_p.exists():
            if cache_p.suffix.lower() == ".npz":
                from src.pipeline import load_result_cache

                cached_res = load_result_cache(cache_p)
            else:
                load_scene_file = cache_p
        else:
            print(f"[WARNING] --load-scene file not found: {cache_p}")

    win = Native3DWindow(
        initial_video=video_path,
        initial_result=cached_res,
        initial_preset=preset,
        initial_start_sec=args.start_sec,
        initial_end_sec=args.end_sec,
    )
    if load_scene_file is not None:
        win.load_saved_scene_file(load_scene_file)
    win.show()
    win.raise_()
    win.activateWindow()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
