from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, List, Optional
import cv2
import numpy as np


# Populated after each extract_keyframes call: which Stage-1 decoder ran and its cost.
LAST_STAGE1_TELEMETRY: dict = {}


def _open_pyav_video(path: Path | str, use_hw: bool = True):
    """Open a video with PyAV, preferring CUDA NVDEC over threaded CPU decode."""
    import av

    if use_hw:
        hwaccel = av.codec.hwaccel.HWAccel(device_type="cuda")
        return av.open(str(path), hwaccel=hwaccel)

    container = av.open(str(path))
    container.streams.video[0].thread_type = "AUTO"
    return container


def _pyav_decode_targets(
    path: Path | str,
    target_frames: List[int],
    out_w: int,
    out_h: int,
    fps: float,
    progress_cb: Optional[Callable[[float, str], None]],
    cancel_cb: Optional[Callable[[], bool]],
    use_hw: bool,
) -> Optional[List[tuple[int, float, np.ndarray, np.ndarray, float]]]:
    """One PyAV pass over `target_frames`; returns None if anything fails."""
    container = _open_pyav_video(path, use_hw=use_hw)
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base) if stream.time_base else (1.0 / fps)
        stream_start = (
            float(stream.start_time * tb) if stream.start_time is not None else 0.0
        )
        n_targets = len(target_frames)
        stride_sec = (
            (target_frames[1] - target_frames[0]) / fps
            if len(target_frames) > 1
            else (1.0 / fps)
        )
        candidates: List[tuple[int, float, np.ndarray, np.ndarray, float]] = []
        frame_iter = None
        last_sec: Optional[float] = None
        est_gop_sec: float = 0.0

        for count_i, f_target in enumerate(target_frames):
            if cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")

            target_sec = stream_start + (f_target / fps)
            target_pts = int(round(target_sec / tb))
            seek_gap = max(35.0 / fps, 0.8, est_gop_sec * 1.15)
            need_seek = (
                frame_iter is None
                or last_sec is None
                or (target_sec - last_sec) > seek_gap
                or target_sec < last_sec
            )
            if need_seek:
                container.seek(
                    target_pts, stream=stream, backward=True, any_frame=False
                )
                frame_iter = container.decode(stream)

            frame = None
            first_seek_t: Optional[float] = None
            for fr in frame_iter:
                if fr.time is None:
                    continue
                if need_seek and first_seek_t is None:
                    first_seek_t = float(fr.time)
                    est_gop_sec = min(12.0, max(est_gop_sec, target_sec - first_seek_t))
                if last_sec is not None and fr.time <= last_sec + (0.25 / fps):
                    continue
                frame = fr
                # A fresh keyframe seek lands on (or just before) an I-frame.
                # Accept it immediately when it is within one sampling step.
                if need_seek and abs(fr.time - target_sec) <= stride_sec * 0.65:
                    break
                if fr.time >= target_sec - (0.35 / fps):
                    break
            if frame is None:
                continue

            rgb = frame.reformat(
                width=out_w, height=out_h, format="rgb24", interpolation="AREA"
            ).to_ndarray()
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            actual_ts = float(frame.time) if frame.time is not None else target_sec
            actual_rel = max(0.0, actual_ts - stream_start)
            actual_idx = max(0, int(round(actual_rel * fps)))

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            sharp = _score_sharpness(gray)

            h_g, w_g = gray.shape
            sc_g = 160.0 / max(h_g, w_g)
            if sc_g < 1.0:
                nw_g, nh_g = max(16, int(w_g * sc_g)), max(16, int(h_g * sc_g))
                gray_thumb = cv2.resize(gray, (nw_g, nh_g))
            else:
                gray_thumb = gray

            candidates.append((actual_idx, actual_rel, rgb, gray_thumb, sharp))
            last_sec = actual_ts

            if progress_cb and count_i % 8 == 0:
                frac = min(0.99, (count_i + 1) / max(1, n_targets))
                progress_cb(
                    frac * 0.25,
                    f"scanning video (PyAV)... ({count_i + 1}/{n_targets} frames)",
                )

        min_required = 1 if n_targets == 1 else max(2, int(0.85 * n_targets))
        if len(candidates) < min_required:
            return None
        return candidates
    finally:
        try:
            container.close()
        except Exception:
            pass


def _decode_frames_pyav(
    path: Path | str,
    start_frame: int,
    end_frame: int,
    scan_stride: int,
    out_w: int,
    out_h: int,
    fps: float,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Optional[List[tuple[int, float, np.ndarray, np.ndarray, float]]]:
    """Decode sparse target frames with PyAV GOP-aware keyframe seeking.

    Hardware (NVDEC) is tried first, then multi-threaded CPU decode.  Returns
    ``None`` on any failure so the caller can fall back to OpenCV.
    """
    try:
        import av  # noqa: F401
    except Exception:
        return None

    scan_stride = max(1, int(scan_stride))
    target_frames = list(range(int(start_frame), int(end_frame), scan_stride))
    if len(target_frames) == 0:
        return None

    for use_hw in (True, False):
        try:
            result = _pyav_decode_targets(
                path,
                target_frames,
                out_w,
                out_h,
                fps,
                progress_cb,
                cancel_cb,
                use_hw,
            )
        except RuntimeError as e:
            if "cancelled by user" in str(e).lower():
                raise
            result = None
        except Exception:
            result = None
        if result is not None:
            return result
    return None


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    sharpness: float
    highres_rgb: Optional[np.ndarray] = None


@dataclass
class VideoData:
    path: Path
    fps: float
    total_frames: int
    duration: float
    orig_size: tuple[int, int]
    frames: List[VideoFrame]


def _score_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    # shrink down so flow doesn't take forever on long videos
    h, w = prev_gray.shape
    scale = 160.0 / max(h, w)
    if scale < 1.0:
        nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
        p_small = cv2.resize(prev_gray, (nw, nh))
        c_small = cv2.resize(curr_gray, (nw, nh))
    else:
        p_small, c_small = prev_gray, curr_gray

    flow = cv2.calcOpticalFlowFarneback(
        p_small, c_small, None, 0.5, 2, 13, 2, 5, 1.1, 0
    )
    mag = np.linalg.norm(flow, axis=-1)
    return float(np.median(mag))


def _probe_nvdec_keyframe_alignment(
    path: Path | str,
    target_frames: List[int],
    fps: float,
    max_packets: int = 90,
) -> bool:
    """Demux up to ``max_packets`` packets (zero decode) to check if targets land on keyframes."""
    import av

    if len(target_frames) < 2:
        return False
    container = None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        tb = float(stream.time_base) if stream.time_base else (1.0 / fps)
        stream_start = (
            float(stream.start_time * tb) if stream.start_time is not None else 0.0
        )
        kf_times: List[float] = []
        last_pkt_t = 0.0
        pkt_count = 0
        for pkt in container.demux(stream):
            if pkt.dts is None:
                continue
            pkt_t = float((pkt.pts if pkt.pts is not None else pkt.dts) * tb)
            if pkt_t > last_pkt_t:
                last_pkt_t = pkt_t
            if pkt.is_keyframe and pkt.pts is not None:
                kf_times.append(float(pkt.pts * tb))
            pkt_count += 1
            if pkt_count >= max_packets:
                break
        if len(kf_times) < 2:
            return False
        tol = 0.45 / max(1.0, float(fps))
        probed_targets = 0
        for f_t in target_frames:
            t_sec = stream_start + (f_t / fps)
            if t_sec > last_pkt_t + (0.1 / max(1.0, float(fps))):
                break
            probed_targets += 1
            if min(abs(t_sec - kt) for kt in kf_times) > tol:
                return False
        return probed_targets >= 2
    except Exception:
        return False
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _iter_grid_nvdec(
    path: Path | str,
    target_frames: List[int],
    fps: float,
    stats: dict,
    cancel_cb: Optional[Callable[[], bool]] = None,
):
    """Yield (index, full-res BGR) for exact target frames via CUDA NVDEC.

    Uses keyframe-only decode (``skip_frame="NONKEY"``) when a quick packet probe
    shows target frames align with keyframes, falling back seamlessly to full-frame
    NVDEC if any target misses mid-video.
    """
    use_nonkey = _probe_nvdec_keyframe_alignment(path, target_frames, fps)
    yielded_count = 0

    if use_nonkey:
        missed = False
        for item in _iter_grid_nvdec_pass(
            path, target_frames, fps, stats, cancel_cb, use_nonkey=True
        ):
            if item[0] == -1:
                missed = True
                break
            yielded_count += 1
            yield item
        if not missed and stats.get("completed", False):
            return
        stats["completed"] = False

    rem_targets = target_frames[yielded_count:]
    if not rem_targets:
        stats["completed"] = True
        return

    for item in _iter_grid_nvdec_pass(
        path, rem_targets, fps, stats, cancel_cb, use_nonkey=False
    ):
        if item[0] != -1:
            yield item


def _iter_grid_nvdec_pass(
    path: Path | str,
    target_frames: List[int],
    fps: float,
    stats: dict,
    cancel_cb: Optional[Callable[[], bool]] = None,
    *,
    use_nonkey: bool = False,
):
    """Single NVDEC decode pass over ``target_frames``."""
    import av

    container = None
    try:
        try:
            container = _open_pyav_video(path, use_hw=True)
            stream = container.streams.video[0]
            codec_fmt = stream.codec_context.format
            if codec_fmt is None or "cuda" not in codec_fmt.name:
                return
            if use_nonkey:
                stream.codec_context.skip_frame = "NONKEY"
        except Exception:
            return

        tb = float(stream.time_base) if stream.time_base else (1.0 / fps)
        stream_start = (
            float(stream.start_time * tb) if stream.start_time is not None else 0.0
        )
        frame_iter = None
        last_sec: Optional[float] = None
        gop_samples: List[float] = []

        for f_target in target_frames:
            if cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")

            target_sec = stream_start + (f_target / fps)
            target_pts = int(round(target_sec / tb))
            est_gop = float(np.median(gop_samples)) if gop_samples else 0.0
            seek_gap = max(15.0 / fps, 0.55, est_gop * 1.15)
            need_seek = (
                frame_iter is None
                or last_sec is None
                or (target_sec - last_sec) > seek_gap
                or target_sec < last_sec
            )

            t0 = time.perf_counter()
            if need_seek:
                container.seek(
                    target_pts, stream=stream, backward=True, any_frame=False
                )
                frame_iter = container.decode(stream)
                stats["seeks"] += 1

            got = None
            sampled_gop = False
            for fr in frame_iter:
                if fr.time is None:
                    continue
                ft = float(fr.time)
                if need_seek and not sampled_gop:
                    gop_samples.append(max(0.0, target_sec - ft))
                    gop_samples = gop_samples[-5:]
                    sampled_gop = True
                if (
                    not need_seek
                    and last_sec is not None
                    and ft <= last_sec + (0.25 / fps)
                ):
                    continue
                got = fr
                if ft >= target_sec - (0.35 / fps):
                    break

            if got is None:
                if use_nonkey:
                    yield (-1, None)
                return
            if use_nonkey and abs(float(got.time) - target_sec) > (0.65 / fps):
                yield (-1, None)
                return
            try:
                bgr = got.to_ndarray(format="bgr24")
            except Exception:
                return
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
            stats["decode_s"] += time.perf_counter() - t0
            stats["decoded_frames"] += 1

            actual_ts = float(got.time) if got.time is not None else target_sec
            last_sec = actual_ts
            pts_map = stats.get("pts_sec")
            if isinstance(pts_map, dict):
                pts_map[f_target] = max(0.0, actual_ts - stream_start)
            yield f_target, bgr

        stats["completed"] = True
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def extract_keyframes(
    video_path: str | Path,
    max_frames: int = 24,
    max_dim: int = 560,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    highres_max_dim: int = 0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> VideoData:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"cant find video at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv failed to open {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    start_frame = max(0, int(round(start_sec * fps)))
    if end_sec > start_sec and total_frames > 0:
        end_frame = min(total_frames, int(round(end_sec * fps)))
    elif end_sec > start_sec:
        end_frame = int(round(end_sec * fps))
    else:
        end_frame = total_frames

    has_known_window = end_frame > start_frame
    window_frames = (end_frame - start_frame) if has_known_window else (total_frames if total_frames > 0 else 100000)

    if max_frames <= 0:
        max_frames = window_frames if has_known_window else (total_frames if total_frames > 0 else 100000)

    # sample enough candidates within the active window so fast camera pans never have 300px gaps
    max_candidates = min(max(max_frames * 2, 220), max(max_frames, 400))
    if has_known_window and max_frames >= window_frames:
        scan_stride = 1
    elif has_known_window:
        max_candidates = min(window_frames, max_candidates)
        scan_stride = max(1, window_frames // max_candidates)
    else:
        max_candidates = max(max_frames * 3, 180)
        scan_stride = 2

    if highres_max_dim > 0:
        hr_s = min(1.0, float(highres_max_dim) / float(max(orig_w, orig_h, 1)))
        hr_px_bytes = max(64, int(round(orig_w * hr_s))) * max(64, int(round(orig_h * hr_s))) * 3
        want_highres = (int(max_candidates) * hr_px_bytes) <= 1_600_000_000
    else:
        want_highres = False
    cap.release()
    candidates: List[tuple[int, float, np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = []

    def _build_candidate(idx: int, bgr_img: np.ndarray, ts_override: Optional[float] = None):
        h, w = bgr_img.shape[:2]
        hr_rgb: Optional[np.ndarray] = None
        if want_highres:
            hr_scale = min(1.0, float(highres_max_dim) / float(max(h, w)))
            hr_w = max(2, int(round(w * hr_scale)))
            hr_h = max(2, int(round(h * hr_scale)))
            hr_w -= hr_w % 2
            hr_h -= hr_h % 2
            bgr_hr = (
                cv2.resize(bgr_img, (hr_w, hr_h), interpolation=cv2.INTER_AREA)
                if (hr_w, hr_h) != (w, h)
                else bgr_img
            )
            hr_rgb = np.ascontiguousarray(
                cv2.cvtColor(bgr_hr, cv2.COLOR_BGR2RGB), dtype=np.uint8
            )

        scale = min(1.0, float(max_dim) / max(h, w))
        nw = max(64, (int(round(w * scale)) // 8) * 8)
        nh = max(64, (int(round(h * scale)) // 8) * 8)
        if (nw, nh) != (w, h):
            bgr_img = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        sharp = _score_sharpness(gray)
        ts = float(ts_override) if ts_override is not None else (idx / fps)

        # store only <= 160px grayscale thumbnail in candidates to save RAM on long videos
        h_g, w_g = gray.shape
        sc_g = 160.0 / max(h_g, w_g)
        if sc_g < 1.0:
            nw_g, nh_g = max(16, int(w_g * sc_g)), max(16, int(h_g * sc_g))
            gray_thumb = cv2.resize(gray, (nw_g, nh_g))
        else:
            gray_thumb = gray

        return (idx, ts, rgb, gray_thumb, sharp, hr_rgb)

    def _process_frame(idx: int, bgr_img: np.ndarray, ts_override: Optional[float] = None):
        nonlocal scan_stride, candidates
        candidates.append(_build_candidate(idx, bgr_img, ts_override=ts_override))
        if (not has_known_window) and len(candidates) >= max_candidates * 2:
            candidates = candidates[::2]
            scan_stride *= 2

    nvdec_ok = False
    LAST_STAGE1_TELEMETRY.clear()
    LAST_STAGE1_TELEMETRY.update(
        {"backend": "OpenCV", "seeks": 0, "decoded_frames": 0, "decode_s": 0.0}
    )

    # Decode the exact uniform grid with NVDEC at 4K, then downscale with
    # cv2.INTER_AREA (identical to the OpenCV path). Fall back to OpenCV when
    # hardware decode is unavailable or the stream ends early.
    if has_known_window and scan_stride >= 8 and max(orig_w, orig_h) >= 480:
        nvdec_targets = list(range(start_frame, end_frame, scan_stride))
        n_before = len(candidates)
        nv_stats = {
            "backend": "NVDEC",
            "seeks": 0,
            "decoded_frames": 0,
            "decode_s": 0.0,
            "completed": False,
            "pts_sec": {},
        }
        try:
            pending = deque()
            with ThreadPoolExecutor(max_workers=1) as proc_pool:
                for nv_i, (f_idx, bgr) in enumerate(
                    _iter_grid_nvdec(path, nvdec_targets, fps, nv_stats, cancel_cb)
                ):
                    ts_pts = nv_stats["pts_sec"].get(f_idx)
                    pending.append(proc_pool.submit(_build_candidate, f_idx, bgr, ts_pts))
                    if len(pending) >= 3:
                        candidates.append(pending.popleft().result())
                    if progress_cb and nv_i % 8 == 0:
                        frac = min(0.99, (nv_i + 1) / max(1, len(nvdec_targets)))
                        progress_cb(
                            frac * 0.25,
                            f"decoding NVDEC... ({nv_i + 1}/{len(nvdec_targets)})",
                        )
                while pending:
                    candidates.append(pending.popleft().result())
            nvdec_ok = nv_stats["completed"] and (
                (len(candidates) - n_before) == len(nvdec_targets)
            )
        except Exception:
            if cancel_cb and cancel_cb():
                raise RuntimeError("build cancelled by user")
            nvdec_ok = False

        if nvdec_ok:
            frame_idx = nvdec_targets[-1] + 1
            LAST_STAGE1_TELEMETRY.update(
                {
                    "backend": "NVDEC",
                    "seeks": nv_stats["seeks"],
                    "decoded_frames": nv_stats["decoded_frames"],
                    "decode_s": nv_stats["decode_s"],
                }
            )
        else:
            del candidates[n_before:]

    # OpenCV fallback keeps the exact uniform grid its motion scoring was tuned on.
    if not nvdec_ok:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"opencv failed to open {path}")
        if scan_stride >= 180 and has_known_window:
            target_frames = list(range(start_frame, end_frame, scan_stride))
            for step_i, f_target in enumerate(target_frames):
                if cancel_cb and cancel_cb():
                    cap.release()
                    raise RuntimeError("build cancelled by user")
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_target)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                _process_frame(f_target, bgr)
                if progress_cb and len(candidates) % 8 == 0:
                    frac = min(0.99, (f_target - start_frame + 1) / max(1, window_frames))
                    progress_cb(frac * 0.25, f"scanning video... ({f_target}/{end_frame} frames)")
            frame_idx = target_frames[-1] + 1 if target_frames else start_frame
        else:
            frame_idx = start_frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while True:
                if cancel_cb and cancel_cb():
                    cap.release()
                    raise RuntimeError("build cancelled by user")

                if end_frame > 0 and frame_idx >= end_frame:
                    break

                if scan_stride > 1 and (frame_idx - start_frame) % scan_stride != 0:
                    ok = cap.grab()
                    if not ok:
                        break
                    frame_idx += 1
                    continue

                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break

                _process_frame(frame_idx, bgr)
                frame_idx += 1

                if progress_cb and window_frames > 0 and len(candidates) % 8 == 0:
                    frac = min(0.99, (frame_idx - start_frame) / max(1, window_frames))
                    progress_cb(frac * 0.25, f"scanning video... ({frame_idx}/{end_frame} frames)")

        cap.release()

    if not candidates:
        raise RuntimeError("no readable frames in video")

    actual_total = max(total_frames, frame_idx) if total_frames > 0 else frame_idx
    duration = (
        (window_frames / fps)
        if (has_known_window and (start_sec > 0.0 or end_sec > start_sec))
        else (actual_total / fps)
    )

    med_sharp = float(np.median([c[4] for c in candidates])) if candidates else 1.0

    if len(candidates) <= max_frames:
        kept_cands = candidates
        if len(candidates) > 16 and med_sharp > 1e-6:
            blur_cut = 0.15 * med_sharp
            sharp_cands = [c for c in candidates if c[4] >= blur_cut]
            if len(sharp_cands) >= max(8, int(round(len(candidates) * 0.88))):
                kept_cands = sharp_cands
        if progress_cb:
            progress_cb(0.28, f"selected {len(kept_cands)} keyframes")
        selected = [
            VideoFrame(index=c[0], timestamp=c[1], rgb=c[2], sharpness=c[4], highres_rgb=c[5])
            for c in kept_cands
        ]
        candidates.clear()
        return VideoData(
            path=path,
            fps=fps,
            total_frames=actual_total,
            duration=duration,
            orig_size=(orig_w, orig_h),
            frames=selected,
        )

    # space keyframes evenly along cumulative camera motion arc-length so pans keep high overlap
    step_motions = [0.0]
    for k in range(1, len(candidates)):
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")
        mot = _motion_score(candidates[k - 1][3], candidates[k][3])
        # blend actual optical flow with a tiny time floor so static clips still progress
        step_motions.append(float(np.clip(mot, 0.15, 18.0)) + 0.25)

    cum_mot = np.cumsum(step_motions)
    target_arcs = np.linspace(0.0, cum_mot[-1], max_frames)

    chosen_indices: List[int] = []
    used: set[int] = set()
    n_cand = len(candidates)
    blur_floor = 0.38 * med_sharp

    for t_idx, target in enumerate(target_arcs):
        if cancel_cb and cancel_cb():
            raise RuntimeError("build cancelled by user")
        center_i = int(np.argmin(np.abs(cum_mot - target)))
        best_i = center_i
        best_sharp = -1.0
        arc_step = cum_mot[-1] / max_frames
        for offset in (0, -1, 1, -2, 2):
            cand_i = center_i + offset
            if 0 <= cand_i < n_cand and cand_i not in used:
                if abs(cum_mot[cand_i] - target) <= arc_step * 0.65 or best_sharp < 0:
                    if candidates[cand_i][4] > best_sharp:
                        best_sharp = candidates[cand_i][4]
                        best_i = cand_i

        # Widen search slightly when the local neighborhood hits a motion-blur burst
        if best_sharp < blur_floor:
            for offset in (-3, 3):
                cand_i = center_i + offset
                if 0 <= cand_i < n_cand and cand_i not in used:
                    if abs(cum_mot[cand_i] - target) <= arc_step * 0.85 and candidates[cand_i][4] > best_sharp:
                        best_sharp = candidates[cand_i][4]
                        best_i = cand_i

        if best_i in used:
            # fallback to nearest unused candidate to center_i
            unused = [c for c in range(n_cand) if c not in used]
            best_i = min(unused, key=lambda c: abs(c - center_i))

        used.add(best_i)
        chosen_indices.append(best_i)

        if progress_cb and (t_idx % 4 == 0 or t_idx == max_frames - 1):
            progress_cb(
                0.25 + 0.03 * ((t_idx + 1) / max_frames),
                f"picked motion-spaced keyframe {t_idx + 1}/{max_frames}",
            )

    chosen_indices.sort()
    selected = [
        VideoFrame(
            index=candidates[i][0],
            timestamp=candidates[i][1],
            rgb=candidates[i][2],
            sharpness=candidates[i][4],
            highres_rgb=candidates[i][5],
        )
        for i in chosen_indices
    ]
    candidates.clear()

    if progress_cb:
        progress_cb(0.28, f"selected {len(selected)} motion-spaced keyframes")

    return VideoData(
        path=path,
        fps=fps,
        total_frames=actual_total,
        duration=duration,
        orig_size=(orig_w, orig_h),
        frames=selected,
    )


def generate_demo_video(
    output_path: str | Path,
    duration_sec: float = 5.0,
    fps: int = 24,
    width: int = 560,
    height: int = 360,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    pts_list = []
    col_list = []

    def add_surface(p0, u_vec, v_vec, nu, nv, base_rgb, checker=True):
        u = np.linspace(0.0, 1.0, nu)[:, None]
        v = np.linspace(0.0, 1.0, nv)[None, :]
        pts = (
            np.asarray(p0)[None, None, :]
            + u[:, :, None] * np.asarray(u_vec)[None, None, :]
            + v[:, :, None] * np.asarray(v_vec)[None, None, :]
        ).reshape(-1, 3)

        cols = np.broadcast_to(np.asarray(base_rgb, dtype=np.float32), pts.shape).copy()
        if checker:
            cu = (u * 8).astype(int)
            cv_grid = (v * 8).astype(int)
            mask = ((cu + cv_grid) % 2 == 0).reshape(-1)
            cols[mask] *= 0.72
        noise = rng.uniform(-18, 18, size=cols.shape)
        cols = np.clip(cols + noise, 15, 245).astype(np.uint8)
        pts_list.append(pts)
        col_list.append(cols)

    add_surface([-4.0, 1.2, 2.0], [8.0, 0.0, 0.0], [0.0, 0.0, 8.0], 130, 130, [150, 145, 135])
    add_surface([-4.0, -2.2, 9.5], [8.0, 0.0, 0.0], [0.0, 3.4, 0.0], 130, 70, [195, 115, 85])
    add_surface([-4.0, -2.2, 2.5], [0.0, 0.0, 7.0], [0.0, 3.4, 0.0], 110, 65, [95, 150, 185])
    add_surface([4.0, -2.2, 2.5], [0.0, 0.0, 7.0], [0.0, 3.4, 0.0], 110, 65, [120, 175, 125])

    add_surface([-1.6, -1.0, 4.5], [0.9, 0.0, 0.0], [0.0, 2.2, 0.0], 45, 60, [220, 185, 75])
    add_surface([-0.7, -1.0, 4.5], [0.0, 0.0, 0.9], [0.0, 2.2, 0.0], 45, 60, [190, 155, 60])
    add_surface([0.8, -0.4, 5.8], [1.3, 0.0, 0.0], [0.0, 1.6, 0.0], 55, 50, [185, 95, 195])
    add_surface([0.8, -0.4, 5.8], [0.0, 0.0, 1.2], [0.0, 1.6, 0.0], 50, 50, [155, 75, 165])

    world_pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    world_cols = np.concatenate(col_list, axis=0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out}")

    total_frames = max(12, int(round(duration_sec * fps)))
    fx = fy = 0.92 * max(width, height)
    cx, cy = width * 0.5, height * 0.5

    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)
        cam_pos = np.array(
            [
                -0.9 + 1.8 * t,
                -0.15 + 0.12 * np.sin(t * np.pi * 2.0),
                1.6 + 1.4 * t,
            ],
            dtype=np.float32,
        )
        yaw = -0.18 + 0.36 * t
        pitch = 0.05

        cyaw, syaw = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        r_yaw = np.array([[cyaw, 0.0, syaw], [0.0, 1.0, 0.0], [-syaw, 0.0, cyaw]], dtype=np.float32)
        r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        r_wc = r_pitch @ r_yaw

        rel = (world_pts - cam_pos[None, :]) @ r_wc.T
        valid = rel[:, 2] > 0.4
        rel = rel[valid]
        cols = world_cols[valid]

        order = np.argsort(rel[:, 2])[::-1]
        rel = rel[order]
        cols = cols[order]

        u = np.round(fx * (rel[:, 0] / rel[:, 2]) + cx).astype(np.int32)
        v = np.round(fy * (rel[:, 1] / rel[:, 2]) + cy).astype(np.int32)
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, cols = u[in_bounds], v[in_bounds], cols[in_bounds]

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        grad = np.linspace(35, 75, height, dtype=np.uint8)[:, None]
        frame[:, :, 0] = grad
        frame[:, :, 1] = grad + 10
        frame[:, :, 2] = grad + 20

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                uu = np.clip(u + dx, 0, width - 1)
                vv = np.clip(v + dy, 0, height - 1)
                frame[vv, uu] = cols[:, ::-1]

        writer.write(frame)

    writer.release()
    return out
