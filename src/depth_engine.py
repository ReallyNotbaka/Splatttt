from concurrent.futures import ThreadPoolExecutor
import os
from typing import Callable, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
from PIL import Image
import torch
from src.video_ingest import VideoFrame

DEPTH_MODELS_INDOOR = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}
DEPTH_MODELS_OUTDOOR = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}
DEPTH_MODELS = DEPTH_MODELS_INDOOR


def _import_da3_class():
    import sys
    import types

    if "depth_anything_3.utils.export" not in sys.modules:
        _exp = types.ModuleType("depth_anything_3.utils.export")
        _exp.export = lambda *a, **kw: None
        sys.modules["depth_anything_3.utils.export"] = _exp

    if "depth_anything_3.utils.pose_align" not in sys.modules:
        _pa = types.ModuleType("depth_anything_3.utils.pose_align")
        _pa.align_poses_umeyama = lambda pred_ext, gt_ext, *a, **kw: (
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            pred_ext,
        )
        sys.modules["depth_anything_3.utils.pose_align"] = _pa

    if "pillow_heif" not in sys.modules:
        _ph = types.ModuleType("pillow_heif")
        _ph.register_heif_opener = lambda *a, **kw: None
        sys.modules["pillow_heif"] = _ph

    from depth_anything_3.api import DepthAnything3
    return DepthAnything3


class DepthEngine:
    def __init__(self, device: Optional[str] = None, model_size: str = "large"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._is_cuda = torch.device(self.device).type == "cuda"
        self.dtype = torch.float16 if self._is_cuda else torch.float32
        self.model_size = model_size if model_size in DEPTH_MODELS else "large"
        self._loaded_key: Optional[tuple[str, str]] = None
        self._processor = None
        self._model = None
        self._da3_model = None
        self.last_used_da3: bool = False

    def _ensure_da3_loaded(self) -> bool:
        if self._da3_model is not None:
            if next(self._da3_model.parameters()).device.type != torch.device(self.device).type:
                self._da3_model.to(self.device)
            return True

        if self._model is not None:
            del self._model
            self._model = None
            if self._is_cuda:
                torch.cuda.empty_cache()

        try:
            DepthAnything3 = _import_da3_class()
            self._da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(self.device).eval()
            return True
        except Exception as e:
            print(f"[WARNING] DA3-BASE load failed ({e}); falling back to DA2")
            return False

    def _ensure_loaded(self, model_size: Optional[str] = None, scene_type: str = "indoor"):
        stype = "outdoor" if str(scene_type).strip().lower() == "outdoor" else "indoor"
        target_dict = DEPTH_MODELS_OUTDOOR if stype == "outdoor" else DEPTH_MODELS_INDOOR
        target_size = model_size if (model_size and model_size in target_dict) else self.model_size
        key = (stype, target_size)

        if self._da3_model is not None:
            del self._da3_model
            self._da3_model = None
            if self._is_cuda:
                torch.cuda.empty_cache()

        if self._model is not None and self._loaded_key == key:
            if next(self._model.parameters()).device.type != torch.device(self.device).type:
                self._model.to(self.device)
            return

        if self._model is not None:
            del self._model
            self._model = None
            if self._is_cuda:
                torch.cuda.empty_cache()

        fallback_order = [target_size]
        if target_size == "large":
            fallback_order.extend(["base", "small"])
        elif target_size == "base":
            fallback_order.append("small")

        last_err = None
        for cand in fallback_order:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()

                model_id = target_dict[cand]
                self._processor = AutoImageProcessor.from_pretrained(model_id)
                try:
                    self._model = AutoModelForDepthEstimation.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                    ).to(self.device, dtype=self.dtype)
                except TypeError:
                    self._model = AutoModelForDepthEstimation.from_pretrained(
                        model_id,
                    ).to(self.device, dtype=self.dtype)
                self._model.eval()
                self._loaded_key = (stype, cand)
                self.model_size = cand
                return
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err

    def offload_to_cpu(self):
        if self._model is not None and self._is_cuda:
            self._model.to("cpu")
        if self._da3_model is not None and self._is_cuda:
            self._da3_model.to("cpu")
        if self._is_cuda:
            torch.cuda.empty_cache()

    def estimate_depths(
        self,
        frames: List[VideoFrame],
        batch_size: Optional[int] = None,
        edge_thresh: float = 0.12,
        model_size: Optional[str] = None,
        scene_type: str = "indoor",
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
        progress_range: tuple[float, float] = (0.56, 0.68),
        poses: Optional[List] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        if not frames:
            return [], []

        h, w = frames[0].rgb.shape[:2]
        is_outdoor = (str(scene_type).strip().lower() == "outdoor")
        target_size = model_size if (model_size and model_size in DEPTH_MODELS) else self.model_size
        use_da3 = (target_size != "small") and (poses is not None and intrinsics is not None and len(poses) == len(frames) and len(frames) >= 2)
        used_da3 = False

        if use_da3 and self._ensure_da3_loaded():
            used_da3 = True
            n = len(frames)
            target_med_z = 4.5 if is_outdoor else 2.6
            da3_res = 644 if target_size == "large" else 504
            win_size = min(6 if target_size == "large" else 8, n)
            win_stride = max(1, win_size - 2)

            if n <= win_size:
                win_starts = [0]
            else:
                win_starts = list(range(0, n - win_size, win_stride))
                if not win_starts or (win_starts[-1] + win_size) < n:
                    win_starts.append(n - win_size)

            depth_acc = [np.zeros((h, w), dtype=np.float32) for _ in range(n)]
            w_acc = [np.zeros((h, w), dtype=np.float32) for _ in range(n)]
            conf_acc = [np.zeros((h, w), dtype=np.float32) for _ in range(n)]
            cnt_acc = [0] * n

            for w_i, start in enumerate(win_starts):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("build cancelled by user")
                w_idxs = list(range(start, start + win_size))
                imgs = [Image.fromarray(frames[k].rgb) for k in w_idxs]

                exts = np.zeros((len(w_idxs), 4, 4), dtype=np.float32)
                for m_i, k in enumerate(w_idxs):
                    r_cw = poses[k].r_wc.T
                    t_cw = -r_cw @ poses[k].t_wc
                    exts[m_i, :3, :3] = r_cw
                    exts[m_i, :3, 3] = t_cw
                    exts[m_i, 3, 3] = 1.0
                ixts = np.repeat(intrinsics.astype(np.float32)[None, :, :], len(w_idxs), axis=0)

                with torch.inference_mode():
                    pred = self._da3_model.inference(
                        image=imgs,
                        extrinsics=exts,
                        intrinsics=ixts,
                        align_to_input_ext_scale=False,
                        process_res=da3_res,
                    )

                d_win = np.asarray(pred.depth, dtype=np.float32)
                c_win = np.asarray(pred.conf, dtype=np.float32) if pred.conf is not None else np.ones_like(d_win)
                s_win = None
                if getattr(pred, "extrinsics", None) is not None and len(w_idxs) >= 2:
                    c_in = np.stack([poses[k].t_wc.astype(np.float64) for k in w_idxs], axis=0)
                    c_pr = np.zeros_like(c_in)
                    for m_i in range(len(w_idxs)):
                        r_p = pred.extrinsics[m_i, :3, :3].astype(np.float64)
                        t_p = pred.extrinsics[m_i, :3, 3].astype(np.float64)
                        c_pr[m_i] = -r_p.T @ t_p
                    ratios = []
                    for a in range(len(w_idxs)):
                        for b in range(a + 1, len(w_idxs)):
                            d_pr = float(np.linalg.norm(c_pr[a] - c_pr[b]))
                            d_in = float(np.linalg.norm(c_in[a] - c_in[b]))
                            if d_pr > 1e-4 and d_in > 1e-4:
                                ratios.append(d_in / d_pr)
                    if ratios:
                        cand_s = float(np.median(ratios))
                        med_d = float(np.median(d_win[d_win > 1e-4])) if np.any(d_win > 1e-4) else 1.0
                        if 0.05 < (cand_s * med_d) < 20.0:
                            s_win = cand_s
                if s_win is None:
                    med_d = float(np.median(d_win[d_win > 1e-4])) if np.any(d_win > 1e-4) else 1.0
                    s_win = 1.0 / max(med_d, 1e-3)
                d_win_scaled = d_win * float(s_win * target_med_z)

                for m_i, k in enumerate(w_idxs):
                    dk = cv2.resize(d_win_scaled[m_i], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    ck = cv2.resize(c_win[m_i], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    wk = np.clip(ck, 0.1, 20.0)
                    depth_acc[k] += dk * wk
                    w_acc[k] += wk
                    conf_acc[k] += ck
                    cnt_acc[k] += 1

                if progress_cb:
                    p0, p1 = progress_range
                    progress_cb(
                        p0 + (p1 - p0) * ((w_i + 1) / len(win_starts)),
                        f"DA3-BASE ({da3_res}px) multi-view depth... (window {w_i + 1}/{len(win_starts)})",
                    )

            if self._is_cuda:
                torch.cuda.empty_cache()

            raw_metric_depths = [depth_acc[k] / np.maximum(w_acc[k], 1e-6) for k in range(n)]
            conf_maps = [conf_acc[k] / max(1, cnt_acc[k]) for k in range(n)]
        else:
            self._ensure_loaded(model_size=target_size, scene_type=scene_type)
            # saturate the 6gb RTX 4050 cleanly (~1.2GB peak at batch=8 for ViT-Base)
            if batch_size is None:
                if self._is_cuda:
                    if self.model_size == "large":
                        batch_size = 4
                    elif self.model_size == "base":
                        batch_size = 8
                    else:
                        batch_size = 16
                else:
                    batch_size = 2

            raw_metric_depths = []
            for start in range(0, len(frames), batch_size):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("build cancelled by user")

                chunk = frames[start : start + batch_size]
                pil_imgs = [Image.fromarray(f.rgb) for f in chunk]
                inputs = self._processor(images=pil_imgs, return_tensors="pt")
                inputs = {
                    k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                    for k, v in inputs.items()
                }

                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    pred = outputs.predicted_depth

                pred_resized = torch.nn.functional.interpolate(
                    pred.unsqueeze(1).float(),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                for arr in pred_resized.cpu().numpy():
                    raw_metric_depths.append(arr.astype(np.float32))

                if progress_cb:
                    done = min(len(frames), start + batch_size)
                    p0, p1 = progress_range
                    progress_cb(
                        p0 + (p1 - p0) * (done / len(frames)),
                        f"GPU depth ({self.model_size}, batch={batch_size})... ({done}/{len(frames)})",
                    )
            conf_maps = [None] * len(frames)

        self.last_used_da3 = used_da3

        k3 = np.ones((3, 3), dtype=np.uint8)
        bh = max(3, int(round(h * 0.015)))
        bw = max(3, int(round(w * 0.015)))
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[bh : h - bh, bw : w - bw] = True
        f_approx = 0.92 * max(w, h)
        y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

        has_pose_rays = (poses is not None and intrinsics is not None and len(poses) == len(frames))
        if has_pose_rays:
            uu_d, vv_d = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            fx_d, fy_d = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx_d, cy_d = float(intrinsics[0, 2]), float(intrinsics[1, 2])
            rx_d = (uu_d - cx_d) / max(fx_d, 1e-3)
            ry_d = (vv_d - cy_d) / max(fy_d, 1e-3)
            inv_rn = 1.0 / np.sqrt(rx_d * rx_d + ry_d * ry_d + 1.0)
            rx_u = rx_d * inv_rn
            ry_u = ry_d * inv_rn
            rz_u = inv_rn
            pose_list = poses
        else:
            rx_u = ry_u = rz_u = None
            pose_list = [None] * len(frames)

        def _postproc_one(args):
            frame, d_raw, conf_map, pose_k = args
            if is_outdoor:
                raw_depth = np.clip(d_raw, 0.35, 65.0).astype(np.float32)
            else:
                raw_depth = np.clip(d_raw, 0.25, 16.0).astype(np.float32)

            d_max = cv2.dilate(raw_depth, k3)
            d_min = cv2.erode(raw_depth, k3)
            ramp = np.minimum(d_max - raw_depth, raw_depth - d_min) / np.maximum(raw_depth, 0.4)

            gx = cv2.Sobel(raw_depth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(raw_depth, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.hypot(gx, gy) / np.maximum(raw_depth, 0.4)

            dz_dx = np.gradient(raw_depth, axis=1)
            dz_dy = np.gradient(raw_depth, axis=0)
            nx = -dz_dx * f_approx
            ny = -dz_dy * f_approx
            nz = -raw_depth
            n_mag = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
            cos_view = np.abs(nz) / n_mag

            gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)

            if is_outdoor:
                cos_ok = np.where(raw_depth < 8.0, cos_view >= 0.08, cos_view >= 0.16)
                igx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                igy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                img_grad = np.hypot(igx, igy)

                cliff = (ramp >= (edge_thresh * 0.55)) | (grad >= (edge_thresh * 3.0))
                dilated_cliff = cv2.dilate(cliff.astype(np.uint8), k3) > 0
                is_fg_plateau = (raw_depth - d_min) < (0.065 * np.maximum(raw_depth, 0.4))
                cliff_mask = cliff | (dilated_cliff & (~is_fg_plateau) & (raw_depth >= 8.0))

                if pose_k is not None:
                    # COLMAP world +Y is down, so world up component of unit ray is -(r_wc[1, :] . ray_c)
                    r1 = pose_k.r_wc[1].astype(np.float32)
                    ray_w_up = -(r1[0] * rx_u + r1[1] * ry_u + r1[2] * rz_u)
                    above_horizon = ray_w_up > 0.02
                else:
                    above_horizon = y_norm < 0.55

                r_ch = frame.rgb[:, :, 0].astype(np.int16)
                b_ch = frame.rgb[:, :, 2].astype(np.int16)
                sky_tint = b_ch >= (r_ch - 10)
                sky_blue = above_horizon & (b_ch > (r_ch + 20)) & (b_ch > 175) & (gray > 140) & (raw_depth > 6.0)

                sky_bright_white = above_horizon & sky_tint & (gray > 205) & (raw_depth > 4.5)
                sky_flat = above_horizon & sky_tint & (raw_depth > 9.0) & (gray > 165) & (img_grad < 16.0) & (grad < 0.03)
                sky_or_far = (
                    (raw_depth >= 55.0)
                    | ((raw_depth > 22.0) & above_horizon & sky_tint & (gray > 150))
                    | sky_flat
                    | sky_bright_white
                    | sky_blue
                )
                if conf_map is not None:
                    sky_lowconf = above_horizon & sky_tint & (raw_depth > 6.5) & (gray > 145) & (conf_map < 2.3)
                    sky_or_far = sky_or_far | sky_lowconf
                    conf_floor = min(1.15, float(np.percentile(conf_map, 5.0)))
                    conf_ok = conf_map >= conf_floor
                else:
                    conf_ok = np.ones((h, w), dtype=bool)

                k5 = np.ones((5, 5), dtype=np.uint8)
                sky_dil = cv2.dilate(sky_or_far.astype(np.uint8), k5) > 0
                sky_halo = sky_dil & ((~is_fg_plateau) | (above_horizon & sky_tint & (gray > 175)))
                sky_mask = sky_or_far | sky_halo

                valid = (
                    border_mask
                    & (~sky_mask)
                    & conf_ok
                    & cos_ok
                    & (~cliff_mask)
                    & (ramp < (edge_thresh * 0.60))
                    & (grad < (edge_thresh * 3.4))
                    & (raw_depth > 0.35)
                    & (raw_depth < 55.0)
                )
                if used_da3:
                    return raw_depth, valid
                smooth_depth = cv2.bilateralFilter(raw_depth, d=5, sigmaColor=0.12, sigmaSpace=2.5)
            else:
                cliff = (ramp >= (edge_thresh * 0.32)) | (grad >= (edge_thresh * 2.0))
                dilated_cliff = cv2.dilate(cliff.astype(np.uint8), k3) > 0
                is_fg_plateau = (raw_depth - d_min) < (0.032 * np.maximum(raw_depth, 0.4))
                cliff_mask = cliff | (dilated_cliff & (~is_fg_plateau))

                dark_void = (gray < 12) & (raw_depth > 4.2)
                valid = (
                    border_mask
                    & (~dark_void)
                    & (cos_view >= 0.12)
                    & (~cliff_mask)
                    & (ramp < (edge_thresh * 0.38))
                    & (grad < (edge_thresh * 2.4))
                    & (raw_depth > 0.28)
                    & (raw_depth < 14.0)
                )
                if conf_map is not None:
                    conf_floor = min(1.20, float(np.percentile(conf_map, 5.0)))
                    conf_ok = conf_map >= conf_floor
                    valid = valid & conf_ok
                if used_da3:
                    return raw_depth, valid
                smooth_depth = cv2.bilateralFilter(raw_depth, d=5, sigmaColor=0.04, sigmaSpace=2.5)
            return smooth_depth, valid

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            results = list(pool.map(_postproc_one, zip(frames, raw_metric_depths, conf_maps, pose_list)))

        depths = [r[0] for r in results]
        masks = [r[1] for r in results]
        return depths, masks
