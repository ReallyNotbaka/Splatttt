from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

import cv2
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from src.pipeline import (
        ReconstructionConfig,
        ReconstructionResult,
        VideoTo3DPipeline,
    )
from src.presets import DEFAULT_PRESET_NAME, Preset, PresetManager

from src.m3_ui import (
    M3_PALETTES,
    THEME_ALIASES,
    M3Button,
    M3ComboBox,
    M3ConnectedGroup,
    M3ExpressiveSlider,
    M3LoadingIndicator,
    M3NavRailItem,
    M3NumericControl,
    M3SwatchButton,
    M3Switch,
    M3VideoPreview,
    M3WavyProgressBar,
    build_m3_stylesheet,
    make_m3_icon,
    resolve_palette,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SCENE_EXTS = {".ply", ".glb", ".gltf", ".obj", ".npz"}

_GPU_INFO_CACHE: Optional[tuple[str, str]] = None


def _detect_gpu_info() -> tuple[str, str]:
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE

    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        try:
            if torch_mod.cuda.is_available():
                full_gpu = torch_mod.cuda.get_device_name(0)
                short_gpu = (
                    full_gpu.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "").strip()
                )
                _GPU_INFO_CACHE = (short_gpu, full_gpu)
                return _GPU_INFO_CACHE
        except Exception:
            pass

    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1.5,
            creationflags=flags,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if lines:
                full_gpu = lines[0]
                short_gpu = (
                    full_gpu.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "").strip()
                )
                _GPU_INFO_CACHE = (short_gpu, full_gpu)
                return _GPU_INFO_CACHE
    except Exception:
        pass

    _GPU_INFO_CACHE = ("CPU Mode", "CPU Mode (No CUDA GPU detected)")
    return _GPU_INFO_CACHE


def configure_hidpi():
    """Ensure Windows 11 renders native dialogs and fonts at 1:1 physical pixels."""
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    if sys.platform == "win32":
        try:
            import ctypes
            # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
    if QApplication.instance() is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )


def _wrap_scroll(content_widget: QWidget) -> QScrollArea:
    """Wrap a vertical page widget in a borderless, smooth-scrolling container."""
    if content_widget.layout() is not None:
        content_widget.layout().setContentsMargins(0, 0, 12, 8)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content_widget)
    return scroll


_COLMAP_WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def _safe_is_gsplat_available() -> bool:
    try:
        from src.gsplat_engine import is_gsplat_available

        return bool(is_gsplat_available())
    except Exception:
        return False


class SplatOrbitCamera:
    """Right-side-up orbit/pan/zoom/fly camera in COLMAP world coords (-Y is up)."""

    def __init__(self):
        self.target = np.zeros(3, dtype=np.float32)
        self.distance: float = 2.5
        self.yaw: float = 0.0
        self.pitch: float = 0.25
        self.fov_deg: float = 50.0
        self._floor_y: Optional[float] = None
        self._floor_margin: float = 0.05
        self._scene_extent: float = 1.0

    def get_c2w(self) -> np.ndarray:
        cp = float(np.cos(self.pitch))
        sp = float(np.sin(self.pitch))
        cy = float(np.cos(self.yaw))
        sy = float(np.sin(self.yaw))

        offset_dir = np.array([cp * sy, -sp, cp * cy], dtype=np.float32)
        eye = self.target + max(0.05, float(self.distance)) * offset_dir

        forward = self.target - eye
        f_norm = float(np.linalg.norm(forward))
        if f_norm < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            forward = forward / f_norm

        right = np.cross(forward, _COLMAP_WORLD_UP)
        r_norm = float(np.linalg.norm(right))
        if r_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / r_norm

        down = np.cross(forward, right)
        d_norm = float(np.linalg.norm(down))
        if d_norm > 1e-6:
            down = down / d_norm

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        return c2w

    def get_intrinsics(self, width: int, height: int) -> np.ndarray:
        w = max(1, int(width))
        h = max(1, int(height))
        fy = 0.5 * float(h) / max(1e-4, float(np.tan(np.deg2rad(self.fov_deg) * 0.5)))
        return np.array(
            [[fy, 0.0, w * 0.5], [0.0, fy, h * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _set_eye_target(self, eye: np.ndarray, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.float32).copy()
        offset = np.asarray(eye, dtype=np.float32) - self.target
        dist = float(np.linalg.norm(offset))
        self.distance = max(0.08, dist)
        if dist > 1e-6:
            u = offset / dist
            self.pitch = float(np.arcsin(np.clip(-u[1], -0.995, 0.995)))
            self.yaw = float(np.arctan2(u[0], u[2]))
        else:
            self.pitch = 0.25
            self.yaw = 0.0

    def set_from_pose(self, pose: Any, look_dist: float = 2.5) -> None:
        eye = np.asarray(pose.t_wc, dtype=np.float32)
        fwd = np.asarray(pose.r_wc, dtype=np.float32) @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        target = eye + fwd * float(look_dist)
        self._set_eye_target(eye, target)

    def set_overview(
        self,
        means_xyz: Optional[np.ndarray],
        poses: Sequence[Any],
        intrinsics: Optional[np.ndarray] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        if intrinsics is not None and frame_size is not None and frame_size[1] > 0:
            fy = float(intrinsics[1, 1])
            if fy > 1.0:
                fov = float(np.rad2deg(2.0 * np.arctan(0.5 * float(frame_size[1]) / fy)))
                self.fov_deg = float(np.clip(fov, 28.0, 85.0))

        valid_poses = [p for p in poses if getattr(p, "valid", True)]
        if not valid_poses:
            valid_poses = list(poses)

        cam_centers = (
            np.stack([np.asarray(p.t_wc, dtype=np.float32) for p in valid_poses], axis=0)
            if len(valid_poses) > 0
            else np.zeros((0, 3), dtype=np.float32)
        )
        cam_center = (
            np.median(cam_centers, axis=0)
            if len(cam_centers) > 0
            else np.zeros(3, dtype=np.float32)
        )
        cam_rad_med = (
            max(0.25, float(np.median(np.linalg.norm(cam_centers - cam_center[None, :], axis=1))))
            if len(cam_centers) > 0
            else 1.0
        )

        ang_cov = 0.0
        if len(cam_centers) >= 8:
            rel_xz = (cam_centers - cam_center[None, :])[:, [0, 2]]
            angs = np.sort(np.arctan2(rel_xz[:, 1], rel_xz[:, 0]))
            gaps = np.diff(angs)
            wrap_gap = float((2.0 * np.pi) - angs[-1] + angs[0])
            max_gap = float(max(float(np.max(gaps)) if len(gaps) > 0 else 2.0 * np.pi, wrap_gap))
            ang_cov = float(2.0 * np.pi - max_gap)

        pts = None
        if means_xyz is not None and len(means_xyz) > 0:
            pts = np.asarray(means_xyz, dtype=np.float32)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(cam_centers) > 0 and len(pts) > 0:
                filter_rad = max(0.8, cam_rad_med * 2.0)
                near_mask = np.linalg.norm(pts - cam_center[None, :], axis=1) <= filter_rad
                if np.sum(near_mask) >= 32:
                    pts = pts[near_mask]

        if pts is not None and len(pts) >= 10:
            p10 = np.percentile(pts, 10, axis=0)
            p20 = np.percentile(pts, 20, axis=0)
            p80 = np.percentile(pts, 80, axis=0)
            center = 0.5 * (p20 + p80)
            # Bias vertical aim upward so tall center subjects are not cropped by dense floor points
            center[1] = 0.5 * (p10[1] + p80[1])
            span = float(np.linalg.norm(p80 - p20))
            radius = max(0.8, span * 0.5)
        elif len(cam_centers) > 0:
            center = np.mean(cam_centers, axis=0)
            span = float(np.max(np.linalg.norm(cam_centers - center[None, :], axis=1)) * 2.0)
            radius = max(1.0, span * 0.5)
        else:
            center = np.zeros(3, dtype=np.float32)
            span = 3.0
            radius = 1.5

        self._scene_extent = max(0.5, float(radius))
        self._floor_y = None
        if len(cam_centers) >= 4 and pts is not None and len(pts) >= 32:
            ring_pts = pts[
                np.linalg.norm(pts[:, [0, 2]] - cam_center[[0, 2]], axis=1) <= max(0.8, cam_rad_med * 1.6)
            ]
            if len(ring_pts) >= 32:
                cand_floor = float(np.percentile(ring_pts[:, 1], 88))
                if cand_floor > float(cam_center[1]) + 0.05 * cam_rad_med:
                    self._floor_y = cand_floor
                    self._floor_margin = max(0.04, 0.08 * cam_rad_med)

        if len(valid_poses) > 0:
            fwd0 = np.asarray(valid_poses[0].r_wc, dtype=np.float32) @ np.array(
                [0.0, 0.0, 1.0], dtype=np.float32
            )
        else:
            fwd0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        horiz_fwd = fwd0.copy()
        horiz_fwd[1] = 0.0
        h_norm = float(np.linalg.norm(horiz_fwd))
        if h_norm < 1e-4:
            horiz_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            horiz_fwd = horiz_fwd / h_norm

        if len(cam_centers) >= 8 and ang_cov > np.deg2rad(210.0):
            # Inward 360 orbit: stay inside/along the camera ring so outer bushes/walls never block overview
            dist_use = max(0.58, min(cam_rad_med * 1.30, max(cam_rad_med * 1.02, span * 0.76)))
            elev = 0.24
            overview_pos = (
                center
                - horiz_fwd * (dist_use * float(np.cos(elev)))
                + np.array([0.0, -dist_use * float(np.sin(elev)), 0.0], dtype=np.float32)
            )
        elif len(cam_centers) >= 4:
            # Open/linear trajectory: frame subject from the observed camera corridor
            look_hf = center - cam_center
            look_hf[1] = 0.0
            ln = float(np.linalg.norm(look_hf))
            dir_hf = (look_hf / ln) if ln > 1e-3 else horiz_fwd
            dist_use = max(0.8, min(cam_rad_med * 1.35, span * 0.65))
            elev = 0.34
            overview_pos = (
                center
                - dir_hf * (dist_use * float(np.cos(elev)))
                + np.array([0.0, -dist_use * float(np.sin(elev)), 0.0], dtype=np.float32)
            )
        else:
            # -Y is UP in COLMAP world, so offset up by -radius * 0.60
            overview_pos = (
                center
                - horiz_fwd * (radius * 1.45)
                + np.array([0.0, -radius * 0.60, 0.0], dtype=np.float32)
            )
        self._set_eye_target(overview_pos, center)

    def _clamp_above_floor(self, adjust_target: bool = False) -> None:
        if self._floor_y is None:
            return
        max_eye_y = float(self._floor_y - self._floor_margin)
        dist = max(0.08, float(self.distance))
        if adjust_target:
            cur_eye_y = float(self.target[1]) - dist * float(np.sin(self.pitch))
            if cur_eye_y > max_eye_y:
                self.target[1] = np.float32(float(self.target[1]) - (cur_eye_y - max_eye_y))
            return
        max_targ_y = max_eye_y + dist * 0.90
        if float(self.target[1]) > max_targ_y:
            self.target[1] = np.float32(max_targ_y)
        cp_sin = (float(self.target[1]) - max_eye_y) / dist
        if cp_sin >= 0.985:
            self.pitch = float(max(self.pitch, 1.40))
            self.target[1] = np.float32(max_eye_y + float(np.sin(self.pitch)) * dist)
        else:
            min_pitch = float(np.arcsin(np.clip(cp_sin, -0.985, 0.985)))
            if self.pitch < min_pitch:
                self.pitch = min_pitch

    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.yaw += float(dx_px) * 0.006
        self.pitch = float(np.clip(self.pitch + float(dy_px) * 0.006, -1.48, 1.48))
        self._clamp_above_floor(adjust_target=False)

    def pan(self, dx_px: float, dy_px: float) -> None:
        c2w = self.get_c2w()
        right = c2w[:3, 0]
        down = c2w[:3, 1]
        scale = max(0.05, float(self.distance)) * 0.0018
        self.target = self.target - right * (float(dx_px) * scale) - down * (float(dy_px) * scale)
        self._clamp_above_floor(adjust_target=True)

    def zoom(self, delta_steps: float) -> None:
        self.distance = max(0.08, float(self.distance) * (0.88 ** float(delta_steps)))
        self._clamp_above_floor(adjust_target=False)

    def fly(self, fwd_amt: float, right_amt: float, up_amt: float, speed_mult: float = 1.0) -> None:
        c2w = self.get_c2w()
        forward = c2w[:3, 2]
        right = c2w[:3, 0]
        step_scale = max(0.06, float(self._scene_extent) * 0.045) * float(speed_mult)
        delta = (
            forward * (float(fwd_amt) * step_scale)
            + right * (float(right_amt) * step_scale)
            + _COLMAP_WORLD_UP * (float(up_amt) * step_scale)
        )
        self.target = self.target + delta
        self._clamp_above_floor(adjust_target=True)


class SplatViewportWidget(QWidget):
    """Live CUDA-rendered 3DGS viewport with adaptive drag scaling, RGBA blit, and held-key flight."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self.camera = SplatOrbitCamera()
        self._splat_scene: Optional[Any] = None
        self._qimage: Optional[QImage] = None
        self._img_buf: Optional[np.ndarray] = None
        self._last_render_ms: float = 0.0
        self._interactive_scale: float = 0.52
        self._preview_hud: Optional[str] = None
        self._last_drag_pos = None
        self._drag_mode: Optional[str] = None
        self._bg_color = QColor("#10141c")
        self._pressed_keys: set[int] = set()
        self._key_mods = Qt.KeyboardModifier.NoModifier

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(120)
        self._debounce_timer.timeout.connect(self.render_full)

        self._key_fly_timer = QTimer(self)
        self._key_fly_timer.setInterval(33)
        self._key_fly_timer.timeout.connect(self._tick_key_fly)

    def set_background_hex(self, hex_color: str) -> None:
        self._bg_color = QColor(hex_color)
        self.update()

    def set_preview_image(self, rgb: np.ndarray, hud_text: str = "Live 3DGS preview...") -> None:
        """Display a progressive RGB uint8 frame inside the 3DGS viewport while training runs."""
        h, w = rgb.shape[:2]
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        self._img_buf = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._qimage = QImage(
            self._img_buf.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_RGBA8888,
        )
        self._preview_hud = hud_text
        self.update()

    def set_scene(
        self,
        splat_scene: Any,
        poses: Sequence[Any],
        intrinsics: Optional[np.ndarray] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self._preview_hud = None
        self._splat_scene = splat_scene
        means_np = None
        if splat_scene is not None and hasattr(splat_scene, "means"):
            try:
                means_np = splat_scene.means.detach().cpu().numpy()
            except Exception:
                means_np = None
        self.camera.set_overview(means_np, poses, intrinsics=intrinsics, frame_size=frame_size)
        self.render_full()

    def clear_scene(self) -> None:
        self._debounce_timer.stop()
        self._key_fly_timer.stop()
        self._pressed_keys.clear()
        self._preview_hud = None
        self._splat_scene = None
        self._qimage = None
        self._img_buf = None
        self.update()

    def render_full(self) -> None:
        self._render(scale=1.0)

    def render_interactive(self) -> None:
        self._render(scale=self._interactive_scale)
        self._debounce_timer.start()

    def _render(self, scale: float = 1.0) -> None:
        if self._splat_scene is None or not _safe_is_gsplat_available():
            return
        w = max(64, int(self.width()))
        h = max(64, int(self.height()))
        dpr = max(1.0, float(self.devicePixelRatioF()))
        rw = max(64, int(round(w * dpr * scale)))
        rh = max(64, int(round(h * dpr * scale)))
        max_edge = max(rw, rh)
        if max_edge > 1600:
            cap = 1600.0 / float(max_edge)
            rw = max(64, int(round(rw * cap)))
            rh = max(64, int(round(rh * cap)))

        from src.gsplat_engine import render_view

        try:
            t0 = time.perf_counter()
            c2w = self.camera.get_c2w()
            k_mat = self.camera.get_intrinsics(rw, rh)
            rgba = render_view(self._splat_scene, c2w=c2w, K=k_mat, width=rw, height=rh, rgba=True)
            self._last_render_ms = (time.perf_counter() - t0) * 1000.0
            if scale < 0.99:
                if self._last_render_ms > 38.0:
                    self._interactive_scale = max(0.35, self._interactive_scale * 0.88)
                elif self._last_render_ms < 16.0:
                    self._interactive_scale = min(0.75, self._interactive_scale * 1.06)
            self._img_buf = np.ascontiguousarray(rgba, dtype=np.uint8)
            self._qimage = QImage(
                self._img_buf.data,
                rw,
                rh,
                rw * 4,
                QImage.Format.Format_RGBA8888,
            )
            self._preview_hud = None
            self.update()
        except Exception as exc:
            _logger.warning("SplatViewportWidget render failed: %s", exc)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)
        if self._qimage is not None and not self._qimage.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), self._qimage)
            hud: Optional[str] = self._preview_hud
            if hud is None and self._splat_scene is not None:
                n_gs = int(getattr(self._splat_scene, "final_splats", 0))
                hud = f"3DGS CUDA  |  {n_gs:,} splats  |  {self._last_render_ms:.1f} ms"
            if hud:
                hud_w = QFontMetrics(painter.font()).horizontalAdvance(hud) + 20
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 145))
                painter.drawRoundedRect(10, 10, hud_w, 26, 6, 6)
                painter.setPen(QColor(235, 240, 248))
                painter.drawText(20, 27, hud)

    def _tick_key_fly(self) -> None:
        if self._splat_scene is None or not self._pressed_keys:
            self._key_fly_timer.stop()
            return
        fwd = float(Qt.Key.Key_W in self._pressed_keys) - float(Qt.Key.Key_S in self._pressed_keys)
        right = float(Qt.Key.Key_D in self._pressed_keys) - float(Qt.Key.Key_A in self._pressed_keys)
        up = float(Qt.Key.Key_E in self._pressed_keys) - float(Qt.Key.Key_Q in self._pressed_keys)
        if fwd == 0.0 and right == 0.0 and up == 0.0:
            self._key_fly_timer.stop()
            return
        mult = 0.45
        if self._key_mods & Qt.KeyboardModifier.ShiftModifier:
            mult *= 2.5
        if self._key_mods & Qt.KeyboardModifier.ControlModifier:
            mult *= 0.35
        self.camera.fly(fwd, right, up, speed_mult=mult)
        self.render_interactive()

    def focusOutEvent(self, event) -> None:
        self._pressed_keys.clear()
        self._key_fly_timer.stop()
        super().focusOutEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._splat_scene is not None:
            self._debounce_timer.start()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        self._last_drag_pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag_mode = "pan"
        else:
            self._drag_mode = None
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_drag_pos is None or self._drag_mode is None or self._splat_scene is None:
            return
        cur = event.position()
        dx = float(cur.x() - self._last_drag_pos.x())
        dy = float(cur.y() - self._last_drag_pos.y())
        self._last_drag_pos = cur
        if self._drag_mode == "orbit":
            self.camera.orbit(dx, dy)
        elif self._drag_mode == "pan":
            self.camera.pan(dx, dy)
        self.render_interactive()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_drag_pos = None
        self._drag_mode = None
        if self._splat_scene is not None:
            self._debounce_timer.start()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._splat_scene is None:
            return
        steps = float(event.angleDelta().y()) / 120.0
        if abs(steps) > 1e-4:
            self.camera.zoom(steps)
            self.render_interactive()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._splat_scene is None:
            super().keyPressEvent(event)
            return
        key = event.key()
        fly_keys = {Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_E, Qt.Key.Key_Q}
        if key in fly_keys:
            self._key_mods = event.modifiers()
            if not event.isAutoRepeat():
                self._pressed_keys.add(key)
                self._tick_key_fly()
                if not self._key_fly_timer.isActive():
                    self._key_fly_timer.start()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self._splat_scene is None:
            super().keyReleaseEvent(event)
            return
        if not event.isAutoRepeat() and event.key() in self._pressed_keys:
            self._pressed_keys.discard(event.key())
            self._key_mods = event.modifiers()
            if not self._pressed_keys:
                self._key_fly_timer.stop()
                self._debounce_timer.start()
            event.accept()
            return
        super().keyReleaseEvent(event)


class ReconstructWorker(QThread):
    progress = Signal(float, str)
    preview_ready = Signal(object)
    finished_ok = Signal(object, float, list)
    failed = Signal(str)

    def __init__(
        self,
        pipeline: Optional[Any],
        video_path: Path,
        config: Optional[Any] = None,
        export_options: Optional[tuple[Path, bool, bool, bool]] = None,
        preset: Optional[Preset] = None,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.config = config
        self.export_options = export_options
        self.preset = preset
        self.start_sec = max(0.0, float(start_sec))
        self.end_sec = max(0.0, float(end_sec))
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def run(self):
        t0 = time.perf_counter()
        try:
            if self.pipeline is None:
                self.progress.emit(0.01, "Loading 3D engine (PyTorch, Open3D, SfM)...")
                from src.pipeline import VideoTo3DPipeline

                self.pipeline = VideoTo3DPipeline()

            if self.config is None and self.preset is not None:
                from src.pipeline import ReconstructionConfig

                self.config = ReconstructionConfig.from_preset(self.preset)
                if self.start_sec > 0.0:
                    self.config.start_sec = self.start_sec
                if self.end_sec > 0.0:
                    self.config.end_sec = self.end_sec
            if self.config is not None and self.export_options:
                self.config.out_dir = str(self.export_options[0])

            res = self.pipeline.run(
                self.video_path,
                config=self.config,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                cancel_cb=lambda: self._cancelled,
                preview_cb=lambda im: self.preview_ready.emit(im),
            )
            exported_paths: List[Path] = []
            if self._cancelled:
                raise RuntimeError("build cancelled by user")
            if getattr(res, "splat_scene", None) is not None:
                if getattr(res, "splat_ply_path", None) is not None:
                    exported_paths.append(Path(res.splat_ply_path))
            elif self.export_options:
                out_dir, auto_ply, auto_glb, auto_obj = self.export_options
                if auto_ply or auto_glb or auto_obj:
                    self.progress.emit(0.98, "Exporting 3D scene files...")
                    from src.scene_builder import export_scene

                    out_dir.mkdir(parents=True, exist_ok=True)
                    if auto_ply and not self._cancelled:
                        p = export_scene(res.scene, out_dir / "scene_cloud.ply", export_mesh=False)
                        exported_paths.append(p)
                    if auto_glb and not self._cancelled:
                        p = export_scene(res.scene, out_dir / "scene_mesh.glb", export_mesh=True)
                        exported_paths.append(p)
                    if auto_obj and not self._cancelled:
                        p = export_scene(res.scene, out_dir / "scene_mesh.obj", export_mesh=True)
                        exported_paths.append(p)
            if self._cancelled:
                raise RuntimeError("build cancelled by user")
            elapsed = time.perf_counter() - t0
            self.finished_ok.emit(res, elapsed, exported_paths)
        except Exception as exc:
            self.failed.emit(str(exc))


class Native3DWindow(QMainWindow):
    def __init__(
        self,
        initial_video: Optional[Path] = None,
        initial_result: Optional[ReconstructionResult] = None,
        initial_preset: Optional[Preset] = None,
        presets_path: str | Path = "presets.json",
        initial_start_sec: float = 0.0,
        initial_end_sec: float = 0.0,
    ):
        super().__init__()
        self.setWindowTitle("arg // video to 3D environment studio")
        self.resize(1380, 840)
        self.setMinimumSize(1024, 640)
        self.setAcceptDrops(True)

        app = QApplication.instance()
        if app is not None:
            app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, False)
            font = QFont("Segoe UI Variable", 10)
            if not font.exactMatch():
                font = QFont("Segoe UI", 10)
            font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
            app.setFont(font)

        self._current_pal = resolve_palette("graphite_cobalt")

        self.preset_mgr = PresetManager(presets_path)
        self._pipeline = None
        self._did_initial_render = False
        self.current_video: Optional[Path] = None
        self.current_video_total_frames: int = 0
        self.current_result: Optional[ReconstructionResult] = None
        self.worker: Optional[ReconstructWorker] = None
        self._syncing_preset = False
        self._active_max_anchors: int = 0
        self._initial_start_sec: float = max(0.0, float(initial_start_sec))
        self._initial_end_sec: float = max(0.0, float(initial_end_sec))

        self._build_start_t = 0.0
        self._last_logged_stage = ""
        self._last_preview_rgb: Optional[np.ndarray] = None
        self._last_fly_kf: int = -1
        self._scene_frustum_scale = 0.16
        self._gsplat_avail: bool = _safe_is_gsplat_available()
        self._active_build_mode: str = "tsdf"

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(100)
        self.elapsed_timer.timeout.connect(self._tick_elapsed_timer)

        self.fly_timer = QTimer(self)
        self.fly_timer.setInterval(33)
        self.fly_timer.timeout.connect(self._step_flythrough)
        self.fly_t = 0.0

        self._pending_scrub_idx = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(35)
        self._scrub_timer.timeout.connect(self._read_raw_video_preview_frame)

        self._build_ui()

        start_preset = initial_preset or self.preset_mgr.get(DEFAULT_PRESET_NAME)
        self.apply_preset(start_preset)

        if initial_result is not None:
            if initial_result.video_data.path.exists():
                self.set_video_path(initial_result.video_data.path)
            self.load_result(initial_result)
        elif initial_video is not None and initial_video.exists():
            self.set_video_path(initial_video)
            QTimer.singleShot(200, self.start_reconstruction)

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            from src.pipeline import VideoTo3DPipeline

            self._pipeline = VideoTo3DPipeline()
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Any):
        self._pipeline = value

    def showEvent(self, event):
        super().showEvent(event)
        if not self._did_initial_render:
            self._did_initial_render = True
            QTimer.singleShot(0, self._safe_initial_render)

    def _safe_initial_render(self):
        if self.isVisible():
            try:
                self.plotter.render()
            except Exception:
                pass

    def _build_ui(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self.main_splitter)

        # =========================================================================
        # Left Shell: 74px Navigation Rail + Fluid Single-Column Inspector Pane
        # =========================================================================
        left_shell = QWidget()
        left_shell.setMinimumWidth(370)
        left_shell.setMaximumWidth(500)
        left_shell_layout = QHBoxLayout(left_shell)
        left_shell_layout.setContentsMargins(0, 0, 0, 0)
        left_shell_layout.setSpacing(0)

        # -------------------------------------------------------------------------
        # 1. Navigation Rail (74px width)
        # -------------------------------------------------------------------------
        self.rail_frame = QFrame()
        self.rail_frame.setObjectName("m3Rail")
        self.rail_frame.setFixedWidth(74)
        rail_layout = QVBoxLayout(self.rail_frame)
        rail_layout.setContentsMargins(5, 12, 5, 12)
        rail_layout.setSpacing(10)
        rail_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Brand Badge (42x42 primaryContainer)
        self.brand_badge = QPushButton()
        self.brand_badge.setFixedSize(42, 42)
        self.brand_badge.setObjectName("m3BrandBadge")
        self.brand_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self.brand_badge.setToolTip("arg - Video to 3D Studio")
        self.brand_badge.setIcon(make_m3_icon("cube", self._current_pal.on_primary_container, 22))
        self.brand_badge.setStyleSheet(
            f"QPushButton {{ background-color: {self._current_pal.primary_container}; "
            f"border: none; border-radius: 14px; }} "
            f"QPushButton:hover {{ background-color: {self._current_pal.primary}; }}"
        )
        self.brand_badge.clicked.connect(lambda: self._on_nav_switched(0))
        rail_layout.addWidget(self.brand_badge, alignment=Qt.AlignmentFlag.AlignCenter)
        rail_layout.addSpacing(6)

        # 4 Navigation Rail Destinations
        self.nav_btn_group = QButtonGroup(self)
        self.nav_btn_group.setExclusive(True)

        self.nav_studio = M3NavRailItem("studio", "Studio")
        self.nav_pipeline = M3NavRailItem("tune", "Pipeline")
        self.nav_export = M3NavRailItem("export", "Export")
        self.nav_activity = M3NavRailItem("activity", "Activity")

        nav_items = [self.nav_studio, self.nav_pipeline, self.nav_export, self.nav_activity]
        for idx, item in enumerate(nav_items):
            self.nav_btn_group.addButton(item, idx)
            rail_layout.addWidget(item, alignment=Qt.AlignmentFlag.AlignCenter)
            item.clicked.connect(lambda _chk=False, i=idx: self._on_nav_switched(i))

        self.nav_studio.setChecked(True)
        rail_layout.addStretch(1)

        # 4 Circular Theme Swatch Buttons at rail bottom (2x2 grid, spacing 6, 22x22)
        swatch_grid = QGridLayout()
        swatch_grid.setContentsMargins(0, 0, 0, 0)
        swatch_grid.setSpacing(6)
        swatch_grid.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.swatch_buttons: Dict[str, M3SwatchButton] = {}
        for s_idx, (k, p) in enumerate(M3_PALETTES.items()):
            s_btn = M3SwatchButton(p.swatch, tooltip=p.label)
            s_btn.clicked.connect(lambda _chk=False, key=p.key: self._on_swatch_clicked(key))
            swatch_grid.addWidget(s_btn, s_idx // 2, s_idx % 2)
            self.swatch_buttons[p.key] = s_btn

        rail_layout.addLayout(swatch_grid)
        left_shell_layout.addWidget(self.rail_frame)

        # -------------------------------------------------------------------------
        # 2. Fluid Single-Column Inspector Pane
        # -------------------------------------------------------------------------
        self.inspector_frame = QFrame()
        self.inspector_frame.setObjectName("m3Inspector")
        inspector_layout = QVBoxLayout(self.inspector_frame)
        inspector_layout.setContentsMargins(14, 12, 14, 12)
        inspector_layout.setSpacing(10)

        # Top Dynamic Header Row
        top_heading_row = QHBoxLayout()
        top_heading_row.setContentsMargins(2, 2, 2, 2)
        top_heading_row.setSpacing(8)

        self.page_title_lbl = QLabel("Studio")
        self.page_title_lbl.setObjectName("m3PageHeading")
        top_heading_row.addWidget(self.page_title_lbl)
        top_heading_row.addStretch(1)

        short_gpu, full_gpu = _detect_gpu_info()
        self.gpu_badge = QLabel(short_gpu)
        self.gpu_badge.setObjectName("m3GpuChip")
        self.gpu_badge.setToolTip(full_gpu)
        top_heading_row.addWidget(self.gpu_badge)
        inspector_layout.addLayout(top_heading_row)

        # 4-page Stack
        self.nav_stack = QStackedWidget()
        inspector_layout.addWidget(self.nav_stack, stretch=1)

        # =========================================================================
        # PAGE 0: Studio (Preview, Info, Actions, Scrubber, Quick Setup)
        # =========================================================================
        page_studio = QWidget()
        ps_layout = QVBoxLayout(page_studio)
        ps_layout.setContentsMargins(0, 0, 12, 8)
        ps_layout.setSpacing(10)

        # Video Input Card
        card_vid = QFrame()
        card_vid.setObjectName("m3Card")
        cv_layout = QVBoxLayout(card_vid)
        cv_layout.setContentsMargins(14, 12, 14, 12)
        cv_layout.setSpacing(8)

        lbl_vtitle = QLabel("Video Source")
        lbl_vtitle.setObjectName("m3SectionTitle")
        cv_layout.addWidget(lbl_vtitle)

        self.preview_widget = M3VideoPreview()
        self.preview_widget.clicked.connect(self.pick_video_file)
        cv_layout.addWidget(self.preview_widget)

        self.video_info_lbl = QLabel(
            "No video loaded. Drop a video file, click to browse, or generate a demo."
        )
        self.video_info_lbl.setObjectName("m3Muted")
        self.video_info_lbl.setWordWrap(True)
        cv_layout.addWidget(self.video_info_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.open_btn = M3Button("Open video / 3D", variant="tonal", icon_name="folder", height=36)
        self.open_btn.setToolTip("Open a video (.mp4/.mov/...) or saved 3D scene (.ply/.glb/.obj/.npz)")
        self.open_btn.clicked.connect(self.pick_video_file)
        btn_row.addWidget(self.open_btn, stretch=1)

        self.demo_btn = M3Button("Demo scene", variant="outlined", icon_name="sparkle", height=36)
        self.demo_btn.clicked.connect(self.make_and_load_demo)
        btn_row.addWidget(self.demo_btn, stretch=1)
        cv_layout.addLayout(btn_row)

        # Inline Frame Scrubber inside Card 1
        kf_top = QHBoxLayout()
        lbl_kftitle = QLabel("Frame preview")
        lbl_kftitle.setObjectName("m3SectionTitle")
        kf_top.addWidget(lbl_kftitle)
        kf_top.addStretch(1)

        self.kf_lbl = QLabel("0 / 0")
        self.kf_lbl.setObjectName("m3ValuePill")
        self.kf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        kf_top.addWidget(self.kf_lbl)
        cv_layout.addLayout(kf_top)

        self.kf_slider = M3ExpressiveSlider()
        self.kf_slider.setRange(0, 0)
        self.kf_slider.valueChanged.connect(self.on_keyframe_scrub)
        cv_layout.addWidget(self.kf_slider)

        ps_layout.addWidget(card_vid)

        # Card 2: Quick setup
        card_setup = QFrame()
        card_setup.setObjectName("m3Card")
        csetup_layout = QVBoxLayout(card_setup)
        csetup_layout.setContentsMargins(14, 12, 14, 12)
        csetup_layout.setSpacing(8)

        lbl_stitle = QLabel("Quick Setup")
        lbl_stitle.setObjectName("m3SectionTitle")
        csetup_layout.addWidget(lbl_stitle)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = M3ComboBox()
        self._refresh_preset_combo_items()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_picked)
        preset_row.addWidget(self.preset_combo, stretch=1)
        csetup_layout.addLayout(preset_row)

        lbl_engine = QLabel("Reconstruction mode:")
        lbl_engine.setObjectName("m3Muted")
        csetup_layout.addWidget(lbl_engine)

        self.recon_mode_combo = M3ConnectedGroup(
            ["Surface Mesh (TSDF)", "3DGS Splats"],
            data=["tsdf", "3dgs"],
        )
        if not self._gsplat_avail and len(self.recon_mode_combo._buttons) > 1:
            self.recon_mode_combo._buttons[1].setEnabled(False)
            self.recon_mode_combo._buttons[1].setToolTip("Requires gsplat CUDA backend")
        csetup_layout.addWidget(self.recon_mode_combo)

        lbl_depth = QLabel("Depth model:")
        lbl_depth.setObjectName("m3Muted")
        csetup_layout.addWidget(lbl_depth)

        self.depth_model_combo = M3ConnectedGroup(
            ["Large", "Base", "Small"],
            data=["large", "base", "small"],
        )
        csetup_layout.addWidget(self.depth_model_combo)

        lbl_scene = QLabel("Scene profile:")
        lbl_scene.setObjectName("m3Muted")
        csetup_layout.addWidget(lbl_scene)

        self.scene_type_combo = M3ConnectedGroup(["Indoor", "Outdoor"], data=["indoor", "outdoor"])
        csetup_layout.addWidget(self.scene_type_combo)

        self.active_model_lbl = QLabel("")
        self.active_model_lbl.setObjectName("m3Muted")
        self.active_model_lbl.setWordWrap(True)
        csetup_layout.addWidget(self.active_model_lbl)

        self.recon_mode_combo.currentIndexChanged.connect(self._on_recon_mode_changed)
        self.depth_model_combo.currentIndexChanged.connect(self._update_active_model_lbl)
        self.scene_type_combo.currentIndexChanged.connect(self._on_scene_type_changed)
        self._on_recon_mode_changed()

        self.all_frames_btn = M3Button("All", variant="compact", height=24)
        self.all_frames_btn.setToolTip("Set keyframes to all frames in video")
        self.all_frames_btn.clicked.connect(self._set_all_frames)

        self.max_frames_spin = M3NumericControl(
            "Max keyframes",
            min_val=4,
            max_val=100000,
            init_val=24,
            step=4,
            slider_max=200,
            extra_widget=self.all_frames_btn,
        )
        csetup_layout.addWidget(self.max_frames_spin)

        ps_layout.addWidget(card_setup)
        ps_layout.addStretch(1)
        self.nav_stack.addWidget(_wrap_scroll(page_studio))

        # =========================================================================
        # PAGE 1: Pipeline (Preset description, Tuning controls, Custom Presets)
        # =========================================================================
        page_pipeline = QWidget()
        pp_layout = QVBoxLayout(page_pipeline)
        pp_layout.setContentsMargins(0, 0, 12, 8)
        pp_layout.setSpacing(10)

        # Preset Card
        card_pre = QFrame()
        card_pre.setObjectName("m3Card")
        cpre_layout = QVBoxLayout(card_pre)
        cpre_layout.setContentsMargins(14, 12, 14, 12)
        cpre_layout.setSpacing(6)

        lbl_ptitle = QLabel("Active Preset")
        lbl_ptitle.setObjectName("m3SectionTitle")
        cpre_layout.addWidget(lbl_ptitle)

        self.preset_desc_lbl = QLabel("")
        self.preset_desc_lbl.setObjectName("m3Muted")
        self.preset_desc_lbl.setWordWrap(True)
        cpre_layout.addWidget(self.preset_desc_lbl)
        pp_layout.addWidget(card_pre)

        # Parameters Card (Single Column M3 Controls)
        card_params = QFrame()
        card_params.setObjectName("m3Card")
        cp_layout = QVBoxLayout(card_params)
        cp_layout.setContentsMargins(14, 12, 14, 12)
        cp_layout.setSpacing(8)

        lbl_ctrls = QLabel("Tuning Controls")
        lbl_ctrls.setObjectName("m3SectionTitle")
        cp_layout.addWidget(lbl_ctrls)

        self.max_dim_spin = M3NumericControl(
            "Resolution",
            min_val=128,
            max_val=1024,
            init_val=512,
            step=32,
            suffix=" px",
        )
        cp_layout.addWidget(self.max_dim_spin)

        self.point_stride_spin = M3NumericControl(
            "Point stride",
            min_val=1,
            max_val=6,
            init_val=2,
            step=1,
        )
        cp_layout.addWidget(self.point_stride_spin)

        self.mesh_stride_spin = M3NumericControl(
            "Mesh stride",
            min_val=2,
            max_val=10,
            init_val=4,
            step=1,
        )
        cp_layout.addWidget(self.mesh_stride_spin)

        self.edge_thresh_spin = M3NumericControl(
            "Edge filter",
            min_val=0.03,
            max_val=0.40,
            init_val=0.12,
            step=0.01,
            decimals=2,
        )
        cp_layout.addWidget(self.edge_thresh_spin)

        self.demo_sec_spin = M3NumericControl(
            "Demo duration",
            min_val=2.0,
            max_val=120.0,
            init_val=5.0,
            step=1.0,
            decimals=1,
            suffix=" s",
            slider_max=30.0,
        )
        cp_layout.addWidget(self.demo_sec_spin)

        self.start_sec_spin = M3NumericControl(
            "Trim start",
            min_val=0.0,
            max_val=3600.0,
            init_val=float(self._initial_start_sec),
            step=0.5,
            decimals=1,
            suffix=" s",
            slider_max=120.0,
        )
        self.start_sec_spin.valueChanged.connect(self._on_trim_changed)
        cp_layout.addWidget(self.start_sec_spin)

        self.end_sec_spin = M3NumericControl(
            "Trim end (0=end)",
            min_val=0.0,
            max_val=3600.0,
            init_val=float(self._initial_end_sec),
            step=0.5,
            decimals=1,
            suffix=" s",
            slider_max=120.0,
        )
        self.end_sec_spin.valueChanged.connect(self._on_trim_changed)
        cp_layout.addWidget(self.end_sec_spin)

        lbl_dev = QLabel("Compute device:")
        lbl_dev.setObjectName("m3Muted")
        cp_layout.addWidget(lbl_dev)

        self.device_combo = M3ConnectedGroup(["Auto", "CUDA", "CPU"], data=["auto", "cuda", "cpu"])
        cp_layout.addWidget(self.device_combo)

        pp_layout.addWidget(card_params)

        # Custom Presets Sub-Card
        card_custom = QFrame()
        card_custom.setObjectName("m3SubCard")
        cc_layout = QVBoxLayout(card_custom)
        cc_layout.setContentsMargins(14, 12, 14, 12)
        cc_layout.setSpacing(8)

        lbl_mtitle = QLabel("Save / Manage Custom Preset")
        lbl_mtitle.setObjectName("m3SectionTitle")
        lbl_mtitle.setWordWrap(True)
        cc_layout.addWidget(lbl_mtitle)

        form = QFormLayout()
        form.setSpacing(6)
        self.preset_name_edit = QLineEdit(DEFAULT_PRESET_NAME)
        self.preset_name_edit.textChanged.connect(self._update_active_model_lbl)
        self.preset_desc_edit = QLineEdit("sweet spot for 6gb vram")
        form.addRow("Name:", self.preset_name_edit)
        form.addRow("Notes:", self.preset_desc_edit)
        cc_layout.addLayout(form)

        crud_btns = QHBoxLayout()
        crud_btns.setSpacing(8)
        self.save_preset_btn = M3Button("Save preset", variant="tonal", icon_name="save", height=34)
        self.save_preset_btn.clicked.connect(self.save_custom_preset_from_ui)
        crud_btns.addWidget(self.save_preset_btn, stretch=1)

        self.del_preset_btn = M3Button("Delete", variant="error_tonal", icon_name="trash", height=34)
        self.del_preset_btn.clicked.connect(self.delete_current_custom_preset)
        crud_btns.addWidget(self.del_preset_btn)
        cc_layout.addLayout(crud_btns)

        pp_layout.addWidget(card_custom)
        pp_layout.addStretch(1)
        self.nav_stack.addWidget(_wrap_scroll(page_pipeline))

        # =========================================================================
        # PAGE 2: View & Export
        # =========================================================================
        page_export = QWidget()
        pe_layout = QVBoxLayout(page_export)
        pe_layout.setContentsMargins(0, 0, 12, 8)
        pe_layout.setSpacing(10)

        # Display & Theme Card
        card_vp = QFrame()
        card_vp.setObjectName("m3Card")
        cvp_layout = QVBoxLayout(card_vp)
        cvp_layout.setContentsMargins(14, 12, 14, 12)
        cvp_layout.setSpacing(8)

        lbl_vpt = QLabel("3D Display & Shading")
        lbl_vpt.setObjectName("m3SectionTitle")
        cvp_layout.addWidget(lbl_vpt)

        pt_top = QHBoxLayout()
        pt_top.addWidget(QLabel("Point size:"))
        pt_top.addStretch(1)
        self.pt_val_lbl = QLabel("4")
        self.pt_val_lbl.setObjectName("m3ValuePill")
        self.pt_val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pt_top.addWidget(self.pt_val_lbl)
        cvp_layout.addLayout(pt_top)

        self.pt_slider = M3ExpressiveSlider()
        self.pt_slider.setRange(2, 14)
        self.pt_slider.setValue(4)
        self.pt_slider.valueChanged.connect(self._on_pt_slider_changed)
        cvp_layout.addWidget(self.pt_slider)

        self.edl_chk = M3Switch("Eye-Dome lighting")
        self.edl_chk.setChecked(False)
        self.edl_chk.toggled.connect(self._apply_edl_toggle)
        cvp_layout.addWidget(self.edl_chk)

        lbl_theme = QLabel("Appearance")
        lbl_theme.setObjectName("m3SectionTitle")
        cvp_layout.addWidget(lbl_theme)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)
        theme_row.addWidget(QLabel("Palette:"))
        self.theme_combo = M3ComboBox()
        for p in M3_PALETTES.values():
            self.theme_combo.addItem(p.label, p.key)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self.theme_combo, stretch=1)
        cvp_layout.addLayout(theme_row)

        pe_layout.addWidget(card_vp)

        # Output & Export Card
        card_exp = QFrame()
        card_exp.setObjectName("m3Card")
        cexp_layout = QVBoxLayout(card_exp)
        cexp_layout.setContentsMargins(14, 12, 14, 12)
        cexp_layout.setSpacing(8)

        lbl_expt = QLabel("Export Settings")
        lbl_expt.setObjectName("m3SectionTitle")
        cexp_layout.addWidget(lbl_expt)

        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        out_row.addWidget(QLabel("Output folder:"))
        self.out_dir_edit = QLineEdit("outputs")
        out_row.addWidget(self.out_dir_edit, stretch=1)
        self.browse_dir_btn = M3Button("", variant="icon_tonal", icon_name="folder", height=34)
        self.browse_dir_btn.setToolTip("Choose output folder")
        self.browse_dir_btn.clicked.connect(self.pick_output_folder)
        out_row.addWidget(self.browse_dir_btn)
        cexp_layout.addLayout(out_row)

        lbl_auto = QLabel("Automatic export:")
        lbl_auto.setObjectName("m3Muted")
        cexp_layout.addWidget(lbl_auto)

        auto_vbox = QVBoxLayout()
        auto_vbox.setSpacing(6)
        self.auto_ply_chk = M3Switch("Auto-export PLY point cloud")
        self.auto_ply_chk.setChecked(True)
        self.auto_glb_chk = M3Switch("Auto-export GLB mesh")
        self.auto_glb_chk.setChecked(True)
        self.auto_obj_chk = M3Switch("Auto-export OBJ mesh")
        self.auto_obj_chk.setChecked(False)
        auto_vbox.addWidget(self.auto_ply_chk)
        auto_vbox.addWidget(self.auto_glb_chk)
        auto_vbox.addWidget(self.auto_obj_chk)
        cexp_layout.addLayout(auto_vbox)

        lbl_man = QLabel("Manual export:")
        lbl_man.setObjectName("m3Muted")
        cexp_layout.addWidget(lbl_man)

        man_row = QHBoxLayout()
        man_row.setSpacing(6)
        self.exp_ply_btn = M3Button("PLY", variant="tonal", icon_name="export", height=34)
        self.exp_ply_btn.clicked.connect(lambda: self.export_current(".ply", False))
        man_row.addWidget(self.exp_ply_btn, stretch=1)

        self.exp_splat_btn = M3Button("SPLAT", variant="tonal", icon_name="export", height=34)
        self.exp_splat_btn.setToolTip("Compact 32-byte/splat web format (3DGS mode)")
        self.exp_splat_btn.clicked.connect(lambda: self.export_current(".splat", False))
        man_row.addWidget(self.exp_splat_btn, stretch=1)

        self.exp_glb_btn = M3Button("GLB", variant="tonal", icon_name="export", height=34)
        self.exp_glb_btn.clicked.connect(lambda: self.export_current(".glb", True))
        man_row.addWidget(self.exp_glb_btn, stretch=1)

        self.exp_obj_btn = M3Button("OBJ", variant="tonal", icon_name="export", height=34)
        self.exp_obj_btn.clicked.connect(lambda: self.export_current(".obj", True))
        man_row.addWidget(self.exp_obj_btn, stretch=1)

        cexp_layout.addLayout(man_row)

        media_row = QHBoxLayout()
        media_row.setSpacing(6)
        self.snap_png_btn = M3Button("Snapshot PNG", variant="outlined", icon_name="camera", height=34)
        self.snap_png_btn.clicked.connect(self.export_snapshot_png)
        media_row.addWidget(self.snap_png_btn, stretch=1)

        self.turntable_btn = M3Button("Turntable MP4", variant="outlined", icon_name="play", height=34)
        self.turntable_btn.clicked.connect(self.export_turntable_mp4)
        media_row.addWidget(self.turntable_btn, stretch=1)
        cexp_layout.addLayout(media_row)

        pe_layout.addWidget(card_exp)
        pe_layout.addStretch(1)
        self.nav_stack.addWidget(_wrap_scroll(page_export))

        # =========================================================================
        # PAGE 3: Activity & Milestones
        # =========================================================================
        page_act = QWidget()
        pa_layout = QVBoxLayout(page_act)
        pa_layout.setContentsMargins(0, 0, 12, 8)
        pa_layout.setSpacing(10)

        # Telemetry Card
        card_tel = QFrame()
        card_tel.setObjectName("m3Card")
        ctel_layout = QVBoxLayout(card_tel)
        ctel_layout.setContentsMargins(14, 12, 14, 12)
        ctel_layout.setSpacing(6)

        lbl_tel = QLabel("Scene Telemetry")
        lbl_tel.setObjectName("m3SectionTitle")
        ctel_layout.addWidget(lbl_tel)

        self.telemetry_lbl = QLabel("No 3D scene built yet.")
        self.telemetry_lbl.setWordWrap(True)
        self.telemetry_lbl.setStyleSheet(
            "font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 11px;"
        )
        ctel_layout.addWidget(self.telemetry_lbl)
        pa_layout.addWidget(card_tel)

        # Milestone Log Card
        card_log = QFrame()
        card_log.setObjectName("m3Card")
        clog_layout = QVBoxLayout(card_log)
        clog_layout.setContentsMargins(14, 12, 14, 12)
        clog_layout.setSpacing(6)

        lbl_log = QLabel("Stage Milestones")
        lbl_log.setObjectName("m3SectionTitle")
        clog_layout.addWidget(lbl_log)

        self.milestone_log = QPlainTextEdit()
        self.milestone_log.setReadOnly(True)
        self.milestone_log.setMaximumBlockCount(40)
        self.milestone_log.setMinimumHeight(130)
        clog_layout.addWidget(self.milestone_log)
        pa_layout.addWidget(card_log, stretch=1)

        self.nav_stack.addWidget(_wrap_scroll(page_act))

        # -------------------------------------------------------------------------
        # Pinned Build Deck (#m3BuildDeck) at Bottom of Inspector
        # -------------------------------------------------------------------------
        self.build_deck = QFrame()
        self.build_deck.setObjectName("m3BuildDeck")
        deck_layout = QVBoxLayout(self.build_deck)
        deck_layout.setContentsMargins(14, 12, 14, 12)
        deck_layout.setSpacing(8)

        # Row 1: Loading Indicator + Status Stack + Timer Pill
        deck_top_row = QHBoxLayout()
        deck_top_row.setSpacing(10)
        deck_top_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.loading_indicator = M3LoadingIndicator(size=44, contained=True)
        deck_top_row.addWidget(self.loading_indicator)

        status_vbox = QVBoxLayout()
        status_vbox.setSpacing(2)
        self.stage_pill = QLabel("Ready")
        self.stage_pill.setObjectName("m3StageChip")
        self.stage_pill.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        status_vbox.addWidget(self.stage_pill, alignment=Qt.AlignmentFlag.AlignLeft)

        self.status_lbl = QLabel("Waiting to start")
        self.status_lbl.setObjectName("m3Muted")
        self.status_lbl.setWordWrap(True)
        status_vbox.addWidget(self.status_lbl)
        deck_top_row.addLayout(status_vbox, stretch=1)

        self.timer_lbl = QLabel("00.0s")
        self.timer_lbl.setObjectName("m3TimerPill")
        self.timer_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.timer_lbl.setFixedHeight(28)
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        deck_top_row.addWidget(self.timer_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        deck_layout.addLayout(deck_top_row)

        # Row 2: M3 Wavy Progress Bar
        self.prog_bar = M3WavyProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        deck_layout.addWidget(self.prog_bar)

        # Row 3: Action Buttons
        deck_btns = QHBoxLayout()
        deck_btns.setSpacing(8)
        self.build_btn = M3Button("Build 3D scene", variant="filled", icon_name="play", height=40)
        self.build_btn.clicked.connect(self.start_reconstruction)
        deck_btns.addWidget(self.build_btn, stretch=1)

        self.cancel_btn = M3Button("Stop", variant="error_tonal", icon_name="stop", height=40)
        self.cancel_btn.setFixedWidth(92)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel current reconstruction")
        self.cancel_btn.clicked.connect(self.request_cancel)
        deck_btns.addWidget(self.cancel_btn)
        deck_layout.addLayout(deck_btns)

        inspector_layout.addWidget(self.build_deck)
        left_shell_layout.addWidget(self.inspector_frame, stretch=1)
        self.main_splitter.addWidget(left_shell)

        # =========================================================================
        # Right Viewport: Floating Viewport Toolbar + PyVista QtInteractor
        # =========================================================================
        viewport_container = QWidget()
        vp_layout = QVBoxLayout(viewport_container)
        vp_layout.setContentsMargins(8, 8, 8, 8)
        vp_layout.setSpacing(8)

        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("m3Toolbar")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(12, 6, 12, 6)
        toolbar_layout.setSpacing(10)

        self.mode_combo = M3ConnectedGroup(["Point Cloud", "Surface Mesh", "Hybrid"])
        self.mode_combo.currentIndexChanged.connect(self.refresh_3d_actors)
        toolbar_layout.addWidget(self.mode_combo)

        self.dollhouse_chk = M3Switch("Dollhouse")
        self.dollhouse_chk.setChecked(True)
        self.dollhouse_chk.toggled.connect(self.refresh_3d_actors)
        toolbar_layout.addWidget(self.dollhouse_chk)

        self.frustums_chk = M3Switch("Cameras")
        self.frustums_chk.setChecked(False)
        self.frustums_chk.toggled.connect(self.refresh_3d_actors)
        toolbar_layout.addWidget(self.frustums_chk)

        toolbar_layout.addStretch(1)

        self.reset_cam_btn = M3Button("Overview", variant="outlined", icon_name="overview", height=34)
        self.reset_cam_btn.clicked.connect(self.reset_overview_camera)
        toolbar_layout.addWidget(self.reset_cam_btn)

        self.snap_cam_btn = M3Button("Snap", variant="outlined", icon_name="camera", height=34)
        self.snap_cam_btn.clicked.connect(self.snap_to_selected_keyframe)
        toolbar_layout.addWidget(self.snap_cam_btn)

        self.fly_btn = M3Button("Fly", variant="tonal", icon_name="fly", height=34)
        self.fly_btn.clicked.connect(self.toggle_flythrough)
        toolbar_layout.addWidget(self.fly_btn)

        vp_layout.addWidget(toolbar_frame)

        self.viewport_stack = QStackedWidget(viewport_container)

        self.plotter = QtInteractor(self.viewport_stack, auto_update=False)
        render_timer = getattr(self.plotter, "render_timer", None)
        if render_timer is not None and render_timer.isActive():
            render_timer.stop()
        self.plotter.add_axes(color="white")
        self.viewport_stack.addWidget(self.plotter.interactor)

        self.splat_viewport = SplatViewportWidget(self.viewport_stack)
        self.viewport_stack.addWidget(self.splat_viewport)
        self.viewport_stack.setCurrentIndex(0)
        vp_layout.addWidget(self.viewport_stack)

        self.main_splitter.addWidget(viewport_container)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setSizes([415, 965])

        self._apply_theme("graphite_cobalt")
        self._apply_edl_toggle(self.edl_chk.isChecked())
        self._log_milestone("Studio ready. Pick a video or click 'Demo scene'.")

    def _on_pt_slider_changed(self, val: int):
        self.pt_val_lbl.setText(str(val))
        self.refresh_3d_actors()

    def _on_nav_switched(self, idx: int):
        titles = [
            "Studio",
            "Reconstruction Pipeline",
            "View & Export",
            "Activity & Milestones",
        ]
        if 0 <= idx < len(titles):
            self.nav_stack.setCurrentIndex(idx)
            self.page_title_lbl.setText(titles[idx])
            items = [self.nav_studio, self.nav_pipeline, self.nav_export, self.nav_activity]
            for i, itm in enumerate(items):
                itm.blockSignals(True)
                itm.setChecked(i == idx)
                itm.blockSignals(False)
                itm.update()

    def _on_swatch_clicked(self, key: str):
        idx = self.theme_combo.findData(key)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        else:
            self._apply_theme(key)

    def _log_milestone(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.milestone_log.appendPlainText(f"[{ts}] {text}")

    def _refresh_preset_combo_items(self, select_name: Optional[str] = None):
        self._syncing_preset = True
        try:
            curr = select_name or self.preset_combo.currentData() or DEFAULT_PRESET_NAME
            self.preset_combo.clear()
            target_idx = 0
            for idx, p in enumerate(self.preset_mgr.list_presets()):
                tag = "custom" if self.preset_mgr.is_custom(p.name) else "built-in"
                self.preset_combo.addItem(f"{p.name} [{tag}]", p.name)
                if p.name.lower() == str(curr).lower():
                    target_idx = idx
            self.preset_combo.setCurrentIndex(target_idx)
        finally:
            self._syncing_preset = False

    def _on_preset_picked(self, _idx: int = 0):
        if self._syncing_preset:
            return
        name = self.preset_combo.currentData()
        if name:
            preset = self.preset_mgr.get(str(name))
            self.apply_preset(preset, preserve_recon_mode=preset.builtin)

    def gather_ui_preset(self) -> Preset:
        mode_reverse = {0: "points", 1: "mesh", 2: "hybrid"}
        dev_val = str(self.device_combo.currentData() or "auto")
        depth_val = str(self.depth_model_combo.currentData() or "large")
        scene_val = str(self.scene_type_combo.currentData() or "indoor")
        theme_val = str(self.theme_combo.currentData() or "graphite_cobalt")
        recon_val = str(self.recon_mode_combo.currentData() or "tsdf")
        if recon_val == "3dgs" and not self._gsplat_avail:
            recon_val = "tsdf"
        return Preset(
            name=self.preset_name_edit.text().strip() or "Custom Preset",
            description=self.preset_desc_edit.text().strip() or "custom user preset",
            max_frames=int(self.max_frames_spin.value()),
            max_dim=int(self.max_dim_spin.value()),
            point_stride=int(self.point_stride_spin.value()),
            mesh_stride=int(self.mesh_stride_spin.value()),
            edge_thresh=float(self.edge_thresh_spin.value()),
            depth_model=depth_val,
            scene_type=scene_val,
            max_anchors=self._active_max_anchors,
            recon_mode=recon_val,
            device=dev_val,
            out_dir=self.out_dir_edit.text().strip() or "outputs",
            export_ply=self.auto_ply_chk.isChecked(),
            export_glb=self.auto_glb_chk.isChecked(),
            export_obj=self.auto_obj_chk.isChecked(),
            render_mode=mode_reverse.get(self.mode_combo.currentIndex(), "mesh"),
            point_size=int(self.pt_slider.value()),
            eye_dome=self.edl_chk.isChecked(),
            dollhouse_cull=self.dollhouse_chk.isChecked(),
            show_frustums=self.frustums_chk.isChecked(),
            theme=theme_val,
        ).sanitized()

    def apply_preset(self, preset: Preset, preserve_recon_mode: bool = False):
        self._syncing_preset = True
        try:
            self._refresh_preset_combo_items(select_name=preset.name)
            self._active_max_anchors = int(preset.max_anchors)
            self.preset_desc_lbl.setText(preset.description)
            self.max_frames_spin.setValue(preset.max_frames)
            self.max_dim_spin.setValue(preset.max_dim)
            self.point_stride_spin.setValue(preset.point_stride)
            self.mesh_stride_spin.setValue(preset.mesh_stride)
            self.edge_thresh_spin.setValue(preset.edge_thresh)

            dev_idx = self.device_combo.findData(preset.device)
            if dev_idx >= 0:
                self.device_combo.setCurrentIndex(dev_idx)

            dep_idx = self.depth_model_combo.findData(preset.depth_model)
            if dep_idx >= 0:
                self.depth_model_combo.setCurrentIndex(dep_idx)

            scene_idx = self.scene_type_combo.findData(preset.scene_type)
            if scene_idx >= 0:
                self.scene_type_combo.setCurrentIndex(scene_idx)
            else:
                self.scene_type_combo.setCurrentIndex(0 if preset.scene_type == "indoor" else 1)

            if not preserve_recon_mode:
                r_mode = getattr(preset, "recon_mode", "tsdf")
                if r_mode == "3dgs" and self._gsplat_avail:
                    self.recon_mode_combo.setCurrentIndex(1)
                else:
                    self.recon_mode_combo.setCurrentIndex(0)

            self.out_dir_edit.setText(preset.out_dir)
            self.auto_ply_chk.setChecked(preset.export_ply)
            self.auto_glb_chk.setChecked(preset.export_glb)
            self.auto_obj_chk.setChecked(preset.export_obj)

            mode_map = {"points": 0, "mesh": 1, "hybrid": 2}
            self.mode_combo.setCurrentIndex(mode_map.get(preset.render_mode, 1))
            self.pt_slider.setValue(preset.point_size)
            self.pt_val_lbl.setText(str(preset.point_size))
            self.edl_chk.setChecked(preset.eye_dome)
            self.dollhouse_chk.setChecked(preset.dollhouse_cull)
            self.frustums_chk.setChecked(preset.show_frustums)

            self.preset_name_edit.setText(preset.name)
            self.preset_desc_edit.setText(preset.description)
            self.preset_name_edit.setCursorPosition(0)
            self.preset_desc_edit.setCursorPosition(0)
            self.del_preset_btn.setEnabled(self.preset_mgr.is_custom(preset.name))

            theme_key = THEME_ALIASES.get(preset.theme, preset.theme)
            t_idx = self.theme_combo.findData(theme_key)
            if t_idx >= 0:
                self.theme_combo.setCurrentIndex(t_idx)
            self._apply_theme(theme_key)
            self._on_recon_mode_changed()
        finally:
            self._syncing_preset = False

    def _on_trim_changed(self, *_args):
        if hasattr(self, "start_sec_spin"):
            self._initial_start_sec = max(0.0, float(self.start_sec_spin.value()))
        if hasattr(self, "end_sec_spin"):
            self._initial_end_sec = max(0.0, float(self.end_sec_spin.value()))

    def _on_recon_mode_changed(self, *_args):
        is_3dgs = str(self.recon_mode_combo.currentData() or "tsdf") == "3dgs"
        self.depth_model_combo.setEnabled(not is_3dgs)
        for ctrl in ("max_dim_spin", "point_stride_spin", "mesh_stride_spin", "edge_thresh_spin"):
            w = getattr(self, ctrl, None)
            if w is not None:
                w.setEnabled(not is_3dgs)
                w.setToolTip(
                    "Controlled automatically by 3DGS preset (TSDF only)"
                    if is_3dgs
                    else ""
                )
        if not self._gsplat_avail and len(self.recon_mode_combo._buttons) > 1:
            self.recon_mode_combo._buttons[1].setEnabled(False)
        self._update_active_model_lbl()

    def _update_active_model_lbl(self, *args):
        if str(self.recon_mode_combo.currentData() or "tsdf") == "3dgs":
            name_edit = getattr(self, "preset_name_edit", None)
            preset_name = (
                (name_edit.text().strip() if name_edit is not None else "")
                or str(self.preset_combo.currentData() or DEFAULT_PRESET_NAME)
            )
            try:
                from src.gsplat_engine import resolve_gsplat_preset

                gs_cfg = resolve_gsplat_preset(preset_name)
                self.active_model_lbl.setText(
                    f"3DGS AbsGS - {gs_cfg.name} ({gs_cfg.cap_max // 1000}k splats, "
                    f"{gs_cfg.steps} steps, {gs_cfg.max_dim}px)"
                )
            except Exception:
                self.active_model_lbl.setText("3DGS AbsGS CUDA (DA3 & TSDF bypassed)")
            return

        stype = str(self.scene_type_combo.currentData() or "indoor").capitalize()
        model_size = str(self.depth_model_combo.currentData() or "large").lower()
        if model_size == "large":
            self.active_model_lbl.setText(f"Depth-Anything-3 Base / Hi-Res 644px ({stype})")
        elif model_size == "base":
            self.active_model_lbl.setText(f"Depth-Anything-3 Base / 504px ({stype})")
        else:
            depth_range = "up to 40m" if stype.lower() == "outdoor" else "up to 9.2m"
            self.active_model_lbl.setText(f"Depth-Anything-V2 {stype} / Small ({depth_range})")

    def _on_scene_type_changed(self, _idx: int = 0):
        self._update_active_model_lbl()
        if not self._syncing_preset:
            is_outdoor = str(self.scene_type_combo.currentData() or "indoor").lower() == "outdoor"
            self.dollhouse_chk.setChecked(not is_outdoor)

    def save_custom_preset_from_ui(self):
        p = self.gather_ui_preset()
        saved = self.preset_mgr.save_preset(p)
        self.apply_preset(saved)
        self._log_milestone(f"Saved custom preset '{saved.name}' to {self.preset_mgr.path.name}")

    def delete_current_custom_preset(self):
        name = self.preset_name_edit.text().strip()
        if self.preset_mgr.delete_preset(name):
            fallback = self.preset_mgr.get(DEFAULT_PRESET_NAME)
            self.apply_preset(fallback)
            self._log_milestone(f"Deleted custom preset '{name}'")

    def _on_theme_changed(self, _idx: int = 0):
        theme_key = str(self.theme_combo.currentData() or "graphite_cobalt")
        self._apply_theme(theme_key)

    def _apply_theme(self, theme_key: str):
        pal = resolve_palette(theme_key)
        self._current_pal = pal
        self.setStyleSheet(build_m3_stylesheet(pal))

        for w in self.findChildren(QWidget):
            if hasattr(w, "set_palette_colors"):
                w.set_palette_colors(pal)

        for k, btn in getattr(self, "swatch_buttons", {}).items():
            btn.blockSignals(True)
            btn.setChecked(k == pal.key)
            btn.blockSignals(False)
            btn.update()

        if hasattr(self, "brand_badge"):
            self.brand_badge.setStyleSheet(
                f"QPushButton {{ background-color: {pal.primary_container}; "
                f"border: none; border-radius: 14px; }} "
                f"QPushButton:hover {{ background-color: {pal.primary}; }}"
            )
            self.brand_badge.setIcon(make_m3_icon("cube", pal.on_primary_container, 22))

        try:
            self.plotter.set_background(pal.vp_bottom, top=pal.vp_top)
            self.plotter.add_axes(color="black" if pal.is_light else "white")
            if self.isVisible():
                self.plotter.render()
        except Exception:
            pass
        if hasattr(self, "splat_viewport"):
            self.splat_viewport.set_background_hex(pal.vp_bottom)

    def _apply_edl_toggle(self, enabled: bool):
        try:
            if enabled:
                self.plotter.enable_eye_dome_lighting()
            else:
                self.plotter.disable_eye_dome_lighting()
            if self.isVisible():
                self.plotter.render()
        except Exception:
            pass

    def _show_rgb_preview(self, rgb: np.ndarray):
        self._last_preview_rgb = rgb
        if hasattr(self, "preview_widget") and self.preview_widget is not None:
            self.preview_widget.set_rgb_frame(rgb)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.suffix.lower() in VIDEO_EXTS or p.suffix.lower() in SCENE_EXTS:
                    event.acceptProposedAction()
                    if hasattr(self, "preview_widget") and self.preview_widget is not None:
                        self.preview_widget.set_drag_hover(True)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        if hasattr(self, "preview_widget") and self.preview_widget is not None:
            self.preview_widget.set_drag_hover(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if hasattr(self, "preview_widget") and self.preview_widget is not None:
            self.preview_widget.set_drag_hover(False)
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file() and (p.suffix.lower() in VIDEO_EXTS or p.suffix.lower() in SCENE_EXTS):
                self.open_input_path(p)
                event.acceptProposedAction()
                return

    def pick_video_file(self):
        dialog = QFileDialog(self, "Open Video or Saved 3D Scene")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, False)
        dialog.setNameFilter(
            "Video & 3D Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.ply *.glb *.gltf *.obj *.npz);;"
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;"
            "3D Scenes (*.ply *.glb *.gltf *.obj *.npz);;All Files (*.*)"
        )
        if dialog.exec():
            files = dialog.selectedFiles()
            if files:
                self.open_input_path(Path(files[0]))

    def open_input_path(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext in VIDEO_EXTS:
            self.set_video_path(path)
        elif ext in SCENE_EXTS:
            self.load_saved_scene_file(path)

    def load_saved_scene_file(self, path: Path) -> None:
        """Open a saved .ply (3DGS or mesh/cloud), .glb/.gltf/.obj mesh, or .npz cache directly."""
        from src.pipeline import ReconstructionResult, load_result_cache
        from src.scene_builder import Scene3D
        from src.video_ingest import VideoData

        ext = path.suffix.lower()
        try:
            if ext == ".npz":
                res = load_result_cache(path)
                if res.video_data.path.exists():
                    self.set_video_path(res.video_data.path)
                self.load_result(res)
                self.stage_pill.setText("Loaded cache")
                self._log_milestone(f"Loaded scene cache '{path.name}'")
                return

            if ext == ".ply" and _safe_is_gsplat_available():
                try:
                    from src.gsplat_engine import load_splats_ply

                    splat_sc, loaded_poses, loaded_k, loaded_fsz = load_splats_ply(path)
                    fsz = loaded_fsz or (1280, 720)
                    k_mat = (
                        loaded_k
                        if loaded_k is not None
                        else np.array([[900.0, 0.0, fsz[0] * 0.5], [0.0, 900.0, fsz[1] * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
                    )
                    xyz_np = splat_sc.means.detach().cpu().numpy().astype(np.float32)
                    scene = Scene3D(
                        points=xyz_np,
                        colors=np.full((len(xyz_np), 3), 180, dtype=np.uint8),
                        normals=np.zeros_like(xyz_np, dtype=np.float32),
                        mesh_vertices=np.zeros((0, 3), dtype=np.float32),
                        mesh_faces=np.zeros((0, 3), dtype=np.int32),
                        mesh_colors=np.zeros((0, 3), dtype=np.uint8),
                        poses=loaded_poses or [],
                        intrinsics=k_mat,
                        frame_size=fsz,
                    )
                    vd = VideoData(path=path, fps=30.0, total_frames=len(loaded_poses or []), duration=0.0, orig_size=fsz, frames=[])
                    res = ReconstructionResult(video_data=vd, scene=scene, splat_scene=splat_sc, splat_ply_path=path)
                    self.load_result(res, exported=[path])
                    self.stage_pill.setText("Loaded 3DGS")
                    self._log_milestone(f"Loaded 3DGS PLY '{path.name}' ({splat_sc.final_splats:,} splats)")
                    return
                except ValueError:
                    pass

            import trimesh

            loaded = trimesh.load(str(path), process=False)
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
                if not geoms:
                    raise ValueError(f"No geometry in {path.name}")
                loaded = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]

            if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
                verts = np.asarray(loaded.vertices, dtype=np.float32)
                faces = np.asarray(loaded.faces, dtype=np.int32)
                m_uvs = None
                m_tex = None
                vcols = np.full((len(verts), 3), 185, dtype=np.uint8)
                vis = getattr(loaded, "visual", None)
                if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
                    mat = getattr(vis, "material", None)
                    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
                    if img is not None:
                        m_tex = np.asarray(img.convert("RGB"), dtype=np.uint8)
                        m_uvs = np.asarray(vis.uv, dtype=np.float32)
                elif getattr(vis, "vertex_colors", None) is not None and len(vis.vertex_colors) == len(verts):
                    vcols = np.asarray(vis.vertex_colors[:, :3], dtype=np.uint8)
                scene = Scene3D(
                    points=verts,
                    colors=vcols,
                    normals=np.asarray(loaded.vertex_normals, dtype=np.float32) if len(loaded.vertex_normals) == len(verts) else np.zeros_like(verts),
                    mesh_vertices=verts,
                    mesh_faces=faces,
                    mesh_colors=vcols,
                    poses=[],
                    intrinsics=np.eye(3, dtype=np.float32),
                    frame_size=(1280, 720),
                    mesh_uvs=m_uvs,
                    mesh_texture=m_tex,
                )
            else:
                pts = np.asarray(getattr(loaded, "vertices", loaded), dtype=np.float32)
                cols = np.full((len(pts), 3), 185, dtype=np.uint8)
                if hasattr(loaded, "colors") and loaded.colors is not None and len(loaded.colors) == len(pts):
                    cols = np.asarray(loaded.colors[:, :3], dtype=np.uint8)
                scene = Scene3D(
                    points=pts,
                    colors=cols,
                    normals=np.zeros_like(pts),
                    mesh_vertices=np.zeros((0, 3), dtype=np.float32),
                    mesh_faces=np.zeros((0, 3), dtype=np.int32),
                    mesh_colors=np.zeros((0, 3), dtype=np.uint8),
                    poses=[],
                    intrinsics=np.eye(3, dtype=np.float32),
                    frame_size=(1280, 720),
                )
            vd = VideoData(path=path, fps=30.0, total_frames=0, duration=0.0, orig_size=(1280, 720), frames=[])
            res = ReconstructionResult(video_data=vd, scene=scene)
            self.load_result(res, exported=[path])
            self.stage_pill.setText("Loaded 3D")
            self._log_milestone(f"Loaded saved 3D scene '{path.name}'")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Could not load {path.name}: {exc}")

    def pick_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.out_dir_edit.text().strip() or "outputs",
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self.out_dir_edit.setText(folder)

    def make_and_load_demo(self):
        out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
        demo_path = out_dir / "demo_courtyard.mp4"
        dur = float(self.demo_sec_spin.value())
        self.stage_pill.setText("Generating demo")
        self.status_lbl.setText(f"Generating {dur:.1f}s 3D courtyard video...")
        self.loading_indicator.start()
        QTimer.singleShot(10, lambda: self._finish_demo_gen(demo_path, dur))

    def _finish_demo_gen(self, demo_path: Path, dur: float):
        from src.video_ingest import generate_demo_video

        generate_demo_video(demo_path, duration_sec=dur, fps=24, width=560, height=360)
        self.loading_indicator.stop()
        self.set_video_path(demo_path)
        self.stage_pill.setText("Ready")
        self.status_lbl.setText(f"Generated {demo_path.name} ({dur:.1f}s)")
        self._log_milestone(f"Generated {dur:.1f}s synthetic 3D video -> {demo_path}")

    def set_video_path(self, path: Path):
        self.current_video = path
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, bgr = cap.read()
        cap.release()

        dur = total / max(fps, 1.0)
        self.current_video_total_frames = total
        if total >= 4:
            self.max_frames_spin.setRange(4, max(4, total))
            self.max_frames_spin.set_slider_max(min(500, max(64, total)))
            self.max_frames_spin.setToolTip(f"Video has {total} frames ({dur:.1f}s)")
        else:
            self.max_frames_spin.setRange(4, 100000)
            self.max_frames_spin.set_slider_max(120)
            self.max_frames_spin.setToolTip("Unknown frame count stream")
        result_matches = (
            self.current_result is not None
            and (self.current_video is None or self.current_result.video_data.path == path)
        )
        if not result_matches and total > 0:
            self.kf_slider.blockSignals(True)
            self.kf_slider.setRange(0, max(0, total - 1))
            self.kf_slider.setValue(0)
            self.kf_slider.blockSignals(False)
            self.kf_lbl.setText(f"1 / {total}")
        self.video_info_lbl.setText(
            f"{path.name} ({dur:.1f}s, {total:,} frames @ {fps:.1f} fps, {w}x{h})"
        )
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._show_rgb_preview(rgb)
        self._log_milestone(f"Loaded video '{path.name}' ({dur:.1f}s, {w}x{h})")

    def _set_all_frames(self):
        if self.current_video_total_frames >= 4:
            self.max_frames_spin.setValue(self.current_video_total_frames)

    def _tick_elapsed_timer(self):
        if self._build_start_t > 0:
            el = time.perf_counter() - self._build_start_t
            self.timer_lbl.setText(f"{el:04.1f}s")

    def start_reconstruction(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.current_video is None or not self.current_video.exists():
            out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
            demo_path = out_dir / "demo_courtyard.mp4"
            dur = float(self.demo_sec_spin.value())
            from src.video_ingest import generate_demo_video

            generate_demo_video(demo_path, duration_sec=dur, fps=24, width=560, height=360)
            self.set_video_path(demo_path)

        self.fly_timer.stop()
        self.fly_btn.setText("Fly")
        self.fly_btn.set_icon_name("fly")
        self._release_active_splat_scene()

        preset = self.gather_ui_preset()
        self._active_build_mode = preset.recon_mode

        self.build_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.demo_btn.setEnabled(False)
        self.prog_bar.setValue(2)
        self.loading_indicator.start()
        self._build_start_t = time.perf_counter()
        self._last_logged_stage = ""
        self.timer_lbl.setText("00.0s")
        self.elapsed_timer.start()

        self.stage_pill.setText("Scanning video")
        self.status_lbl.setText("Scanning video keyframes...")
        self._log_milestone(
            f"Started build on {self.current_video.name} (keyframes={preset.max_frames}, res={preset.max_dim}px)"
        )

        out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
        export_opts = (
            out_dir,
            self.auto_ply_chk.isChecked(),
            self.auto_glb_chk.isChecked(),
            self.auto_obj_chk.isChecked(),
        )

        if preset.recon_mode == "3dgs":
            self.plotter.clear_actors()
            self._set_viewport_mode(True)
            self.splat_viewport.set_preview_image(
                self._last_preview_rgb if self._last_preview_rgb is not None else np.zeros((270, 480, 3), dtype=np.uint8),
                hud_text="Preparing 3DGS views...",
            )

        self.worker = ReconstructWorker(
            self._pipeline,
            self.current_video,
            config=None,
            export_options=export_opts,
            preset=preset,
            start_sec=self._initial_start_sec,
            end_sec=self._initial_end_sec,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.preview_ready.connect(self._on_worker_preview)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_preview(self, rgb: Any):
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and hasattr(self, "splat_viewport"):
            if self.viewport_stack.currentIndex() != 1:
                self.viewport_stack.setCurrentIndex(1)
            self.splat_viewport.set_preview_image(rgb, hud_text=f"Live 3DGS preview  |  {self.status_lbl.text()}")

    def _on_worker_progress(self, frac: float, msg: str):
        self.prog_bar.setValue(int(round(frac * 100)))
        self.status_lbl.setText(msg)

        if frac < 0.28:
            stage_key = "Scanning video"
            milestone_txt = "Scanning video & scoring keyframe sharpness/parallax..."
        elif frac < 0.56:
            stage_key = "Pose tracking"
            milestone_txt = "Matching GPU features, anchor SfM & chained bidirectional 3D-PnP..."
        elif self._active_build_mode == "3dgs":
            if frac < 0.62:
                stage_key = "Decoding 3DGS views"
                milestone_txt = "Decoding high-res training views via PyAV NVDEC..."
            elif frac < 1.0:
                stage_key = "Training 3DGS"
                milestone_txt = "Training 3D Gaussian Splats (AbsGS CUDA) & exporting scene_splats.ply..."
            else:
                stage_key = "Complete"
                milestone_txt = ""
        elif frac < 0.76:
            stage_key = "Depth alignment"
            milestone_txt = "Estimating diverse ViT metric depth & multi-view affine consensus..."
        elif frac < 1.0:
            stage_key = "TSDF fusion"
            milestone_txt = "Integrating high-res TSDF voxels & orienting dollhouse surface mesh..."
        else:
            stage_key = "Complete"
            milestone_txt = ""

        self.stage_pill.setText(stage_key)
        if stage_key != self._last_logged_stage and milestone_txt:
            self._last_logged_stage = stage_key
            self._log_milestone(milestone_txt)

    def request_cancel(self):
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.status_lbl.setText("Cancelling build...")
            self.stage_pill.setText("Cancelling")
            self.worker.request_cancel()
            self.cancel_btn.setEnabled(False)

    def _on_worker_done(
        self,
        result: ReconstructionResult,
        elapsed: float,
        exported: Optional[List[Path]] = None,
    ):
        if self.worker is not None and self.worker.pipeline is not None:
            self._pipeline = self.worker.pipeline
        if exported is None:
            exported = []
        self.loading_indicator.stop()
        self.elapsed_timer.stop()
        self.timer_lbl.setText(f"{elapsed:04.1f}s")
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self.demo_btn.setEnabled(True)
        self.prog_bar.setValue(100)
        self.stage_pill.setText("Complete")

        self.load_result(result, elapsed=elapsed, exported=exported)

        exp_names = ", ".join(p.name for p in exported) if exported else "no auto-exports"
        splat_sc = getattr(result, "splat_scene", None)
        if splat_sc is not None:
            val_msg = (
                f", val {splat_sc.val_psnr_db:.2f} dB"
                if getattr(splat_sc, "val_psnr_db", None) is not None
                else ""
            )
            self._log_milestone(
                f"Completed 3DGS in {elapsed:.1f}s -> {splat_sc.final_splats:,} splats, "
                f"train PSNR {splat_sc.train_psnr_db:.2f} dB{val_msg}, peak VRAM {splat_sc.peak_vram_gb:.2f} GB ({exp_names})"
            )
        else:
            self._log_milestone(
                f"Completed in {elapsed:.1f}s -> {len(result.scene.points):,} pts, "
                f"{len(result.scene.mesh_faces):,} tris ({exp_names})"
            )

    def _on_worker_failed(self, err: str):
        if self.worker is not None and self.worker.pipeline is not None:
            self._pipeline = self.worker.pipeline
        self.loading_indicator.stop()
        self.elapsed_timer.stop()
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self.demo_btn.setEnabled(True)
        if "cancel" in err.lower():
            self.stage_pill.setText("Cancelled")
            self.status_lbl.setText("Build cancelled by user")
            self._log_milestone("Build cancelled by user")
        else:
            self.stage_pill.setText("Error")
            self.status_lbl.setText(f"Failed: {err}")
            self._log_milestone(f"ERROR: {err}")
            QMessageBox.critical(self, "Reconstruction Failed", err)
        prev_splat = getattr(self.current_result, "splat_scene", None) if self.current_result is not None else None
        if prev_splat is not None and _safe_is_gsplat_available():
            self._set_viewport_mode(True)
            self.splat_viewport.set_scene(
                prev_splat,
                self.current_result.scene.poses,
                intrinsics=self.current_result.scene.intrinsics,
                frame_size=self.current_result.scene.frame_size,
            )
        elif self.current_result is not None:
            self._set_viewport_mode(False)
            self.refresh_3d_actors()
        else:
            self.splat_viewport.clear_scene()
            self._set_viewport_mode(False)

    def _release_active_splat_scene(self):
        if hasattr(self, "splat_viewport"):
            self.splat_viewport.clear_scene()
        cur = getattr(self, "current_result", None)
        splat_sc = getattr(cur, "splat_scene", None) if cur is not None else None
        if splat_sc is not None and hasattr(splat_sc, "release"):
            try:
                splat_sc.release()
            except Exception:
                pass

    def _set_viewport_mode(self, is_3dgs: bool):
        self.viewport_stack.setCurrentIndex(1 if is_3dgs else 0)
        tsdf_enabled = not is_3dgs
        for w in (
            self.mode_combo,
            self.dollhouse_chk,
            self.frustums_chk,
            self.pt_slider,
            self.edl_chk,
            self.exp_glb_btn,
            self.exp_obj_btn,
        ):
            w.setEnabled(tsdf_enabled)
        if hasattr(self, "exp_splat_btn"):
            self.exp_splat_btn.setEnabled(is_3dgs)

    def load_result(
        self,
        result: ReconstructionResult,
        elapsed: float = 0.0,
        exported: Optional[List[Path]] = None,
    ):
        prev_splat = (
            getattr(self.current_result, "splat_scene", None)
            if self.current_result is not None
            else None
        )
        new_splat = getattr(result, "splat_scene", None)
        if prev_splat is not None and prev_splat is not new_splat:
            self._release_active_splat_scene()

        self.current_result = result
        n_kf = len(result.video_data.frames)
        self.kf_slider.blockSignals(True)
        self.kf_slider.setRange(0, max(0, n_kf - 1))
        self.kf_slider.setValue(0)
        self.kf_slider.blockSignals(False)
        self.kf_lbl.setText(f"1 / {n_kf}")

        if len(result.scene.points) > 10:
            p5 = np.percentile(result.scene.points, 5, axis=0)
            p95 = np.percentile(result.scene.points, 95, axis=0)
            span = float(np.linalg.norm(p95 - p5))
            self._scene_frustum_scale = float(np.clip(span * 0.030, 0.04, 0.14))
        else:
            self._scene_frustum_scale = 0.10

        if n_kf > 0:
            self._show_rgb_preview(result.video_data.frames[0].rgb)

        self._update_telemetry_card(result, elapsed=elapsed, exported=exported)

        if new_splat is not None:
            self.status_lbl.setText(
                f"{new_splat.final_splats:,} 3DGS splats, train PSNR {new_splat.train_psnr_db:.2f} dB, {n_kf} keyframes"
            )
            self.plotter.clear_actors()
            self._set_viewport_mode(True)
            self.splat_viewport.set_scene(
                new_splat,
                result.scene.poses,
                intrinsics=result.scene.intrinsics,
                frame_size=result.scene.frame_size,
            )
            self.splat_viewport.setFocus()
        else:
            self.status_lbl.setText(
                f"{len(result.scene.points):,} 3D pts, {len(result.scene.mesh_faces):,} tris, {n_kf} keyframes"
            )
            self.splat_viewport.clear_scene()
            self._set_viewport_mode(False)
            self.refresh_3d_actors()
            self.reset_overview_camera()

    def _update_telemetry_card(
        self,
        result: ReconstructionResult,
        elapsed: float = 0.0,
        exported: Optional[List[Path]] = None,
    ):
        vd = result.video_data
        sc = result.scene
        valid_poses = [p for p in sc.poses if p.valid]
        if len(valid_poses) >= 2:
            centers = np.stack([p.t_wc for p in valid_poses], axis=0)
            path_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
        else:
            path_len = 0.0

        exp_str = ", ".join(p.name for p in exported) if exported else "none"
        splat_sc = getattr(result, "splat_scene", None)
        if splat_sc is not None:
            ply_p = getattr(result, "splat_ply_path", None)
            if ply_p is not None and exp_str == "none":
                exp_str = Path(ply_p).name
            val_str = (
                f" | val {splat_sc.val_psnr_db:.2f} dB"
                if getattr(splat_sc, "val_psnr_db", None) is not None
                else ""
            )
            text = (
                f"Video Source   : {vd.path.name} (n = {vd.duration:.1f}s, {vd.total_frames} frames)\n"
                f"Keyframes Used : {len(valid_poses)}/{len(vd.frames)} aligned @ {sc.frame_size[0]}x{sc.frame_size[1]}px\n"
                f"3DGS Splats    : {splat_sc.init_splats:,} -> {splat_sc.final_splats:,} (SH{splat_sc.sh_degree})\n"
                f"Render Quality : train {splat_sc.train_psnr_db:.2f} dB{val_str} | peak VRAM {splat_sc.peak_vram_gb:.2f} GB\n"
                f"Camera Path    : {len(valid_poses)} poses (span = {path_len:.2f} m)\n"
                f"Build Time     : {elapsed:.2f} s (train {splat_sc.train_s:.2f} s)\n"
                f"Auto-Exported  : {exp_str}"
            )
        else:
            text = (
                f"Video Source   : {vd.path.name} (n = {vd.duration:.1f}s, {vd.total_frames} frames)\n"
                f"Keyframes Used : {len(valid_poses)}/{len(vd.frames)} aligned @ {sc.frame_size[0]}x{sc.frame_size[1]}px\n"
                f"3D Point Cloud : {len(sc.points):,} fused RGB points\n"
                f"Surface Mesh   : {len(sc.mesh_faces):,} tris ({len(sc.mesh_vertices):,} verts)\n"
                f"Camera Path    : {len(valid_poses)} poses (span = {path_len:.2f} m)\n"
                f"Build Time     : {elapsed:.2f} s\n"
                f"Auto-Exported  : {exp_str}"
            )
        self.telemetry_lbl.setText(text)

    def refresh_3d_actors(self):
        if self.current_result is None or getattr(self.current_result, "splat_scene", None) is not None:
            return

        scene = self.current_result.scene
        mode = self.mode_combo.currentIndex()
        dpr = max(1.0, float(self.devicePixelRatioF()))
        pt_size = max(3.5, float(self.pt_slider.value()) * dpr * 1.15)

        self.plotter.clear_actors()
        axes_col = (
            "black"
            if getattr(self, "_current_pal", None) and self._current_pal.is_light
            else "white"
        )
        self.plotter.add_axes(color=axes_col)

        if mode in (0, 2) and len(scene.points) > 0:
            cloud = pv.PolyData(scene.points)
            cloud["rgb"] = scene.colors
            self.plotter.add_mesh(
                cloud,
                scalars="rgb",
                rgb=True,
                lighting=False,
                point_size=pt_size,
                render_points_as_spheres=True,
                name="point_cloud",
            )

        if mode in (1, 2) and len(scene.mesh_faces) > 0:
            n_f = len(scene.mesh_faces)
            vtk_faces = np.hstack(
                [np.full((n_f, 1), 3, dtype=np.int64), scene.mesh_faces.astype(np.int64)]
            ).ravel()
            mesh = pv.PolyData(scene.mesh_vertices, vtk_faces)
            if (
                scene.mesh_uvs is not None
                and scene.mesh_texture is not None
                and len(scene.mesh_uvs) == len(scene.mesh_vertices)
            ):
                mesh.active_texture_coordinates = scene.mesh_uvs.astype(np.float32)
                tex = pv.numpy_to_texture(np.ascontiguousarray(scene.mesh_texture))
                tex.interpolate = True
                tex.mipmap = True
                self.plotter.add_mesh(
                    mesh,
                    texture=tex,
                    lighting=True,
                    ambient=0.86,
                    diffuse=0.14,
                    specular=0.0,
                    smooth_shading=True,
                    backface_culling=self.dollhouse_chk.isChecked(),
                    opacity=0.92 if mode == 2 else 1.0,
                    name="surface_mesh",
                )
            else:
                mesh["rgb"] = scene.mesh_colors
                self.plotter.add_mesh(
                    mesh,
                    scalars="rgb",
                    rgb=True,
                    lighting=True,
                    ambient=0.84,
                    diffuse=0.14,
                    specular=0.0,
                    smooth_shading=True,
                    backface_culling=self.dollhouse_chk.isChecked(),
                    opacity=0.92 if mode == 2 else 1.0,
                    name="surface_mesh",
                )

        if self.frustums_chk.isChecked() and len(scene.poses) > 0:
            from src.scene_builder import WORLD_FLIP, build_frustum_lines

            valid_poses = [p for p in scene.poses if p.valid]
            frust_step = max(1, len(valid_poses) // 24)
            all_f_pts = []
            all_f_lines = []
            pt_offset = 0

            for p_i, pose in enumerate(valid_poses):
                if p_i % frust_step == 0:
                    f_pts, f_lines = build_frustum_lines(
                        pose, scene.frame_size, scene.intrinsics, scale=self._scene_frustum_scale
                    )
                    lines_shifted = f_lines.copy()
                    lines_shifted[1::3] += pt_offset
                    lines_shifted[2::3] += pt_offset
                    all_f_pts.append(f_pts)
                    all_f_lines.append(lines_shifted)
                    pt_offset += len(f_pts)

            if all_f_pts:
                frust_poly = pv.PolyData()
                frust_poly.points = np.vstack(all_f_pts)
                frust_poly.lines = np.concatenate(all_f_lines)
                self.plotter.add_mesh(
                    frust_poly,
                    color="#00f5ff",
                    line_width=1.8,
                    lighting=False,
                    name="camera_frustums",
                )

            valid_poses_idx = [i for i, p in enumerate(scene.poses) if p.valid]
            if len(valid_poses_idx) >= 2:
                consec = [
                    float(
                        np.linalg.norm(
                            scene.poses[valid_poses_idx[k + 1]].t_wc
                            - scene.poses[valid_poses_idx[k]].t_wc
                        )
                    )
                    for k in range(len(valid_poses_idx) - 1)
                ]
                med_step = float(np.median(consec)) if consec else 0.15
            else:
                med_step = 0.15
            max_seg_step = max(0.65, 2.5 * med_step)

            valid_segs = []
            pts_list = []
            for k in range(len(scene.poses) - 1):
                p1 = scene.poses[k]
                p2 = scene.poses[k + 1]
                if p1.valid and p2.valid:
                    c1 = WORLD_FLIP @ p1.t_wc.astype(np.float32)
                    c2 = WORLD_FLIP @ p2.t_wc.astype(np.float32)
                    if np.linalg.norm(c2 - c1) <= max_seg_step:
                        idx_base = len(pts_list)
                        pts_list.append(c1)
                        pts_list.append(c2)
                        valid_segs.extend([2, idx_base, idx_base + 1])

            if pts_list:
                path_poly = pv.PolyData()
                path_poly.points = np.array(pts_list, dtype=np.float32)
                path_poly.lines = np.array(valid_segs, dtype=np.int32)
                self.plotter.add_mesh(
                    path_poly,
                    color="#ff2a85",
                    line_width=3.2,
                    lighting=False,
                    name="camera_path",
                )

            self._update_active_frustum_highlight()

        if self.isVisible():
            self.plotter.render()

    def _update_active_frustum_highlight(self):
        if self.current_result is None or not self.frustums_chk.isChecked():
            return
        from src.scene_builder import build_frustum_lines

        scene = self.current_result.scene
        idx = int(self.kf_slider.value())
        if 0 <= idx < len(scene.poses):
            f_pts, f_lines = build_frustum_lines(
                scene.poses[idx],
                scene.frame_size,
                scene.intrinsics,
                scale=self._scene_frustum_scale * 1.35,
            )
            active_poly = pv.PolyData()
            active_poly.points = f_pts
            active_poly.lines = f_lines
            self.plotter.add_mesh(
                active_poly,
                color="#ffe600",
                line_width=4.0,
                lighting=False,
                name="active_frustum",
            )

    def on_keyframe_scrub(self, idx: int, render: bool = True):
        if self.current_result is not None and (
            self.current_video is None or self.current_result.video_data.path == self.current_video
        ):
            frames = self.current_result.video_data.frames
            if 0 <= idx < len(frames):
                self.kf_lbl.setText(f"{idx + 1} / {len(frames)}")
                self._show_rgb_preview(frames[idx].rgb)
                if self.viewport_stack.currentIndex() == 0:
                    self._update_active_frustum_highlight()
                    if render:
                        self.plotter.render()
        elif (
            self.current_video is not None
            and self.current_video.exists()
            and self.current_video_total_frames > 0
        ):
            self.kf_lbl.setText(f"{idx + 1} / {self.current_video_total_frames}")
            self._pending_scrub_idx = idx
            self._scrub_timer.start()

    def _read_raw_video_preview_frame(self):
        if self.current_video is None or not self.current_video.exists():
            return
        cap = cv2.VideoCapture(str(self.current_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self._pending_scrub_idx))
        ok, bgr = cap.read()
        cap.release()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._show_rgb_preview(rgb)

    def _splat_look_dist(self) -> float:
        splat_sc = getattr(self.current_result, "splat_scene", None) if self.current_result is not None else None
        sc = float(getattr(splat_sc, "scene_scale", 1.0)) if splat_sc is not None else 1.0
        return float(np.clip(sc * 0.85, 0.4, 2.5))

    def snap_to_selected_keyframe(self):
        if self.current_result is None:
            return
        idx = int(self.kf_slider.value())
        poses = self.current_result.scene.poses
        if 0 <= idx < len(poses):
            if self.viewport_stack.currentIndex() == 1 and getattr(self.current_result, "splat_scene", None) is not None:
                self.splat_viewport.camera.set_from_pose(poses[idx], look_dist=self._splat_look_dist())
                self.splat_viewport.render_full()
                return
            from src.scene_builder import pose_to_world_camera

            cam_pos, focal, up = pose_to_world_camera(poses[idx], look_dist=2.5)
            self.plotter.camera.position = tuple(cam_pos.tolist())
            self.plotter.camera.focal_point = tuple(focal.tolist())
            self.plotter.camera.up = tuple(up.tolist())
            self.plotter.reset_camera_clipping_range()
            if self.isVisible():
                self.plotter.render()

    def reset_overview_camera(self):
        if self.viewport_stack.currentIndex() == 1 and self.current_result is not None:
            splat_sc = getattr(self.current_result, "splat_scene", None)
            if splat_sc is not None:
                means_np = splat_sc.means.detach().cpu().numpy() if hasattr(splat_sc, "means") else None
                self.splat_viewport.camera.set_overview(
                    means_np,
                    self.current_result.scene.poses,
                    intrinsics=self.current_result.scene.intrinsics,
                    frame_size=self.current_result.scene.frame_size,
                )
                self.splat_viewport.render_full()
                return
        if self.current_result is None or len(self.current_result.scene.points) == 0:
            self.plotter.reset_camera()
            return
        from src.scene_builder import pose_to_world_camera

        scene = self.current_result.scene

        p5 = np.percentile(scene.points, 5, axis=0)
        p95 = np.percentile(scene.points, 95, axis=0)
        center = 0.5 * (p5 + p95)
        span = float(np.linalg.norm(p95 - p5))
        radius = max(1.2, span * 0.5)

        if scene.poses:
            cam0, focal0, _ = pose_to_world_camera(scene.poses[0], look_dist=2.5)
            horiz_fwd = focal0 - cam0
        else:
            horiz_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        horiz_fwd[1] = 0.0
        if np.linalg.norm(horiz_fwd) < 1e-4:
            horiz_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            horiz_fwd = horiz_fwd / np.linalg.norm(horiz_fwd)

        overview_pos = (
            center
            - horiz_fwd * (radius * 2.35)
            + np.array([0.0, radius * 1.15, 0.0], dtype=np.float32)
        )
        self.plotter.camera.position = tuple(overview_pos.tolist())
        self.plotter.camera.focal_point = tuple(center.tolist())
        self.plotter.camera.up = (0.0, 1.0, 0.0)
        self.plotter.reset_camera_clipping_range()
        if self.isVisible():
            self.plotter.render()

    def toggle_flythrough(self):
        if self.current_result is None or len(self.current_result.scene.poses) < 2:
            return
        if self.fly_timer.isActive():
            self.fly_timer.stop()
            self.fly_btn.setText("Fly")
            self.fly_btn.set_icon_name("fly")
            if self.viewport_stack.currentIndex() == 1:
                self.splat_viewport.render_full()
        else:
            self._last_fly_kf = -1
            self.fly_t = 0.0
            self.fly_btn.setText("Stop Fly")
            self.fly_btn.set_icon_name("stop")
            self.fly_timer.start()

    def _step_flythrough(self):
        if self.current_result is None:
            self.fly_timer.stop()
            return
        poses = self.current_result.scene.poses
        n = len(poses)
        if n < 2:
            self.fly_timer.stop()
            return

        fly_step = max(0.04, float(n) / 450.0)
        self.fly_t += fly_step
        if self.fly_t >= n - 1:
            self.fly_t = 0.0

        i0 = int(np.floor(self.fly_t))
        i1 = min(n - 1, i0 + 1)
        alpha = float(self.fly_t - i0)

        from src.scene_builder import pose_to_world_camera

        p0, f0, u0 = pose_to_world_camera(poses[i0], look_dist=2.5)
        p1, f1, u1 = pose_to_world_camera(poses[i1], look_dist=2.5)

        pos = (1.0 - alpha) * p0 + alpha * p1
        foc = (1.0 - alpha) * f0 + alpha * f1
        up = (1.0 - alpha) * u0 + alpha * u1
        up_norm = up / max(1e-6, float(np.linalg.norm(up)))

        if i0 != self._last_fly_kf:
            self._last_fly_kf = i0
            self.kf_slider.blockSignals(True)
            self.kf_slider.setValue(i0)
            self.kf_slider.blockSignals(False)
            self.on_keyframe_scrub(i0, render=False)

        if self.viewport_stack.currentIndex() == 1 and getattr(self.current_result, "splat_scene", None) is not None:
            look_d = self._splat_look_dist()
            p0_c = np.asarray(poses[i0].t_wc, dtype=np.float32)
            p1_c = np.asarray(poses[i1].t_wc, dtype=np.float32)
            f0_c = np.asarray(poses[i0].r_wc, dtype=np.float32) @ np.array([0.0, 0.0, look_d], dtype=np.float32) + p0_c
            f1_c = np.asarray(poses[i1].r_wc, dtype=np.float32) @ np.array([0.0, 0.0, look_d], dtype=np.float32) + p1_c
            eye_c = (1.0 - alpha) * p0_c + alpha * p1_c
            foc_c = (1.0 - alpha) * f0_c + alpha * f1_c
            self.splat_viewport.camera._set_eye_target(eye_c, foc_c)
            self.splat_viewport.render_interactive()
            return

        self.plotter.camera.position = tuple(pos.tolist())
        self.plotter.camera.focal_point = tuple(foc.tolist())
        self.plotter.camera.up = tuple(up_norm.tolist())
        self.plotter.reset_camera_clipping_range()
        if self.isVisible():
            self.plotter.render()

    def export_current(self, ext: str, as_mesh: bool):
        if self.current_result is None:
            QMessageBox.information(self, "Nothing to Export", "Build a 3D scene first.")
            return
        out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
        default_name = f"scene_export{ext}"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {ext.upper()}",
            str(out_dir / default_name),
            f"3D File (*{ext})",
        )
        if path_str:
            splat_sc = getattr(self.current_result, "splat_scene", None)
            if splat_sc is not None and ext.lower() == ".splat":
                from src.gsplat_engine import export_splats_dot_splat

                out = export_splats_dot_splat(splat_sc, Path(path_str))
            elif splat_sc is not None and ext.lower() == ".ply":
                from src.gsplat_engine import export_splats_ply

                out = export_splats_ply(
                    splat_sc,
                    Path(path_str),
                    poses=self.current_result.scene.poses,
                    intrinsics=self.current_result.scene.intrinsics,
                    frame_size=self.current_result.scene.frame_size,
                )
            else:
                from src.scene_builder import export_scene

                out = export_scene(self.current_result.scene, Path(path_str), export_mesh=as_mesh)
            self.status_lbl.setText(f"Saved {out.name}")
            self._log_milestone(f"Manual export -> {out}")

    def export_snapshot_png(self, target_path: Optional[Path] = None) -> Optional[Path]:
        """Save the current 3D viewport view as a full-resolution PNG image."""
        if self.current_result is None:
            if target_path is None:
                QMessageBox.information(self, "Nothing to Snapshot", "Build or load a 3D scene first.")
            return None
        out_path = Path(target_path) if target_path is not None else None
        if out_path is None:
            out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save Viewport Snapshot",
                str(out_dir / "snapshot.png"),
                "PNG Image (*.png)",
            )
            if not path_str:
                return None
            out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        splat_sc = getattr(self.current_result, "splat_scene", None)
        if self.viewport_stack.currentIndex() == 1 and splat_sc is not None and _safe_is_gsplat_available():
            from src.gsplat_engine import render_view

            if self.splat_viewport.isVisible() and self.splat_viewport.width() >= 200 and self.splat_viewport.height() >= 150:
                aspect = float(self.splat_viewport.width()) / float(self.splat_viewport.height())
            else:
                fsz = getattr(self.current_result.scene, "frame_size", (1280, 720))
                aspect = float(max(1, int(fsz[0]))) / float(max(1, int(fsz[1])))
            rw = 1600
            rh = max(128, int(round(rw / max(0.25, aspect))) // 2 * 2)
            rgb = render_view(
                splat_sc,
                c2w=self.splat_viewport.camera.get_c2w(),
                K=self.splat_viewport.camera.get_intrinsics(rw, rh),
                width=rw,
                height=rh,
            )
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            self.plotter.screenshot(filename=str(out_path))

        self.status_lbl.setText(f"Saved snapshot {out_path.name}")
        self._log_milestone(f"Saved PNG snapshot -> {out_path}")
        return out_path

    def export_turntable_mp4(
        self,
        target_path: Optional[Path] = None,
        n_frames: int = 90,
        fps: int = 30,
    ) -> Optional[Path]:
        """Render a smooth 360-degree orbit video around the current camera target."""
        if self.current_result is None:
            if target_path is None:
                QMessageBox.information(self, "Nothing to Export", "Build or load a 3D scene first.")
            return None
        out_path = Path(target_path) if target_path is not None else None
        if out_path is None:
            out_dir = Path(self.out_dir_edit.text().strip() or "outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Export 360 Turntable Video",
                str(out_dir / "turntable.mp4"),
                "MP4 Video (*.mp4)",
            )
            if not path_str:
                return None
            out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_frames = max(12, int(n_frames))
        fps = max(10, int(fps))
        splat_sc = getattr(self.current_result, "splat_scene", None)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.viewport_stack.currentIndex() == 1 and splat_sc is not None and _safe_is_gsplat_available():
            from src.gsplat_engine import render_view

            rw, rh = 1280, 720
            cam = self.splat_viewport.camera
            orig_yaw = float(cam.yaw)
            orig_pitch = float(cam.pitch)
            writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (rw, rh))
            try:
                for k in range(n_frames):
                    cam.yaw = orig_yaw + (2.0 * np.pi * k / float(n_frames))
                    cam._clamp_above_floor(adjust_target=False)
                    k_mat = cam.get_intrinsics(rw, rh)
                    rgb = render_view(splat_sc, c2w=cam.get_c2w(), K=k_mat, width=rw, height=rh)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if k % 15 == 0:
                        self.status_lbl.setText(f"Rendering turntable... ({k + 1}/{n_frames})")
                        QApplication.processEvents()
            finally:
                cam.yaw = orig_yaw
                cam.pitch = orig_pitch
                writer.release()
            self.splat_viewport.render_full()
        else:
            pos0 = np.array(self.plotter.camera.position, dtype=np.float64)
            foc0 = np.array(self.plotter.camera.focal_point, dtype=np.float64)
            up0 = np.array(self.plotter.camera.up, dtype=np.float64)
            rel0 = pos0 - foc0
            first_img = self.plotter.screenshot(return_img=True)
            rh, rw = int(first_img.shape[0]), int(first_img.shape[1])
            writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (rw, rh))
            try:
                for k in range(n_frames):
                    ang = 2.0 * np.pi * k / float(n_frames)
                    ca, sa = float(np.cos(ang)), float(np.sin(ang))
                    rx = rel0[0] * ca + rel0[2] * sa
                    rz = -rel0[0] * sa + rel0[2] * ca
                    self.plotter.camera.position = tuple((foc0 + np.array([rx, rel0[1], rz])).tolist())
                    self.plotter.reset_camera_clipping_range()
                    img = self.plotter.screenshot(return_img=True)
                    writer.write(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
                    if k % 15 == 0:
                        self.status_lbl.setText(f"Rendering turntable... ({k + 1}/{n_frames})")
                        QApplication.processEvents()
            finally:
                self.plotter.camera.position = tuple(pos0.tolist())
                self.plotter.camera.focal_point = tuple(foc0.tolist())
                self.plotter.camera.up = tuple(up0.tolist())
                self.plotter.reset_camera_clipping_range()
                if self.isVisible():
                    self.plotter.render()
                writer.release()

        self.status_lbl.setText(f"Saved turntable {out_path.name}")
        self._log_milestone(f"Saved 360 turntable ({n_frames}f @ {fps}fps) -> {out_path}")
        return out_path

    def closeEvent(self, event):
        self.elapsed_timer.stop()
        self.fly_timer.stop()
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            try:
                self.worker.request_cancel()
                self.worker.wait(5000)
            except Exception:
                pass
        self._release_active_splat_scene()
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
