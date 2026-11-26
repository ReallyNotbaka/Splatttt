from dataclasses import dataclass
import math
from typing import Dict, List, Optional
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class M3Palette:
    key: str
    label: str
    swatch: str
    is_light: bool
    surface: str
    surface_container_lowest: str
    surface_container_low: str
    surface_container: str
    surface_container_high: str
    surface_container_highest: str
    on_surface: str
    on_surface_variant: str
    outline: str
    outline_variant: str
    primary: str
    on_primary: str
    primary_container: str
    on_primary_container: str
    secondary: str
    secondary_container: str
    on_secondary_container: str
    tertiary: str
    tertiary_container: str
    on_tertiary_container: str
    error: str
    on_error: str
    error_container: str
    on_error_container: str
    vp_bottom: str
    vp_top: str


M3_PALETTES: Dict[str, M3Palette] = {
    "graphite_cobalt": M3Palette(
        key="graphite_cobalt",
        label="Graphite Cobalt",
        swatch="#8AB4F8",
        is_light=False,
        surface="#101216",
        surface_container_lowest="#0B0D11",
        surface_container_low="#161920",
        surface_container="#1C2029",
        surface_container_high="#252A36",
        surface_container_highest="#303645",
        on_surface="#E6E9F0",
        on_surface_variant="#A9B1C2",
        outline="#7D8597",
        outline_variant="#343B4A",
        primary="#8AB4F8",
        on_primary="#062E6F",
        primary_container="#1D3A6D",
        on_primary_container="#D3E3FD",
        secondary="#B8C4DC",
        secondary_container="#2A3347",
        on_secondary_container="#D7E2F8",
        tertiary="#86D6CB",
        tertiary_container="#1B3E3A",
        on_tertiary_container="#BCECE5",
        error="#FFB4AB",
        on_error="#690005",
        error_container="#5F1C22",
        on_error_container="#FFDAD6",
        vp_bottom="#0C0E13",
        vp_top="#19202D",
    ),
    "warm_terracotta": M3Palette(
        key="warm_terracotta",
        label="Warm Terracotta",
        swatch="#F0B095",
        is_light=False,
        surface="#161311",
        surface_container_lowest="#100D0C",
        surface_container_low="#1E1A17",
        surface_container="#26211D",
        surface_container_high="#312B26",
        surface_container_highest="#3D3630",
        on_surface="#EFE6DF",
        on_surface_variant="#C3B5AA",
        outline="#93857A",
        outline_variant="#463D36",
        primary="#F0B095",
        on_primary="#4E200E",
        primary_container="#693420",
        on_primary_container="#FFDBC9",
        secondary="#D6C0B4",
        secondary_container="#463932",
        on_secondary_container="#F1E1D8",
        tertiary="#C8D49E",
        tertiary_container="#3A4429",
        on_tertiary_container="#E2ECC1",
        error="#FFB4AB",
        on_error="#690005",
        error_container="#5F1C22",
        on_error_container="#FFDAD6",
        vp_bottom="#110E0C",
        vp_top="#251D19",
    ),
    "forest_mineral": M3Palette(
        key="forest_mineral",
        label="Forest Mineral",
        swatch="#8ED5A8",
        is_light=False,
        surface="#101512",
        surface_container_lowest="#0B0F0D",
        surface_container_low="#161D19",
        surface_container="#1D2520",
        surface_container_high="#27312B",
        surface_container_highest="#323E37",
        on_surface="#E4ECE6",
        on_surface_variant="#AEBDB3",
        outline="#7D8E83",
        outline_variant="#344239",
        primary="#8ED5A8",
        on_primary="#09381F",
        primary_container="#1C5133",
        on_primary_container="#C2F2D2",
        secondary="#B3C8B9",
        secondary_container="#2C3B32",
        on_secondary_container="#D3E7D9",
        tertiary="#95D0E8",
        tertiary_container="#1E3C49",
        on_tertiary_container="#C5E9F8",
        error="#FFB4AB",
        on_error="#690005",
        error_container="#5F1C22",
        on_error_container="#FFDAD6",
        vp_bottom="#0B100D",
        vp_top="#18241E",
    ),
    "alabaster_light": M3Palette(
        key="alabaster_light",
        label="Alabaster Slate",
        swatch="#1B5E9E",
        is_light=True,
        surface="#EFF2F6",
        surface_container_lowest="#FFFFFF",
        surface_container_low="#F8FAFC",
        surface_container="#E9EEF4",
        surface_container_high="#DFE6EE",
        surface_container_highest="#D1DAE5",
        on_surface="#151B24",
        on_surface_variant="#4B5668",
        outline="#758296",
        outline_variant="#C8D2DE",
        primary="#1B5E9E",
        on_primary="#FFFFFF",
        primary_container="#D3E4FF",
        on_primary_container="#001D38",
        secondary="#4C5E74",
        secondary_container="#D7E3F3",
        on_secondary_container="#12263C",
        tertiary="#196A62",
        tertiary_container="#CBECE7",
        on_tertiary_container="#002E29",
        error="#BA1A1A",
        on_error="#FFFFFF",
        error_container="#F9DEDC",
        on_error_container="#410E0B",
        vp_bottom="#DCE3EC",
        vp_top="#F3F6FA",
    ),
}

THEME_ALIASES: Dict[str, str] = {
    "cyber_blue": "graphite_cobalt",
    "obsidian_indigo": "graphite_cobalt",
    "neon_ember": "warm_terracotta",
    "terracotta_ember": "warm_terracotta",
    "matrix_green": "forest_mineral",
    "sage_botanical": "forest_mineral",
    "lavender_light": "alabaster_light",
}


def resolve_palette(theme_key: str) -> M3Palette:
    norm = THEME_ALIASES.get(theme_key, theme_key)
    return M3_PALETTES.get(norm, M3_PALETTES["graphite_cobalt"])


def _blend_hex(c1: str, c2: str, t: float) -> str:
    q1, q2 = QColor(c1), QColor(c2)
    r = int(round(q1.red() * (1.0 - t) + q2.red() * t))
    g = int(round(q1.green() * (1.0 - t) + q2.green() * t))
    b = int(round(q1.blue() * (1.0 - t) + q2.blue() * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_m3_icon(p: QPainter, name: str, rect: QRectF, color: QColor, stroke: float = 1.8):
    """Crisp Material Symbols Rounded vector icon renderer (24x24 normalized grid)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    side = min(rect.width(), rect.height())
    ox = rect.x() + (rect.width() - side) * 0.5
    oy = rect.y() + (rect.height() - side) * 0.5
    sc = side / 24.0
    p.translate(ox, oy)
    p.scale(sc, sc)

    pen = QPen(color, stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    if name == "studio":
        p.drawRoundedRect(QRectF(3, 5, 18, 14), 3, 3)
        tri = QPainterPath()
        tri.moveTo(10.2, 9.2)
        tri.lineTo(15.2, 12.0)
        tri.lineTo(10.2, 14.8)
        tri.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawPath(tri)
    elif name == "tune":
        for y, kx in ((7.0, 9.0), (12.0, 15.5), (17.0, 11.0)):
            p.drawLine(QPointF(4, y), QPointF(kx - 2.3, y))
            p.drawLine(QPointF(kx + 2.3, y), QPointF(20, y))
            p.drawEllipse(QPointF(kx, y), 2.1, 2.1)
    elif name == "cube":
        pts = [
            QPointF(12, 3),
            QPointF(20, 7.5),
            QPointF(20, 16.5),
            QPointF(12, 21),
            QPointF(4, 16.5),
            QPointF(4, 7.5),
        ]
        path = QPainterPath()
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(QPointF(4, 7.5), QPointF(12, 12))
        p.drawLine(QPointF(20, 7.5), QPointF(12, 12))
        p.drawLine(QPointF(12, 12), QPointF(12, 21))
    elif name == "activity":
        path = QPainterPath()
        path.moveTo(3, 13)
        path.lineTo(7.5, 13)
        path.lineTo(10, 6.5)
        path.lineTo(14, 17.5)
        path.lineTo(16.5, 13)
        path.lineTo(21, 13)
        p.drawPath(path)
    elif name == "play":
        tri = QPainterPath()
        tri.moveTo(7.5, 5.5)
        tri.lineTo(18.5, 12.0)
        tri.lineTo(7.5, 18.5)
        tri.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawPath(tri)
    elif name == "stop":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2.5, 2.5)
    elif name == "folder":
        path = QPainterPath()
        path.moveTo(3.5, 7.5)
        path.lineTo(3.5, 17.5)
        path.quadTo(3.5, 19, 5, 19)
        path.lineTo(19, 19)
        path.quadTo(20.5, 19, 20.5, 17.5)
        path.lineTo(20.5, 9.5)
        path.quadTo(20.5, 8, 19, 8)
        path.lineTo(11.5, 8)
        path.lineTo(9.5, 6)
        path.lineTo(5, 6)
        path.quadTo(3.5, 6, 3.5, 7.5)
        p.drawPath(path)
    elif name == "upload":
        p.drawLine(QPointF(12, 15.5), QPointF(12, 5.5))
        p.drawLine(QPointF(8.2, 9.3), QPointF(12, 5.5))
        p.drawLine(QPointF(15.8, 9.3), QPointF(12, 5.5))
        path = QPainterPath()
        path.moveTo(5, 15.5)
        path.lineTo(5, 17.8)
        path.quadTo(5, 19, 6.2, 19)
        path.lineTo(17.8, 19)
        path.quadTo(19, 19, 19, 17.8)
        path.lineTo(19, 15.5)
        p.drawPath(path)
    elif name == "sparkle":
        path = QPainterPath()
        path.moveTo(12, 3.5)
        path.quadTo(13.2, 10.8, 20.5, 12)
        path.quadTo(13.2, 13.2, 12, 20.5)
        path.quadTo(10.8, 13.2, 3.5, 12)
        path.quadTo(10.8, 10.8, 12, 3.5)
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawPath(path)
    elif name == "overview":
        p.drawLine(QPointF(4, 9), QPointF(4, 5))
        p.drawLine(QPointF(4, 5), QPointF(8, 5))
        p.drawLine(QPointF(16, 5), QPointF(20, 5))
        p.drawLine(QPointF(20, 5), QPointF(20, 9))
        p.drawLine(QPointF(4, 15), QPointF(4, 19))
        p.drawLine(QPointF(4, 19), QPointF(8, 19))
        p.drawLine(QPointF(16, 19), QPointF(20, 19))
        p.drawLine(QPointF(20, 19), QPointF(20, 15))
        p.drawEllipse(QPointF(12, 12), 2.5, 2.5)
    elif name == "camera":
        p.drawRoundedRect(QRectF(3.5, 7, 17, 12), 3, 3)
        p.drawLine(QPointF(8, 7), QPointF(9.5, 4.8))
        p.drawLine(QPointF(9.5, 4.8), QPointF(14.5, 4.8))
        p.drawLine(QPointF(14.5, 4.8), QPointF(16, 7))
        p.drawEllipse(QPointF(12, 13), 3.1, 3.1)
    elif name == "fly":
        path = QPainterPath()
        path.moveTo(4.5, 11.5)
        path.lineTo(19.5, 5.0)
        path.lineTo(13.2, 19.5)
        path.lineTo(11.0, 13.2)
        path.closeSubpath()
        p.drawPath(path)
    elif name == "export":
        p.drawLine(QPointF(12, 5.5), QPointF(12, 15.5))
        p.drawLine(QPointF(8.2, 11.7), QPointF(12, 15.5))
        p.drawLine(QPointF(15.8, 11.7), QPointF(12, 15.5))
        p.drawLine(QPointF(5, 19), QPointF(19, 19))
    elif name == "save":
        p.drawRoundedRect(QRectF(4.5, 4.5, 15, 15), 2.5, 2.5)
        p.drawRect(QRectF(8, 4.5, 8, 5))
        p.drawRect(QRectF(7.5, 13, 9, 6.5))
    elif name == "trash":
        p.drawLine(QPointF(5, 7), QPointF(19, 7))
        p.drawLine(QPointF(9, 7), QPointF(9.8, 4.8))
        p.drawLine(QPointF(9.8, 4.8), QPointF(14.2, 4.8))
        p.drawLine(QPointF(14.2, 4.8), QPointF(15, 7))
        p.drawRoundedRect(QRectF(6.5, 7, 11, 12.5), 2, 2)
        p.drawLine(QPointF(10, 10.5), QPointF(10, 16))
        p.drawLine(QPointF(14, 10.5), QPointF(14, 16))
    elif name == "minus":
        p.drawLine(QPointF(6.5, 12), QPointF(17.5, 12))
    elif name == "plus":
        p.drawLine(QPointF(6.5, 12), QPointF(17.5, 12))
        p.drawLine(QPointF(12, 6.5), QPointF(12, 17.5))
    elif name == "chevron_down":
        p.drawLine(QPointF(7, 9.5), QPointF(12, 14.5))
        p.drawLine(QPointF(12, 14.5), QPointF(17, 9.5))
    elif name == "check":
        p.drawLine(QPointF(5.5, 12.5), QPointF(10, 17))
        p.drawLine(QPointF(10, 17), QPointF(18.5, 7.5))

    p.restore()


def make_m3_icon(name: str, color_hex: str, size: int = 18) -> QIcon:
    dpr = 2.0
    px = int(round(size * dpr))
    pm = QPixmap(px, px)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    draw_m3_icon(p, name, QRectF(0, 0, size, size), QColor(color_hex), stroke=2.0)
    p.end()
    return QIcon(pm)


def build_m3_stylesheet(pal: M3Palette) -> str:
    prim_hover = _blend_hex(pal.primary, pal.on_primary, 0.10)
    tonal_hover = _blend_hex(pal.secondary_container, pal.on_secondary_container, 0.10)
    surf_hover = _blend_hex(pal.surface_container_high, pal.on_surface, 0.08)
    err_hover = _blend_hex(pal.error_container, pal.on_error_container, 0.12)
    return f"""
    QMainWindow, QWidget {{
        background-color: {pal.surface};
        color: {pal.on_surface};
        font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
        font-size: 13px;
    }}
    QScrollArea, QScrollArea > QWidget > QWidget {{
        background-color: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 4px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {pal.outline_variant};
        border-radius: 3px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {pal.outline};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
        height: 0px;
    }}
    QSplitter::handle {{
        background-color: {pal.surface_container};
        width: 2px;
    }}

    /* Borderless M3 Expressive Tonal Surfaces */
    QFrame#m3Rail {{
        background-color: {pal.surface_container_low};
        border: none;
    }}
    QFrame#m3Inspector {{
        background-color: {pal.surface};
        border: none;
    }}
    QFrame#m3Card {{
        background-color: {pal.surface_container_low};
        border: none;
        border-radius: 20px;
    }}
    QFrame#m3SubCard {{
        background-color: {pal.surface_container};
        border: none;
        border-radius: 16px;
    }}
    QFrame#m3BuildDeck {{
        background-color: {pal.surface_container_low};
        border: none;
        border-radius: 22px;
    }}
    QFrame#m3Toolbar {{
        background-color: {pal.surface_container_low};
        border: none;
        border-radius: 22px;
    }}
    QLabel {{
        background-color: transparent;
    }}
    QLabel#m3PageHeading {{
        color: {pal.on_surface};
        font-size: 18px;
        font-weight: 700;
    }}
    QLabel#m3SectionTitle {{
        color: {pal.on_surface};
        font-size: 13px;
        font-weight: 700;
    }}
    QLabel#m3Muted {{
        color: {pal.on_surface_variant};
        font-size: 12px;
    }}
    QLabel#m3StageChip {{
        background-color: {pal.tertiary_container};
        color: {pal.on_tertiary_container};
        border-radius: 10px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#m3TimerPill {{
        background-color: {pal.surface_container_high};
        color: {pal.on_surface};
        border-radius: 10px;
        padding: 3px 9px;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 12px;
        font-weight: 600;
    }}
    QLabel#m3ValuePill {{
        background-color: {pal.surface_container_high};
        color: {pal.on_surface};
        border-radius: 9px;
        padding: 2px 8px;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 12px;
        font-weight: 600;
    }}
    QLabel#m3GpuChip {{
        background-color: {pal.surface_container};
        color: {pal.on_surface_variant};
        border-radius: 11px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* M3 Expressive Pill-to-Squircle Buttons */
    QPushButton {{
        background-color: {pal.secondary_container};
        color: {pal.on_secondary_container};
        border: none;
        border-radius: 16px;
        padding: 6px 14px;
        min-height: 22px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {tonal_hover};
    }}
    QPushButton:pressed {{
        border-radius: 10px;
    }}
    QPushButton:disabled {{
        background-color: {pal.surface_container_high};
        color: {pal.outline};
    }}

    /* Clean Borderless Inputs */
    QLineEdit {{
        background-color: {pal.surface_container_high};
        color: {pal.on_surface};
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 6px 12px;
        min-height: 22px;
        selection-background-color: {pal.primary_container};
        selection-color: {pal.on_primary_container};
    }}
    QLineEdit:focus {{
        border: 2px solid {pal.primary};
        padding: 5px 11px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {pal.surface_container_high};
        color: {pal.on_surface};
        border: 1px solid {pal.outline_variant};
        border-radius: 14px;
        padding: 4px;
        selection-background-color: {pal.secondary_container};
        selection-color: {pal.on_secondary_container};
        outline: none;
    }}
    QPlainTextEdit {{
        background-color: {pal.surface_container_lowest};
        color: {pal.on_surface};
        border: none;
        border-radius: 14px;
        padding: 10px;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 11px;
    }}
    """


def _smooth_circular(arr: np.ndarray, sigma: float = 3.6) -> np.ndarray:
    radius = int(math.ceil(sigma * 3.0))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    padded = np.concatenate([arr[-radius:], arr, arr[:radius]])
    return np.convolve(padded, k, mode="valid")


def _build_m3_expressive_shapes(n_pts: int = 128) -> List[np.ndarray]:
    """
    Official Material 3 Expressive LoadingIndicator 7-shape sequence:
    SoftBurst -> Cookie9Sided -> Pentagon -> Pill -> Sunny -> Cookie4Sided -> Oval
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    def _from_r(r: np.ndarray) -> np.ndarray:
        r_n = r / max(1e-6, float(np.max(r)))
        return np.stack([r_n * cos_t, r_n * sin_t], axis=-1).astype(np.float64)

    soft_burst = _from_r(0.81 + 0.19 * np.cos(10.0 * theta))
    cookie9 = _from_r(0.88 + 0.12 * np.cos(9.0 * theta))

    sec5 = 2.0 * np.pi / 5.0
    phi5 = ((theta + np.pi * 0.5) % sec5) - (sec5 * 0.5)
    r_pent = np.cos(np.pi / 5.0) / np.cos(phi5)
    pentagon = _from_r(_smooth_circular(r_pent, sigma=4.2))

    b_pill = 0.62
    l_pill = 1.0 - b_pill
    abs_sin = np.abs(sin_t)
    abs_cos = np.abs(cos_t)
    on_flat = (abs_sin > 1e-6) & ((abs_cos / np.maximum(abs_sin, 1e-6)) <= (l_pill / b_pill))
    r_flat = b_pill / np.maximum(abs_sin, 1e-6)
    r_cap = l_pill * abs_cos + np.sqrt(np.maximum(0.0, b_pill * b_pill - (l_pill * sin_t) ** 2))
    pill = _from_r(np.where(on_flat, r_flat, r_cap))

    sunny = _from_r(0.77 + 0.23 * np.cos(8.0 * theta))
    cookie4 = _from_r(0.80 + 0.20 * np.cos(4.0 * theta))

    a_ov, b_ov = 1.0, 0.72
    r_oval = (a_ov * b_ov) / np.sqrt((b_ov * cos_t) ** 2 + (a_ov * sin_t) ** 2)
    oval = _from_r(r_oval)

    return [soft_burst, cookie9, pentagon, pill, sunny, cookie4, oval]


_M3_SHAPES = _build_m3_expressive_shapes(128)


def _expressive_spring(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    val = 1.0 - math.exp(-6.2 * u) * (math.cos(7.8 * u) + 0.45 * math.sin(7.8 * u))
    end_norm = 1.0 - math.exp(-6.2) * (math.cos(7.8) + 0.45 * math.sin(7.8))
    return val / end_norm


class M3Button(QPushButton):
    """
    True Material 3 Expressive spring-morphing button:
    - Full pill resting shape (radius = h/2) -> spring-morphs to 9px squircle on press with overshoot on release
    - Smooth animated tonal hover/press state layer
    - Custom vector icon support (zero QSS border-radius breakage)
    """

    def __init__(
        self,
        text: str = "",
        variant: str = "tonal",
        icon_name: Optional[str] = None,
        height: int = 38,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._variant = variant  # filled, tonal, outlined, error_tonal, compact, icon_tonal
        self._icon_name = icon_name
        self._pal = resolve_palette("graphite_cobalt")
        self._press_pos = 0.0
        self._press_vel = 0.0
        self._press_target = 0.0
        self._hover_pos = 0.0
        self._hover_target = 0.0

        self.setFixedHeight(int(height))
        if variant == "icon_tonal":
            self.setFixedWidth(int(height))
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed if variant == "icon_tonal" else QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def set_icon_name(self, icon_name: Optional[str]):
        self._icon_name = icon_name
        self.update()

    def _paint_font(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(11 if self._variant == "compact" else 13)
        f.setWeight(
            QFont.Weight.Bold
            if self._variant in ("filled", "error_tonal", "compact")
            else QFont.Weight.DemiBold
        )
        return f

    def sizeHint(self) -> QSize:
        if self._variant == "icon_tonal":
            return QSize(self.height(), self.height())
        fm = QFontMetrics(self._paint_font())
        tw = fm.horizontalAdvance(self.text()) if self.text() else 0
        ic_w = (16 + (7 if tw > 0 else 0)) if self._icon_name else 0
        pad_x = 18 if self._variant == "compact" else 28
        return QSize(max(self.height(), tw + ic_w + pad_x), self.height())

    def minimumSizeHint(self) -> QSize:
        if self._variant == "icon_tonal":
            return QSize(self.height(), self.height())
        return QSize(max(self.height(), 44), self.height())

    def _ensure_anim(self):
        if not self._timer.isActive():
            self._timer.start()

    def enterEvent(self, event):
        if self.isEnabled():
            self._hover_target = 1.0
            self._ensure_anim()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_target = 0.0
        self._press_target = 0.0
        self._ensure_anim()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._press_target = 1.0
            self._ensure_anim()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_target = 0.0
        self._ensure_anim()
        super().mouseReleaseEvent(event)

    def _tick(self):
        dt = 0.016
        k, c = 420.0, 22.0  # M3 Expressive fast spatial damped spring
        acc = k * (self._press_target - self._press_pos) - c * self._press_vel
        self._press_vel += acc * dt
        self._press_pos += self._press_vel * dt
        self._hover_pos += (self._hover_target - self._hover_pos) * 0.28

        if (
            abs(self._press_target - self._press_pos) < 0.004
            and abs(self._press_vel) < 0.015
            and abs(self._hover_target - self._hover_pos) < 0.01
        ):
            self._press_pos = self._press_target
            self._press_vel = 0.0
            self._hover_pos = self._hover_target
            self._timer.stop()
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        pal = self._pal
        enabled = self.isEnabled()
        if not enabled:
            bg_hex, fg_hex, border_hex = pal.surface_container_high, pal.outline, None
        elif self._variant == "filled":
            bg_hex, fg_hex, border_hex = pal.primary, pal.on_primary, None
        elif self._variant == "error_tonal":
            bg_hex, fg_hex, border_hex = pal.error_container, pal.on_error_container, None
        elif self._variant == "outlined":
            bg_hex, fg_hex, border_hex = None, pal.on_surface, pal.outline_variant
        elif self._variant == "icon_tonal":
            bg_hex, fg_hex, border_hex = pal.surface_container_high, pal.on_surface, None
        else:  # tonal / compact
            bg_hex, fg_hex, border_hex = pal.secondary_container, pal.on_secondary_container, None

        w, h = float(self.width()), float(self.height())
        # Subtle expressive horizontal width breath on press
        inset_x = max(0.5, 1.5 - 1.0 * self._press_pos)
        inset_y = 1.0
        rect = QRectF(inset_x, inset_y, max(2.0, w - 2.0 * inset_x), max(2.0, h - 2.0 * inset_y))

        pill_r = min(rect.width(), rect.height()) * 0.5
        squircle_r = min(pill_r, 9.0 if h >= 32 else 6.0)
        p_clamped = max(-0.15, min(1.15, self._press_pos))
        radius = max(4.0, min(pill_r, pill_r + (squircle_r - pill_r) * p_clamped - 1.2 * self._hover_pos))

        state_amt = 0.08 * self._hover_pos + 0.12 * max(0.0, p_clamped)
        if bg_hex is not None:
            fill = QColor(_blend_hex(bg_hex, fg_hex, min(0.24, state_amt))) if enabled else QColor(bg_hex)
            p.setBrush(QBrush(fill))
        else:
            if enabled and state_amt > 0.005:
                ov = QColor(fg_hex)
                ov.setAlphaF(min(0.16, state_amt))
                p.setBrush(QBrush(ov))
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)

        if border_hex is not None:
            bcol = QColor(pal.outline if (enabled and self._hover_pos > 0.3) else border_hex)
            p.setPen(QPen(bcol, 1.2))
        else:
            p.setPen(Qt.PenStyle.NoPen)

        p.drawRoundedRect(rect, radius, radius)

        # Draw centered vector icon + label
        fg = QColor(fg_hex)
        f = self._paint_font()
        p.setFont(f)
        fm = QFontMetrics(f)

        txt = self.text()
        ic_sz = 14.0 if self._variant in ("compact", "icon_tonal") else 16.0
        gap = 6.0 if (self._icon_name and txt) else 0.0
        tw = float(fm.horizontalAdvance(txt)) if txt else 0.0
        total_w = (ic_sz if self._icon_name else 0.0) + gap + tw
        avail_w = max(8.0, rect.width() - 14.0)
        if total_w > avail_w and txt:
            max_tw = max(4.0, avail_w - (ic_sz + gap if self._icon_name else 0.0))
            txt = fm.elidedText(txt, Qt.TextElideMode.ElideRight, int(max_tw))
            tw = float(fm.horizontalAdvance(txt))
            total_w = (ic_sz if self._icon_name else 0.0) + gap + tw

        x = rect.center().x() - total_w * 0.5
        cy = rect.center().y()
        if self._icon_name:
            draw_m3_icon(p, self._icon_name, QRectF(x, cy - ic_sz * 0.5, ic_sz, ic_sz), fg, stroke=2.0)
            x += ic_sz + gap
        if txt:
            p.setPen(fg)
            p.drawText(QRectF(x, rect.y(), tw + 2.0, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)

        p.end()


class M3SwatchButton(QPushButton):
    """Crisp circular Material 3 palette swatch button for the Navigation Rail."""

    def __init__(self, swatch_hex: str, tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._swatch = QColor(swatch_hex)
        self._pal = resolve_palette("graphite_cobalt")
        self._hovered = False
        self.setCheckable(True)
        self.setFixedSize(22, 22)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)
        cx, cy = self.width() * 0.5, self.height() * 0.5
        if self.isChecked():
            p.setPen(QPen(QColor(self._pal.on_surface), 1.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), 9.0, 9.0)
            r = 5.8
        else:
            r = 7.4 if self._hovered else 6.4
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._swatch))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.end()


class M3LoadingIndicator(QWidget):
    """Official Material 3 Expressive 7-shape spring-morphing Loading Indicator."""

    def __init__(self, size: int = 44, contained: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._size = int(size)
        self._contained = bool(contained)
        self._active = False
        self._elapsed = 0.0
        self._step_sec = 0.72
        self._pal = resolve_palette("graphite_cobalt")

        self.setFixedSize(self._size, self._size)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def start(self):
        if not self._active:
            self._active = True
            self._timer.start()
            self.update()

    def stop(self):
        self._active = False
        self._timer.stop()
        self.update()

    def _tick(self):
        self._elapsed += 0.016
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        side = float(min(self.width(), self.height()))
        cx = self.width() * 0.5
        cy = self.height() * 0.5

        if self._contained:
            p.setPen(Qt.PenStyle.NoPen)
            bg = QColor(self._pal.primary_container if self._active else self._pal.surface_container_high)
            p.setBrush(QBrush(bg))
            r_bg = (side - 2.0) * 0.5
            p.drawEllipse(QPointF(cx, cy), r_bg, r_bg)

        if self._active:
            total_steps = self._elapsed / self._step_sec
            idx = int(math.floor(total_steps)) % len(_M3_SHAPES)
            nxt = (idx + 1) % len(_M3_SHAPES)
            local_u = total_steps - math.floor(total_steps)
            morph_t = _expressive_spring(min(1.0, local_u / 0.82))
            pts = (1.0 - morph_t) * _M3_SHAPES[idx] + morph_t * _M3_SHAPES[nxt]
            rot_deg = (self._elapsed * 52.0 + (idx + morph_t) * 42.0) % 360.0
            fill_col = QColor(self._pal.on_primary_container if self._contained else self._pal.primary)

            r_shape = side * (0.37 if self._contained else 0.44)
            p.save()
            p.translate(cx, cy)
            p.rotate(rot_deg)
            path = QPainterPath()
            path.moveTo(float(pts[0, 0] * r_shape), float(pts[0, 1] * r_shape))
            for k in range(1, len(pts)):
                path.lineTo(float(pts[k, 0] * r_shape), float(pts[k, 1] * r_shape))
            path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(fill_col))
            p.drawPath(path)
            p.restore()
        else:
            # Calm resting M3 vector cube emblem when idle
            ic = side * 0.48
            draw_m3_icon(
                p,
                "cube",
                QRectF(cx - ic * 0.5, cy - ic * 0.5, ic, ic),
                QColor(self._pal.on_surface_variant),
                stroke=1.8,
            )
        p.end()


class M3WavyProgressBar(QWidget):
    """Official Material 3 Expressive LinearWavyProgressIndicator."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._val = 0
        self._phase = 0.0
        self._pal = resolve_palette("graphite_cobalt")

        self.setFixedHeight(16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def setRange(self, lo: int, hi: int):
        self._min = int(lo)
        self._max = max(self._min + 1, int(hi))
        self.update()

    def setValue(self, val: int):
        self._val = max(self._min, min(self._max, int(val)))
        frac = (self._val - self._min) / float(self._max - self._min)
        if 0.0 < frac < 1.0:
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()
        self.update()

    def value(self) -> int:
        return self._val

    def _tick(self):
        self._phase = (self._phase + 0.42) % 24.0
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        w = float(self.width())
        cy = float(self.height()) * 0.5
        pad = 4.0
        span = max(1.0, w - 2.0 * pad)
        frac = (self._val - self._min) / float(self._max - self._min)

        stroke_w = 4.0
        gap = 6.0
        active_end = pad + span * frac
        active_col = QColor(self._pal.primary)
        track_col = QColor(self._pal.surface_container_highest)

        track_start = (active_end + gap) if frac > 0.0 else pad
        if track_start < (w - pad):
            pen_t = QPen(track_col, stroke_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen_t)
            p.drawLine(QPointF(track_start, cy), QPointF(w - pad, cy))

        if frac < 0.99:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(active_col))
            p.drawEllipse(QPointF(w - pad, cy), 2.2, 2.2)

        if frac > 0.005:
            wavelength = 24.0
            env = min(1.0, frac / 0.08) * min(1.0, (1.0 - frac) / 0.05)
            amp = 3.0 * env
            active_len = max(1.0, active_end - pad)
            steps = max(2, int(active_len / 1.5))

            path = QPainterPath()
            for s in range(steps + 1):
                x = pad + (active_len * s / steps)
                u = s / float(steps)
                edge = min(1.0, u * 10.0) * min(1.0, (1.0 - u) * 10.0) if amp > 0.01 else 0.0
                y = cy + amp * edge * math.sin(2.0 * math.pi * (x - self._phase) / wavelength)
                if s == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)

            pen_a = QPen(
                active_col,
                stroke_w,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            p.setPen(pen_a)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.end()


class M3ExpressiveSlider(QWidget):
    """Android 16 / Material 3 Expressive thick-track slider with vertical pill handle."""

    valueChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._val = 0
        self._dragging = False
        self._pal = resolve_palette("graphite_cobalt")
        self.setFixedHeight(26)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def setRange(self, lo: int, hi: int):
        self._min = int(lo)
        self._max = max(self._min, int(hi))
        self._val = max(self._min, min(self._max, self._val))
        self.update()

    def value(self) -> int:
        return self._val

    def setValue(self, val: int):
        clamped = max(self._min, min(self._max, int(val)))
        if clamped != self._val:
            self._val = clamped
            self.update()
            if not self.signalsBlocked():
                self.valueChanged.emit(self._val)
        else:
            self.update()

    def _val_from_x(self, x: float) -> int:
        if self._max <= self._min:
            return self._min
        pad = 6.0
        span = max(1.0, float(self.width()) - 2.0 * pad)
        frac = max(0.0, min(1.0, (x - pad) / span))
        return int(round(self._min + frac * (self._max - self._min)))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setValue(self._val_from_x(event.position().x()))
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.setValue(self._val_from_x(event.position().x()))

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        w = float(self.width())
        cy = float(self.height()) * 0.5
        pad = 6.0
        span = max(1.0, w - 2.0 * pad)
        frac = (
            (self._val - self._min) / float(self._max - self._min)
            if self._max > self._min
            else 0.0
        )
        hx = pad + span * frac

        track_h = 10.0
        y_top = cy - track_h * 0.5
        gap = 5.5
        active_col = QColor(self._pal.primary)
        inactive_col = QColor(self._pal.surface_container_highest)

        p.setPen(Qt.PenStyle.NoPen)

        # Left active thick pill segment
        act_w = (hx - gap) - pad
        if act_w > 2.0:
            p.setBrush(QBrush(active_col))
            p.drawRoundedRect(QRectF(pad, y_top, act_w, track_h), 5.0, 5.0)

        # Right inactive thick pill segment
        inact_x = hx + gap
        inact_w = (w - pad) - inact_x
        if inact_w > 2.0:
            p.setBrush(QBrush(inactive_col))
            p.drawRoundedRect(QRectF(inact_x, y_top, inact_w, track_h), 5.0, 5.0)
            p.setBrush(QBrush(active_col))
            p.drawEllipse(QPointF(w - pad - 4.0, cy), 2.0, 2.0)

        # Tall vertical M3 Expressive pill handle
        hw = 5.0 if self._dragging else 4.0
        hh = 24.0 if self._dragging else 20.0
        p.setBrush(QBrush(active_col))
        p.drawRoundedRect(QRectF(hx - hw * 0.5, cy - hh * 0.5, hw, hh), hw * 0.5, hw * 0.5)
        p.end()


class M3NumericControl(QWidget):
    """Modern M3 Expressive slider + stepper + value readout replacing retro QSpinBox/QDoubleSpinBox."""

    valueChanged = Signal(object)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        init_val: float,
        step: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        slider_max: Optional[float] = None,
        extra_widget: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._is_float = decimals > 0
        self._decimals = int(decimals)
        self._min = float(min_val)
        self._max = float(max_val)
        self._slider_max = float(slider_max) if slider_max is not None else float(max_val)
        self._step = float(step)
        self._suffix = suffix
        self._val = float(init_val)
        self._pal = resolve_palette("graphite_cobalt")
        self._syncing = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        self.title_lbl = QLabel(label)
        top.addWidget(self.title_lbl)
        top.addStretch(1)

        if extra_widget is not None:
            top.addWidget(extra_widget)

        self.val_lbl = QLabel()
        self.val_lbl.setObjectName("m3ValuePill")
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.val_lbl)

        self.dec_btn = M3Button("", variant="icon_tonal", icon_name="minus", height=24)
        self.dec_btn.clicked.connect(lambda: self.setValue(self._val - self._step))
        top.addWidget(self.dec_btn)

        self.inc_btn = M3Button("", variant="icon_tonal", icon_name="plus", height=24)
        self.inc_btn.clicked.connect(lambda: self.setValue(self._val + self._step))
        top.addWidget(self.inc_btn)

        root.addLayout(top)

        self.slider = M3ExpressiveSlider()
        self.slider.valueChanged.connect(self._on_slider_moved)
        root.addWidget(self.slider)

        self._rebuild_slider_range()
        self._refresh_display()

    def _scale_factor(self) -> int:
        return int(round(10 ** self._decimals)) if self._is_float else 1

    def _rebuild_slider_range(self):
        sf = self._scale_factor()
        s_lo = int(round(self._min * sf))
        s_hi = int(round(max(self._min, min(self._max, self._slider_max)) * sf))
        self._syncing = True
        self.slider.setRange(s_lo, s_hi)
        self.slider.setValue(int(round(min(self._val, self._slider_max) * sf)))
        self._syncing = False

    def _on_slider_moved(self, raw_int: int):
        if self._syncing:
            return
        sf = self._scale_factor()
        val = raw_int / float(sf)
        if not self._is_float and self._step > 1.0:
            val = round(val / self._step) * self._step
        val = max(self._min, min(self._max, val))
        self.setValue(val)

    def _refresh_display(self):
        if self._is_float:
            txt = f"{self._val:.{self._decimals}f}{self._suffix}"
        else:
            txt = f"{int(round(self._val))}{self._suffix}"
        self.val_lbl.setText(txt)

    def minimumSizeHint(self) -> QSize:
        return QSize(100, 58)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.slider.set_palette_colors(pal)
        self.dec_btn.set_palette_colors(pal)
        self.inc_btn.set_palette_colors(pal)

    def set_slider_max(self, smax: float):
        self._slider_max = max(self._min + self._step, float(smax))
        self._rebuild_slider_range()

    def setRange(self, lo: float, hi: float):
        self._min = float(lo)
        self._max = float(hi)
        if self._slider_max > self._max:
            self._slider_max = self._max
        self._val = max(self._min, min(self._max, self._val))
        self._rebuild_slider_range()
        self._refresh_display()

    def setSingleStep(self, step: float):
        self._step = max(1e-6, float(step))

    def setDecimals(self, d: int):
        self._decimals = int(d)
        self._is_float = self._decimals > 0
        self._rebuild_slider_range()
        self._refresh_display()

    def value(self):
        return float(self._val) if self._is_float else int(round(self._val))

    def setValue(self, val: float):
        clamped = max(self._min, min(self._max, float(val)))
        if not self._is_float:
            clamped = float(int(round(clamped)))
        changed = abs(clamped - self._val) > 1e-7
        self._val = clamped
        if self._val > self._slider_max and self._val <= self._max:
            self._slider_max = self._val
        self._rebuild_slider_range()
        self._refresh_display()
        if changed and not self.signalsBlocked():
            self.valueChanged.emit(self.value())


class M3ComboBox(QComboBox):
    """Custom-painted Material 3 dropdown with elided text and crisp vector chevron (zero retro chrome)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pal = resolve_palette("graphite_cobalt")
        self._hovered = False
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(160, 34)

    def minimumSizeHint(self) -> QSize:
        return QSize(80, 34)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        rect = QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0)
        bg = QColor(
            _blend_hex(self._pal.surface_container_high, self._pal.on_surface, 0.06)
            if self._hovered
            else self._pal.surface_container_high
        )
        p.setBrush(QBrush(bg))
        if self.hasFocus():
            p.setPen(QPen(QColor(self._pal.primary), 2.0))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, 12.0, 12.0)

        text_rect = QRectF(12.0, 0.0, max(10.0, self.width() - 38.0), float(self.height()))
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self.currentText(), Qt.TextElideMode.ElideRight, int(text_rect.width()))
        p.setPen(QColor(self._pal.on_surface))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)

        ic_rect = QRectF(self.width() - 26.0, (self.height() - 16.0) * 0.5, 16.0, 16.0)
        draw_m3_icon(p, "chevron_down", ic_rect, QColor(self._pal.on_surface_variant), stroke=2.0)
        p.end()


class M3Switch(QCheckBox):
    """Material 3 Expressive animated toggle switch (drop-in QCheckBox subclass)."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self._pos = 1.0 if self.isChecked() else 0.0
        self._target = self._pos
        self._pal = resolve_palette("graphite_cobalt")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate)
        self.toggled.connect(self._on_toggled)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        if self.signalsBlocked():
            self._target = 1.0 if checked else 0.0
            self._pos = self._target
            self.update()

    def _on_toggled(self, checked: bool):
        self._target = 1.0 if checked else 0.0
        if not self._timer.isActive():
            self._timer.start()

    def _animate(self):
        diff = self._target - self._pos
        if abs(diff) < 0.02:
            self._pos = self._target
            self._timer.stop()
        else:
            self._pos += diff * 0.34
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        tw = fm.horizontalAdvance(self.text()) if self.text() else 0
        gap = 8 if tw > 0 else 0
        return QSize(42 + gap + tw + 6, max(26, fm.height() + 6))

    def minimumSizeHint(self) -> QSize:
        return QSize(54, 26)

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        track_w, track_h = 40.0, 22.0
        y0 = (self.height() - track_h) * 0.5
        rect = QRectF(1.0, y0, track_w, track_h)

        if not self._timer.isActive():
            self._target = 1.0 if self.isChecked() else 0.0
            self._pos = self._target

        t = self._pos
        c_off_bg = QColor(self._pal.surface_container_highest)
        c_on_bg = QColor(self._pal.primary)
        bg = QColor(
            int(c_off_bg.red() * (1 - t) + c_on_bg.red() * t),
            int(c_off_bg.green() * (1 - t) + c_on_bg.green() * t),
            int(c_off_bg.blue() * (1 - t) + c_on_bg.blue() * t),
        )

        p.setBrush(QBrush(bg))
        if t < 0.5:
            p.setPen(QPen(QColor(self._pal.outline), 1.5))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, track_h * 0.5, track_h * 0.5)

        thumb_d = 12.0 + 5.0 * t
        thumb_r = thumb_d * 0.5
        cx_off = 1.0 + track_h * 0.5
        cx_on = 1.0 + track_w - track_h * 0.5
        cx = cx_off * (1.0 - t) + cx_on * t
        cy = y0 + track_h * 0.5

        c_off_th = QColor(self._pal.outline)
        c_on_th = QColor(self._pal.on_primary)
        th_col = QColor(
            int(c_off_th.red() * (1 - t) + c_on_th.red() * t),
            int(c_off_th.green() * (1 - t) + c_on_th.green() * t),
            int(c_off_th.blue() * (1 - t) + c_on_th.blue() * t),
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(th_col))
        p.drawEllipse(QPointF(cx, cy), thumb_r, thumb_r)

        if t > 0.65:
            draw_m3_icon(p, "check", QRectF(cx - 5.5, cy - 5.5, 11.0, 11.0), QColor(self._pal.primary), stroke=2.6)

        if self.text():
            p.setPen(QColor(self._pal.on_surface))
            avail = max(4.0, self.width() - track_w - 9.0)
            text_rect = QRectF(track_w + 9.0, 0.0, avail, float(self.height()))
            fm = QFontMetrics(self.font())
            elided = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, int(avail))
            p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)
        p.end()


class _M3ConnectedItem(QPushButton):
    """Single animated segment inside M3ConnectedGroup."""

    def __init__(self, label: str, pos_idx: int, total: int, parent: Optional[QWidget] = None):
        super().__init__(label, parent)
        self._label = label
        self._pos_idx = pos_idx
        self._total = total
        self._pal = resolve_palette("graphite_cobalt")
        self._act_pos = 0.0
        self._act_vel = 0.0
        self._act_target = 0.0
        self._press_pos = 0.0
        self._press_target = 0.0
        self._hover_pos = 0.0
        self._hover_target = 0.0

        self.setCheckable(True)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def set_active_state(self, active: bool, animate: bool = True):
        self.blockSignals(True)
        self.setChecked(active)
        self.blockSignals(False)
        self._act_target = 1.0 if active else 0.0
        if not animate:
            self._act_pos = self._act_target
            self._act_vel = 0.0
            self.update()
        else:
            if not self._timer.isActive():
                self._timer.start()

    def _paint_font(self, bold: bool) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.Bold if bold else QFont.Weight.DemiBold)
        return f

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self._paint_font(True))
        tw = fm.horizontalAdvance(self._label)
        return QSize(tw + 42, 34)

    def minimumSizeHint(self) -> QSize:
        return QSize(28, 34)

    def enterEvent(self, event):
        if self.isEnabled():
            self._hover_target = 1.0
            if not self._timer.isActive():
                self._timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_target = 0.0
        self._press_target = 0.0
        if not self._timer.isActive():
            self._timer.start()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._press_target = 1.0
            if not self._timer.isActive():
                self._timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_target = 0.0
        if not self._timer.isActive():
            self._timer.start()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._hover_target = 0.0
            self._press_target = 0.0
            self.update()
        super().changeEvent(event)

    def _tick(self):
        dt = 0.016
        k, c = 380.0, 23.0
        acc = k * (self._act_target - self._act_pos) - c * self._act_vel
        self._act_vel += acc * dt
        self._act_pos += self._act_vel * dt
        self._press_pos += (self._press_target - self._press_pos) * 0.38
        self._hover_pos += (self._hover_target - self._hover_pos) * 0.28

        if (
            abs(self._act_target - self._act_pos) < 0.004
            and abs(self._act_vel) < 0.015
            and abs(self._press_target - self._press_pos) < 0.01
            and abs(self._hover_target - self._hover_pos) < 0.01
        ):
            self._act_pos = self._act_target
            self._act_vel = 0.0
            self._press_pos = self._press_target
            self._hover_pos = self._hover_target
            self._timer.stop()
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        pal = self._pal
        rect = QRectF(0.5, 1.0, self.width() - 1.0, self.height() - 2.0)
        t = max(0.0, min(1.0, self._act_pos))

        pill_r = rect.height() * 0.5
        inner_base = 6.0
        inner_r = max(4.0, min(pill_r, inner_base + (pill_r - inner_base) * self._act_pos - 2.5 * self._press_pos))
        outer_r = max(6.0, min(pill_r, pill_r - 5.0 * self._press_pos))

        tl = outer_r if self._pos_idx == 0 else inner_r
        bl = outer_r if self._pos_idx == 0 else inner_r
        tr = outer_r if self._pos_idx == self._total - 1 else inner_r
        br = outer_r if self._pos_idx == self._total - 1 else inner_r

        # Custom per-corner rounded path
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        path = QPainterPath()
        path.moveTo(x + tl, y)
        path.lineTo(x + w - tr, y)
        path.quadTo(x + w, y, x + w, y + tr)
        path.lineTo(x + w, y + h - br)
        path.quadTo(x + w, y + h, x + w - br, y + h)
        path.lineTo(x + bl, y + h)
        path.quadTo(x, y + h, x, y + h - bl)
        path.lineTo(x, y + tl)
        path.quadTo(x, y, x + tl, y)
        path.closeSubpath()

        bg_hex = _blend_hex(pal.surface_container_high, pal.primary_container, t)
        fg_hex = _blend_hex(pal.on_surface_variant, pal.on_primary_container, t)
        if self._hover_pos > 0.01 or self._press_pos > 0.01:
            bg_hex = _blend_hex(bg_hex, fg_hex, 0.07 * self._hover_pos + 0.08 * self._press_pos)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(bg_hex)))
        p.drawPath(path)

        f = self._paint_font(t > 0.5)
        p.setFont(f)
        fm = QFontMetrics(f)
        txt = self._label
        tw = float(fm.horizontalAdvance(txt))
        ic_sz = 13.0 * t
        gap = 5.0 * t
        if ic_sz + gap + tw > rect.width() - 10.0:
            max_tw = max(4.0, rect.width() - 10.0 - (ic_sz + gap))
            txt = fm.elidedText(txt, Qt.TextElideMode.ElideRight, int(max_tw))
            tw = float(fm.horizontalAdvance(txt))
        total_w = ic_sz + gap + tw
        sx = rect.center().x() - total_w * 0.5
        cy = rect.center().y()

        fg = QColor(fg_hex)
        if ic_sz > 2.0:
            draw_m3_icon(p, "check", QRectF(sx, cy - ic_sz * 0.5, ic_sz, ic_sz), fg, stroke=2.2)
            sx += ic_sz + gap
        p.setPen(fg)
        p.drawText(QRectF(sx, rect.y(), tw + 2.0, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)
        p.end()


class M3ConnectedGroup(QWidget):
    """Official Material 3 Expressive Connected Button Group with spring-morphing corners and data API."""

    currentIndexChanged = Signal(int)

    def __init__(
        self,
        items: List[str],
        data: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._items = list(items)
        self._data = list(data) if data is not None else list(items)
        self._index = 0
        self._pal = resolve_palette("graphite_cobalt")
        self._buttons: List[_M3ConnectedItem] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        n = len(self._items)
        for idx, label in enumerate(self._items):
            btn = _M3ConnectedItem(label, idx, n)
            btn.clicked.connect(lambda _chk=False, i=idx: self.setCurrentIndex(i))
            layout.addWidget(btn, stretch=1)
            self._buttons.append(btn)

        self._sync_buttons(animate=False)

    def currentIndex(self) -> int:
        return self._index

    def currentData(self):
        return self._data[self._index] if 0 <= self._index < len(self._data) else None

    def findData(self, val) -> int:
        for i, d in enumerate(self._data):
            if d == val:
                return i
        return -1

    def setCurrentIndex(self, idx: int):
        if idx < 0 or idx >= len(self._buttons):
            return
        changed = idx != self._index
        self._index = idx
        self._sync_buttons(animate=True)
        if changed and not self.signalsBlocked():
            self.currentIndexChanged.emit(self._index)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        for btn in self._buttons:
            btn.set_palette_colors(pal)

    def _sync_buttons(self, animate: bool = True):
        for i, btn in enumerate(self._buttons):
            btn.set_active_state(i == self._index, animate=animate)


class M3NavRailItem(QPushButton):
    """Single destination inside the Material 3 Navigation Rail."""

    def __init__(self, icon_name: str, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._icon_name = icon_name
        self._label = label
        self._pal = resolve_palette("graphite_cobalt")
        self._hovered = False
        self.setCheckable(True)
        self.setFixedSize(64, 54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)

        active = self.isChecked()
        pill_rect = QRectF(4.0, 2.0, 56.0, 32.0)

        p.setPen(Qt.PenStyle.NoPen)
        if active:
            p.setBrush(QBrush(QColor(self._pal.secondary_container)))
            p.drawRoundedRect(pill_rect, 16.0, 16.0)
        elif self._hovered:
            hv = QColor(self._pal.on_surface)
            hv.setAlpha(18)
            p.setBrush(QBrush(hv))
            p.drawRoundedRect(pill_rect, 16.0, 16.0)

        ic_col = QColor(self._pal.on_secondary_container if active else self._pal.on_surface_variant)
        draw_m3_icon(p, self._icon_name, QRectF(22.0, 8.0, 20.0, 20.0), ic_col, stroke=2.0 if active else 1.7)

        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.Bold if active else QFont.Weight.Medium)
        p.setFont(f)
        p.setPen(QColor(self._pal.on_surface if active else self._pal.on_surface_variant))
        p.drawText(QRectF(0.0, 35.0, float(self.width()), 18.0), Qt.AlignmentFlag.AlignCenter, self._label)
        p.end()


class M3VideoPreview(QWidget):
    """Fluid responsive 16:9 rounded video preview and M3 drop-zone."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._qimage: Optional[QImage] = None
        self._pal = resolve_palette("graphite_cobalt")
        self._drag_hover = False
        self.setMinimumHeight(140)
        self.setMaximumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(320, 175)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, 112)

    def set_palette_colors(self, pal: M3Palette):
        self._pal = pal
        self.update()

    def set_drag_hover(self, active: bool):
        self._drag_hover = active
        self.update()

    def set_rgb_frame(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        rgb_c = np.ascontiguousarray(rgb)
        self._qimage = QImage(rgb_c.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        rect = QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 16.0, 16.0)
        p.setClipPath(clip)

        p.fillRect(rect, QColor(self._pal.surface_container_lowest if self._qimage else self._pal.surface_container))

        if self._qimage is not None and not self._qimage.isNull():
            iw, ih = float(self._qimage.width()), float(self._qimage.height())
            scale = min(rect.width() / max(1.0, iw), rect.height() / max(1.0, ih))
            dw, dh = iw * scale, ih * scale
            dx = rect.x() + (rect.width() - dw) * 0.5
            dy = rect.y() + (rect.height() - dh) * 0.5
            p.drawImage(QRectF(dx, dy, dw, dh), self._qimage)
        else:
            cx, cy = rect.center().x(), rect.center().y()
            badge_r = 22.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(self._pal.secondary_container)))
            p.drawEllipse(QPointF(cx, cy - 16.0), badge_r, badge_r)
            draw_m3_icon(
                p,
                "upload",
                QRectF(cx - 10.0, cy - 26.0, 20.0, 20.0),
                QColor(self._pal.on_secondary_container),
                stroke=1.9,
            )

            f_title = QFont(self.font())
            f_title.setPixelSize(13)
            f_title.setWeight(QFont.Weight.DemiBold)
            p.setFont(f_title)
            p.setPen(QColor(self._pal.on_surface))
            p.drawText(
                QRectF(12.0, cy + 14.0, rect.width() - 24.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                "Drop a video or click to browse",
            )

            f_sub = QFont(self.font())
            f_sub.setPixelSize(11)
            p.setFont(f_sub)
            p.setPen(QColor(self._pal.on_surface_variant))
            p.drawText(
                QRectF(12.0, cy + 34.0, rect.width() - 24.0, 18.0),
                Qt.AlignmentFlag.AlignCenter,
                "MP4, MOV, MKV, or WebM of any length",
            )

        p.setClipping(False)
        if self._drag_hover:
            pen = QPen(QColor(self._pal.primary), 2.0, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, 16.0, 16.0)
        p.end()
