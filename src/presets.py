from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List


@dataclass
class Preset:
    name: str
    description: str = "custom user preset"
    max_frames: int = 24
    max_dim: int = 512
    point_stride: int = 2
    mesh_stride: int = 4
    edge_thresh: float = 0.12
    depth_model: str = "large"    # large (best geometry), base (sharp), small (fast)
    device: str = "auto"          # auto, cuda, cpu
    out_dir: str = "outputs"
    export_ply: bool = True
    export_glb: bool = True
    export_obj: bool = False
    render_mode: str = "mesh"     # points, mesh, hybrid
    point_size: int = 4
    eye_dome: bool = False
    dollhouse_cull: bool = True
    show_frustums: bool = False
    theme: str = "cyber_blue"     # cyber_blue, neon_ember, matrix_green
    builtin: bool = False
    scene_type: str = "indoor"    # indoor, outdoor
    max_anchors: int = 0          # 0 = auto adaptive (140-360), or explicit 12..500
    recon_mode: str = "tsdf"      # tsdf, 3dgs

    def sanitized(self) -> "Preset":
        clean_dim = max(128, min(1024, (int(self.max_dim) // 8) * 8))
        valid_themes = (
            "graphite_cobalt",
            "warm_terracotta",
            "forest_mineral",
            "alabaster_light",
            "obsidian_indigo",
            "terracotta_ember",
            "sage_botanical",
            "lavender_light",
            "cyber_blue",
            "neon_ember",
            "matrix_green",
        )
        clean_scene_type = "outdoor" if str(self.scene_type).strip().lower() == "outdoor" else "indoor"
        try:
            raw_anchors = int(self.max_anchors)
        except (ValueError, TypeError):
            raw_anchors = 0
        clean_max_anchors = 0 if raw_anchors <= 0 else max(12, min(500, raw_anchors))
        clean_recon_mode = "3dgs" if str(self.recon_mode).strip().lower() == "3dgs" else "tsdf"

        return Preset(
            name=self.name.strip() or "Custom Preset",
            description=self.description.strip() or "custom user preset",
            max_frames=max(4, min(100000, int(self.max_frames))),
            max_dim=clean_dim,
            point_stride=max(1, min(6, int(self.point_stride))),
            mesh_stride=max(2, min(10, int(self.mesh_stride))),
            edge_thresh=max(0.03, min(0.40, float(self.edge_thresh))),
            depth_model=self.depth_model if self.depth_model in ("large", "base", "small") else "large",
            device=self.device if self.device in ("auto", "cuda", "cpu") else "auto",
            out_dir=self.out_dir.strip() or "outputs",
            export_ply=bool(self.export_ply),
            export_glb=bool(self.export_glb),
            export_obj=bool(self.export_obj),
            render_mode=self.render_mode if self.render_mode in ("points", "mesh", "hybrid") else "mesh",
            point_size=max(2, min(14, int(self.point_size))),
            eye_dome=bool(self.eye_dome),
            dollhouse_cull=bool(self.dollhouse_cull),
            show_frustums=bool(self.show_frustums),
            theme=self.theme if self.theme in valid_themes else "graphite_cobalt",
            builtin=bool(self.builtin),
            scene_type=clean_scene_type,
            max_anchors=clean_max_anchors,
            recon_mode=clean_recon_mode,
        )


DEFAULT_PRESET_NAME = "High"


BUILTIN_PRESETS: List[Preset] = [
    Preset(
        name="Fast Draft",
        description="quick low-res pass with ViT-Small depth to preview layout",
        max_frames=12,
        max_dim=384,
        point_stride=3,
        mesh_stride=6,
        edge_thresh=0.14,
        depth_model="small",
        render_mode="mesh",
        point_size=5,
        builtin=True,
    ),
    Preset(
        name="Balanced",
        description="ViT-Large metric depth with TSDF mesh",
        max_frames=32,
        max_dim=512,
        point_stride=2,
        mesh_stride=4,
        edge_thresh=0.12,
        depth_model="large",
        render_mode="mesh",
        point_size=4,
        builtin=True,
    ),
    Preset(
        name="High",
        description="DA3-BASE 644px multi-view depth with TSDF mesh",
        max_frames=140,
        max_dim=640,
        point_stride=2,
        mesh_stride=3,
        edge_thresh=0.11,
        depth_model="large",
        render_mode="mesh",
        point_size=4,
        theme="graphite_cobalt",
        builtin=True,
    ),
    Preset(
        name="Ultra Sharp",
        description="high-density ViT-Large reconstruction for detailed rooms",
        max_frames=48,
        max_dim=640,
        point_stride=1,
        mesh_stride=3,
        edge_thresh=0.10,
        depth_model="large",
        render_mode="mesh",
        point_size=4,
        builtin=True,
    ),
    Preset(
        name="Long Video (n)",
        description="samples more keyframes across long multi-minute walkthroughs",
        max_frames=70,
        max_dim=512,
        point_stride=2,
        mesh_stride=4,
        edge_thresh=0.12,
        depth_model="large",
        render_mode="mesh",
        point_size=4,
        builtin=True,
    ),
    Preset(
        name="Mesh Focus",
        description="prioritizes fine connected mesh + auto-exports ply, glb & obj",
        max_frames=36,
        max_dim=560,
        point_stride=1,
        mesh_stride=2,
        edge_thresh=0.10,
        depth_model="large",
        export_ply=True,
        export_glb=True,
        export_obj=True,
        render_mode="mesh",
        point_size=4,
        builtin=True,
    ),
]

BUILTIN_KEYS = {p.name.lower() for p in BUILTIN_PRESETS}


class PresetManager:
    def __init__(self, path: str | Path = "presets.json"):
        self.path = Path(path)
        self._custom: Dict[str, Preset] = {}
        self.reload()

    def reload(self):
        self._custom.clear()
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("presets", []) if isinstance(raw, dict) else []
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    item["builtin"] = False
                    valid_keys = {f for f in Preset.__dataclass_fields__}
                    filtered = {k: v for k, v in item.items() if k in valid_keys}
                    p = Preset(**filtered).sanitized()
                    if p.name.lower() not in BUILTIN_KEYS:
                        self._custom[p.name.lower()] = p
        except Exception:
            pass

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"presets": [asdict(p) for p in self._custom.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_presets(self) -> List[Preset]:
        out: List[Preset] = [Preset(**asdict(b)) for b in BUILTIN_PRESETS]
        for p in self._custom.values():
            out.append(p)
        return out

    def get(self, name: str) -> Preset:
        key = name.strip().lower()
        if key in self._custom:
            return Preset(**asdict(self._custom[key]))
        for b in BUILTIN_PRESETS:
            if b.name.lower() == key:
                return Preset(**asdict(b))
        for b in BUILTIN_PRESETS:
            if b.name.lower() == DEFAULT_PRESET_NAME.lower():
                return Preset(**asdict(b))
        return Preset(**asdict(BUILTIN_PRESETS[0]))

    def save_preset(self, preset: Preset) -> Preset:
        clean = preset.sanitized()
        if clean.name.lower() in BUILTIN_KEYS:
            clean.name = f"{clean.name} (Custom)"
        clean.builtin = False
        self._custom[clean.name.lower()] = clean
        self._write()
        return clean

    def delete_preset(self, name: str) -> bool:
        key = name.strip().lower()
        if key in self._custom:
            del self._custom[key]
            self._write()
            return True
        return False

    def is_custom(self, name: str) -> bool:
        return name.strip().lower() in self._custom
