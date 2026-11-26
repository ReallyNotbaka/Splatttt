# Splatttt

video to 3d scene app using 3d gaussian splatting (gsplat) and tsdf mesh reconstruction. made with pyside6 and cuda, mostly tested on a 6gb rtx 4050 laptop.

## what it does

- grabs motion spaced keyframes using pyav and nvdec
- estimates camera poses with pycolmap sfm
- trains 3dgs splats (with absgs, da3 depth priors and a sky dome for outdoor clips) or fuses a textured tsdf mesh
- lets you orbit, snap to keyframes, fly around and export .ply, .glb and .obj files

## setup and running

needs python 3.12, an nvidia gpu with cuda for 3dgs, and uv. on windows run `scripts/patch_gsplat_win.py` if gsplat complains about building extensions.

```powershell
# install deps
uv sync

# open the desktop app
uv run python main.py

# generate a quick demo video and load it
uv run python main.py --demo

# list presets
uv run python main.py --list-presets

# run headless cli mode
uv run python main.py --cli --demo --demo-duration 6.0 --preset "Balanced"
```

## tests

```powershell
uv run pytest -q
```
