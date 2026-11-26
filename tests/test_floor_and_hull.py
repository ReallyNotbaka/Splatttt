import numpy as np
from src.scene_builder import _fill_floor_grid_holes


def test_fill_floor_grid_holes_no_false_exterior_extension():
    # Construct a synthetic floor mesh
    # A 2m x 2m room with vertices at y=0, with an enclosed 40cm hole in the middle
    grid_coords = []
    cell = 0.025
    for x in np.arange(-1.0, 1.0, cell):
        for z in np.arange(-1.0, 1.0, cell):
            # Hole in center: |x| < 0.2 and |z| < 0.2
            if abs(x) < 0.20 and abs(z) < 0.20:
                continue
            grid_coords.append([x, 0.0, z])

    verts = np.array(grid_coords, dtype=np.float32)
    n_v = len(verts)
    cols = np.full((n_v, 3), 160, dtype=np.uint8)
    norms = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n_v, 1))

    # Dummy faces
    faces = np.zeros((1000, 3), dtype=np.int32)
    faces[:, 0] = np.arange(1000) % (n_v - 2)
    faces[:, 1] = faces[:, 0] + 1
    faces[:, 2] = faces[:, 0] + 2

    new_v, new_f, new_c, new_n = _fill_floor_grid_holes(verts, faces, cols, norms)

    # Should have patched the enclosed hole
    assert len(new_v) > len(verts), "Enclosed hole was not patched!"
    # Ensure all patched vertices stay strictly within the room bounding box [-1.0, 1.0] (no exterior protrusion)
    patched_v = new_v[len(verts):]
    assert np.all(patched_v[:, 0] >= -1.05) and np.all(patched_v[:, 0] <= 1.05)
    assert np.all(patched_v[:, 2] >= -1.05) and np.all(patched_v[:, 2] <= 1.05)
