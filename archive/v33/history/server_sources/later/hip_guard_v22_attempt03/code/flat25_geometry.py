"""Large level patches for full-duration flat behavior evaluation only."""
import numpy as np
import trimesh


def flat_patch(difficulty, cfg):
    del difficulty
    size = np.asarray(cfg.size, dtype=float)
    if size.shape != (2,) or (size < 80.).any():
        raise ValueError("Full-duration evaluation needs at least an 80 m square patch")
    mesh = trimesh.creation.box(extents=(size[0], size[1], .2))
    mesh.apply_translation((size[0] / 2., size[1] / 2., -.1))
    return [mesh], np.array([size[0] / 2., size[1] / 2., 0.])
