"""Explicit radial route variants in five existing stair columns, with no pit allocation."""
from __future__ import annotations

import numpy as np
import trimesh

PROFILE = 'radial_routes_v30'
VARIANTS = {6: 'single_curb_up', 7: 'short_tread_up', 9: 'long_tread_up',
            10: 'ramp_then_stairs_up', 12: 'short_tread_down'}


def segments(kind, level):
    """Return (inner radius, outer radius, inner height, outer height).

    A vertical riser occurs between adjacent rings with different endpoint heights.
    Central spawn disk is flat for 1.5m in every cardinal direction.
    """
    if kind not in VARIANTS.values() or not 0 <= level <= 9:
        raise ValueError('Unknown route or invalid curriculum level')
    h = .05 + .15 * level / 9
    rings = [(0., 1.5, 0., 0.)]
    if kind == 'single_curb_up':
        return rings + [(1.5, 4.5, h, h)]
    if kind == 'ramp_then_stairs_up':
        slope_height = .7 * np.tan(np.deg2rad(5. + 10. * level / 9))
        rings += [(1.5, 2.2, 0., float(slope_height)), (2.2, 2.5, float(slope_height), float(slope_height))]
        start, base, tread, count = 2.5, float(slope_height), .3, 3
    else:
        start, base = 1.5, 0.
        tread = .35 if kind == 'long_tread_up' else .25
        count = 4
    sign = -1. if kind == 'short_tread_down' else 1.
    for index in range(count):
        z = base + sign * h * (index + 1)
        rings.append((start + index * tread,
                      start + (index + 1) * tread if index < count - 1 else 4.5, z, z))
    return rings


def mesh_for(kind, level):
    rings = segments(kind, level)
    vertices, faces = [], []

    def quad(points):
        offset = len(vertices)
        vertices.extend(points)
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))

    def square(radius, z):
        return [(radius, radius, z), (-radius, radius, z),
                (-radius, -radius, z), (radius, -radius, z)]

    previous_z = 0.
    for a, b, za, zb in rings:
        outer = square(b, zb)
        if a == 0:
            quad(outer)
        else:
            inner = square(a, za)
            for side in range(4):
                nxt = (side + 1) % 4
                quad((inner[side], outer[side], outer[nxt], inner[nxt]))
            if za != previous_z:
                low = square(a, previous_z)
                for side in range(4):
                    nxt = (side + 1) % 4
                    quad((low[side], inner[side], inner[nxt], low[nxt]))
        previous_z = zb
    bottom_z = min(min(r[2:]) for r in rings) - .2
    bottom, top = square(4.5, bottom_z), square(4.5, previous_z)
    quad(bottom[::-1])
    for side in range(4):
        nxt = (side + 1) % 4
        quad((bottom[side], bottom[nxt], top[nxt], top[side]))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or np.any(mesh.area_faces <= 0):
        raise ValueError('Route mesh is not a closed nondegenerate collision surface')
    return mesh


class VarietyMixin:
    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if not hasattr(self, 'variety_receipts'):
            self.variety_receipts = []
        if col in VARIANTS:
            mesh = mesh_for(VARIANTS[col], min(row, 9))
            origin = np.array([0., 0., 0.])
            self.variety_receipts.append(dict(row=int(row), column=int(col),
                kind=VARIANTS[col], level=min(int(row), 9),
                rings=segments(VARIANTS[col], min(row, 9)),
                watertight=bool(mesh.is_watertight)))
        result = super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        self.cfg.variety_receipts = self.variety_receipts
        return result


def configure(cfg):
    terrain = cfg.scene.terrain.terrain_generator
    if terrain.num_cols != 40 or terrain.num_rows != 22:
        raise ValueError('V30 requires the established atlas')
    parent = terrain.class_type
    cls = type('VarietyTerrainGenerator', (VarietyMixin, parent), {'__module__': __name__})
    globals()[cls.__name__] = cls
    terrain.class_type = cls
    cfg.terrain_variety_profile = PROFILE


def snapshot(env):
    if getattr(env.cfg, 'terrain_variety_profile', 'off') != PROFILE:
        return {'profile': 'off'}
    terrain = env.scene.terrain
    values = {}
    for col, kind in VARIANTS.items():
        mask = terrain.terrain_types == col
        values[kind] = dict(column=col, count=int(mask.sum()),
                           mean_level=float(terrain.terrain_levels[mask].float().mean()) if bool(mask.any()) else None)
    signs = getattr(env, 'stair_motion_sign', None)
    return dict(profile=PROFILE, variants=values,
                reverse_envs=int((signs < 0).sum()) if signs is not None else 0,
                episode_statistics=(env.variety_episode_stats.cpu().tolist()
                                    if hasattr(env, 'variety_episode_stats') else None),
                statistic_axes=['variant_in_column_order', 'teacher_student', 'forward_backward',
                                'episodes_radius_exit_physical_failure'],
                interpretation='Training distribution; not successful traversal counts')


def record(env, command, ids, passed, failed):
    if getattr(env.cfg, 'terrain_variety_profile', 'off') != PROFILE:
        return
    import torch
    if not hasattr(env, 'variety_episode_stats'):
        env.variety_episode_stats = torch.zeros((5, 2, 2, 3), dtype=torch.long, device=env.device)
    for index, col in enumerate(VARIANTS):
        mask = env.scene.terrain.terrain_types[ids] == col
        selected = ids[mask]
        key = command.learner[selected] * 2 + (env.stair_motion_sign[selected] < 0).long()
        values = torch.stack((torch.ones_like(key), passed[mask].long(), failed[mask].long()), -1)
        env.variety_episode_stats[index].view(4, 3).index_add_(0, key, values)
