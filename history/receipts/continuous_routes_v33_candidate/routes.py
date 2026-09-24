"""Meter-based continuous courses; export identical visual/collision triangles."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import trimesh


@dataclass(frozen=True)
class Course:
    kind: str
    height: float = .1
    slope_deg: float = 15.
    platform: float = 1.5
    gap: float = 1.
    count: int = 3
    seed: int = 0
    width: float = 6.


def profile(cfg):
    if cfg.kind not in ('curbs', 'blocks', 'ramp_stairs_ramp'):
        raise ValueError('Unknown course')
    if not (.05 <= cfg.height <= .4 and 5 <= cfg.slope_deg <= 45):
        raise ValueError('Height or slope outside design envelope')
    if not (2 <= cfg.count <= 5 and 1.2 <= cfg.platform <= 1.8
            and .8 <= cfg.gap <= 1.5 and cfg.width >= 3.):
        raise ValueError('Invalid platform, gap, count or width')
    if cfg.kind == 'ramp_stairs_ramp' and (cfg.height > .3 or cfg.count > 3):
        raise ValueError('Ramp course is limited to three 30 cm risers')
    rng = np.random.default_rng(cfg.seed)
    points = [(0., 0.), (2., 0.)]
    events = []

    def add(x, z, name):
        points.append((float(x), float(z)))
        events.append(dict(x=float(x), z=float(z), name=name))

    x = 2.
    if cfg.kind in ('curbs', 'blocks'):
        for i in range(cfg.count):
            h = cfg.height if cfg.kind == 'curbs' else rng.uniform(.5, 1.) * cfg.height
            length = cfg.platform if cfg.kind == 'curbs' else rng.uniform(1.2, 1.8)
            gap = cfg.gap if cfg.kind == 'curbs' else rng.uniform(.8, 1.5)
            add(x, h, f'up_{i}')
            x += length
            add(x, h, f'top_end_{i}')
            add(x, 0., f'down_{i}')
            if i < cfg.count - 1:
                x += gap
                add(x, 0., f'gap_end_{i}')
    else:
        tangent = math.tan(math.radians(cfg.slope_deg))
        # Fixed ramp run; descending length follows accumulated elevation.
        x += 1.2
        z = 1.2 * tangent
        add(x, z, 'uphill_end')
        x += 1.
        add(x, z, 'transition_end')
        for i in range(cfg.count):
            z += cfg.height
            add(x, z, f'stair_{i}')
            x += .4 if i < cfg.count - 1 else cfg.platform
            add(x, z, f'tread_end_{i}')
        x += z / tangent
        add(x, 0., 'downhill_end')
    add(x + 2., 0., 'finish')
    return np.asarray(points), events


def triangulate_polygon(poly):
    """Ear clipping of our simple X-monotone side profile, no external mesher."""
    def cross(a, b, c):
        u, v = b-a, c-a
        return u[0]*v[1]-u[1]*v[0]
    area = sum(poly[i, 0]*poly[(i+1) % len(poly), 1] - poly[i, 1]*poly[(i+1) % len(poly), 0] for i in range(len(poly)))
    ids = list(range(len(poly))) if area > 0 else list(reversed(range(len(poly))))
    triangles = []
    while len(ids) > 3:
        for k, b in enumerate(ids):
            a, c = ids[k-1], ids[(k+1) % len(ids)]
            if cross(poly[a], poly[b], poly[c]) <= 1e-10:
                continue
            others = [j for j in ids if j not in (a, b, c)]
            inside = any(min(cross(poly[a], poly[b], poly[j]),
                             cross(poly[b], poly[c], poly[j]),
                             cross(poly[c], poly[a], poly[j])) >= -1e-10 for j in others)
            if inside:
                continue
            triangles.append((a, b, c))
            ids.pop(k)
            break
        else:
            raise ValueError('Invalid or degenerate course polygon')
    triangles.append(tuple(ids))
    return triangles


def generate(cfg):
    top, events = profile(cfg)
    side = np.vstack((top, [top[-1, 0], -.25], [0., -.25]))
    n = len(side)
    vertices = np.asarray([(x, y, z) for y in (-cfg.width/2, cfg.width/2) for x, z in side])
    triangles = triangulate_polygon(side)
    faces = list(triangles) + [tuple(n+i for i in reversed(t)) for t in triangles]
    for i in range(n):
        j = (i+1) % n
        faces.extend(((i, j, n+j), (i, n+j, n+i)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Course must be a closed, consistently oriented solid')
    if mesh.area_faces.min() <= 1e-10:
        raise ValueError('Degenerate collision triangle')
    delta = np.diff(top, axis=0)
    vertical = np.abs(delta[np.abs(delta[:, 0]) < 1e-9, 1])
    if len(vertical) and vertical.max() > .400001:
        raise ValueError('Adjacent vertical discontinuity exceeds 40 cm')
    # A finite-width lane cannot support an arbitrary 45-degree straight ray.
    # This bound reserves 0.6 m on either side for the robot envelope.
    safe_angle = math.degrees(math.atan2(cfg.width/2-.6, top[-1, 0]-1.))
    metadata = dict(config=asdict(cfg), units='m', up_axis='Z', forward_axis='X',
                    spawn=[1., 0., 0.], profile=top.tolist(), events=events,
                    length_m=float(top[-1, 0]), max_step_m=float(vertical.max()),
                    conservative_straight_yaw_deg=safe_angle,
                    required_width_for_yaw45_m=2*(top[-1, 0]-1.+.6),
                    seconds_at_03mps_without_margin=float((top[-1, 0]-1.)/.3),
                    training_integrated=False)
    return mesh, metadata
