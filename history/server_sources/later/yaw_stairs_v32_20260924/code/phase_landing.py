"""Keep original stair risers but reserve a landing within each 9m tile."""
import numpy as np


def capped_vertices(vertices, kind):
    if kind not in ('stairs_up', 'stairs_down'):
        raise ValueError(kind)
    result = vertices.copy()
    radius = np.abs(vertices[:, :2]).max(-1)
    # The upstream height-field mesh is centered before placement in the atlas.
    if abs(vertices[:, 0].min()+4.5) > .2 or abs(vertices[:, 0].max()-4.5) > .2:
        raise ValueError('Landing requires the original centered 9m stair tile')
    band = (radius >= 2.7) & (radius <= 2.9)
    values, counts = np.unique(np.round(vertices[band, 2], 5), return_counts=True)
    if not len(values):
        raise ValueError('No stair surface at the planned landing transition')
    cap = values[counts.argmax()]
    center = np.median(vertices[radius < 1., 2])
    if (kind == 'stairs_up' and cap <= center) or (kind == 'stairs_down' and cap >= center):
        raise ValueError('Stair direction differs from expected outward traversal')
    result[:, 2] = np.minimum(vertices[:, 2], cap) if kind == 'stairs_up' else np.maximum(vertices[:, 2], cap)
    return result


def cap_stair_mesh(mesh, kind):
    mesh = mesh.copy()
    mesh.vertices = capped_vertices(np.asarray(mesh.vertices), kind)
    # Capping makes the unused outer vertical faces degenerate; remove only
    # zero-area triangles, retaining the original inner risers and landing.
    mesh.update_faces(mesh.area_faces > 1e-10)
    mesh.remove_unreferenced_vertices()
    return mesh
