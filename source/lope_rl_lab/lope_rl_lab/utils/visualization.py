"""Flat-shaded two-tone terrain overlay for crisp Newton GL playback videos."""
from __future__ import annotations


def _build_edge_rods(segments, radius: float = 0.0028, sections: int = 4):
    """Build crisp flat-shaded bevel rods along sharp terrain creases."""
    import numpy as np

    starts = segments[:, 0].astype(np.float64)
    ends = segments[:, 1].astype(np.float64)
    dirs = ends - starts
    lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(lengths, 1.0e-8)
    helper = np.zeros_like(dirs)
    helper[np.arange(len(dirs)), np.argmin(np.abs(dirs), axis=1)] = 1.0
    side = np.cross(dirs, helper)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1.0e-8)
    up = np.cross(dirs, side)

    angles = np.arange(sections) * (2.0 * np.pi / sections) + (np.pi / 4.0)
    ring = (
        np.cos(angles)[None, :, None] * side[:, None, :]
        + np.sin(angles)[None, :, None] * up[:, None, :]
    ) * radius
    verts = np.empty((len(segments), 2 * sections, 3), dtype=np.float32)
    verts[:, :sections] = starts[:, None, :] + ring
    verts[:, sections:] = ends[:, None, :] + ring

    ri = np.arange(sections)
    ni = (ri + 1) % sections
    rod_faces = np.concatenate(
        (
            np.stack((ri, ni, sections + ni), axis=1),
            np.stack((ri, sections + ni, sections + ri), axis=1),
        ),
        axis=0,
    )
    faces = (rod_faces[None] + (np.arange(len(segments)) * (2 * sections))[:, None, None]).reshape(-1, 3)
    tri_pts = verts.reshape(-1, 3)[faces]
    face_normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1.0e-8)
    return tri_pts.reshape(-1, 3).astype(np.float32), np.repeat(face_normals, 3, axis=0).astype(np.float32)


def prepare_crisp_terrain_visualizer(raw_env, patch_radius_m: float = 40.0):
    """Replace smooth-shaded Newton GL terrain with crisp flat-shaded treads, risers, and dark edges.

    Returns an idempotent callback returning ``True`` once installed (or when not applicable).
    """
    visualizers = getattr(getattr(raw_env, "sim", None), "visualizers", None) or []
    gl_viz = next(
        (
            viz
            for viz in visualizers
            if hasattr(viz, "log_mesh")
            and hasattr(viz, "_viewer")
            and "newton" in type(viz).__name__.lower()
        ),
        None,
    )
    if gl_viz is None:
        return lambda: True

    stage = getattr(getattr(raw_env, "scene", None), "stage", None)
    if stage is None:
        return lambda: True
    prim = stage.GetPrimAtPath("/World/ground/terrain/mesh")
    if not prim or not prim.IsValid():
        return lambda: True

    import numpy as np
    import trimesh
    import warp as wp
    from pxr import UsdGeom

    mesh_prim = UsdGeom.Mesh(prim)
    raw_pts = mesh_prim.GetPointsAttr().Get()
    raw_idx = mesh_prim.GetFaceVertexIndicesAttr().Get()
    if not raw_pts or not raw_idx:
        return lambda: True

    pts = np.asarray(raw_pts, dtype=np.float32)
    idx = np.asarray(raw_idx, dtype=np.int32).reshape(-1, 3)
    tri_pts = pts[idx]
    face_normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid = fn_len[:, 0] > 1.0e-8
    idx_valid = idx[valid]
    tri_pts = tri_pts[valid]
    face_normals = face_normals[valid] / fn_len[valid]
    is_top = np.abs(face_normals[:, 2]) > 0.7

    env_origin = raw_env.scene.env_origins[0].detach().cpu().numpy()
    tri_ctr = tri_pts.mean(axis=1)
    near = (
        (np.abs(tri_ctr[:, 0] - env_origin[0]) <= patch_radius_m)
        & (np.abs(tri_ctr[:, 1] - env_origin[1]) <= patch_radius_m)
    )
    if not np.any(near):
        return lambda: True

    local_faces = idx_valid[near]
    local_fn = face_normals[near]
    tm = trimesh.Trimesh(vertices=pts, faces=local_faces, process=False)
    adj_pairs = tm.face_adjacency
    rod_v, rod_n = None, None
    if len(adj_pairs) > 0:
        crease_mask = (
            (np.degrees(tm.face_adjacency_angles) > 30.0)
            & (local_fn[adj_pairs[:, 0], 2] > -0.5)
            & (local_fn[adj_pairs[:, 1], 2] > -0.5)
        )
        if np.any(crease_mask):
            segs = tm.vertices[tm.face_adjacency_edges[crease_mask]]
            lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
            segs = segs[lengths > 1.0e-4]
            if len(segs) > 0:
                rod_v, rod_n = _build_edge_rods(segs, radius=0.0028, sections=4)

    installed = False

    def ensure_installed() -> bool:
        nonlocal installed
        if installed:
            return True
        viewer = getattr(gl_viz, "_viewer", None)
        if viewer is None:
            return False
        dev = viewer.device
        if hasattr(viewer, "renderer"):
            viewer.renderer.spotlight_enabled = False
        if hasattr(gl_viz, "set_camera_view") and hasattr(viewer, "camera"):
            cam_pos = viewer.camera.pos
            cam_xy = np.array([float(cam_pos[0]), float(cam_pos[1])], dtype=np.float32)
            if float(np.linalg.norm(cam_xy - env_origin[:2])) > 12.0:
                eye_z = max(float(env_origin[2]) + 2.2, 0.85)
                gl_viz.set_camera_view(
                    eye=(float(env_origin[0]) + 2.8, float(env_origin[1]) - 2.8, eye_z),
                    target=(float(env_origin[0]), float(env_origin[1]), float(env_origin[2]) + 0.30),
                )
        static_batches = []
        if hasattr(viewer, "_shape_instances"):
            static_batches.extend(viewer._shape_instances.values())
        for group in getattr(viewer, "_packed_groups", ()) or ():
            if len(group) >= 2:
                static_batches.append(group[1])
        for batch in static_batches:
            worlds = batch.worlds.numpy() if hasattr(batch.worlds, "numpy") else np.asarray(batch.worlds)
            if len(worlds) > 0 and worlds[0] == -1:
                if hasattr(batch, "flags"):
                    batch.flags = 0
                if batch.name in viewer.objects:
                    viewer.objects[batch.name].hidden = True
                if batch.mesh in viewer.objects:
                    viewer.objects[batch.mesh].hidden = True
        for name, mask, color, roughness in (
            ("/lope_crisp_terrain/treads", near & is_top, (0.85, 0.86, 0.87), 0.88),
            ("/lope_crisp_terrain/risers", near & (~is_top), (0.43, 0.49, 0.58), 0.90),
        ):
            if not np.any(mask):
                continue
            v = tri_pts[mask].reshape(-1, 3).astype(np.float32)
            n = np.repeat(face_normals[mask], 3, axis=0).astype(np.float32)
            inds = np.arange(len(v), dtype=np.int32)
            gl_viz.log_mesh(
                name,
                wp.array(v, dtype=wp.vec3, device=dev),
                wp.array(inds, dtype=wp.int32, device=dev),
                normals=wp.array(n, dtype=wp.vec3, device=dev),
                color=color,
                roughness=roughness,
                metallic=0.0,
            )
        if rod_v is not None and len(rod_v) > 0:
            gl_viz.log_mesh(
                "/lope_crisp_terrain/edges",
                wp.array(rod_v, dtype=wp.vec3, device=dev),
                wp.array(np.arange(len(rod_v), dtype=np.int32), dtype=wp.int32, device=dev),
                normals=wp.array(rod_n, dtype=wp.vec3, device=dev),
                color=(0.12, 0.15, 0.20),
                roughness=0.92,
                metallic=0.0,
            )
        installed = True
        print(
            f"[INFO] Installed crisp flat-shaded terrain visualizer "
            f"(treads={int(np.sum(near & is_top))}, risers={int(np.sum(near & (~is_top)))}, "
            f"edge_verts={0 if rod_v is None else len(rod_v)})",
            flush=True,
        )
        return True

    return ensure_installed
