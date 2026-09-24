"""Conservative terminal-stair support witness from Thunder collision geometry.

The stock per-wheel ContactSensor reports terrain force but not contact XY.
This provider uses each wheel collider link origin and an over-approximated
wheel envelope, plus sampled URDF collision surfaces for the rest of the body.
It is a geometric inference, never labelled as a measured contact point.
"""

import hashlib
import math
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


SIDES = ("fr", "fl", "rr", "rl")
AXES = ("positive_x", "positive_y", "negative_x", "negative_y")


@lru_cache(maxsize=4)
def validated_wheel_envelope(urdf_path):
    """Bound wheel collider XY extent about its link origin for any orientation."""
    path = Path(urdf_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ET.parse(path).getroot()
    envelope = 0.0
    for side in SIDES:
        links = [link for link in tree.findall("link") if link.get("name") == f"{side}_foot_Link"]
        if len(links) != 1:
            raise ValueError(f"Expected exactly one {side} wheel collider link")
        collisions = links[0].findall("collision")
        if len(collisions) != 1:
            raise ValueError(f"Unexpected {side} wheel collision geometry")
        origin = collisions[0].find("origin")
        cylinder = collisions[0].find("geometry/cylinder")
        if origin is None or cylinder is None:
            raise ValueError(f"Expected a cylinder with an explicit origin for {side}")
        xyz = [float(part) for part in origin.get("xyz").split()]
        radius, length = float(cylinder.get("radius")), float(cylinder.get("length"))
        if len(xyz) != 3 or not all(map(math.isfinite, (*xyz, radius, length))):
            raise ValueError("Nonfinite wheel collision dimensions")
        if not (0.08 <= radius <= 0.11 and 0 < length < 0.08):
            raise ValueError("Unexpected wheel collision dimensions")
        # Bounding sphere around the link origin conservatively encloses the
        # rotated cylinder, including its URDF collision-origin displacement.
        envelope = max(envelope, math.dist(xyz, (0.0, 0.0, 0.0)) + radius + length / 2)
    if not 0.09 < envelope < 0.20:
        raise ValueError("Invalid wheel collision envelope")
    return digest, envelope


def platform_world_z(env_origins_z, receipts, rows, columns, axes, device):
    """Map measured tile mesh plateau Z through the saved terrain origin Z."""
    offsets = []
    for row, column, axis in zip(rows, columns, axes):
        receipt = receipts[(row, column)]
        if not receipt["selected"] or not receipt["origin_unchanged"]:
            raise ValueError("Stair platform receipt lacks a stable terrain origin")
        runs = receipt["geometry_after"][AXES[axis]]["surface_runs"]
        offsets.append(float(runs[-1]["z_m"]) - float(receipt["origin_after"][2]))
    return env_origins_z + torch.tensor(offsets, dtype=env_origins_z.dtype, device=device)


def conservative_contact_provider(env, ids):
    """Return a provenance-marked support observation for MixedStairCommandMixin."""
    command = env.command_manager.get_term("base_velocity")
    pit = command.pit
    robot = env.scene["robot"]
    collision = pit.geometry
    urdf_path = env.cfg.scene.robot.spawn.asset_path
    urdf_hash, wheel_envelope = validated_wheel_envelope(urdf_path)
    if collision.urdf_sha256 != urdf_hash or abs(collision.spacing - 0.04) > 1e-9:
        raise ValueError("Collision hull differs from the verified Thunder URDF")
    if [robot.body_names[i] for i in pit.wheel_ids] != [f"{side}_foot_Link" for side in SIDES]:
        raise ValueError("Wheel body order differs from the contact-sensor order")
    if len(ids) == 0:
        raise ValueError("Support provider expects nonempty environment IDs")
    wheel_centers = robot.data.body_pos_w[ids][:, pit.wheel_ids]
    hull = collision.world_points(robot.data.joint_pos[ids], robot.data.root_state_w[ids, :7])
    hull = hull[:, ~collision.wheel_points, :2]
    if hull.shape[1] < 4 or not torch.isfinite(hull).all():
        raise ValueError("No finite physical body/leg collision hull")
    force = torch.stack([
        env.scene[f"pit_contact_{side}"].data.force_matrix_w[ids, 0, 0, 2].clamp_min(0)
        for side in SIDES
    ], dim=1)
    terrain = env.scene.terrain
    row = terrain.terrain_levels[ids].tolist()
    column = terrain.terrain_types[ids].tolist()
    axis = env.stair_axis[ids].tolist()
    plateau_z = platform_world_z(env.scene.env_origins[ids, 2], command._mixed_receipts,
                                 row, column, axis, env.device)
    return {
        "kind": "conservative_center_hull",
        "validated_collision_geometry": True,
        "collision_geometry_id": f"urdf:{urdf_hash}:sample_spacing:{collision.spacing}:wheel_envelope_v1",
        "wheel_radius_m": wheel_envelope,
        "wheel_centers_world_xyz": wheel_centers,
        "body_collision_hull_world_xy": hull,
        "platform_world_z_m": plateau_z,
        "terrain_forces_n": force,
    }
