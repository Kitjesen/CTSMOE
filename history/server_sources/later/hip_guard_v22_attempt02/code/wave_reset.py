"""Place wave resets near the actual collision surface without stepping simulation."""

import torch

from reset_geometry import CollisionGeometry, placement_delta


PROFILE = 'wave_surface_v1'
SURFACE_SPACING_M = 0.02
CLEARANCE_M = 0.03


def align_wave_reset(env, env_ids):
    # This event must execute after reset_robot_joints AND reset_base.
    from robot_lab.tasks.go2.mdp.utils import is_env_assigned_to_terrain
    from isaaclab.utils.warp import raycast_mesh

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[is_env_assigned_to_terrain(env, 'wave')[ids]]
    if not len(ids):
        return
    robot = env.scene['robot']
    if not hasattr(env, 'wave_reset_geometry'):
        spawn = robot.cfg.spawn
        if spawn.replace_cylinders_with_capsules or spawn.collision_from_visuals:
            raise ValueError('Wave reset geometry requires the URDF primitive collision model')
        env.wave_reset_geometry = CollisionGeometry(spawn.asset_path, robot.joint_names, env.device, SURFACE_SPACING_M)
        env.wave_reset_count = 0
        env.wave_reset_last = {}
    geometry = env.wave_reset_geometry
    pose = robot.data.root_state_w[ids, :7].clone()
    # Root and DOF buffers have been written by the preceding reset events.
    # FK avoids reading stale body transforms before the framework's sim.forward().
    points = geometry.world_points(robot.data.joint_pos[ids], pose)
    starts = points.clone()
    starts[..., 2] += 10.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    scanner = env.scene['height_scanner']
    hits = raycast_mesh(starts, directions, scanner.meshes[scanner.cfg.mesh_prim_paths[0]])[0]
    delta = placement_delta(points, hits[..., 2], CLEARANCE_M)
    pose[:, 2] += delta
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    env.wave_reset_count += len(ids)
    # Keep GPU scalars during rollout; materialize only at ordinary logging time.
    gaps = points[..., 2] + delta[:, None] - hits[..., 2]
    env.wave_reset_last = dict(count=len(ids), delta_min=delta.min(), delta_max=delta.max(),
                               collision_gap_min=gaps.min(),
                               nearest_wheel_gap_max=gaps[:, geometry.wheel_points].amin(1).max())


def reset_snapshot(env):
    geometry = getattr(env, 'wave_reset_geometry', None)
    return dict(profile=getattr(env.cfg, 'wave_reset_profile', 'original'),
                count=getattr(env, 'wave_reset_count', 0), clearance_m=CLEARANCE_M,
                surface_spacing_m=SURFACE_SPACING_M,
                geometry_sha256=geometry.urdf_sha256 if geometry else None,
                surface_points=len(geometry.owners) if geometry else 0,
                last={key: float(value) if isinstance(value, torch.Tensor) else value
                      for key, value in getattr(env, 'wave_reset_last', {}).items()})


def configure_wave_reset(cfg, profile):
    if profile not in ('original', PROFILE):
        raise ValueError(f'Unknown wave reset profile: {profile}')
    cfg.wave_reset_profile = profile
    if profile == 'original':
        return
    from isaaclab.managers import EventTermCfg

    # Preserve the original pose/joint/velocity draws and all non-wave resets.
    # This final reset event replaces only the wave root z with surface clearance.
    cfg.events.wave_surface = EventTermCfg(func=align_wave_reset, mode='reset')
