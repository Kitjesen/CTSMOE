"""Compare FK with PhysX and dense collision clearance across randomized wave resets."""
import json
import os
from pathlib import Path
import sys

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_reset_20260919'
sys.path[:0] = [str(root / 'code_wave_reset_20260919'), str(root / 'upstream/source/rsl_rl'),
               str(root / 'upstream/source/robot_lab'), str(root / 'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher  # noqa: E402
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import torch
    from isaaclab.utils.warp import raycast_mesh
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.utils import _get_terrain_column_range
    from config import make_config
    from stair_speed import configure_stair_speed
    from train import apply_command_curriculum_start
    from wave_reset import configure_wave_reset, reset_snapshot, align_wave_reset
    from reset_geometry import CollisionGeometry

    torch.set_num_threads(2)
    cfg = make_config(96, root/'cache/usd', actuator_profile='rs0402_slew12_v3')
    cfg.seed = 42
    apply_command_curriculum_start(cfg.commands.base_velocity, 'intermediate')
    configure_stair_speed(cfg, 'straight_speed_v1')
    configure_wave_reset(cfg, 'wave_surface_v1')
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    raw = Go2Env(cfg)
    try:
        robot, terrain = raw.scene['robot'], raw.scene.terrain
        names = ['wave']*80 + ['stairs_up']*8 + ['stairs_down']*8
        cols = [_get_terrain_column_range(cfg.scene.terrain.terrain_generator, name, raw.device)[0] for name in names]
        terrain.terrain_types[:] = torch.tensor(cols, device=raw.device)
        terrain.terrain_levels[:] = torch.arange(96, device=raw.device) % 10
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        command = raw.command_manager.get_term('base_velocity')
        command._init_terrain_infos()
        command._update_env_command_ranges()
        dense = CollisionGeometry(robot.cfg.spawn.asset_path, robot.joint_names, raw.device, spacing=.005)
        scanner = raw.scene['height_scanner']
        records = []
        for repeat in range(8):
            raw.reset()
            raw.sim.forward()
            raw.scene.update(raw.physics_dt)
            geometry = raw.wave_reset_geometry
            pos, rot = geometry.transforms(robot.data.joint_pos, robot.data.root_state_w[:, :7])
            common = [name for name in robot.body_names if name in geometry.link_names]
            gi, bi = [geometry.link_names.index(n) for n in common], [robot.body_names.index(n) for n in common]
            fk_error = (pos[:, gi] - robot.data.body_link_pos_w[:, bi]).abs().max().item()
            assert fk_error < 2e-4, fk_error
            minimum, nearest_wheels, bad_body = [], [], 0
            for start in range(0, 80, 4):
                ids = slice(start, start+4)
                points = dense.world_points(robot.data.joint_pos[ids], robot.data.root_state_w[ids, :7])
                rays = points.clone()
                rays[..., 2] += 10
                directions = torch.zeros_like(rays)
                directions[..., 2] = -1
                hits = raycast_mesh(rays, directions, scanner.meshes[scanner.cfg.mesh_prim_paths[0]])[0]
                gaps = points[..., 2]-hits[..., 2]
                assert torch.isfinite(gaps).all()
                minimum.extend(gaps.amin(1).cpu().tolist())
                wg = gaps[:, dense.wheel_points].amin(1)
                nearest_wheels.extend(wg.cpu().tolist())
                bad_body += int((wg > .06).sum())
            assert min(minimum) > 0, min(minimum)
            # Calling the wave placement event on staircase ids must be an exact no-op.
            before = robot.data.root_state_w[80:].clone()
            align_wave_reset(raw, torch.arange(80, 96, device=raw.device))
            torch.testing.assert_close(robot.data.root_state_w[80:], before, rtol=0, atol=0)
            records.append(dict(repeat=repeat, fk_error_m=fk_error, dense_min_gap_m=min(minimum),
                                nearest_wheel_max_m=max(nearest_wheels), wheels_over_6cm=bad_body,
                                snapshot=reset_snapshot(raw)))
            print(json.dumps(records[-1]), flush=True)
        result = dict(passed=True, randomized_wave_resets=640, unchanged_stair_resets=128,
                      placement_surface_spacing_m=.02, independent_check_spacing_m=.005,
                      points=len(dense.owners), records=records)
        (folder/'geometry_validation.json').write_text(json.dumps(result, indent=2))
    finally:
        raw.close()
finally:
    app.close()
