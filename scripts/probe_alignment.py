"""Optimizer-free 200 Hz fixed-target isolation for the fresh Go2-aligned task.

Run nominal and randomized in separate processes with different output paths.
The randomized variant retains mass/CoM/material/gains/zero-offset uncertainty;
both disable pushes and reset pose noise to isolate actuator/load stability.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def qualified_class(value):
    """Lab resolves config class strings in place when constructing the simulator."""
    if isinstance(value, str):
        resolver = getattr(value, '_resolve', None)
        if resolver is not None:
            return qualified_class(resolver())
        return value
    cls = value if isinstance(value, type) else type(value)
    return f'{cls.__module__}:{cls.__qualname__}'


def summarize(data, dt, wheel_ids, leg_ids, base_ids):
    """Measure first episodes and exclude reset discontinuities from sign changes."""
    import numpy as np
    force = np.linalg.norm(data['contact_normal'], axis=-1)
    first = data['episode_index'] == 0
    steady = first & ((np.arange(len(first)) + 1) * dt >= 2.)[:, None]
    tau = data['torque_proxy'][..., :12]
    adjacent = first[1:] & first[:-1]
    active = (np.abs(tau[1:]) > 5.) & (np.abs(tau[:-1]) > 5.)
    flips = (tau[1:] * tau[:-1] < 0.) & active & adjacent[..., None]
    gravity = data['gravity'][steady]
    def peak(ids):
        return float(force[..., ids][first].max()) if ids and first.any() else None
    metrics = {
        'first_episode_seconds_per_env': (first.sum(axis=0) * dt).tolist(),
        'leg_torque_proxy_abs_peak_Nm': np.abs(tau[first]).max(axis=0).tolist(),
        'leg_torque_120Nm_samples': ((np.abs(tau) >= 119.9) & first[..., None]).sum(axis=(0, 1)).tolist(),
        'adjacent_leg_torque_signflip_over5Nm_counts': flips.sum(axis=(0, 1)).tolist(),
        'signflip_env_ids': np.flatnonzero(flips.any(axis=(0, 2))).tolist(),
        'wheel_normal_peak_N': peak(wheel_ids),
        'leg_normal_peak_N': peak(leg_ids),
        'base_normal_peak_N': peak(base_ids),
        'first_episode_physics_env_samples': int(first.sum()),
        'steady_first_episode_physics_env_samples': int(steady.sum()),
    }
    if steady.any():
        roll = np.degrees(np.arctan2(-gravity[:, 1], -gravity[:, 2]))
        pitch = np.degrees(np.arctan2(gravity[:, 0], np.linalg.norm(gravity[:, 1:], axis=-1)))
        support = (force[..., wheel_ids][steady] > 20.).sum(axis=-1)
        metrics.update(
            root_z_steady_median=float(np.median(data['root_position'][..., 2][steady])),
            roll_abs_p99_deg=float(np.quantile(np.abs(roll), .99)),
            pitch_abs_p99_deg=float(np.quantile(np.abs(pitch), .99)),
            wheel_support_count_fraction_0_to_4=[float((support == i).mean()) for i in range(5)],
            leg_contact_over5N_steady_fraction=float((force[..., leg_ids][steady] > 5.).any(axis=-1).mean()),
        )
    return metrics


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('nominal', 'randomized'), required=True)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1250)
    parser.add_argument('--seed', type=int, default=42)
    go2 = project / 'upstream/go2_rl_lab'
    sys.path[:0] = [str(project / 'source'), str(go2 / 'source/rsl_rl'), str(go2 / 'source/go2_rl_lab')]
    from isaacsim.asset.importer.urdf import URDFImporter  # noqa: F401
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 4 or args.num_envs % 4:
        parser.error('steps must be positive; num-envs must be positive and divisible by four')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['NEWPOSE_URDF'] = str(project / 'source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf')
    os.environ['NEWPOSE_USD_CACHE'] = str(output / 'converted')
    os.environ.pop('NEWPOSE_CANDIDATE_OVERRIDES', None)

    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp
    from thunder_lab3.tasks.thunder import FRESH_TASK_ID
    from thunder_lab3.tasks.thunder.env_cfg import ThunderFreshEnvCfg, CommandsCfg
    from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS, WHEEL_BODIES
    from thunder_lab3.training.history import validate_observations

    cfg = ThunderFreshEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
    if (cfg.sim.dt, cfg.decimation, cfg.episode_length_s) != (.005, 4, 25.):
        raise ValueError('This probe requires the fresh 5 ms x4 / 25 s configuration')
    cfg.curriculum = None
    flat = cfg.scene.terrain.terrain_generator.sub_terrains['flat']
    flat.proportion = 1.
    cfg.scene.terrain.terrain_generator.sub_terrains = {'flat': flat}
    cfg.scene.terrain.fixed_level, cfg.gradual_boxes = 0, False
    cfg.events.reset_root.params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.reset_root.params['velocity_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    if getattr(cfg.events, 'reset_robot_joints', None) is not None:
        cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    if args.variant == 'nominal':
        for name in vars(cfg.events):
            if name.startswith('randomize_'):
                setattr(cfg.events, name, None)
        cfg.sim.physics.solver_cfg.body_com_buckets = None
    cfg.events.randomize_push_robot = None
    cfg.commands = CommandsCfg()
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0., 0.)
    command.rel_standing_envs, command.rel_pure_yaw_envs, command.rel_pure_lateral_envs = 1., 0., 0.

    rows = []
    resets = np.zeros(args.num_envs, dtype=np.int32)
    early = np.zeros_like(resets)
    result = dict(completed=False, training_started=False, optimizer_updates=0,
                  variant=args.variant, num_envs=args.num_envs, seed=args.seed,
                  physics_dt=cfg.sim.dt, decimation=cfg.decimation, episode_length_s=cfg.episode_length_s,
                  steps_requested=args.steps,
                  overrides='flat L0; zero commands/actions; deterministic pose/velocity reset; no curriculum or interval pushes',
                  scope='Fixed targets with optional domain uncertainty; not learned movement or policy acceptance. Applied effort is an actuator proxy, not a torque sensor.')
    step = 0
    with launch_simulation(cfg.sim, args):
        env = gym.make(FRESH_TASK_ID, cfg=cfg)
        base = env.unwrapped
        try:
            obs, _ = env.reset()
            validate_observations(obs)
            robot, contact = base.scene['robot'], base.scene['contact_forces']
            joints = [robot.joint_names.index(name) for name in JOINTS]
            names = list(contact.body_names)
            wheels = [names.index(name) for name in WHEEL_BODIES]
            legs = [i for i, name in enumerate(names) if '_thigh_' in name or '_calf_' in name]
            bodies = [i for i, name in enumerate(names) if name == 'base_link']
            result.update(joint_names=list(JOINTS), body_names=list(robot.body_names), contact_body_names=names,
                          backend=type(robot).__module__, termination_terms=list(base.termination_manager.active_terms))
            result['physics_manager_class'] = qualified_class(base.sim.physics_manager)
            result['configured_physics_class'] = qualified_class(cfg.sim.physics.class_type)
            result['configured_solver_class'] = qualified_class(cfg.sim.physics.solver_cfg.class_type)
            if result['physics_manager_class'] != result['configured_solver_class']:
                raise RuntimeError('Actual physics manager does not match the configured randomized solver')

            def snapshot(label):
                model = base.sim.physics_manager.get_model()
                values = {
                    'body_mass': robot.data.body_mass.torch,
                    'body_com_pos_b': robot.data.body_com_pos_b.torch,
                    'leg_stiffness': robot.actuators['legs'].stiffness,
                    'leg_damping': robot.actuators['legs'].damping,
                    'joint_solver_stiffness': robot.data.joint_stiffness.torch,
                    'joint_solver_damping': robot.data.joint_damping.torch,
                    'leg_action_offsets': base.action_manager.get_term('joint_pos')._offset,
                    'shape_friction_mu': wp.to_torch(robot._root_view.get_attribute('shape_material_mu', model)[:, 0]),
                    'shape_restitution': wp.to_torch(robot._root_view.get_attribute('shape_material_restitution', model)[:, 0]),
                }
                arrays = {name: value.detach().cpu().numpy().copy() for name, value in values.items()}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise RuntimeError('Nonfinite actual randomization parameter')
                base_id = robot.body_names.index('base_link')
                base_com = arrays['body_com_pos_b'][:, base_id]
                com_spread = np.ptp(base_com, axis=0)
                result[f'base_com_{label}'] = dict(minimum=base_com.min(axis=0).tolist(),
                    maximum=base_com.max(axis=0).tolist(), spread=com_spread.tolist())
                np.savez_compressed(output / f'parameters_{label}.npz', **arrays)
                result[f'parameters_{label}'] = {name: dict(shape=list(value.shape), minimum=float(value.min()), maximum=float(value.max()))
                                                for name, value in arrays.items()}
                if args.variant == 'randomized' and not np.all(com_spread > 1.e-6):
                    raise RuntimeError('Configured randomized CoM did not produce per-environment base CoM variation')
            snapshot('initial')
            result['original_physics_handles_decimation'] = bool(base._physics_handles_decimation)
            if base._physics_handles_decimation:
                base.sim.physics_manager.set_decimation(1)
                base._physics_handles_decimation = False
            result['diagnostic_unfused'] = result['original_physics_handles_decimation']
            original_update = base.scene.update
            def update(dt):
                original_update(dt)
                if abs(dt - cfg.sim.dt) > 1.e-9:
                    raise RuntimeError(f'Expected native dt {cfg.sim.dt}; got {dt}')
                values = dict(root_position=robot.data.root_pos_w.torch-base.scene.env_origins,
                              gravity=robot.data.projected_gravity_b.torch,
                              angular_velocity=robot.data.root_ang_vel_b.torch,
                              joint_pos=robot.data.joint_pos.torch[:, joints],
                              joint_vel=robot.data.joint_vel.torch[:, joints],
                              torque_proxy=robot.actuators.applied_effort.torch[:, joints],
                              contact_normal=contact.data.net_normal_forces_w.torch,
                              targets=base.action_manager.get_term('joint_pos').processed_actions)
                if any(not bool(torch.isfinite(v).all()) for v in values.values()):
                    raise RuntimeError(f'Nonfinite native state at control step {step}')
                row = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}
                row['episode_index'] = resets.copy()
                rows.append(row)
            base.scene.update = update
            actions = torch.zeros(args.num_envs, 16, device=args.device)
            term_counts = {name: 0 for name in base.termination_manager.active_terms}
            for step in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(actions)
                validate_observations(obs)
                if not bool(torch.isfinite(reward).all()):
                    raise RuntimeError('Nonfinite reward')
                if base.command_manager.get_command('base_velocity').abs().max().item() > 1.e-8:
                    raise RuntimeError('Stance command is not zero')
                resets += (terminated | truncated).cpu().numpy().astype(np.int32)
                early += terminated.cpu().numpy().astype(np.int32)
                for name in term_counts:
                    term_counts[name] += int(base.termination_manager.get_term(name).sum())
            snapshot('final')
            result.update(completed=True, control_steps_completed=step+1, termination_counts=term_counts)
        except Exception as exc:
            result['failure'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            result.update(samples_recorded=len(rows), reset_counts=resets.tolist(), early_termination_counts=early.tolist())
            if rows:
                data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                np.savez_compressed(output / 'native_trajectory.npz', **data)
                result['metrics'] = summarize(data, cfg.sim.dt, wheels, legs, bodies)
                result['native_sample_count_matches'] = len(rows) == args.steps * cfg.decimation
                metrics = result['metrics']
                result['fixed_target_stance_gate'] = bool(result['completed'] and result['native_sample_count_matches']
                    and not early.any() and not metrics['signflip_env_ids']
                    and sum(metrics['leg_torque_120Nm_samples']) == 0
                    and metrics.get('leg_contact_over5N_steady_fraction', 1.) == 0.
                    and metrics.get('roll_abs_p99_deg', 180.) < 5.
                    and metrics.get('pitch_abs_p99_deg', 180.) < 5.)
            (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            env.close()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
