"""Zero-update evaluation for early checkpoints under one explicit V14 recipe.

Run ``terrain83`` and ``flat25`` into separate output directories.  The files
written in either directory retain the V13 manifest/traces schema consumed by
the existing analyze.py and joint_assessment.py scripts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, '/root/autodl-tmp/cts-v15-evaluation-audit-20260922')

from early_protocol import (build_cases as original_build_cases, checkpoint_metadata, load_protocol,
                            load_selection, source_contracts)
from audit_helpers import assign_columns, update_geometric_climb
from focus_protocol import build_focus_cases, brake_factor


def parse_args():
    parser = argparse.ArgumentParser()
    for key in ("root", "output", "selection", "protocol", "code"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--suite", choices=("brake25", "stairend"), required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Three physics control steps; not a behavior evaluation")
    return parser.parse_args()


def run(args) -> None:
    protocol = load_protocol(args.protocol)
    selection = load_selection(args.selection, protocol)
    contracts = source_contracts(protocol)
    cases = build_focus_cases(selection, args.suite, original_build_cases)
    runtime = protocol["runtime"]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.code), str(args.root / "upstream/source/rsl_rl"),
                    str(args.root / "upstream/source/robot_lab"),
                    str(args.root / "thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, device="cuda:0").app

    import numpy as np
    import torch
    from isaaclab.envs.mdp.events import reset_joints_by_scale
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS

    from actuator_guard import ActuatorGuard
    from config import make_config
    from mixed_task import configure_mixed, mixed_reset_root
    from pit_posture import configure as configure_pit_posture
    from reset_targets import install_reset_target_sync
    from stair_speed import configure_stair_speed
    from stance_strength import configure as configure_stance_strength
    from startup_repair import configure as configure_startup
    from surface_posture import configure as configure_surface_posture
    from train import apply_command_curriculum_start
    from training_profiles import configure_base_height_penalty
    from wave_reset import configure_wave_reset, reset_snapshot

    torch.set_num_threads(2)

    class FirstEpisode(Go2Env):
        retain_terminal = False

        def _reset_idx(self, ids):
            if hasattr(self, "audit_reset_count"):
                self.audit_reset_count[ids] += 1
            if not self.retain_terminal:
                super()._reset_idx(ids)

    def reset_start(env, env_ids, pose_range, velocity_range, asset_cfg):
        robot = env.scene["robot"]
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids],
            torch.zeros_like(robot.data.default_joint_pos[env_ids]), env_ids=env_ids)
        mixed_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg)
        pit = env.command_manager.get_term("base_velocity").pit
        pit_ids = env_ids[pit.active[env_ids]]
        if len(pit_ids):
            pose = robot.data.default_root_state[pit_ids, :7].clone()
            origins = env.scene.env_origins[pit_ids]
            pose[:, :3] += origins
            pose[:, 3:] = 0.
            pose[:, 3] = 1.
            points = pit.geometry.world_points(robot.data.default_joint_pos[pit_ids], pose)
            pose[:, 2] += origins[:, 2] + .03 - points[..., 2].amin(1)
            assert bool((points[..., 0].amax(1) < origins[:, 0] + 1.4).all())
            robot.write_root_pose_to_sim(pose, pit_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(pit_ids), 6, device=env.device), pit_ids)
            env.pit_valid_spawn[pit_ids] = True
        other_ids = env_ids[~pit.active[env_ids]]
        state = robot.data.root_state_w[other_ids].clone()
        state[:, 3:7] = 0.
        state[:, 3] = 1.
        robot.write_root_state_to_sim(state, env_ids=other_ids)
        env.stair_axis[other_ids] = 0

    cfg = make_config(len(cases), args.root / "cache/usd",
                      actuator_profile=runtime["actuator_profile"])
    cfg.seed, cfg.log_dir, cfg.episode_length_s = 123, str(args.output), 25.
    apply_command_curriculum_start(cfg.commands.base_velocity,
                                   runtime["command_curriculum_start"])
    configure_base_height_penalty(cfg, runtime["base_height_penalty"])
    configure_stair_speed(cfg, runtime["stair_command_profile"])
    configure_wave_reset(cfg, runtime["wave_reset_profile"])
    configure_mixed(cfg, slope_profile=runtime["slope_profile"],
                    pit_training_profile=runtime["pit_training_profile"])
    configure_pit_posture(cfg, runtime["pit_posture_profile"])
    configure_surface_posture(cfg, runtime["surface_posture_profile"])
    configure_stance_strength(cfg, runtime["stance_strength_profile"])
    configure_startup(cfg, runtime["startup_repair_profile"])
    cfg.terminations.startup_window = None
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    from evaluation_reset_events import retain_deterministic_reset_events
    retained_reset_events = retain_deterministic_reset_events(cfg)
    for group in (cfg.observations.policy, cfg.observations.single_obs,
                  cfg.observations.critic):
        group.enable_corruption = False
    cfg.events.reset_base.func = reset_start
    cfg.events.reset_robot_joints.func = reset_joints_by_scale
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.),
                                                velocity_range=(0., 0.))
    for key in ("pose_range", "velocity_range"):
        for name in cfg.events.reset_base.params[key]:
            cfg.events.reset_base.params[key][name] = (0., 0.)
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = actuator.max_delay = 2  # 2 x 5 ms physics steps
    cfg.rewards.lin_vel_z_l2.weight = 0.
    if args.suite == 'stairend':
        cfg.terminations.stair_route_exit = None
        cfg.scene.terrain.terrain_generator.use_cache = False
        for name in ('stairs_up', 'stairs_down'):
            cfg.scene.terrain.terrain_generator.sub_terrains[name].phase_landing_kind = name
    urdf = Path(cfg.scene.robot.spawn.asset_path)
    (args.output / "runtime.urdf").write_bytes(urdf.read_bytes())
    if args.suite == "brake25":
        from flat25_geometry import flat_patch
        from isaaclab.terrains import TerrainGenerator
        generator = cfg.scene.terrain.terrain_generator
        generator.class_type = TerrainGenerator
        generator.size = (80., 80.)
        generator.num_rows = 1
        generator.use_cache = False
        cfg.scene.terrain.max_init_terrain_level = 0
        for terrain_cfg in generator.sub_terrains.values():
            terrain_cfg.function = flat_patch

    raw = FirstEpisode(cfg=cfg)
    raw.audit_reset_count = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
    guard = ActuatorGuard(raw)
    reset_receipt = install_reset_target_sync(raw)
    raw.common_step_counter = runtime["common_step_counter"]
    terrain = raw.scene.terrain
    from robot_lab.tasks.go2.mdp.utils import _get_terrain_column_range, is_env_assigned_to_terrain
    terrain_cfg = cfg.scene.terrain.terrain_generator
    terrain_names = list(terrain_cfg.sub_terrains)
    ranges = {name: _get_terrain_column_range(terrain_cfg, name, raw.device) for name in terrain_names}
    terrain.terrain_types[:] = torch.tensor(
        assign_columns(cases, ranges), device=raw.device)
    terrain.terrain_levels[:] = torch.tensor([case["level"] for case in cases], device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    expected_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].clone()
    torch.testing.assert_close(terrain.env_origins, expected_origins, atol=0., rtol=0.)
    raw.scene.env_origins[:] = terrain.env_origins
    command = raw.command_manager.get_term("base_velocity")
    command.refresh_membership()
    pit = command.pit
    for name in terrain_names:
        expected = torch.tensor([case["terrain"] == name for case in cases], device=raw.device)
        assert bool(torch.equal(is_env_assigned_to_terrain(raw, name), expected))
    assert bool(torch.equal(pit.active, torch.tensor([case["terrain"] == "pit" for case in cases], device=raw.device)))
    for case_id, case in enumerate(cases):
        case["terrain_atlas_column"] = int(terrain.terrain_types[case_id])
        case["terrain_origin"] = terrain.env_origins[case_id].cpu().tolist()
    pit.fixed_levels = terrain.terrain_levels.clone()
    settle = torch.tensor([2. if case["mode"] == "settle" else 0. for case in cases],
                          device=raw.device)
    velocities = torch.tensor([case["velocity"] for case in cases], device=raw.device)

    landing_stop = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    stop_on_landing = torch.tensor([c.get('landing_mode') == 'stop' for c in cases], device=raw.device)

    def update_commands():
        elapsed = raw.episode_length_buf * raw.step_dt
        factor = brake_factor(elapsed) if args.suite == 'brake25' else torch.ones_like(elapsed)
        command.commands[:] = velocities * ((elapsed >= settle).float()*factor)[:, None]
        command.commands[landing_stop] = 0.
        command.commands[pit.active & pit.stop_on_platform] = 0.
        pit.commands[:] = command.commands
        pit.held_speed[:] = command.commands[:, 0]

    def resample(ids):
        command.time_left[ids] = 1e9
        update_commands()

    command._resample, command._update_command = resample, update_commands
    raw.pit_valid_spawn = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
    agent = MoECTSRunnerCfg().to_dict()
    agent["robogauge"] = {"enabled": False}
    runner = OnPolicyRunnerCTS(env, agent, log_dir=str(args.output / "loader"), device=raw.device)
    policy_template = runner.alg.policy
    policies, history = [], []
    for item in selection:
        state = torch.load(args.root / item["path"], map_location="cpu", weights_only=False)
        identity = checkpoint_metadata(state, contracts[item["path"]])
        model = copy.deepcopy(policy_template)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        policies.append(model)
        scalar_fields = {key: value for key, value in state.items()
                         if isinstance(value, (str, int, float, bool))}
        history.append(dict(**item, fields=scalar_fields, explicit_source_contract=identity))
        del state
    runner.eval_mode()
    indices = [torch.tensor([case["id"] for case in cases if case["policy"] == index],
                            device=raw.device) for index in range(len(policies))]

    def act(model, observations, teacher):
        if not teacher:
            return model.act_inference(observations)
        critic = model.critic_obs_normalizer(model.get_critic_obs(observations))
        single = model.single_obs_normalizer(observations["single_obs"])
        return model.actor(torch.cat((model.teacher_encoder(critic), single), dim=-1))

    pit.learner.fill_(1)
    command.learner.fill_(1)
    raw.retain_terminal = True
    robot = raw.scene["robot"]
    origins = raw.scene.env_origins.clone()
    initial = robot.data.root_pos_w.clone()
    initial_q = robot.data.joint_pos.clone()
    relative = robot.data.root_state_w.clone()
    relative[:, :3] -= origins
    expected_commands = velocities * (settle == 0.)[:, None]
    torch.testing.assert_close(initial_q, robot.data.default_joint_pos, atol=1e-6, rtol=0.)
    torch.testing.assert_close(robot.data.joint_pos_target, robot.data.joint_pos, atol=1e-6, rtol=0.)
    torch.testing.assert_close(command.commands, expected_commands, atol=1e-6, rtol=0.)
    assert bool((raw.episode_length_buf == 0).all())
    for ids in indices[1:]:
        torch.testing.assert_close(relative[ids], relative[indices[0]], atol=2e-4, rtol=0.)
        torch.testing.assert_close(initial_q[ids], initial_q[indices[0]], atol=1e-6, rtol=0.)
    wave_receipt = reset_snapshot(raw)
    expected_waves = sum(case["terrain"] == "wave" for case in cases)
    assert runtime["wave_reset_profile"] == "wave_surface_v1"
    assert wave_receipt["count"] >= expected_waves

    active = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    trace_names = ("body_pos", "body_quat", "joint_pos", "joint_vel", "target", "torque",
                   "command", "velocity", "root_clearance", "climb", "stable", "active",
                   "angular_velocity", "raw_action", "requested", "torque_peak", "speed_peak",
                   "hard_excess", "saturation", "gravity", "base_contact", "ground_relief",
                   "ground_scan_valid", "wheel_clearance", "wheel_ground_force", "pit_front_gap",
                   "wheel_ground_force_z", "flat_wheel_clearance", "platform_stop", "pit_all_supported", "pit_body_clear", "pit_upright",
                   "pit_quiet", "pit_hold_s", "phase_weight", "phase_ready", "phase_stable_s", "world_velocity",
                   "stair_edge_window", "stair_platform_hull", "stair_platform_supported")
    traces = {key: [] for key in trace_names}
    scanner = raw.scene["height_scanner"]
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    step_stats = {key: torch.zeros_like(robot.data.joint_pos)
                  for key in ("torque_peak", "speed_peak", "hard_excess", "saturation")}
    original_measure = guard.measure
    terminal_targets = torch.zeros_like(robot.data.joint_pos)
    captured_reset_count = torch.zeros_like(raw.audit_reset_count)
    platform_hold = torch.zeros(raw.num_envs, device=raw.device)
    platform_seen = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    geom_hold = torch.zeros(raw.num_envs, dtype=torch.float32, device=raw.device)
    geom_seen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)

    def measure_substep():
        original_measure()
        data = robot.data
        terminal_targets.copy_(data.joint_pos_target)
        captured_reset_count.copy_(raw.audit_reset_count)
        step_stats["torque_peak"].copy_(torch.maximum(step_stats["torque_peak"], data.applied_torque.abs()))
        step_stats["speed_peak"].copy_(torch.maximum(step_stats["speed_peak"], data.joint_vel.abs()))
        excess = torch.maximum(guard.hard[..., 0] - data.joint_pos,
                               data.joint_pos - guard.hard[..., 1]).clamp_min(0) * guard.leg
        step_stats["hard_excess"].copy_(torch.maximum(step_stats["hard_excess"], excess))
        step_stats["saturation"].add_((data.computed_torque - data.applied_torque).abs().gt(.001)
                                       / raw.cfg.decimation)

    guard.measure = measure_substep
    flat = torch.tensor([case["terrain"] == "flat" for case in cases], device=raw.device)
    observations = env.get_observations()
    with torch.inference_mode():
        for step in range(3 if args.smoke else 1250):
            issued = command.commands.clone()
            actions = torch.cat([
                act(model, {key: value[indices[index]] for key, value in observations.items()},
                    selection[index]["teacher"])
                for index, model in enumerate(policies)
            ], dim=0)
            actions[~active] = 0.
            for value in step_stats.values():
                value.zero_()
            observations, _, _, _ = env.step(actions)
            data = robot.data
            start = torch.cat((data.root_pos_w[:, None, :], data.body_pos_w[:, pit.wheel_ids]), dim=1).clone()
            start[:, :, 2] += 10.
            direction = torch.zeros_like(start)
            direction[:, :, 2] = -1.
            hits = raycast_mesh(start, direction, mesh)[0]
            ray_z = scanner.data.ray_hits_w[..., 2]
            valid = torch.isfinite(ray_z).all(-1)
            relief = torch.where(valid, ray_z.amax(-1) - ray_z.amin(-1),
                                 torch.full_like(ray_z[:, 0], float("nan")))
            ground_force = torch.stack([sensor.data.force_matrix_w[:, 0, 0].norm(dim=-1)
                                        for sensor in pit.ground_contacts], dim=1)
            pit_all_supported = pit.supported.all(-1)
            pit_body_clear = pit.platform_clear
            pit_upright = -data.projected_gravity_b[:, 2] > np.cos(np.deg2rad(15.))
            pit_quiet = (torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=-1) < .15)
            pit_quiet &= torch.linalg.vector_norm(data.root_ang_vel_w, dim=-1) < .3
            geom_ok = pit_all_supported & pit_body_clear & pit_upright & pit.active
            geom_ok &= ~pit.physical & ~pit.outside
            geom_hold, geom_seen = update_geometric_climb(geom_hold, geom_seen, geom_ok, raw.step_dt)
            front_gap = origins[:, 0] + 1.5 - (data.body_pos_w[:, pit.wheel_ids, 0].amax(-1) + .093)
            from phase_transition import measure as measure_phase
            phase = measure_phase(raw)
            hull = pit.geometry.world_points(data.joint_pos, data.root_state_w[:, :7])
            local = hull - origins[:, None, :]
            # Straight cases all move +X. Validate the full collision hull and
            # wheel support on the actual outer landing, not just root distance.
            on_platform = (local[..., 0].amin(1) >= 2.9) & (local[..., 0].amax(1) <= 4.3)
            on_platform &= local[..., 1].abs().amax(1) <= 4.3
            supported_platform = on_platform & phase['ready'] & phase['stairs']
            edge_window = (local[..., 0].amax(1) >= 2.6) & phase['stairs']
            platform_seen |= supported_platform
            platform_hold = torch.where(supported_platform, platform_hold+raw.step_dt, 0.)
            landing_stop |= supported_platform & stop_on_landing
            values = dict(
                phase_weight=phase['weight'], phase_ready=phase['ready'], phase_stable_s=phase['stable_s'],
                world_velocity=data.root_lin_vel_w,
                stair_edge_window=edge_window, stair_platform_hull=on_platform & phase['stairs'],
                stair_platform_supported=supported_platform,
                ground_relief=relief, ground_scan_valid=valid,
                wheel_clearance=data.body_pos_w[:, pit.wheel_ids, 2] - hits[:, 1:, 2] - .093,
                wheel_ground_force=ground_force,
                wheel_ground_force_z=torch.stack([sensor.data.force_matrix_w[:, 0, 0, 2] for sensor in pit.ground_contacts], dim=1),
                flat_wheel_clearance=data.body_link_pos_w[:, pit.wheel_ids, 2] - origins[:, 2:3] - .093,
                pit_front_gap=torch.where(pit.active, front_gap, torch.full_like(front_gap, float("nan"))),
                platform_stop=pit.stop_on_platform, pit_all_supported=pit_all_supported,
                pit_body_clear=pit_body_clear, pit_upright=pit_upright,
                pit_quiet=pit_quiet, pit_hold_s=pit.hold,
                body_pos=data.body_pos_w - origins[:, None, :],
                body_quat=data.body_quat_w, joint_pos=data.joint_pos, joint_vel=data.joint_vel,
                target=terminal_targets, torque=data.applied_torque, command=issued,
                velocity=data.root_lin_vel_b, angular_velocity=data.root_ang_vel_b,
                raw_action=actions, requested=raw.action_manager.get_term("joint_pos").requested_actions,
                gravity=data.projected_gravity_b,
                base_contact=pit.contact.data.net_forces_w_history[:, :, pit.base_contact_ids]
                    .norm(dim=-1).amax(dim=(1, 2)), **step_stats,
                root_clearance=data.root_pos_w[:, 2] - hits[:, 0, 2],
                climb=geom_seen | pit.success, stable=pit.success, active=active)
            for key, value in values.items():
                traces[key].append(value.clone())
            dx = data.root_pos_w[:, 0] - initial[:, 0]
            elapsed = (step + 1) * raw.step_dt
            flat_budget = 25. if args.suite == "brake25" else 5.
            ended = active & (raw.termination_manager.terminated |
                              raw.termination_manager.time_outs |
                              (flat & (elapsed >= flat_budget)) |
                              ((~flat) & (platform_hold >= 2.)) |
                              ((~flat) & (data.root_pos_w[:, 0]-origins[:, 0] >= 4.1)))
            for case_id in ended.nonzero().flatten().tolist():
                physical = bool(raw.termination_manager.terminated[case_id])
                if physical:
                    end_reason = 'physical_termination'
                elif bool(raw.termination_manager.time_outs[case_id]):
                    end_reason = 'timeout'
                elif bool(platform_hold[case_id] >= 2.):
                    end_reason = 'supported_platform_dwell'
                elif not bool(flat[case_id]):
                    end_reason = 'safe_tile_endpoint'
                else:
                    end_reason = 'time_budget' 
                terms = [name for name in raw.termination_manager.active_terms
                         if bool(raw.termination_manager.get_term(name)[case_id])]
                cases[case_id].update(
                    end_step=step, seconds=elapsed, physical=physical, terms=terms,
                    end_reason=end_reason, platform_hold_s=float(platform_hold[case_id]),
                    supported_platform_reached=bool(platform_seen[case_id]),
                    actuator_or_base_contact=bool(guard.violation[case_id] | pit.body_contact[case_id]),
                    passed_4m=bool(dx[case_id] >= 4. and not physical), dx_m=float(dx[case_id]),
                    climb=bool(geom_seen[case_id] | pit.success[case_id]),
                    terminal_target=terminal_targets[case_id].cpu().tolist(),
                    reset_metadata={"episode_length": int(raw.episode_length_buf[case_id]),
                                    "reset_count": int(raw.audit_reset_count[case_id]),
                                    "captured_reset_count": int(captured_reset_count[case_id]),
                                    "reset_occurred_after_capture": bool(raw.audit_reset_count[case_id] > captured_reset_count[case_id])},
                    stable=bool(pit.success[case_id]),
                    checkpoint=selection[cases[case_id]["policy"]]["path"],
                    violation_causes={key: bool(value[case_id])
                                      for key, value in guard.violation_causes.items()})
            active[ended] = False
            if step % 200 == 0:
                print("PROGRESS", step, int(active.sum()), flush=True)
            if not bool(active.any()):
                break
    for case in cases:
        if "end_step" not in case:
            case.update(end_step=step, seconds=(step + 1) * raw.step_dt,
                        physical=False, terms=["budget"], passed_4m=False,
                        end_reason='smoke_budget' if args.smoke else 'time_budget',
                        platform_hold_s=float(platform_hold[case['id']]),
                        supported_platform_reached=bool(platform_seen[case['id']]))
    arrays = {key: torch.stack(value).cpu().numpy() for key, value in traces.items()}
    base_id = robot.body_names.index("base_link")
    for case in cases:
        count = case["end_step"] + 1
        case["max_side_m"] = float(abs(arrays["body_pos"][:count, case["id"], base_id, 1]).max())
        case["straight_4m"] = case["passed_4m"] and case["max_side_m"] <= .5
    np.savez_compressed(args.output / "traces.npz", initial_q=initial_q.cpu().numpy(),
                        initial_root=relative.cpu().numpy(), **arrays)
    manifest = dict(
        evaluation_revision="speed_brake_v20_focus_20260922",
        protocol_id=args.suite,
        protocol_config_id=protocol["protocol_id"], protocol_config=protocol,
        historical_training_reconstruction=False,
        evaluation_runtime_source="explicit protocol JSON; checkpoint profiles are not inherited",
        zero_update_evaluation=True, optimizer_steps=0, training_updates=0,
        checkpoint=selection[-1]["path"],
        completed_updates=contracts[selection[-1]["path"]]["completed_updates"],
        history=history, cases=cases, joint_names=robot.joint_names, body_names=robot.body_names,
        dt=raw.step_dt, default_q=robot.data.default_joint_pos[0].cpu().tolist(),
        request_joint_ids=raw.action_manager.get_term("joint_pos").joint_indices,
        actuator_limits=guard.limits, initial_state_matched=True,
        default_pose_initial_verified=True, smoke_only=args.smoke,
        initial_scripted_commands_verified=True, initial_flat_origins_verified=True,
        initial_episode_age_zero_verified=True, reset_targets=reset_receipt,
        ramp_diagnostic=False, direct_gate_protocol=True,
        max_episode_length_steps=raw.max_episode_length, retained_reset_events=retained_reset_events,
        wave_reset_receipt=wave_receipt,
        wave_surface_correction_verified=wave_receipt["count"] >= expected_waves,
        terrain_mapping={"atlas_columns": {name: list(bounds) for name, bounds in ranges.items()},
                         "membership_asserted": True, "pit_active_asserted": True,
                         "origins_assigned_after_mapping": True,
                         "origins_match_atlas_asserted": True},
        slope_geometry=getattr(cfg.scene.terrain.terrain_generator, "slope_geometry_receipts", []),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        protocol=("Common early-checkpoint physics evaluation, not a reconstruction of historical "
                  "training configuration. Model weights and each model's serialized observation "
                  "normalizers are loaded; no optimizer or training update is executed. Deterministic "
                  "default-pose reset with synchronized targets, no noise or pushes, fixed 10 ms delay. "
                  + ("36 all-direction speed/braking cases per policy on 80m patches, 25s M-S-reverse-S-M."
                     if args.suite == "brake25" else
                     "Stairs use a dedicated capped outer landing; retain real phase and full collision-hull entry, "
                     "end after supported landing dwell, safe tile endpoint, or physical/25s termination.")))
    if args.suite == "brake25":
        manifest["flat_patch_size_m"] = [80., 80.]
        manifest["flat_boundary_crossed"] = [
            case["id"] for case in cases
            if np.abs(arrays["body_pos"][:case["end_step"] + 1, case["id"], base_id, :2]).max() > 38.
        ]
        if manifest["flat_boundary_crossed"]:
            raise RuntimeError("a full-duration case left the validated flat evaluation area")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DONE", len(cases), flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
