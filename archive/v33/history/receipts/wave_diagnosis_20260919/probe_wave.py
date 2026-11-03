"""Frozen-policy stair failure observations over a wave-only 72-case grid; explicit isolated ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def evaluate(args):
    import torch
    from tensordict import TensorDict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from stair_speed import configure_stair_speed
    from training_profiles import configure_base_height_penalty
    from train import apply_command_curriculum_start

    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from policy_contract import checkpoint_profile
    assert checkpoint_profile(saved) == args.actuator_profile
    columns = saved["terrain_column_names"]
    cases = []
    for learner in ("teacher", "student"):
        for name in ("wave",):
            available = [i for i, value in enumerate(columns) if value == name]
            for level in (0, 3, 6, 9):
                for speed in (0.3, 0.6, 0.9):
                    for repeat, lateral in enumerate((-0.2, 0.0, 0.2)):
                        cases.append(dict(learner=learner, terrain=name, level=level, speed=speed,
                                          repeat=repeat, start_lateral_m=0.0, column=available[repeat % len(available)]))

    class FirstEpisodeEnv(Go2Env):
        retain_terminal = False

        def _reset_idx(self, env_ids):
            if not self.retain_terminal:
                super()._reset_idx(env_ids)

    count = len(cases)
    cfg = make_config(count, args.root / "cache/usd", actuator_profile=args.actuator_profile)
    cfg.seed = 42
    cfg.log_dir = str(args.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, "intermediate")
    configure_base_height_penalty(cfg, "off")
    configure_stair_speed(cfg, "straight_speed_v1")
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    cfg.rewards.lin_vel_z_l2.weight = 0
    if args.diagnostic_no_actuator_termination:
        # Isolated diagnostic only: keep actuator limits, action guards, all
        # measurements and other terminations; suppress this one stop condition.
        cfg.terminations.actuator_violation = None
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    for name in vars(cfg.events):
        if name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.func = reset_root_state_uniform
    cfg.events.reset_base.params["pose_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_base.params["velocity_range"] = dict(cfg.events.reset_base.params["pose_range"])
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cfg.events.reset_base.params["pose_range"]["z"] = (args.reset_z, args.reset_z)
    import guard_actions
    original_bound = guard_actions.bounded_position_target
    capture = {}

    def observe_bound(requested, previous, lower, upper, max_delta):
        capture["requested"] = requested.clone()
        capture["previous"] = previous.clone()
        return original_bound(requested, previous, lower, upper, max_delta)

    guard_actions.bounded_position_target = observe_bound
    raw = FirstEpisodeEnv(cfg=cfg)
    if args.actuator_profile != "original":
        from actuator_guard import ActuatorGuard
        ActuatorGuard(raw)
    try:
        raw.common_step_counter = saved["common_step_counter"]
        terrain = raw.scene.terrain
        terrain.terrain_levels[:] = torch.tensor([c["level"] for c in cases], device=raw.device)
        terrain.terrain_types[:] = torch.tensor([c["column"] for c in cases], device=raw.device)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        raw.stair_axis = torch.zeros(count, device=raw.device, dtype=torch.long)
        command = raw.command_manager.get_term("base_velocity")
        command._init_terrain_infos()
        command._update_env_command_ranges()
        speeds = torch.tensor([c["speed"] for c in cases], device=raw.device)

        def fixed_resample(ids):
            command.commands[ids] = 0
            command.commands[ids, 0] = speeds[ids]
            command.held_speed[ids] = speeds[ids]
            command.speed_bins[ids] = torch.bucketize(speeds[ids], torch.tensor([0.4, 0.6, 0.8], device=raw.device))
            command.needs_speed[ids] = False
            command.time_left[ids] = 1.0e9

        command._resample = fixed_resample
        original_command_update = command._update_command

        def command_update():
            original_command_update()
            if args.settle_seconds:
                command.commands[:, 0] = torch.where(
                    raw.episode_length_buf * raw.step_dt < args.settle_seconds, 0.0, speeds)
            if args.mode == "heading":
                q = raw.scene["robot"].data.root_quat_w
                yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
                command.commands[:, 2] = (-1.5 * yaw).clamp(-0.6, 0.6)

        command._update_command = command_update
        raw.reset()
        robot = raw.scene["robot"]
        initial = robot.data.root_state_w.clone()
        robot.write_root_state_to_sim(initial)
        raw.sim.forward()
        raw.scene.update(raw.physics_dt)
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
        agent = MoECTSRunnerCfg().to_dict()
        agent["robogauge"] = {"enabled": False}
        runner = OnPolicyRunnerCTS(env, agent, log_dir=str(args.output / "loader"), device=raw.device)
        policy = runner.alg.policy
        policy.load_state_dict(saved["model_state_dict"])
        policy.eval()
        # The wrapper constructor resets the environment. Capture the actual start
        # AFTER the wrapper and runner are initialized; never use a pre-reset pose.
        initial = robot.data.root_state_w.clone()
        torch.testing.assert_close(initial[:, :2], raw.scene.env_origins[:, :2], rtol=0, atol=1e-5)
        names = robot.joint_names
        leg_mask = torch.tensor([not n.endswith("foot_joint") for n in names], device=raw.device)
        hard = robot.root_physx_view.get_dof_limits().to(raw.device).clone()
        soft = robot.data.soft_joint_pos_limits.clone()
        vel_limit = robot.root_physx_view.get_dof_max_velocities().to(raw.device).clone()
        torque_limit = robot.root_physx_view.get_dof_max_forces().to(raw.device).clone()
        assert hard.shape == (count, len(names), 2)
        metrics = {k: torch.zeros((count, len(names)), device=raw.device) for k in (
            "max_hard_excess_rad", "max_soft_excess_rad", "max_target_hard_excess_rad",
            "max_speed_rad_s", "max_applied_torque_nm", "max_computed_torque_nm",
            "hard_excess_samples", "soft_excess_samples", "target_excess_samples",
            "speed_excess_samples", "torque_excess_samples", "torque_clipped_samples")}
        samples = torch.zeros(count, device=raw.device)
        active = torch.ones(count, dtype=torch.bool, device=raw.device)
        update_scene = raw.scene.update
        # Exact horizontal frame decomposition, including motion on pitched terrain.
        # vy_world = sin(yaw)*v_forward_horizontal + cos(yaw)*v_side_horizontal.
        sums = {k: torch.zeros(count, device=raw.device) for k in (
            "heading_command_m", "heading_speed_error_m", "lateral_motion_m", "world_velocity_y_m",
            "body_vx_abs_error_integral", "body_vy_abs_integral", "body_wz_abs_error_integral",
            "body_wz_signed_integral", "world_vy_abs_integral", "heading_abs_integral")}
        peak_sums = {k: torch.zeros(count, device=raw.device) for k in sums}
        peak_cross_signed = torch.zeros(count, device=raw.device)
        peak_heading = torch.zeros(count, device=raw.device)
        peak_time = torch.zeros(count, device=raw.device)
        maximum_heading = torch.zeros(count, device=raw.device)
        final_cross = torch.zeros(count, device=raw.device)
        trace_ids = [i for i, c in enumerate(cases) if c["learner"] == "student" and c["level"] == 9
                     and ((c["terrain"] == "wave" and c["speed"] == 0.3 and c["repeat"] == 2)
                          or (c["terrain"] == "stairs_up" and c["repeat"] == 2)
                          or (c["terrain"] == "stairs_down" and c["speed"] == 0.6 and c["repeat"] == 0))]
        traces = []

        # Read-only 50 Hz capture for every student stair case. Keep tensors on
        # the device throughout stepping, then transfer and save once at the end.
        stair_trace_ids = list(range(count))
        assert count == 72 and len(stair_trace_ids) == 72 and len(names) == 16
        assert abs(raw.step_dt - 0.02) < 1e-9
        stair_indices = torch.tensor(stair_trace_ids, device=raw.device, dtype=torch.long)
        wheel_body_names = [name for name in robot.body_names if name.endswith("_foot_Link")]
        assert len(wheel_body_names) == 4, f"Expected four wheel bodies, got {wheel_body_names}"
        wheel_body_ids = [robot.body_names.index(name) for name in wheel_body_names]
        contact = raw.scene["contact_forces"]
        wheel_sensor_ids = [contact.body_names.index(name) for name in wheel_body_names]
        term_names = list(raw.termination_manager.active_terms)
        assert "time_out" in term_names and "illegal_contact" in term_names
        if args.actuator_profile != "original" and not args.diagnostic_no_actuator_termination:
            assert "actuator_violation" in term_names
        first_term_values = torch.zeros((count, len(term_names)), device=raw.device, dtype=torch.bool)
        first_end_flags = torch.zeros((count, 3), device=raw.device, dtype=torch.bool)
        first_end_step = torch.full((count,), -1, device=raw.device, dtype=torch.long)
        leg_term = raw.action_manager.get_term("joint_pos")
        leg_joint_names = list(leg_term._joint_names)
        if args.target_rate is not None:
            leg_term._max_delta = args.target_rate * raw.step_dt
        # Read-only raycasts under wheel centers; gap uses nominal 0.093 m radius.
        from isaaclab.utils.warp import raycast_mesh
        initial_points = torch.cat((robot.data.root_pos_w[:, None], robot.data.body_link_pos_w[:, wheel_body_ids]), 1).clone()
        ray_starts = initial_points.clone()
        ray_starts[..., 2] += 10.0
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0
        scanner = raw.scene["height_scanner"]
        hits = raycast_mesh(ray_starts, ray_directions, scanner.meshes[scanner.cfg.mesh_prim_paths[0]])[0]
        assert torch.isfinite(hits).all()
        start_geometry = dict(points_w=initial_points.cpu().tolist(), terrain_hits_w=hits.cpu().tolist(),
                              wheel_bottom_vertical_gap_m=(initial_points[:, 1:, 2] - hits[:, 1:, 2] - 0.093).cpu().tolist(),
                              root_height_above_local_surface_m=(initial_points[:, 0, 2] - hits[:, 0, 2]).cpu().tolist())
        early = []
        base_sensor_id = contact.body_names.index("base_link")
        reward_names = list(raw.reward_manager.active_terms)
        trace_shapes = {
            "requested_leg_target_rad": (12,),
            "previous_leg_target_rad": (12,),
            "reward_weighted_rate": (len(reward_names),),
            "root_pos_relative_w_m": (3,), "root_quat_wxyz": (4,),
            "root_link_lin_vel_w_mps": (3,), "root_lin_vel_b_mps": (3,),
            "root_ang_vel_w_rad_s": (3,), "root_ang_vel_b_rad_s": (3,),
            "joint_pos_rad": (16,), "joint_vel_rad_s": (16,),
            "joint_applied_torque_nm": (16,), "joint_computed_torque_nm": (16,),
            "joint_pos_target_rad": (16,), "joint_vel_target_rad_s": (16,),
            "wheel_pos_relative_w_m": (4, 3), "wheel_pos_root_relative_w_m": (4, 3),
            "wheel_contact_force_norm_n": (4,), "command_body": (3,),
            "active_before_step": (), "ended_this_step": (),
            "termination_values": (len(term_names),),
            "terminated": (), "truncated": (), "forward_exit": (),
        }
        bool_fields = {"active_before_step", "ended_this_step", "termination_values",
                       "terminated", "truncated", "forward_exit"}
        stair_traces = {
            key: torch.empty((raw.max_episode_length, len(stair_trace_ids), *shape), device=raw.device,
                             dtype=torch.bool if key in bool_fields else robot.data.joint_pos.dtype)
            for key, shape in trace_shapes.items()
        }
        recorded_steps = 0

        def measure(dt):
            update_scene(dt)
            if len(early) < 200:
                early.append(torch.cat((robot.data.root_pos_w - initial[:, :3],
                    robot.data.root_lin_vel_b, robot.data.joint_vel, robot.data.applied_torque,
                    torch.linalg.vector_norm(contact.data.net_forces_w[:, [base_sensor_id] + wheel_sensor_ids], dim=-1),
                    active[:, None]), dim=-1).clone())
            root_q = robot.data.root_quat_w
            yaw = torch.atan2(2*(root_q[:, 0]*root_q[:, 3]+root_q[:, 1]*root_q[:, 2]),
                              1-2*(root_q[:, 2]**2+root_q[:, 3]**2))
            # Match root_pos_w's link frame. Legacy root_lin_vel_w is COM velocity.
            world_v = robot.data.root_link_lin_vel_w
            body_v = robot.data.root_lin_vel_b
            forward = yaw.cos()*world_v[:, 0] + yaw.sin()*world_v[:, 1]
            side = -yaw.sin()*world_v[:, 0] + yaw.cos()*world_v[:, 1]
            terms = {
                "heading_command_m": yaw.sin()*speeds,
                "heading_speed_error_m": yaw.sin()*(forward-speeds),
                "lateral_motion_m": yaw.cos()*side,
                "world_velocity_y_m": world_v[:, 1],
                "body_vx_abs_error_integral": (body_v[:, 0]-speeds).abs(),
                "body_vy_abs_integral": body_v[:, 1].abs(),
                "body_wz_abs_error_integral": (robot.data.root_ang_vel_b[:, 2]-command.commands[:, 2]).abs(),
                "body_wz_signed_integral": robot.data.root_ang_vel_b[:, 2],
                "world_vy_abs_integral": world_v[:, 1].abs(),
                "heading_abs_integral": yaw.abs(),
            }
            for key, value in terms.items():
                sums[key].add_(value * dt * active)
            y = robot.data.root_pos_w[:, 1] - initial[:, 1]
            final_cross.copy_(torch.where(active, y, final_cross))
            new_peak = active & (y.abs() > peak_cross_signed.abs())
            peak_cross_signed[new_peak] = y[new_peak]
            peak_heading[new_peak] = yaw[new_peak]
            peak_time[new_peak] = (samples[new_peak] + 1) * dt
            maximum_heading.copy_(torch.maximum(maximum_heading, torch.where(active, yaw.abs(), 0)))
            for key in sums:
                peak_sums[key][new_peak] = sums[key][new_peak]
            trace = torch.stack((samples*dt+dt, active, robot.data.root_pos_w[:, 0]-initial[:, 0], y,
                                  yaw, body_v[:, 0], body_v[:, 1], robot.data.root_ang_vel_b[:, 2],
                                  sums["heading_command_m"], sums["heading_speed_error_m"],
                                  sums["lateral_motion_m"], command.commands[:, 2]), dim=1)
            traces.append(trace[trace_ids].clone())
            q = robot.data.joint_pos
            dq = robot.data.joint_vel.abs()
            effort = robot.data.applied_torque.abs()
            computed = robot.data.computed_torque.abs()
            target = robot.data.joint_pos_target
            def excess(v, limits):
                return torch.maximum(limits[..., 0] - v, v - limits[..., 1]).clamp_min(0) * leg_mask
            hard_error, soft_error = excess(q, hard), excess(q, soft)
            target_error = excess(target, hard)
            keep = active[:, None]
            for key, value in (("max_hard_excess_rad", hard_error), ("max_soft_excess_rad", soft_error),
                               ("max_target_hard_excess_rad", target_error), ("max_speed_rad_s", dq),
                               ("max_applied_torque_nm", effort), ("max_computed_torque_nm", computed)):
                metrics[key] = torch.maximum(metrics[key], torch.where(keep, value, 0))
            for key, condition in (("hard_excess_samples", hard_error > 1e-3),
                                   ("soft_excess_samples", soft_error > 1e-3),
                                   ("target_excess_samples", target_error > 1e-3),
                                   ("speed_excess_samples", dq > vel_limit + 1e-3),
                                   ("torque_excess_samples", effort > torque_limit + 1e-3),
                                   ("torque_clipped_samples", (computed - effort).abs() > 1e-3)):
                metrics[key] += condition & keep
            samples.add_(active)

        raw.scene.update = measure
        raw.retain_terminal = True
        half = count // 2
        steps = torch.zeros(count, device=raw.device)
        error = torch.zeros_like(steps)
        cross = torch.zeros_like(steps)
        progress = torch.zeros_like(steps)
        roll_pitch = torch.zeros_like(steps)
        body_z_speed = torch.zeros_like(steps)
        failed = torch.zeros_like(active)
        passed = torch.zeros_like(active)
        obs = TensorDict(raw.obs_buf, batch_size=[count])
        with torch.inference_mode():
            for step in range(raw.max_episode_length):
                torch.testing.assert_close(obs["single_obs"][:, 6:9], command.command, rtol=0, atol=0)
                teacher_obs = obs[:half]
                latent = policy.teacher_encoder(policy.critic_obs_normalizer(policy.get_critic_obs(teacher_obs)))
                action_t = policy.actor(torch.cat((latent, policy.single_obs_normalizer(teacher_obs["single_obs"])), dim=-1))
                if policy.state_dependent_std:
                    action_t = action_t[..., 0, :]
                action = torch.cat((action_t, policy.act_inference(obs[half:])))
                assert torch.isfinite(action).all()
                if agent["clip_actions"] is not None:
                    action.clamp_(-agent["clip_actions"], agent["clip_actions"])
                action[~active] = 0
                issued_command = command.command.clone()
                next_obs, _, terminated, truncated, _ = raw.step(action)
                offset = robot.data.root_pos_w[:, :2] - initial[:, :2]
                steps += active
                error += (robot.data.root_lin_vel_b[:, 0] - speeds).abs() * active
                cross = torch.maximum(cross, torch.where(active, offset[:, 1].abs(), 0))
                progress = torch.where(active, offset[:, 0], progress)
                q = robot.data.root_quat_w
                roll = torch.atan2(2*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3]), 1-2*(q[:, 1]**2+q[:, 2]**2))
                pitch = torch.asin((2*(q[:, 0]*q[:, 2]-q[:, 3]*q[:, 1])).clamp(-1, 1))
                roll_pitch = torch.maximum(roll_pitch, torch.where(active, torch.maximum(roll.abs(), pitch.abs()), 0))
                body_z_speed += robot.data.root_lin_vel_b[:, 2].abs() * active
                ended = active & (terminated | truncated | (offset[:, 0] > 4))
                term_values = torch.stack([raw.termination_manager.get_term(name) for name in term_names], dim=1)
                first_term_values.copy_(torch.where(ended[:, None], term_values, first_term_values))
                end_flags = torch.stack((terminated, truncated, offset[:, 0] > 4), dim=1)
                first_end_flags.copy_(torch.where(ended[:, None], end_flags, first_end_flags))
                first_end_step.copy_(torch.where(ended, step, first_end_step))
                wheel_pos = robot.data.body_pos_w[:, wheel_body_ids]
                control_values = {
                    "requested_leg_target_rad": capture["requested"],
                    "previous_leg_target_rad": capture["previous"],
                    "reward_weighted_rate": raw.reward_manager._step_reward,
                    "root_pos_relative_w_m": robot.data.root_pos_w - initial[:, :3],
                    "root_quat_wxyz": robot.data.root_quat_w,
                    "root_link_lin_vel_w_mps": robot.data.root_link_lin_vel_w,
                    "root_lin_vel_b_mps": robot.data.root_lin_vel_b,
                    "root_ang_vel_w_rad_s": robot.data.root_ang_vel_w,
                    "root_ang_vel_b_rad_s": robot.data.root_ang_vel_b,
                    "joint_pos_rad": robot.data.joint_pos,
                    "joint_vel_rad_s": robot.data.joint_vel,
                    "joint_applied_torque_nm": robot.data.applied_torque,
                    "joint_computed_torque_nm": robot.data.computed_torque,
                    "joint_pos_target_rad": robot.data.joint_pos_target,
                    "joint_vel_target_rad_s": robot.data.joint_vel_target,
                    "wheel_pos_relative_w_m": wheel_pos - initial[:, None, :3],
                    "wheel_pos_root_relative_w_m": wheel_pos - robot.data.root_pos_w[:, None, :],
                    "wheel_contact_force_norm_n": torch.linalg.vector_norm(
                        contact.data.net_forces_w[:, wheel_sensor_ids], dim=-1),
                    "command_body": issued_command,
                    "active_before_step": active,
                    "ended_this_step": ended,
                    "termination_values": term_values,
                    "terminated": terminated,
                    "truncated": truncated,
                    "forward_exit": offset[:, 0] > 4,
                }
                for key, value in control_values.items():
                    stair_traces[key][step].copy_(value.index_select(0, stair_indices))
                recorded_steps = step + 1
                failed[ended] = terminated[ended]
                passed[ended] = (offset[ended, 0] > 4) & ~terminated[ended]
                active &= ~ended
                obs = TensorDict(next_obs, batch_size=[count])
                if step % 250 == 0:
                    (args.output / "status.json").write_text(json.dumps(dict(state="evaluating", step=step, active=int(active.sum()))))
                if not active.any():
                    break
        assert torch.equal(samples, steps * cfg.decimation), "Every active physics step must be sampled"
        measured = {k: v.cpu().tolist() for k, v in metrics.items()}
        decomposition = {k: v.cpu().tolist() for k, v in sums.items()}
        peak_decomposition = {k: v.cpu().tolist() for k, v in peak_sums.items()}
        decomposition_error = (sums["heading_command_m"] + sums["heading_speed_error_m"]
                               + sums["lateral_motion_m"] - sums["world_velocity_y_m"]).abs()
        assert float(decomposition_error.max()) < 1e-4, "Horizontal decomposition identity failed"
        peak_values = torch.stack((peak_cross_signed, peak_heading, peak_time, maximum_heading,
                                   final_cross), dim=1).cpu().tolist()
        scalar = torch.stack((passed, failed, error / steps.clamp_min(1), cross, progress,
                              steps * raw.step_dt, roll_pitch * (180/3.141592653589793),
                              body_z_speed / steps.clamp_min(1), samples), dim=1).cpu().tolist()
        keys = ("forward_exit", "physical_failure", "vx_mae_mps", "max_cross_m", "final_forward_m",
                "duration_s", "max_roll_or_pitch_deg", "mean_abs_body_vz_mps", "physics_samples")
        termination_rows = first_term_values.cpu().tolist()
        end_flag_rows = first_end_flags.cpu().tolist()
        end_step_rows = first_end_step.cpu().tolist()
        for index, case in enumerate(cases):
            case.update(zip(keys, scalar[index]))
            case["joints"] = {k: rows[index] for k, rows in measured.items()}
            case["decomposition_final"] = {k: rows[index] for k, rows in decomposition.items()}
            case["decomposition_at_peak"] = {k: rows[index] for k, rows in peak_decomposition.items()}
            case.update(zip(("peak_cross_signed_m", "heading_at_peak_rad", "peak_time_s", "max_heading_rad",
                             "final_y_m"), peak_values[index]))
            case["peak_integration_residual_m"] = (case["peak_cross_signed_m"]
                                                   - case["decomposition_at_peak"]["world_velocity_y_m"])
            case["termination_reasons"] = {
                "time_out": False, "actuator_violation": False, "illegal_contact": False,
                **dict(zip(term_names, termination_rows[index])),
            }
            case["first_end"] = dict(
                control_step_index=end_step_rows[index],
                terminated=end_flag_rows[index][0], truncated=end_flag_rows[index][1],
                forward_exit=end_flag_rows[index][2],
                evaluation_horizon_reached_without_end=end_step_rows[index] < 0,
            )
        import numpy as np
        (args.output / "initial_geometry.json").write_text(json.dumps(start_geometry, indent=2))
        np.savez_compressed(args.output / "early_200hz.npz", values=torch.stack(early).cpu().numpy(),
                            fields=np.array(["dx", "dy", "dz", "vx", "vy", "vz"] +
                            ["speed_"+n for n in names] + ["torque_"+n for n in names] +
                            ["contact_base"] + ["contact_"+n for n in wheel_body_names] + ["active"]))
        np.savez_compressed(args.output / "selected_traces.npz", values=torch.stack(traces).cpu().numpy(),
                            case_indices=np.array(trace_ids), fields=np.array([
                                "time_s", "active", "forward_m", "cross_m", "yaw_rad", "body_vx", "body_vy",
                                "body_wz", "heading_command_m", "heading_speed_error_m", "lateral_motion_m", "cmd_wz"]))
        np.savez_compressed(
            args.output / "stair_failure_traces.npz",
            **{key: value[:recorded_steps].cpu().numpy() for key, value in stair_traces.items()},
            time_s=(np.arange(recorded_steps, dtype=np.float64) + 1) * raw.step_dt,
            case_indices=np.array(stair_trace_ids, dtype=np.int64),
            joint_names=np.array(names), wheel_body_names=np.array(wheel_body_names),
            leg_action_joint_names=np.array(leg_joint_names), reward_names=np.array(reward_names),
            termination_term_names=np.array(term_names),
            initial_root_pos_w_m=initial[stair_indices, :3].cpu().numpy(),
            terrain_origin_w_m=raw.scene.env_origins[stair_indices].cpu().numpy(),
        )
        trace_metadata = {
            "file": "stair_failure_traces.npz",
            "sample_period_s": raw.step_dt, "recorded_control_steps": recorded_steps,
            "case_indices": stair_trace_ids,
            "array_layout": "All time series use [control_step, selected_case, ...]; case_indices maps to results.json cases.",
            "validity": "Use active_before_step. It includes the first ended sample; later samples for that case are invalid.",
            "sample_time": "Post-control-step state; time_s starts at 0.02 s. command_body is the command issued before that step.",
            "joint_order": names, "wheel_order": wheel_body_names,
            "termination_order": term_names,
            "array_trailing_shapes": {key: list(shape) for key, shape in trace_shapes.items()},
            "frames": {
                "root_pos_relative_w_m": "Root link position minus its initial world position; world XYZ axes.",
                "root_quat_wxyz": "Root world quaternion, scalar-first [w, x, y, z].",
                "root_link_lin_vel_w_mps": "Root link-origin linear velocity in world XYZ.",
                "root_lin_vel_b_mps": "Existing articulation root COM linear velocity in body XYZ.",
                "root_ang_vel_w_rad_s": "Root angular velocity in world XYZ.",
                "root_ang_vel_b_rad_s": "Root angular velocity in body XYZ.",
                "wheel_pos_relative_w_m": "Wheel body origins minus initial root position; world XYZ axes, not contact points.",
                "wheel_pos_root_relative_w_m": "Wheel body origins minus current root position; world XYZ axes, not body axes.",
                "wheel_contact_force_norm_n": "Norm of contact_forces.data.net_forces_w for the matching wheel body; last physics substep, not a substep maximum.",
                "command_body": "[vx m/s, vy m/s, wz rad/s] in body coordinates.",
            },
            "termination_values": "Unmodified termination_manager.get_term values at each control step; columns follow termination_term_names.",
            "termination_reasons": "results.json stores each raw term at the case's first termination/truncation/4 m exit; absent required terms are false and not in termination_term_names.",
            "substep_caution": "These traces sample at 50 Hz. Original joints maxima/counters in results.json retain all active 200 Hz substeps, including spikes absent from control samples.",
            "diagnostic_no_actuator_termination": args.diagnostic_no_actuator_termination,
            "diagnostic_ablation": dict(target_rate=args.target_rate, reset_z=args.reset_z, settle_seconds=args.settle_seconds),
        }
        (args.output / "stair_failure_traces.json").write_text(json.dumps(trace_metadata, indent=2))
        result = dict(ablation=dict(target_rate=args.target_rate, reset_z=args.reset_z, settle_seconds=args.settle_seconds),
                      actuator_profile=args.actuator_profile, checkpoint=str(args.checkpoint), completed_updates=saved["completed_updates"],
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      measurement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      conditions="Nominal dynamics, no observation noise or parameter/push randomization; original action and actuator delays retained; three nominal-center repeats; seed 42; actual start captured after wrapper reset",
                      scope="Forward 4 m or first termination, maximum 25 s; no learning; not a full terrain success certification",
                      diagnostic_mode=args.mode, heading_feedback="wz_cmd=clip(-1.5*yaw,-0.6,0.6)" if args.mode == "heading" else None,
                      diagnostic_no_actuator_termination=args.diagnostic_no_actuator_termination,
                      termination_term_names=term_names, stair_failure_trace_metadata="stair_failure_traces.json",
                      decomposition_frame="Root link horizontal velocity projected on yaw-only axes; body tracking errors use original root COM body velocity",
                      decomposition_identity_max_error_m=float(decomposition_error.max()),
                      physics_dt=raw.physics_dt, control_dt=raw.step_dt, violation_tolerance=0.001,
                      leg_action_joint_names=leg_joint_names,
                      reward_weights={name: raw.reward_manager.get_term_cfg(name).weight for name in reward_names},
                      request_capture="After action scaling/base clip, immediately before soft-bound and rate projection; action-term joint order. No mutation.",
                      reward_capture="Weighted instantaneous reward rates before dt; diagnostic configuration weights, not assumed identical to formal reward curricula.",
                      joint_names=names, hard_limits_rad=hard[0].cpu().tolist(),
                      soft_limits_rad=soft[0].cpu().tolist(), velocity_limits_rad_s=vel_limit[0].cpu().tolist(),
                      effort_limits_nm=torque_limit[0].cpu().tolist(), cases=cases)
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
        (args.output / "status.json").write_text(json.dumps(dict(state="complete", trials=count)))
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actuator-profile", choices=("original", "rs0402_guard_v1", "rs0402_peak_v2"), default="rs0402_peak_v2")
    for name in ("root", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-rate", type=float, default=None)
    parser.add_argument("--reset-z", type=float, default=0.0)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--mode", choices=("baseline", "heading"), default="baseline")
    parser.add_argument("--diagnostic-no-actuator-termination", action="store_true",
                        help="Isolated diagnostic: disable actuator_violation termination only; retain all actuator limits and measurements.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "code_peak_20260918"), str(Path(__file__).resolve().parent),
                   str(args.root / "upstream/source/rsl_rl"),
                   str(args.root / "upstream/source/robot_lab"), str(args.root / "thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        evaluate(args)
    except BaseException as error:
        traceback.print_exc()
        (args.output / "status.json").write_text(json.dumps(dict(state="failed", error=str(error))))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
