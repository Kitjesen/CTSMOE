"""Frozen-policy trajectory decomposition and heading-feedback diagnostic."""
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
    from train_straight_speed_20260918 import apply_command_curriculum_start

    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    columns = saved["terrain_column_names"]
    cases = []
    for learner in ("teacher", "student"):
        for name in ("wave", "stairs_up", "stairs_down"):
            available = [i for i, value in enumerate(columns) if value == name]
            for level in (3, 6, 9):
                for speed in (0.3, 0.6, 0.9):
                    for repeat, lateral in enumerate((-0.2, 0.0, 0.2)):
                        cases.append(dict(learner=learner, terrain=name, level=level, speed=speed,
                                          start_lateral_m=lateral, column=available[repeat % len(available)]))

    class FirstEpisodeEnv(Go2Env):
        retain_terminal = False

        def _reset_idx(self, env_ids):
            if not self.retain_terminal:
                super()._reset_idx(env_ids)

    count = len(cases)
    cfg = make_config(count, args.root / "cache/usd")
    cfg.seed = 42
    cfg.log_dir = str(args.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, "intermediate")
    configure_base_height_penalty(cfg, "off")
    configure_stair_speed(cfg, "straight_speed_v1")
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    cfg.rewards.lin_vel_z_l2.weight = 0
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    for name in vars(cfg.events):
        if name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.func = reset_root_state_uniform
    cfg.events.reset_base.params["pose_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_base.params["velocity_range"] = dict(cfg.events.reset_base.params["pose_range"])
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    raw = FirstEpisodeEnv(cfg=cfg)
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
            if args.mode == "heading":
                q = raw.scene["robot"].data.root_quat_w
                yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
                command.commands[:, 2] = (-1.5 * yaw).clamp(-0.6, 0.6)

        command._update_command = command_update
        raw.reset()
        robot = raw.scene["robot"]
        initial = robot.data.root_state_w.clone()
        initial[:, 1] += torch.tensor([c["start_lateral_m"] for c in cases], device=raw.device)
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
                     and ((c["terrain"] == "wave" and c["speed"] == 0.3 and c["start_lateral_m"] == 0.2)
                          or (c["terrain"] == "stairs_up" and c["start_lateral_m"] == 0.2)
                          or (c["terrain"] == "stairs_down" and c["speed"] == 0.6 and c["start_lateral_m"] == -0.2))]
        traces = []

        def measure(dt):
            update_scene(dt)
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
        for index, case in enumerate(cases):
            case.update(zip(keys, scalar[index]))
            case["joints"] = {k: rows[index] for k, rows in measured.items()}
            case["decomposition_final"] = {k: rows[index] for k, rows in decomposition.items()}
            case["decomposition_at_peak"] = {k: rows[index] for k, rows in peak_decomposition.items()}
            case.update(zip(("peak_cross_signed_m", "heading_at_peak_rad", "peak_time_s", "max_heading_rad",
                             "final_y_m"), peak_values[index]))
            case["peak_integration_residual_m"] = (case["peak_cross_signed_m"]
                                                   - case["decomposition_at_peak"]["world_velocity_y_m"])
        import numpy as np
        np.savez_compressed(args.output / "selected_traces.npz", values=torch.stack(traces).cpu().numpy(),
                            case_indices=np.array(trace_ids), fields=np.array([
                                "time_s", "active", "forward_m", "cross_m", "yaw_rad", "body_vx", "body_vy",
                                "body_wz", "heading_command_m", "heading_speed_error_m", "lateral_motion_m", "cmd_wz"]))
        result = dict(checkpoint=str(args.checkpoint), completed_updates=saved["completed_updates"],
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      measurement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      conditions="Nominal dynamics, no observation noise or parameter/push randomization; original action and actuator delays retained; initial lateral offsets -0.2/0/0.2 m; seed 42",
                      scope="Forward 4 m or first termination, maximum 25 s; no learning; not a full terrain success certification",
                      diagnostic_mode=args.mode, heading_feedback="wz_cmd=clip(-1.5*yaw,-0.6,0.6)" if args.mode == "heading" else None,
                      decomposition_frame="Root link horizontal velocity projected on yaw-only axes; body tracking errors use original root COM body velocity",
                      decomposition_identity_max_error_m=float(decomposition_error.max()),
                      physics_dt=raw.physics_dt, control_dt=raw.step_dt, violation_tolerance=0.001,
                      joint_names=names, hard_limits_rad=hard[0].cpu().tolist(),
                      soft_limits_rad=soft[0].cpu().tolist(), velocity_limits_rad_s=vel_limit[0].cpu().tolist(),
                      effort_limits_nm=torque_limit[0].cpu().tolist(), cases=cases)
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
        (args.output / "status.json").write_text(json.dumps(dict(state="complete", trials=count)))
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "heading"), default="baseline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "code"), str(args.root / "upstream/source/rsl_rl"),
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
