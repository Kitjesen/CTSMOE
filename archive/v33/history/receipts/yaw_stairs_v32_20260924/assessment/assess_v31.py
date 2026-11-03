"""Matched eight-terrain, deterministic student assessment with zero updates."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from state_guard import MutationGuard, state_sha256  # noqa: E402
from v31_common import ENVS, args as training_args, digest, frozen  # noqa: E402

STEPS = 1250  # One 25-second episode at 50 Hz; normal resets remain enabled.
TYPES = ("flat", "stairs_up", "stairs_down", "wave", "slope_up",
         "slope_down", "rough_slope", "obstacles")


def make_assessment(checkpoint, output, receipt):
    def assess(self, updates, init_at_random_ep_len=False):
        import torch
        from hip_contract import INWARD_SIGNS
        from terrain_allocation import NO_PIT, NO_PIT_PROFILE, reassign

        raw = self.env.unwrapped
        command = raw.command_manager.get_term("base_velocity")
        terrain = raw.scene.terrain
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = torch.load(receipt["source_path"], map_location="cpu", weights_only=False)
        if (updates or init_at_random_ep_len or self.env.num_envs != ENVS
                or self.current_learning_iteration != saved["completed_updates"]
                or self.alg.storage.step != 0):
            raise RuntimeError("Assessment must be a fresh zero-update 4096-env run")
        if digest(checkpoint) != receipt["selected_sha256"]:
            raise RuntimeError("Selected model changed")
        student = self.alg.student_env_idxs
        groups = {"teacher": self.alg.teacher_env_idxs.cpu().tolist(),
                  "student": student.cpu().tolist()}
        levels, columns, migration = reassign(
            source["terrain_levels"].tolist(), source["terrain_types"].tolist(),
            source["terrain_column_names"], groups, NO_PIT,
            seed=receipt["source_completed_updates"], profile=NO_PIT_PROFILE)
        if migration["group_counts"]["student"]["pit"]:
            raise RuntimeError("Assessment assigned high-platform environment")
        terrain.terrain_levels[:] = torch.tensor(levels, device=raw.device)
        terrain.terrain_types[:] = torch.tensor(columns, device=raw.device)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        command._init_terrain_infos()
        command._update_env_command_ranges()
        command.refresh_membership()
        raw.common_step_counter = source["common_step_counter"]
        torch.manual_seed(230923)
        torch.cuda.manual_seed_all(230923)
        original_compute = raw.curriculum_manager.compute
        original_reset_idx = raw._reset_idx
        raw.curriculum_manager.compute = lambda env_ids: None
        try:
            raw.reset()
            if bool(command.pit.active.any()) or bool((terrain.terrain_types >= 20).any()):
                raise RuntimeError("Assessment pit allocation or command leaked")
            torch.testing.assert_close(terrain.terrain_levels.cpu(), torch.tensor(levels), atol=0, rtol=0)
            torch.testing.assert_close(terrain.terrain_types.cpu(), torch.tensor(columns), atol=0, rtol=0)
            if hasattr(raw, 'variety_episode_stats'):
                raw.variety_episode_stats.zero_()
            obs = self.env.get_observations().to(self.device)
            robot = raw.scene["robot"]
            term = raw.action_manager.get_term("joint_pos")
            hip_names = list(INWARD_SIGNS)
            hip_ids = torch.tensor([robot.joint_names.index(name) for name in hip_names], device=raw.device)
            hip_action_ids = torch.tensor([list(term.joint_indices).index(int(i)) for i in hip_ids], device=raw.device)
            signs = torch.tensor(list(INWARD_SIGNS.values()), device=raw.device)
            leg_ids = torch.tensor(list(term.joint_indices), device=raw.device)
            wheel_ids = torch.tensor([i for i, name in enumerate(robot.joint_names)
                                      if name.endswith("foot_joint")], device=raw.device)
            if len(wheel_ids) != 4:
                raise RuntimeError("Expected four wheel joints")
            lookup = torch.tensor([TYPES.index(source["terrain_column_names"][col]) for col in columns],
                                  device=raw.device)
            kind = lookup[student]
            count = torch.zeros(len(TYPES), device=raw.device)
            err_xy = torch.zeros_like(count)
            err_yaw = torch.zeros_like(count)
            near = torch.zeros_like(count)
            physical = torch.zeros_like(count)
            timeouts = torch.zeros_like(count)
            pose_sq = torch.zeros_like(count)
            start_sq = torch.zeros_like(count)
            start_count = torch.zeros_like(count)
            command_zero = torch.zeros_like(count)
            body_height_sum = torch.zeros_like(count)
            flat_pure_yaw = torch.tensor(0., device=raw.device)
            flat_pure_side = torch.tensor(0., device=raw.device)
            reverse_stair_samples = torch.tensor(0., device=raw.device)
            flat_command_abs_max = torch.zeros(3, device=raw.device)
            wheel_peak = torch.tensor(0., device=raw.device)
            torque_peak = torch.tensor(0., device=raw.device)
            requested_peak = torch.full((4,), -torch.inf, device=raw.device)
            target_peak = requested_peak.clone()
            actual_peak = requested_peak.clone()
            # Same fixed environment indices for source/candidate; preserve
            # per-joint trajectories rather than only an aggregate pose RMS.
            trace_ids = torch.cat([student[(kind == i).nonzero().flatten()[:8]] for i in range(len(TYPES))])
            trace = {k: [] for k in ('q', 'requested', 'target', 'joint_velocity', 'torque',
                                    'command', 'root_velocity', 'root_position', 'wheel_force', 'done')}
            trace_wheels = [raw.scene['landing_contacts'].body_names.index(f'{s}_foot_Link') for s in ('fr', 'fl', 'rr', 'rl')]
            force_windows = []
            before = {name: state_sha256(value, torch) for name, value in (
                ("model", self.alg.policy.state_dict()),
                ("ppo_optimizer", self.alg.optimizer.state_dict()),
                ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
            guard = MutationGuard(self)
            from terrain_variety import VARIANTS
            route_events = torch.zeros((5, 5, 4), dtype=torch.long, device=raw.device)
            def observe_terminal_before_reset(env_ids):
                ids = torch.as_tensor(env_ids, device=raw.device, dtype=torch.long)
                ids = ids[torch.isin(ids, student) & (raw.episode_length_buf[ids] > 0)]
                for index, col in enumerate(VARIANTS):
                    selected = ids[terrain.terrain_types[ids] == col]
                    physical_end = raw.termination_manager.terminated[selected]
                    values = torch.stack((torch.ones_like(selected),
                        (command.route_complete[selected] & ~physical_end).long(), physical_end.long(),
                        raw.termination_manager.time_outs[selected].long()), -1)
                    route_events[index].index_add_(0, raw.omni_route_group[selected], values)
                return original_reset_idx(env_ids)
            raw._reset_idx = observe_terminal_before_reset
            self.alg.policy.eval()
            with guard, torch.inference_mode():
                for step in range(STEPS):
                    issued = command.commands[student].clone()
                    reverse_stair_samples += (((kind == 1) | (kind == 2)) & (issued[:, 0] < -.01)).sum()
                    actions = self.alg.policy.act_inference(obs)
                    obs, rewards, done, _extras = self.env.step(actions.to(raw.device))
                    obs = obs.to(self.device)
                    wheel_force = raw.scene['landing_contacts'].data.net_forces_w_history[:, :, trace_wheels, 2].clamp_min(0).amax(1)
                    force_windows.append(wheel_force[student].cpu())
                    for name, value in (
                        ('q', robot.data.joint_pos[trace_ids][:, leg_ids]),
                        ('requested', term.requested_actions[trace_ids]),
                        ('target', term.processed_actions[trace_ids]),
                        ('joint_velocity', robot.data.joint_vel[trace_ids]),
                        ('torque', robot.data.applied_torque[trace_ids]),
                        ('command', command.commands[trace_ids]),
                        ('root_velocity', robot.data.root_lin_vel_w[trace_ids]),
                        ('root_position', robot.data.root_pos_w[trace_ids]),
                        ('wheel_force', wheel_force[trace_ids]),
                        ('done', done[trace_ids])):
                        trace[name].append(value.cpu())
                    if not bool(torch.isfinite(rewards).all()):
                        raise FloatingPointError("Nonfinite assessment reward")
                    q = robot.data.joint_pos[student]
                    actual = q[:, hip_ids] * signs
                    requested = term.requested_actions[student][:, hip_action_ids] * signs
                    target = term.processed_actions[student][:, hip_action_ids] * signs
                    requested_peak = torch.maximum(requested_peak, requested.amax(0))
                    target_peak = torch.maximum(target_peak, target.amax(0))
                    actual_peak = torch.maximum(actual_peak, actual.amax(0))
                    actual_xy = robot.data.root_lin_vel_b[student, :2]
                    actual_yaw = robot.data.root_ang_vel_b[student, 2]
                    relative_height = robot.data.root_pos_w[student, 2] - terrain.env_origins[student, 2]
                    flat_issued = issued[kind == 0]
                    if len(flat_issued):
                        flat_command_abs_max = torch.maximum(flat_command_abs_max, flat_issued.abs().amax(0))
                        flat_pure_yaw += ((flat_issued[:, :2].abs() < .01).all(-1)
                                          & (flat_issued[:, 2].abs() > .01)).sum()
                        flat_pure_side += ((flat_issued[:, (0, 2)].abs() < .01).all(-1)
                                           & (flat_issued[:, 1].abs() > .01)).sum()
                    pose_error_sq = ((q[:, leg_ids] - robot.data.default_joint_pos[student][:, leg_ids]) ** 2).mean(-1)
                    for destination, values in (
                        (count, torch.ones_like(kind, dtype=torch.float)),
                        (err_xy, (actual_xy - issued[:, :2]).norm(dim=-1)),
                        (err_yaw, (actual_yaw - issued[:, 2]).abs()),
                        (near, (actual >= .39).any(-1).float()),
                        (physical, raw.termination_manager.terminated[student].float()),
                        (timeouts, raw.termination_manager.time_outs[student].float()),
                        (pose_sq, pose_error_sq),
                        (body_height_sum, relative_height),
                        (command_zero, (issued.abs() < .01).all(-1).float())):
                        destination.index_add_(0, kind, values)
                    early = (raw.episode_length_buf[student] <= 50).float()
                    start_sq.index_add_(0, kind, pose_error_sq * early)
                    start_count.index_add_(0, kind, early)
                    wheel_peak = torch.maximum(wheel_peak, robot.data.joint_vel[student][:, wheel_ids].abs().amax())
                    torque_peak = torch.maximum(torque_peak, robot.data.applied_torque[student][:, leg_ids].abs().amax())
                    if step % 125 == 0:
                        print("V29_ASSESS", step, flush=True)
            after = {name: state_sha256(value, torch) for name, value in (
                ("model", self.alg.policy.state_dict()),
                ("ppo_optimizer", self.alg.optimizer.state_dict()),
                ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
            if before != after or guard.calls or not guard.restored or self.current_learning_iteration != saved["completed_updates"]:
                raise RuntimeError("Assessment altered model or optimizer")
            rows = {}
            for index, name in enumerate(TYPES):
                n = float(count[index])
                rows[name] = dict(control_samples=int(n),
                    xy_tracking_mae_mps=float(err_xy[index] / n),
                    yaw_tracking_mae_radps=float(err_yaw[index] / n),
                    any_hip_inward_ge_039_fraction=float(near[index] / n),
                    physical_termination_events=int(physical[index]),
                    timeout_events=int(timeouts[index]),
                    posture_rms_rad=float(torch.sqrt(pose_sq[index] / n)),
                    mean_body_height_from_tile_origin_m=float(body_height_sum[index] / n),
                    first_second_posture_rms_rad=float(torch.sqrt(start_sq[index] / start_count[index].clamp_min(1))),
                    all_axis_zero_command_fraction=float(command_zero[index] / n))
            output.mkdir(parents=True, exist_ok=False)
            import numpy as np
            np.savez_compressed(output/'joint_and_contact_traces.npz',
                **{k: torch.stack(v).numpy() for k, v in trace.items()},
                env_ids=trace_ids.cpu().numpy(), leg_joint_names=np.array([robot.joint_names[i] for i in leg_ids]),
                all_joint_names=np.array(robot.joint_names), control_dt=raw.step_dt)
            force_values = torch.stack(force_windows).flatten(0, 1)
            force_summary = dict(p95_per_wheel_N=force_values.quantile(.95, dim=0).tolist(),
                                 max_per_wheel_N=force_values.amax(0).tolist(),
                                 interpretation='200Hz force-history maxima per 50Hz control frame; includes support, not landing-only quantiles')
            from terrain_variety import snapshot as variety_snapshot
            (output / "FINALIZED.json").write_text(json.dumps(dict(
                state="complete", optimizer_updates=0, reverse_stair_control_samples=int(reverse_stair_samples),
                variety=variety_snapshot(raw), selected=str(checkpoint),
                selected_sha256=receipt["selected_sha256"],
                common_terrain_source_sha256=receipt["source_sha256"],
                same_initial_terrain_assignment=True, student_environments=len(student),
                steps=STEPS, control_period_s=raw.step_dt, terrain=rows,
                hip_names=hip_names, requested_inward_peak_rad=requested_peak.tolist(),
                limited_target_inward_peak_rad=target_peak.tolist(),
                actual_inward_peak_rad=actual_peak.tolist(),
                max_student_wheel_speed_radps=float(wheel_peak),
                max_student_leg_torque_nm=float(torque_peak),
                flat_pure_yaw_control_samples=int(flat_pure_yaw),
                flat_pure_side_control_samples=int(flat_pure_side),
                flat_command_abs_max=flat_command_abs_max.tolist(),
                landing_v31=__import__('landing_v31').snapshot(raw),
                force_window_summary=force_summary,
                fixed_route_events=route_events.cpu().tolist(),
                fixed_route_axes=['variant_in_column_order', 'command_group', 'episodes_supported_complete_physical_timeout'],
                omni_command_samples=command.route_commands_seen.cpu().tolist(),
                state_hashes_unchanged=before == after,
                meaning="matched initial terrain and simulator seed; fixed zero-update student rollout, not real-world validation",
            ), indent=2))
        finally:
            raw.curriculum_manager.compute = original_compute
            raw._reset_idx = original_reset_idx

    return assess


def main():
    global STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(3, 1250), default=1250)
    parsed = parser.parse_args()
    STEPS = parsed.steps
    receipt = frozen()
    if parsed.output.exists():
        raise FileExistsError("Assessment output already exists")
    receipt["selected_sha256"] = digest(parsed.checkpoint)
    import early_restart

    original = early_restart.configure

    def configure(cfg, profile):
        original(cfg, profile)
        from rsl_rl.runners import OnPolicyRunnerCTS

        OnPolicyRunnerCTS.learn = make_assessment(parsed.checkpoint, parsed.output, receipt)

    early_restart.configure = configure
    sys.argv = [str(STAGE / "code/train.py"),
                *training_args(parsed.checkpoint, parsed.output.with_name(parsed.output.name + "_runtime"), 0)]
    from train import main as train_main

    train_main()
    if not (parsed.output / "FINALIZED.json").is_file():
        raise RuntimeError("Assessment missing finalized receipt")


if __name__ == "__main__":
    sys.path.insert(0, str(STAGE / "code"))
    main()
