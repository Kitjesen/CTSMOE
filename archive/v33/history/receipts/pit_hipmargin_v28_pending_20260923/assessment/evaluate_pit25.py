"""Evaluate a V27 checkpoint at 25 cm without optimizer updates."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from preflight import SHARED_LOCK, training_args, verify_frozen  # noqa: E402
from state_guard import MutationGuard, state_sha256  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['learner']}:{row['speed_bin']}"].append(row)
    result = {}
    for key, values in groups.items():
        entered = [row for row in values if row["entered"]]
        measured = [row["top_mean_abs_speed_error_mps"] for row in entered
                    if row["top_mean_abs_speed_error_mps"] is not None]
        result[key] = dict(
            episodes=len(values), complete=sum(row["success"] for row in values),
            body_cleared=len(entered), physical=sum(row["physical"] for row in values),
            overspeed=sum(row["overspeed"] for row in values),
            outside=sum(row["outside"] for row in values),
            mean_top_speed_error_mps=sum(measured) / len(measured) if measured else None,
            mean_supported_tracking_s=(sum(row["supported_tracking_s"] for row in entered)
                                       / len(entered)) if entered else None,
            max_actual_hip_inward_rad=max(row["max_actual_hip_inward_rad"] for row in values),
            near_hard_hip_seconds=sum(row["near_hard_hip_seconds"] for row in values),
        )
    return result


def make_probe(args, source_hash):
    def probe(self, updates, init_at_random_ep_len=False):
        import torch
        from hip_contract import INWARD_SIGNS
        from pit_task import PitCommand

        raw = self.env.unwrapped
        if (updates != 0 or init_at_random_ep_len or raw.num_envs != 4096
                or self.current_learning_iteration != args.completed
                or raw.max_episode_length != 1250):
            raise RuntimeError("Probe requires the full episode, reviewed checkpoint, zero updates")
        command = raw.command_manager.get_term("base_velocity")
        pit = command.pit
        if pit.training_profile != "platform_traverse_v1":
            raise RuntimeError("Probe is not evaluating the continuous-command task")
        allocation = getattr(raw, "terrain_allocation", None)
        if allocation is not None and allocation.pending:
            raise RuntimeError("Pending terrain moves would change the fixed-case cohort")
        terrain = raw.scene.terrain
        selected = pit.active.clone()
        terrain.terrain_levels[selected] = 2  # 15 + 5*2 = 25 cm.
        terrain.env_origins[selected] = terrain.terrain_origins[2, terrain.terrain_types[selected]]
        raw.scene.env_origins[selected] = terrain.env_origins[selected]
        pit.fixed_levels = torch.full_like(terrain.terrain_levels, 2)
        raw.reset()
        if not bool((terrain.terrain_levels[pit.active] == 2).all()):
            raise RuntimeError("25 cm physical pit assignment did not survive reset")
        names = pit.robot.joint_names
        hip_names = tuple(INWARD_SIGNS)
        hip_ids = [names.index(name) for name in hip_names]
        hip_signs = torch.tensor([INWARD_SIGNS[name] for name in hip_names], device=raw.device)
        leg_ids = [i for i, name in enumerate(names)
                   if name.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
        wheel_ids = [i for i, name in enumerate(names) if name.endswith("_foot_joint")]
        n = raw.num_envs
        hip_max = torch.zeros(n, device=raw.device)
        hip_nearhard_steps = torch.zeros(n, device=raw.device)
        hip_nearhard_pre_steps = torch.zeros(n, device=raw.device)
        hip_nearhard_post_steps = torch.zeros(n, device=raw.device)
        leg_torque_max = torch.zeros(n, device=raw.device)
        wheel_torque_max = torch.zeros(n, device=raw.device)
        leg_speed_max = torch.zeros(n, device=raw.device)
        wheel_speed_max = torch.zeros(n, device=raw.device)
        post_entry_s = torch.zeros(n, device=raw.device)
        entry_rel_y = torch.zeros(n, device=raw.device)
        abs_body_vy_m = torch.zeros(n, device=raw.device)
        abs_yaw_rate_rad = torch.zeros(n, device=raw.device)
        heading_side_m = torch.zeros(n, device=raw.device)
        body_side_m = torch.zeros(n, device=raw.device)
        world_side_m = torch.zeros(n, device=raw.device)
        original_refresh = PitCommand.refresh
        original_record = PitCommand.record_and_resample_levels
        rows = []
        old_clock = raw.common_step_counter
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}

        def refreshed(c):
            if c is not pit or c.last_refresh == raw.common_step_counter:
                return original_refresh(c)
            original_refresh(c)
            data = c.robot.data
            inward = data.joint_pos[:, hip_ids] * hip_signs
            hip_max.copy_(torch.maximum(hip_max, inward.amax(-1)))
            nearhard = (inward >= .39).any(-1).float() * c.active.float()
            hip_nearhard_steps.add_(nearhard)
            hip_nearhard_pre_steps.add_(nearhard * (~c.platform_entered).float())
            hip_nearhard_post_steps.add_(nearhard * c.platform_entered.float())
            torque = data.applied_torque.abs()
            speed = data.joint_vel.abs()
            leg_torque_max.copy_(torch.maximum(leg_torque_max, torque[:, leg_ids].amax(-1)))
            wheel_torque_max.copy_(torch.maximum(wheel_torque_max, torque[:, wheel_ids].amax(-1)))
            leg_speed_max.copy_(torch.maximum(leg_speed_max, speed[:, leg_ids].amax(-1)))
            wheel_speed_max.copy_(torch.maximum(wheel_speed_max, speed[:, wheel_ids].amax(-1)))
            on_top = c.active & c.platform_entered
            entry = on_top & (post_entry_s == 0)
            entry_rel_y[entry] = (data.root_pos_w[entry, 1]
                                  - raw.scene.env_origins[entry, 1])
            quat = data.root_quat_w
            yaw = torch.atan2(
                2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                1 - 2 * (quat[:, 2].square() + quat[:, 3].square()),
            )
            dt = raw.step_dt * on_top.float()
            post_entry_s.add_(dt)
            abs_body_vy_m.add_(data.root_lin_vel_b[:, 1].abs() * dt)
            abs_yaw_rate_rad.add_(data.root_ang_vel_b[:, 2].abs() * dt)
            heading_side_m.add_(yaw.sin() * data.root_lin_vel_b[:, 0] * dt)
            body_side_m.add_(yaw.cos() * data.root_lin_vel_b[:, 1] * dt)
            world_side_m.add_(data.root_lin_vel_w[:, 1] * dt)

        def recorded(c, ids):
            if c is not pit:
                return original_record(c, ids)
            ids = ids[(raw.episode_length_buf[ids] > 0) & c.active[ids]]
            c.refresh()
            causes = raw.actuator_guard.violation_causes
            for i in ids.cpu().tolist():
                supported_time = float(c.platform_tracked_s[i])
                rel_x = float(c.robot.data.root_pos_w[i, 0] - raw.scene.env_origins[i, 0])
                rel_y = float(c.robot.data.root_pos_w[i, 1] - raw.scene.env_origins[i, 1])
                rows.append(dict(
                    env=i, learner="student" if int(c.learner[i]) else "teacher",
                    level=int(terrain.terrain_levels[i]), height_m=.25,
                    speed_bin=int(c.speed_bins[i]), command_vx_mps=float(c.held_speed[i]),
                    age_s=float(raw.episode_length_buf[i]*raw.step_dt),
                    entered=bool(c.platform_entered[i]),
                    entry_x_m=float(c.platform_entry_x[i]) if bool(c.platform_entered[i]) else None,
                    terminal_root_x_m=rel_x, terminal_root_y_m=rel_y,
                    outside_x=rel_x > 4. or rel_x < -.8,
                    outside_y=abs(rel_y) > 1.,
                    front_supported=bool(c.supported[i, :2].any()),
                    rear_supported=bool(c.supported[i, 2:].any()),
                    body_clear_at_end=bool(c.platform_clear[i]),
                    actual_vx_at_end_mps=float(c.robot.data.root_lin_vel_b[i, 0]),
                    post_entry_s=float(post_entry_s[i]),
                    entry_rel_y_m=float(entry_rel_y[i]) if bool(c.platform_entered[i]) else None,
                    post_entry_side_displacement_m=rel_y - float(entry_rel_y[i])
                    if bool(c.platform_entered[i]) else None,
                    abs_body_vy_integral_m=float(abs_body_vy_m[i]),
                    abs_yaw_rate_integral_rad=float(abs_yaw_rate_rad[i]),
                    heading_side_component_m=float(heading_side_m[i]),
                    body_side_component_m=float(body_side_m[i]),
                    world_side_velocity_integral_m=float(world_side_m[i]),
                    top_travel_m=rel_x - float(c.platform_entry_x[i])
                    if bool(c.platform_entered[i]) else 0.,
                    top_elapsed_s=float(c.platform_elapsed_s[i]),
                    supported_tracking_s=supported_time,
                    top_mean_abs_speed_error_mps=(float(c.platform_abs_error_m[i]/supported_time)
                                                  if supported_time > 0 else None),
                    success=bool(c.success[i]), physical=bool(c.physical[i]),
                    overspeed=bool(causes["speed"][i]),
                    torque_violation=bool(causes["torque"][i]),
                    position_violation=bool(causes["position"][i]),
                    body_contact=bool(c.body_contact[i]), outside=bool(c.outside[i]),
                    timeout=bool(raw.episode_length_buf[i] >= raw.max_episode_length-1),
                    max_actual_hip_inward_rad=float(hip_max[i]),
                    near_hard_hip_seconds=float(hip_nearhard_steps[i]*raw.step_dt),
                    near_hard_hip_pre_entry_seconds=float(hip_nearhard_pre_steps[i]*raw.step_dt),
                    near_hard_hip_post_entry_seconds=float(hip_nearhard_post_steps[i]*raw.step_dt),
                    max_leg_torque_Nm=float(leg_torque_max[i]),
                    max_wheel_torque_Nm=float(wheel_torque_max[i]),
                    max_leg_speed_rad_s=float(leg_speed_max[i]),
                    max_wheel_speed_rad_s=float(wheel_speed_max[i]),
                ))
            original_record(c, ids)
            for values in (hip_max, hip_nearhard_steps, hip_nearhard_pre_steps,
                           hip_nearhard_post_steps, leg_torque_max,
                           wheel_torque_max, leg_speed_max, wheel_speed_max,
                           post_entry_s, entry_rel_y, abs_body_vy_m,
                           abs_yaw_rate_rad, heading_side_m, body_side_m,
                           world_side_m):
                values[ids] = 0

        PitCommand.refresh = refreshed
        PitCommand.record_and_resample_levels = recorded
        guard = MutationGuard(self)
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        try:
            with guard, torch.inference_mode():
                for step in range(args.steps):
                    actions = self.alg.act(obs)
                    if args.mean_action:
                        means = torch.zeros_like(actions)
                        teachers, students = self.alg.teacher_env_idxs, self.alg.student_env_idxs
                        means[teachers] = self.alg.transition.action_mean[:self.alg.teacher_num_envs]
                        means[students] = self.alg.transition.action_mean[self.alg.teacher_num_envs:]
                        actions[pit.active] = means[pit.active]
                    obs, reward, done, extras = self.env.step(actions.to(raw.device))
                    obs, reward, done = obs.to(self.device), reward.to(self.device), done.to(self.device)
                    self.alg.process_env_step(obs, reward, done, extras)
                    if bool(pit.stop_on_platform[pit.active].any()):
                        raise RuntimeError("Probe switched a pit environment into stop mode")
                    entered = pit.active & pit.platform_entered
                    if bool(entered.any()) and not bool(torch.equal(
                            command.commands[entered, 0], pit.held_speed[entered])):
                        raise RuntimeError("Post-entry command deviated from the sampled speed")
                    if (step+1) % 24 == 0:
                        self.alg.storage.clear()
                    if step % 300 == 0:
                        print("PIT25_LATERAL", args.completed, args.mean_action, step,
                              "episodes", len(rows), flush=True)
                self.alg.storage.clear()
        finally:
            PitCommand.refresh = original_refresh
            PitCommand.record_and_resample_levels = original_record
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if (before != after or guard.calls or not guard.restored
                or digest(args.source) != source_hash
                or self.current_learning_iteration != args.completed
                or raw.common_step_counter != old_clock + args.steps
                or list(args.output.rglob("*.pt")) or not rows
                or any(row["level"] != 2 for row in rows)):
            raise RuntimeError("Zero-update evidence, height assignment or optimizer proof failed")
        (args.output / "FINALIZED.json").write_text(json.dumps(dict(
            state="complete", optimizer_updates=0, completed_updates=args.completed,
            action="pit_mean_only" if args.mean_action else "training_stochastic",
            heading_command="yaw_world_to_zero_kp1.5_clip0.6",
            source_sha256=source_hash, source_path=str(args.source), steps=args.steps,
            full_episode_s=25, forced_height_m=.25, fixed_cases=False,
            independent_rollout=True, model_and_dual_optimizer_unchanged=True,
            hashes_before=before, hashes_after=after,
            evaluated_episodes=len(rows), by_learner_and_speed=summarise(rows),
            episodes=rows,
            limitation="Seeded batched simulation episodes; not real-robot validation or matched fixed-case pairs",
        ), indent=2))

    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--completed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--mean-action", action="store_true")
    args = parser.parse_args()
    if sys.flags.optimize or args.steps < 750 or args.steps > 3000 or args.output.exists():
        raise ValueError("Diagnosis needs a fresh output and at least 750 control steps")
    verify_frozen()
    if digest(args.source) != args.source_sha256:
        raise ValueError("Selected source checkpoint hash changed")
    with SHARED_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU is occupied; no parallel evaluation")
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original = early_restart.configure

        def configure(cfg, profile):
            original(cfg, profile)
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = make_probe(args, args.source_sha256)

        early_restart.configure = configure
        sys.argv = [str(STAGE / "code/train.py"),
                    *training_args(args.source, args.output)]
        from train import main as train_main

        train_main()
        if not (args.output / "FINALIZED.json").is_file():
            raise RuntimeError("Assessment returned without a zero-update receipt")


if __name__ == "__main__":
    main()
