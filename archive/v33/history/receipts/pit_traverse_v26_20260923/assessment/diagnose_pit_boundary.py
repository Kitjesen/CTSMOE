"""Zero-update 25-cm boundary and support diagnosis for one checkpoint."""

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
        leg_torque_max = torch.zeros(n, device=raw.device)
        wheel_torque_max = torch.zeros(n, device=raw.device)
        leg_speed_max = torch.zeros(n, device=raw.device)
        wheel_speed_max = torch.zeros(n, device=raw.device)
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
            hip_nearhard_steps.add_((inward >= .39).any(-1).float() * c.active.float())
            torque = data.applied_torque.abs()
            speed = data.joint_vel.abs()
            leg_torque_max.copy_(torch.maximum(leg_torque_max, torque[:, leg_ids].amax(-1)))
            wheel_torque_max.copy_(torch.maximum(wheel_torque_max, torque[:, wheel_ids].amax(-1)))
            leg_speed_max.copy_(torch.maximum(leg_speed_max, speed[:, leg_ids].amax(-1)))
            wheel_speed_max.copy_(torch.maximum(wheel_speed_max, speed[:, wheel_ids].amax(-1)))

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
                    max_leg_torque_Nm=float(leg_torque_max[i]),
                    max_wheel_torque_Nm=float(wheel_torque_max[i]),
                    max_leg_speed_rad_s=float(leg_speed_max[i]),
                    max_wheel_speed_rad_s=float(wheel_speed_max[i]),
                ))
            original_record(c, ids)
            for values in (hip_max, hip_nearhard_steps, leg_torque_max,
                           wheel_torque_max, leg_speed_max, wheel_speed_max):
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
                        print("PIT25_BOUNDARY", args.completed, args.mean_action, step,
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
