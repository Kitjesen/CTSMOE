"""Run 3000 physical control steps with no optimizer update before V31."""

import fcntl
import json
import subprocess
import sys

from state_guard import MutationGuard, state_sha256
from v31_common import ENVS, LOCK, STAGE, STEPS, args, digest, frozen


def preflight_factory(source, output, receipt):
    def preflight(self, updates, init_at_random_ep_len=False):
        import torch

        raw = self.env.unwrapped
        command = raw.command_manager.get_term("base_velocity")
        terrain = raw.scene.terrain
        if (updates or init_at_random_ep_len or self.env.num_envs != ENVS
                or self.current_learning_iteration != receipt["source_completed_updates"]
                or self.alg.storage.step != 0 or self.cfg["num_steps_per_env"] != 24):
            raise RuntimeError("V31 preflight age, shape or rollout state changed")
        run = json.loads((output / "run.json").read_text())
        if (run["starting_completed_updates"] != receipt["source_completed_updates"]
                or run["additional_updates"] != 0
                or run["no_pit_profile"] != "omni_eight_v1"
                or run["hip_margin"]["profile"] != "actual_prehard_v1"
                or run["mixed_stair_profile"] != "mixed_stair_short_pulses_v1"
                or run["mixed_checks"]["pit_count"] != 0
                or run["terrain_allocation"]["actual_counts"].get("pit", 0) != 0):
            raise RuntimeError("Eight-terrain, hip and mixed-stair configuration missing")
        expected_types = {"flat", "stairs_up", "stairs_down", "wave",
                          "slope_up", "slope_down", "rough_slope", "obstacles"}
        observed_types = set(run["terrain_by_type"])
        if observed_types != expected_types:
            raise RuntimeError(f"Unexpected terrain types: {observed_types}")
        if bool(command.pit.active.any()) or bool((terrain.terrain_types >= 20).any()):
            raise RuntimeError("High-platform environment survived migration")
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        initial_clock = raw.common_step_counter
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        guard = MutationGuard(self)
        mixed_steps = 0
        mixed_nonzero_yaw_steps = 0
        mixed_nonzero_side_steps = 0
        with guard, torch.inference_mode():
            for step in range(STEPS):
                actions = self.alg.act(obs)
                obs, rewards, done, extras = self.env.step(actions.to(raw.device))
                obs, rewards, done = obs.to(self.device), rewards.to(self.device), done.to(self.device)
                self.alg.process_env_step(obs, rewards, done, extras)
                if not all(bool(torch.isfinite(item).all()) for item in (
                        actions, rewards, raw.scene["robot"].data.joint_pos)):
                    raise FloatingPointError("Nonfinite preflight trajectory")
                if bool(command.pit.active.any()) or bool((terrain.terrain_types >= 20).any()):
                    raise RuntimeError("High-platform assignment reappeared")
                mixed = command.mixed_terminal_mask() & command.mixed_stair_course.active
                mixed_steps += int(mixed.sum())
                mixed_nonzero_yaw_steps += int((command.commands[mixed, 2].abs() > .01).sum())
                mixed_nonzero_side_steps += int((command.commands[mixed, 1].abs() > .01).sum())
                if (step + 1) % 24 == 0:
                    self.alg.storage.clear()
                if step % 300 == 0:
                    print("V31_ZERO_UPDATE", step, flush=True)
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if (before != after or guard.calls or not guard.restored
                or self.alg.storage.step != 0
                or self.current_learning_iteration != receipt["source_completed_updates"]
                or raw.common_step_counter != initial_clock + STEPS
                or mixed_steps == 0 or mixed_nonzero_yaw_steps == 0
                or mixed_nonzero_side_steps == 0
                or list(output.rglob("*.pt")) or digest(source) != receipt["source_sha256"]):
            raise RuntimeError("Zero-update or mixed-direction preflight failed")
        from terrain_variety import snapshot
        variety = snapshot(raw)
        if variety['reverse_envs'] == 0 or len(raw.cfg.scene.terrain.terrain_generator.variety_receipts) != 110:
            raise RuntimeError('Missing reverse commands or variant geometry')
        if any(v['count'] == 0 for v in variety['variants'].values()):
            raise RuntimeError('Unrepresented variant')
        from landing_v31 import snapshot as landing_snapshot
        impact = landing_snapshot(raw)
        seen = command.route_commands_seen.cpu().tolist()
        if min(seen) <= 0 or sum(impact['landings']) == 0 or impact['force_cost_sum'] <= 0 or impact['speed_cost_sum'] <= 0:
            raise RuntimeError('New command groups or landing rewards not exercised')
        if raw.reward_manager.get_term_cfg('landing_force_v31').weight != -.5 or raw.reward_manager.get_term_cfg('landing_velocity_v31').weight != -1.:
            raise RuntimeError('New landing reward weights absent')
        frozen()
        (output / "FINALIZED.json").write_text(json.dumps({
            "landing": impact, "new_command_samples": seen, "variety": variety, "state": "complete", "optimizer_updates": 0, "steps": STEPS, "envs": ENVS,
            "source_sha256": receipt["source_sha256"],
            "model_and_dual_optimizer_before": before,
            "model_and_dual_optimizer_after": after,
            "forbidden_calls": guard.calls,
            "mutation_guard_restored": guard.restored,
            "pit_active_final": int(command.pit.active.sum()),
            "mixed_steps": mixed_steps,
            "mixed_nonzero_yaw_steps": mixed_nonzero_yaw_steps,
            "mixed_nonzero_side_steps": mixed_nonzero_side_steps,
            "terrain_counts": run["terrain_allocation"]["actual_counts"],
            "interpretation": "sampling and configuration only; no learned improvement",
        }, indent=2))

    return preflight


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions must remain enabled")
    receipt = frozen()
    source = receipt["source_path"]
    output = STAGE / "preflight_3000"
    if output.exists():
        raise FileExistsError("Preflight already attempted")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if gpu.strip():
            raise RuntimeError("GPU occupied")
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original = early_restart.configure

        def configure(cfg, profile):
            original(cfg, profile)
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = preflight_factory(source, output, receipt)

        early_restart.configure = configure
        sys.argv = [str(STAGE / "code/train.py"), *args(source, output, 0)]
        from train import main as train_main

        train_main()
        if not (output / "FINALIZED.json").is_file():
            raise RuntimeError("Zero-update rollout did not finalize")


if __name__ == "__main__":
    main()
