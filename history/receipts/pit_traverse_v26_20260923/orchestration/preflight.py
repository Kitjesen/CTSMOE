"""Run a locked 3000-step pit-traverse rollout without optimizer updates."""

import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from state_guard import MutationGuard, state_sha256


STAGE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
SHARED_LOCK = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922/full_episode_v12_pipeline.lock")
STEPS = 3000
ENVS = 4096


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_frozen():
    receipt = json.loads((HERE / "FROZEN.json").read_text())
    actual = {path.relative_to(STAGE / "code").as_posix(): digest(path)
              for path in (STAGE / "code").rglob("*.py")}
    if actual != receipt["code"]:
        raise ValueError("Isolated pit-traverse source has changed since review")
    scripts = {name: digest(HERE / name) for name in (
        "preflight.py", "state_guard.py", "verify_preflight.py", "driver.py")}
    if scripts != receipt["orchestration"]:
        raise ValueError("Pit-traverse preflight scripts changed since review")
    source = Path(receipt["source_path"])
    if digest(source) != receipt["source_sha256"]:
        raise ValueError("Reviewed source checkpoint changed")
    if receipt["source_pit_profile"] not in ("strict_v1", "climb_first_v2"):
        raise ValueError("Expected a reviewed stationary/climb-first pit source")
    return receipt


def training_args(source, output, updates=0):
    return [
        "--root", str(BASE), "--output", str(output), "--num-envs", str(ENVS),
        "--updates", str(updates), "--mixed-pit", "--resume", str(source),
        "--early-restart-profile", "early_phase_v14_no_reward",
        "--startup-repair-profile", "off", "--pit-posture-profile", "off",
        "--surface-posture-profile", "off", "--stance-strength-profile", "off",
        "--allocation-profile", "balanced_adaptive_v1",
        "--pit-training-profile", "platform_traverse_v1",
        "--pit-course-profile", "global_v1",
        "--actuator-profile", "rs0402_transient_v4",
        "--slope-profile", "slope45_v1", "--command-curriculum-start", "intermediate",
        "--base-height-penalty", "original", "--terrain-training-profile", "stairs80_v1",
        "--stair-command-profile", "straight_speed_v1",
        "--wave-reset-profile", "wave_surface_v1",
        "--hip-request-profile", "off", "--mixed-stair-profile", "off",
    ]


def make_preflight(source, output, frozen):
    def preflight(self, updates, init_at_random_ep_len=False):
        import torch

        raw = self.env.unwrapped
        if (updates != 0 or init_at_random_ep_len or self.env.num_envs != ENVS
                or self.current_learning_iteration != frozen["source_completed_updates"]
                or self.alg.storage.step != 0 or self.cfg["num_steps_per_env"] != 24):
            raise RuntimeError("Unreviewed preflight age, shape, storage or update count")
        initial_clock = raw.common_step_counter
        receipt = json.loads((output / "run.json").read_text())
        required = ("model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
                    "command_terrain_membership", "terrain_restored_exactly",
                    "terrain_allocation", "stair_episode_totals", "pit_course_gate_reset")
        if (receipt["resume_sha256"] != frozen["source_sha256"]
                or receipt["pit_training_profile"] != "platform_traverse_v1"
                or receipt["starting_completed_updates"] != frozen["source_completed_updates"]
                or receipt["additional_updates"] != 0
                or not all(receipt["resume_checks"].get(name) is True for name in required)):
            raise RuntimeError("Preflight did not restore the source and reset the new pit gate")
        command = raw.command_manager.get_term("base_velocity")
        pit = command.pit
        if pit.training_profile != "platform_traverse_v1" or pit.stop_on_platform.any():
            raise RuntimeError("The intended no-stop pit profile is not active")
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        guard = MutationGuard(self)
        platform_samples = 0
        platform_error_sum = 0.0
        support_samples = 0
        with guard, torch.inference_mode():
            for step in range(STEPS):
                actions = self.alg.act(obs)
                obs, rewards, done, extras = self.env.step(actions.to(raw.device))
                obs, rewards, done = obs.to(self.device), rewards.to(self.device), done.to(self.device)
                self.alg.process_env_step(obs, rewards, done, extras)
                if not all(bool(torch.isfinite(item).all()) for item in (
                        actions, rewards, raw.scene["robot"].data.joint_pos)):
                    raise FloatingPointError("Nonfinite state during zero-update rollout")
                entered = pit.active & pit.platform_entered
                if bool(entered.any()):
                    selected = command.commands[entered, 0]
                    if (not bool(torch.allclose(selected, pit.held_speed[entered], rtol=0, atol=0))
                            or not bool((selected >= .2).all())
                            or bool(pit.stop_on_platform[entered].any())):
                        raise RuntimeError("Pit command switched to zero after platform entry")
                    platform_samples += int(entered.sum())
                    platform_error_sum += float((
                        raw.scene["robot"].data.root_lin_vel_b[entered, 0] - selected).abs().sum())
                    support_samples += int((pit.supported[entered, :2].any(-1)
                                            & pit.supported[entered, 2:].any(-1)).sum())
                if (step + 1) % 24 == 0:
                    self.alg.storage.clear()
                if step % 300 == 0:
                    print("PIT_TRAVERSE_PREFLIGHT", step, flush=True)
        if self.alg.storage.step != 0 or platform_samples == 0:
            raise RuntimeError("Incomplete rollout or no observed top-platform entry")
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if (before != after or guard.calls or not guard.restored
                or raw.common_step_counter != initial_clock + STEPS
                or self.current_learning_iteration != frozen["source_completed_updates"]
                or list(output.rglob("*.pt")) or digest(source) != frozen["source_sha256"]):
            raise RuntimeError("Zero-update preflight mutated source or optimizer state")
        verify_frozen()
        (output / "FINALIZED.json").write_text(json.dumps(dict(
            state="complete", optimizer_updates=0, steps=STEPS, envs=ENVS,
            source_sha256=frozen["source_sha256"], source_completed=frozen["source_completed_updates"],
            model_and_dual_optimizer_hashes_before=before,
            model_and_dual_optimizer_hashes_after=after,
            forbidden_calls=guard.calls, mutation_guard_restored=guard.restored,
            post_entry_control_samples=platform_samples,
            post_entry_mean_abs_vx_error_mps=platform_error_sum / platform_samples,
            post_entry_front_rear_support_fraction=support_samples / platform_samples,
            pit_training_statistics=pit.snapshot(),
            meaning="runtime command and sample verification; no policy improvement claim",
        ), indent=2))

    return preflight


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions must remain enabled")
    frozen = verify_frozen()
    source = Path(frozen["source_path"])
    output = STAGE / "preflight_3000"
    if output.exists():
        raise FileExistsError("Preflight already attempted; preserve its evidence")
    with SHARED_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU is occupied; do not overlap current training/evaluation")
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original = early_restart.configure

        def configure(cfg, profile):
            original(cfg, profile)
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = make_preflight(source, output, frozen)

        early_restart.configure = configure
        sys.argv = [str(STAGE / "code/train.py"), *training_args(source, output)]
        from train import main as train_main

        train_main()
        if not (output / "FINALIZED.json").is_file():
            raise RuntimeError("Training entry returned without zero-update finalization")


if __name__ == "__main__":
    main()
