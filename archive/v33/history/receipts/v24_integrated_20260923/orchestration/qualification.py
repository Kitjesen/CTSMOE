"""3000 control-step, zero-optimizer V24 hip or opt-in mixed-stair preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from contract import (
    BASE, NUM_ENVS, ORCH, PREFLIGHT_STEPS, STAGE,
    command as train_args, digest, verify_frozen,
)
from state_guard import MutationGuard, state_sha256


MODE_OUTPUT = {"hip_only": "preflight_hip_3000", "mixed_diagnostic": "preflight_mixed_3000"}


def validate_mixed_initial(raw, torch) -> dict:
    cmd = raw.command_manager.get_term("base_velocity")
    course = cmd.mixed_stair_course
    selected = cmd.mixed_terminal_mask()
    active = selected & course.active
    if not bool(selected.any()) or not torch.equal(active, selected):
        raise RuntimeError("First-reset V21 selected stair environments have no mixed course")
    counts = torch.bincount(course.group[active], minlength=3).cpu().tolist()
    if len(counts) != 3 or min(counts) <= 0:
        raise RuntimeError(f"Missing first-reset straight/yaw/lateral cohorts: {counts}")
    by_learner = {}
    for label in (0, 1):
        mask = active & (cmd.learner == label)
        if not bool(mask.any()):
            raise RuntimeError(f"No selected stair environment for learner {label}")
        by_learner[str(label)] = torch.bincount(course.group[mask], minlength=3).cpu().tolist()
    return {"selected_first_reset": int(selected.sum()),
            "groups_first_reset": counts, "groups_by_learner_first_reset": by_learner}


def validate_contact_provider(raw, torch) -> dict:
    from stair_platform_witness import conservative_contact_provider

    cmd = raw.command_manager.get_term("base_velocity")
    active = cmd.mixed_terminal_mask() & cmd.mixed_stair_course.active
    ids = (active & (cmd.mixed_stair_course.group != 0)).nonzero().flatten()
    if not len(ids):
        raise RuntimeError("Mixed preflight sampled no nonstraight stair cases")
    witness = conservative_contact_provider(raw, ids)
    count = len(ids)
    expected = {
        "wheel_centers_world_xyz": (count, 4, 3),
        "terrain_forces_n": (count, 4),
        "platform_world_z_m": (count,),
    }
    if (witness.get("kind") != "conservative_center_hull"
            or witness.get("validated_collision_geometry") is not True
            or not witness.get("collision_geometry_id")):
        raise RuntimeError("Mixed stair support lacks a validated geometry witness")
    for key, shape in expected.items():
        value = witness[key]
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid measured force / inferred collider value: {key}")
    hull = witness["body_collision_hull_world_xy"]
    if (hull.ndim != 3 or hull.shape[0] != count or hull.shape[1] < 4 or hull.shape[2] != 2
            or not bool(torch.isfinite(hull).all())):
        raise RuntimeError("Invalid conservative physical body hull")
    if (witness["terrain_forces_n"] < 0).any():
        raise RuntimeError("Terrain-filtered force cannot be negative")
    return dict(kind=witness["kind"], inferred_not_measured_contact_xy=True,
                collision_geometry_id=witness["collision_geometry_id"],
                wheel_count=4, environment_count=count, hull_points_per_environment=hull.shape[1],
                positive_terrain_force_samples=int((witness["terrain_forces_n"] > 0).sum()))


def make_preflight(source: Path, output: Path, mode: str):
    def preflight(self, updates, init_at_random_ep_len=False):
        import torch

        frozen = verify_frozen()
        source_completed = frozen["source"]["completed_updates"]
        common_step = frozen["source"]["common_step_counter"]
        if (updates != 0 or init_at_random_ep_len or self.env.num_envs != NUM_ENVS
                or self.current_learning_iteration != source_completed
                or self.cfg["num_steps_per_env"] != 24 or self.alg.storage.step != 0):
            raise RuntimeError("Preflight requires reviewed source age, 4096 envs, empty storage, zero updates")
        raw = self.env.unwrapped
        if raw.common_step_counter != common_step:
            raise RuntimeError("Resumed common-step clock differs from source checkpoint")
        receipt = json.loads((output / "run.json").read_text())
        if (receipt.get("resume_sha256") != frozen["source"]["sha256"]
                or receipt.get("starting_completed_updates") != source_completed
                or receipt.get("target_completed_updates") != source_completed
                or receipt.get("additional_updates") != 0
                or not all(receipt.get("resume_checks", {}).get(key) is True
                           for key in ("model", "ppo_optimizer", "student_optimizer",
                                       "terrain_assignment", "command_terrain_membership",
                                       "terrain_restored_exactly", "terrain_allocation",
                                       "stair_episode_totals"))):
            raise RuntimeError("Source model, optimizer or terrain state was not restored exactly")
        from hip_request_excess import PROFILE as HIP_PROFILE, WEIGHT, reward as hip_excess
        hip_term = "hip_inward_unreachable_request" in raw.reward_manager.active_terms
        if mode == "hip_only":
            if (receipt.get("hip_request", {}).get("profile") != HIP_PROFILE or not hip_term
                    or raw.reward_manager.get_term_cfg("hip_inward_unreachable_request").weight != WEIGHT
                    or receipt.get("mixed_stair_profile") != "off"):
                raise RuntimeError("The intended hip-only reward factor is not active")
            mixed_initial = None
        else:
            if (receipt.get("hip_request", {}).get("profile") != "off" or hip_term
                    or receipt.get("mixed_stair_profile") != "mixed_stair_short_pulses_v1"):
                raise RuntimeError("The diagnostic changed the hip-only reward factor")
            mixed_initial = validate_mixed_initial(raw, torch)
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        guard = MutationGuard(self)
        excess_sum, excess_positive = 0.0, 0
        contact = None
        with guard, torch.inference_mode():
            for step in range(PREFLIGHT_STEPS):
                actions = self.alg.act(obs)
                obs, rewards, done, extras = self.env.step(actions.to(raw.device))
                obs, rewards, done = obs.to(self.device), rewards.to(self.device), done.to(self.device)
                self.alg.process_env_step(obs, rewards, done, extras)
                if not all(bool(torch.isfinite(item).all()) for item in (
                        actions, rewards, raw.scene["robot"].data.joint_pos)):
                    raise FloatingPointError("Nonfinite state in zero-update rollout")
                if mode == "hip_only":
                    cost = hip_excess(raw)
                    excess_sum += float(cost.sum())
                    excess_positive += int((cost > 0).sum())
                elif step >= 1 and step % 100 == 0:
                    sample = validate_contact_provider(raw, torch)
                    if contact is None:
                        contact = sample
                    else:
                        if sample["collision_geometry_id"] != contact["collision_geometry_id"]:
                            raise RuntimeError("Support collider contract changed during rollout")
                        contact["positive_terrain_force_samples"] += sample["positive_terrain_force_samples"]
                if (step + 1) % 24 == 0:
                    self.alg.storage.clear()
                if step % 300 == 0:
                    print("PREFLIGHT", mode, step, flush=True)
            if self.alg.storage.step != 0:
                raise RuntimeError("3000 control steps must end at an empty rollout boundary")
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if before != after or guard.calls or not guard.restored:
            raise RuntimeError("Policy or either optimizer mutated in preflight")
        if (raw.common_step_counter != common_step + PREFLIGHT_STEPS
                or self.current_learning_iteration != source_completed
                or list(output.rglob("*.pt")) or digest(source) != frozen["source"]["sha256"]):
            raise RuntimeError("Preflight changed checkpoint age or wrote an optimizer artifact")
        if mode == "hip_only" and excess_positive <= 0:
            raise RuntimeError("Hip request factor received no nonzero exposure")
        if mode == "mixed_diagnostic" and (contact is None or
                contact["positive_terrain_force_samples"] <= 0):
            raise RuntimeError("Mixed diagnostic lacks verified terrain-force exposure")
        verify_frozen()
        (output / "FINALIZED.json").write_text(json.dumps(dict(
            state="complete", mode=mode, source_completed=source_completed,
            optimizer_updates=0, steps=PREFLIGHT_STEPS, num_envs=NUM_ENVS,
            source_sha256=frozen["source"]["sha256"],
            common_step_before=common_step, common_step_after=raw.common_step_counter,
            runner_iteration_after=self.current_learning_iteration,
            state_hashes_before=before, state_hashes_after=after,
            mutation_guard_restored=guard.restored, forbidden_calls=guard.calls,
            checkpoint_artifacts_written=[],
            hip_request_excess_sum=excess_sum, hip_request_positive_env_steps=excess_positive,
            mixed_first_reset=mixed_initial, conservative_support_witness=contact,
            qualification="runtime and sampling only, not learned behavior improvement",
        ), indent=2))

    return preflight


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_OUTPUT), default="hip_only")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Preflight assertions must remain enabled")
    frozen = verify_frozen()
    source = Path(frozen["source"]["path"])
    output = ORCH / MODE_OUTPUT[args.mode]
    if output.exists() or (ORCH / f"{args.mode}_VERIFIED.json").exists():
        raise FileExistsError("Preflight already attempted; do not overwrite or silently retry")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU already occupied")
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original_configure = early_restart.configure

        def configure(cfg, profile):
            original_configure(cfg, profile)
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = make_preflight(source, output, args.mode)

        early_restart.configure = configure
        saved_argv = sys.argv[:]
        sys.argv = [str(STAGE / "code/train.py"), *train_args(
            source, output, 0, mixed=args.mode == "mixed_diagnostic")]
        try:
            from train import main as train_main

            train_main()
        finally:
            sys.argv = saved_argv
            early_restart.configure = original_configure
        if not (output / "FINALIZED.json").is_file():
            raise RuntimeError("Training entry returned without zero-update finalization")
        verify_frozen()


if __name__ == "__main__":
    main()
