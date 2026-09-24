"""One-shot live Isaac qualification: 4096 envs, 3000 steps, zero updates."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from contract import (BASE, HERE, NUM_ENVS, PIT_ENVS, PREFLIGHT_STEPS,
                      RESTORE_CHECKS, ROUTE_ENVS, SOURCE, SOURCE_COMPLETED,
                      SOURCE_SHA256, STAGE, digest, load_json, training_args,
                      verify_frozen, write_new)
from state_guard import MutationGuard, state_sha256


OUTPUT = HERE / "qualification_3000"


def validate_initial(raw, runner, source_saved, torch) -> dict:
    """Reproduce the 205-donor migration and preserve every old assignment."""
    from composite_atlas import migrate_assignments, NEW_COLUMNS, OLD_COLUMNS

    command = raw.command_manager.get_term("base_velocity")
    terrain = raw.scene.terrain
    groups = {"teacher": runner.alg.teacher_env_idxs.cpu().tolist(),
              "student": runner.alg.student_env_idxs.cpu().tolist()}
    expected_levels, expected_columns, migration = migrate_assignments(
        source_saved["terrain_levels"].tolist(),
        source_saved["terrain_types"].tolist(), groups)
    if (len(expected_levels) != NUM_ENVS or migration["selected_count"] != ROUTE_ENVS
            or migration["preserved_pit_count"] != PIT_ENVS
            or not torch.equal(terrain.terrain_levels.cpu(), torch.tensor(expected_levels))
            or not torch.equal(terrain.terrain_types.cpu(), torch.tensor(expected_columns))
            or int(command.composite.active.sum()) != ROUTE_ENVS
            or int(command.pit.active.sum()) != PIT_ENVS
            or bool((command.composite.active & command.pit.active).any())):
        raise RuntimeError("V25 first-reset route/pit quota or inherited terrain state differs")
    for learner, ids in groups.items():
        if len(ids) != (3072 if learner == "teacher" else 1024):
            raise RuntimeError("Teacher/student split changed")
    old_counts = {name: sum(OLD_COLUMNS[int(x)] == name for x in source_saved["terrain_types"])
                  for name in set(OLD_COLUMNS)}
    new_counts = {name: sum(NEW_COLUMNS[int(x)] == name for x in expected_columns)
                  for name in set(NEW_COLUMNS)}
    if any(new_counts[name] != count for name, count in old_counts.items() if name != "flat"):
        raise RuntimeError("An original non-flat terrain quota changed")
    if new_counts["flat"] != old_counts["flat"] - ROUTE_ENVS:
        raise RuntimeError("Composite donors were not only original flat environments")
    return dict(route=ROUTE_ENVS, pit=PIT_ENVS, learner_counts={k: len(v) for k, v in groups.items()},
                old_counts=old_counts, new_counts=new_counts,
                selected_from_flat=migration["selected_from_flat"],
                exact_migrated_assignments=True)


def aligned_resets(raw, ids, torch) -> bool:
    if not len(ids):
        return True
    data = raw.scene["robot"].data
    origin = raw.scene.env_origins[ids]
    relative = data.root_pos_w[ids, :2] - origin[:, :2]
    w, x, y, z = data.root_quat_w[ids].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return bool((relative.abs() <= .11).all() and (yaw.abs() <= .01).all())


@contextmanager
def reject_outside_before_reset(raw, route):
    """Catch a route exit before Isaac clears episode flags during reset."""
    original_reset = raw._reset_idx
    audit = {"route_resets_checked": 0, "restored": False}

    def checked_reset(ids):
        route.refresh()
        active = route.active if ids is None else route.active[ids]
        outside = route.outside if ids is None else route.outside[ids]
        audit["route_resets_checked"] += int(active.sum())
        if bool((active & outside).any()):
            raise RuntimeError("A route collision surface crossed its safe boundary before reset")
        return original_reset(ids)

    raw._reset_idx = checked_reset
    try:
        yield audit
    finally:
        raw._reset_idx = original_reset
        audit["restored"] = raw._reset_idx is original_reset


def make_qualification(output: Path):
    def qualify(self, updates, init_at_random_ep_len=False):
        import torch

        frozen = verify_frozen()
        if (updates != 0 or init_at_random_ep_len or self.env.num_envs != NUM_ENVS
                or self.current_learning_iteration != SOURCE_COMPLETED
                or self.cfg["num_steps_per_env"] != 24 or self.alg.storage.step != 0):
            raise RuntimeError("Only reviewed V21, 4096 environments and zero updates are allowed")
        raw = self.env.unwrapped
        if raw.common_step_counter != SOURCE_COMPLETED * 24:
            raise RuntimeError("V21 control-step clock was not restored")
        receipt = load_json(output / "run.json")
        if (receipt.get("resume_sha256") != SOURCE_SHA256
                or receipt.get("starting_completed_updates") != SOURCE_COMPLETED
                or receipt.get("additional_updates") != 0
                or receipt.get("target_completed_updates") != SOURCE_COMPLETED
                or receipt.get("hip_request_profile") != "off"
                or receipt.get("mixed_stair_profile") != "off"
                or receipt.get("composite_route_profile") != "composite_flat35_v25"
                or any(receipt.get("resume_checks", {}).get(key) is not True
                       for key in RESTORE_CHECKS)
                or receipt.get("resume_checks", {}).get("terrain_restored_exactly") is not False):
            raise RuntimeError("V25 run receipt did not prove exact policy/optimizer and migrated terrain restore")
        saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
        allocation = validate_initial(raw, self, saved, torch)
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if before != frozen["source_state_sha256"]:
            raise RuntimeError("V25 did not restore the three V21 state trees exactly")
        del saved
        command = raw.command_manager.get_term("base_velocity")
        route = command.composite
        route_ids = route.active.nonzero().flatten()
        if not aligned_resets(raw, route_ids, torch):
            raise RuntimeError("Route did not spawn aligned to atlas +X")
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        guard = MutationGuard(self)
        vertical_samples = witnessed_samples = stop_samples = 0
        resets_checked = len(route_ids)
        from composite_runtime import terrain_contact_forces, terrain_wheel_support

        with guard, reject_outside_before_reset(raw, route) as reset_audit, torch.inference_mode():
            for step in range(PREFLIGHT_STEPS):
                route_ids = route.active.nonzero().flatten()
                if len(route_ids) != ROUTE_ENVS or int(command.pit.active.sum()) != PIT_ENVS:
                    raise RuntimeError("Composite/pit quota drifted during zero-update rollout")
                issued = command.commands[route_ids]
                stopped = route.stop_on_platform[route_ids]
                if (not bool((issued[:, 1:] == 0).all())
                        or not bool((issued[~stopped, 0] > 0).all())
                        or not bool((issued[stopped, 0] == 0).all())):
                    raise RuntimeError("Route forward/stop command transition is invalid")
                stop_samples += int(stopped.sum())
                actions = self.alg.act(obs)
                obs, rewards, done, extras = self.env.step(actions.to(raw.device))
                obs, rewards, done = obs.to(self.device), rewards.to(self.device), done.to(self.device)
                self.alg.process_env_step(obs, rewards, done, extras)
                if not all(bool(torch.isfinite(value).all()) for value in (
                        actions, rewards, raw.scene["robot"].data.joint_pos)):
                    raise FloatingPointError("Nonfinite simulation state in zero-update rollout")
                route_ids = route.active.nonzero().flatten()
                if len(route_ids) != ROUTE_ENVS or int(command.pit.active.sum()) != PIT_ENVS:
                    raise RuntimeError("Composite/pit quota drifted after a reset")
                if bool(route.outside[route_ids].any()):
                    raise RuntimeError("A route collision surface crossed its tile's safe boundary")
                reset_ids = route_ids[raw.episode_length_buf[route_ids] == 0]
                if not aligned_resets(raw, reset_ids, torch):
                    raise RuntimeError("A later route episode lost +X spawn alignment")
                resets_checked += len(reset_ids)
                if step % 20 == 0:
                    wheel_world = command.robot.data.body_pos_w[route_ids][:, command.pit.wheel_ids]
                    origins = raw.scene.env_origins[route_ids]
                    wheel_tile = wheel_world - origins[:, None]
                    wheel_tile[..., :2] += torch.tensor((1.1, 4.5), device=raw.device)
                    force = terrain_contact_forces(command.pit.ground_contacts, route_ids, NUM_ENVS)
                    support = terrain_wheel_support(
                        wheel_tile, force, raw.scene.terrain.terrain_levels[route_ids])
                    vertical_samples += int((force[..., 2] > 5).sum())
                    witnessed_samples += int(support.sum())
                if (step + 1) % 24 == 0:
                    self.alg.storage.clear()
                if step % 300 == 0:
                    print("V25_ZERO_UPDATE", step, flush=True)
            if self.alg.storage.step != 0:
                raise RuntimeError("3000 steps must end on an empty rollout boundary")
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if (before != after or guard.calls or not guard.restored
                or not reset_audit["restored"] or reset_audit["route_resets_checked"] <= 0
                or self.current_learning_iteration != SOURCE_COMPLETED
                or raw.common_step_counter != SOURCE_COMPLETED * 24 + PREFLIGHT_STEPS
                or list(output.rglob("*.pt")) or digest(SOURCE) != SOURCE_SHA256
                or vertical_samples <= 0 or witnessed_samples <= 0):
            raise RuntimeError("V25 zero-update identity/contact qualification failed")
        verify_frozen()
        write_new(output / "FINALIZED.json", dict(
            state="complete", optimizer_updates=0, control_steps=PREFLIGHT_STEPS,
            num_envs=NUM_ENVS, source_sha256=SOURCE_SHA256,
            source_completed_updates=SOURCE_COMPLETED,
            common_step_before=SOURCE_COMPLETED * 24,
            common_step_after=raw.common_step_counter,
            runner_iteration_after=self.current_learning_iteration,
            state_hashes_before=before, state_hashes_after=after,
            mutation_guard_restored=guard.restored, forbidden_calls=guard.calls,
            reset_boundary_guard_restored=reset_audit["restored"],
            route_resets_checked=reset_audit["route_resets_checked"],
            checkpoint_artifacts_written=[], allocation=allocation,
            aligned_spawn_samples=resets_checked,
            terrain_vertical_force_samples=vertical_samples,
            conservative_wheel_support_samples=witnessed_samples,
            stop_command_samples=stop_samples,
            qualification="runtime and contact only; source route behavior is assessed separately"))

    return qualify


def main():
    import fcntl

    if sys.flags.optimize:
        raise RuntimeError("Qualification assertions must remain enabled")
    verify_frozen()
    if OUTPUT.exists() or (HERE / "STARTED_V25.json").exists():
        raise FileExistsError("V25 qualification already attempted; no automatic retry")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU already occupied")
        write_new(HERE / "STARTED_V25.json", dict(source=str(SOURCE),
                  source_sha256=SOURCE_SHA256, control_steps=PREFLIGHT_STEPS,
                  optimizer_updates=0, automatic_training=False))
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original_configure = early_restart.configure

        def configure(cfg, profile):
            original_configure(cfg, profile)
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = make_qualification(OUTPUT)

        early_restart.configure = configure
        previous_argv = sys.argv[:]
        sys.argv = [str(STAGE / "code/train.py"), *training_args(SOURCE, OUTPUT, 0)]
        try:
            from train import main as train_main

            train_main()
        finally:
            early_restart.configure = original_configure
            sys.argv = previous_argv
        if not (OUTPUT / "FINALIZED.json").is_file():
            raise RuntimeError("Isaac returned without a zero-update qualification receipt")


if __name__ == "__main__":
    main()
