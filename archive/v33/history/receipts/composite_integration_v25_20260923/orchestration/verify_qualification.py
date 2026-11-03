"""Separate CPU process rehashes the source and verifies Isaac's final receipt."""

from __future__ import annotations

import json
import re
import sys

from contract import (HERE, NUM_ENVS, PIT_ENVS, PREFLIGHT_STEPS, RESTORE_CHECKS,
                      ROUTE_ENVS, SOURCE, SOURCE_COMPLETED, SOURCE_SHA256,
                      STAGE, STATE_KEYS, digest, load_json, verify_frozen, write_new)
from state_guard import state_sha256


OUTPUT = HERE / "qualification_3000"


def verify() -> dict:
    import torch

    frozen = verify_frozen()
    final_path = OUTPUT / "FINALIZED.json"
    final = load_json(final_path)
    run = load_json(OUTPUT / "run.json")
    migration = load_json(OUTPUT / "composite_migration.json")
    before = final.get("state_hashes_before")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("control_steps") != PREFLIGHT_STEPS
            or final.get("num_envs") != NUM_ENVS
            or final.get("source_sha256") != SOURCE_SHA256
            or final.get("source_completed_updates") != SOURCE_COMPLETED
            or final.get("common_step_before") != SOURCE_COMPLETED * 24
            or final.get("common_step_after") != SOURCE_COMPLETED * 24 + PREFLIGHT_STEPS
            or final.get("runner_iteration_after") != SOURCE_COMPLETED
            or final.get("state_hashes_after") != before
            or final.get("mutation_guard_restored") is not True
            or final.get("reset_boundary_guard_restored") is not True
            or final.get("route_resets_checked", 0) <= 0
            or final.get("forbidden_calls") != []
            or final.get("checkpoint_artifacts_written") != []
            or list(OUTPUT.rglob("*.pt"))):
        raise ValueError("Incomplete or state-mutating V25 zero-update qualification")
    if (run.get("resume") != str(SOURCE) or run.get("resume_sha256") != SOURCE_SHA256
            or run.get("starting_completed_updates") != SOURCE_COMPLETED
            or run.get("additional_updates") != 0
            or run.get("target_completed_updates") != SOURCE_COMPLETED
            or run.get("hip_request_profile") != "off"
            or run.get("mixed_stair_profile") != "off"
            or run.get("composite_route_profile") != "composite_flat35_v25"
            or any(run.get("resume_checks", {}).get(key) is not True for key in RESTORE_CHECKS)
            or run.get("resume_checks", {}).get("terrain_restored_exactly") is not False):
        raise ValueError("V25 run receipt does not prove sole-source restore and one-time migration")
    allocation = final.get("allocation", {})
    if (allocation.get("route") != ROUTE_ENVS
            or allocation.get("pit") != PIT_ENVS
            or allocation.get("learner_counts") != {"teacher": 3072, "student": 1024}
            or allocation.get("exact_migrated_assignments") is not True
            or len(allocation.get("selected_from_flat", [])) != ROUTE_ENVS
            or migration.get("selected_from_flat") != allocation["selected_from_flat"]
            or migration.get("selected_count") != ROUTE_ENVS
            or migration.get("preserved_pit_count") != PIT_ENVS
            or run.get("terrain_by_type", {}).get("composite", {}).get("count") != ROUTE_ENVS
            or run.get("terrain_by_type", {}).get("pit", {}).get("count") != PIT_ENVS):
        raise ValueError("Route/pit quota or donor migration evidence differs")
    if (final.get("terrain_vertical_force_samples", 0) <= 0
            or final.get("conservative_wheel_support_samples", 0) <= 0
            or final.get("aligned_spawn_samples", 0) < ROUTE_ENVS):
        raise ValueError("Live contact or aligned-spawn evidence is missing")
    if (not isinstance(before, dict) or set(before) != set(STATE_KEYS)
            or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in before.values())):
        raise ValueError("Missing model/PPO/student optimizer fingerprints")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_hashes = {name: state_sha256(saved[key], torch) for name, key in STATE_KEYS.items()}
    if source_hashes != before or source_hashes != frozen["source_state_sha256"]:
        raise ValueError("Independent source checkpoint state hash differs from Isaac")
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("Reviewed source file bytes changed")
    sys.path.insert(0, str(STAGE / "code"))
    from composite_atlas import OLD_COLUMNS

    old_counts = {name: sum(OLD_COLUMNS[int(column)] == name
                            for column in saved["terrain_types"])
                  for name in set(OLD_COLUMNS)}
    new_counts = allocation["new_counts"]
    if (allocation["old_counts"] != old_counts
            or any(new_counts[name] != count for name, count in old_counts.items()
                   if name != "flat")
            or new_counts["flat"] != old_counts["flat"] - ROUTE_ENVS
            or migration["old_type_counts"] != old_counts
            or migration["new_type_counts"] != new_counts):
        raise ValueError("One of the original nine terrain quotas changed")
    record = dict(state="verified_not_training", source_sha256=SOURCE_SHA256,
                  optimizer_updates=0, control_steps=PREFLIGHT_STEPS,
                  finalized_sha256=digest(final_path), run_sha256=digest(OUTPUT / "run.json"),
                  migration_sha256=digest(OUTPUT / "composite_migration.json"),
                  independent_state_sha256=source_hashes,
                  route_count=ROUTE_ENVS, pit_count=PIT_ENVS,
                  old_nine_counts_preserved_except_flat_donors=True,
                  behavior_eligibility=False, automatic_training=False)
    write_new(HERE / "QUALIFIED_V25.json", record)
    return record


if __name__ == "__main__":
    print(json.dumps(verify()))
