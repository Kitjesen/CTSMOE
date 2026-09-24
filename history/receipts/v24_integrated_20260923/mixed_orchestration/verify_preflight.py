"""Independent CPU check of the zero-update mixed-stair preflight receipt."""

from __future__ import annotations

import json
from pathlib import Path
import re

from contract import (HERE, NUM_ENVS, PREFLIGHT_STEPS, PROFILE, RESTORE,
                      digest, load_json, verify_frozen, write_new)
from state_guard import state_sha256


def verify() -> dict:
    import torch

    frozen = verify_frozen()
    source = Path(frozen["source"]["path"])
    source_age = frozen["source"]["completed_updates"]
    output = HERE / "preflight_mixed_3000"
    final = load_json(output / "FINALIZED.json")
    run = load_json(output / "run.json")
    hashes = final.get("state_hashes_before", {})
    witness = final.get("conservative_support_witness")
    first = final.get("mixed_first_reset")
    if (final.get("state") != "complete"
            or final.get("mode") != "mixed_training_preflight"
            or final.get("optimizer_updates") != 0
            or final.get("steps") != PREFLIGHT_STEPS
            or final.get("num_envs") != NUM_ENVS
            or final.get("source_completed") != source_age
            or final.get("source_sha256") != frozen["source"]["sha256"]
            or final.get("common_step_before") != source_age * 24
            or final.get("common_step_after") != source_age * 24 + PREFLIGHT_STEPS
            or final.get("runner_iteration_after") != source_age
            or final.get("state_hashes_after") != hashes
            or final.get("mutation_guard_restored") is not True
            or final.get("forbidden_calls") != []
            or final.get("checkpoint_artifacts_written") != []
            or list(output.rglob("*.pt"))):
        raise ValueError("3000-step mixed preflight is incomplete or mutated weights")
    if (run.get("resume") != str(source)
            or run.get("resume_sha256") != digest(source)
            or run.get("starting_completed_updates") != source_age
            or run.get("target_completed_updates") != source_age
            or run.get("additional_updates") != 0
            or run.get("mixed_stair_profile") != PROFILE
            or run.get("hip_request", {}).get("profile") != frozen["hip_request_profile"]
            or not all(run.get("resume_checks", {}).get(key) is True for key in RESTORE)):
        raise ValueError("Mixed preflight used a different source, factor or restored state")
    if (not isinstance(first, dict) or first.get("selected_first_reset", 0) <= 0
            or len(first.get("groups_first_reset", [])) != 3
            or min(first["groups_first_reset"]) <= 0
            or sum(first["groups_first_reset"]) != first["selected_first_reset"]
            or not isinstance(witness, dict)
            or witness.get("kind") != "conservative_center_hull"
            or witness.get("inferred_not_measured_contact_xy") is not True
            or not witness.get("collision_geometry_id")
            or witness.get("wheel_count") != 4
            or witness.get("positive_terrain_force_samples", 0) <= 0):
        raise ValueError("Mixed groups or actual terrain-force/geometry witness missing")
    if (set(hashes) != {"model", "ppo_optimizer", "student_optimizer"}
            or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())):
        raise ValueError("Model and dual optimizer fingerprints missing")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    for label, key in (("model", "model_state_dict"),
                       ("ppo_optimizer", "optimizer_state_dict"),
                       ("student_optimizer", "optimizer_stu_enc_state_dict")):
        if state_sha256(saved[key], torch) != hashes[label]:
            raise ValueError(f"Preflight did not start from the selected {label}")
    receipt = dict(state="verified_not_training", optimizer_updates=0,
                   source_sha256=digest(source), source_completed=source_age,
                   finalized_sha256=digest(output / "FINALIZED.json"),
                   collision_geometry_id=witness["collision_geometry_id"],
                   witness_kind=witness["kind"])
    write_new(HERE / "PREFLIGHT_VERIFIED.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(verify()))
