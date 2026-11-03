"""Independent CPU proof for V24 zero-update qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from contract import (
    NUM_ENVS, ORCH, PREFLIGHT_STEPS,
    digest, load_json, verify_frozen, write_new,
)
from qualification import MODE_OUTPUT
from state_guard import state_sha256


def verify(mode: str) -> dict:
    import torch

    frozen = verify_frozen()
    source_completed = frozen["source"]["completed_updates"]
    common_step = frozen["source"]["common_step_counter"]
    output = ORCH / MODE_OUTPUT[mode]
    final = load_json(output / "FINALIZED.json")
    run = load_json(output / "run.json")
    source = Path(frozen["source"]["path"])
    if (final.get("state") != "complete" or final.get("mode") != mode
            or final.get("optimizer_updates") != 0
            or final.get("steps") != PREFLIGHT_STEPS
            or final.get("num_envs") != NUM_ENVS
            or final.get("source_completed") != source_completed
            or final.get("source_sha256") != frozen["source"]["sha256"]
            or final.get("common_step_before") != common_step
            or final.get("common_step_after") != common_step + PREFLIGHT_STEPS
            or final.get("runner_iteration_after") != source_completed
            or final.get("state_hashes_before") != final.get("state_hashes_after")
            or final.get("mutation_guard_restored") is not True
            or final.get("forbidden_calls") != []
            or final.get("checkpoint_artifacts_written") != []
            or list(output.rglob("*.pt"))):
        raise ValueError("Incomplete or mutated 3000-step, zero-update qualification")
    if (run.get("resume") != str(source) or run.get("resume_sha256") != digest(source)
            or run.get("starting_completed_updates") != source_completed
            or run.get("additional_updates") != 0
            or run.get("target_completed_updates") != source_completed
            or run.get("hip_guard_profile") != "hip_default012_v23"
            or run.get("mixed_stair_profile") != (
                "off" if mode == "hip_only" else "mixed_stair_short_pulses_v1")):
        raise ValueError("Preflight used the wrong source, age or training factors")
    if not all(run.get("resume_checks", {}).get(key) is True for key in (
            "model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
            "command_terrain_membership", "terrain_restored_exactly", "terrain_allocation",
            "stair_episode_totals")):
        raise ValueError("Preflight failed exact checkpoint/terrain restore")
    if mode == "hip_only":
        if (run.get("hip_request", {}).get("profile") != "inward_unreachable_request_v1"
                or run["hip_request"]["weight"] != -0.02
                or final.get("hip_request_positive_env_steps", 0) <= 0
                or final.get("mixed_first_reset") is not None):
            raise ValueError("Hip-only reward was missing or mixed with another factor")
    else:
        first = final.get("mixed_first_reset")
        witness = final.get("conservative_support_witness")
        if (run.get("hip_request", {}).get("profile") != "off"
                or not isinstance(first, dict) or first.get("selected_first_reset", 0) <= 0
                or len(first.get("groups_first_reset", [])) != 3
                or min(first["groups_first_reset"]) <= 0
                or sum(first["groups_first_reset"]) != first["selected_first_reset"]
                or not isinstance(witness, dict)
                or witness.get("kind") != "conservative_center_hull"
                or witness.get("inferred_not_measured_contact_xy") is not True
                or witness.get("wheel_count") != 4
                or witness.get("positive_terrain_force_samples", 0) <= 0):
            raise ValueError("Mixed stair first-reset or conservative support evidence missing")
    hashes = final.get("state_hashes_before", {})
    if (set(hashes) != {"model", "ppo_optimizer", "student_optimizer"}
            or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())):
        raise ValueError("Missing independent model/dual-optimizer fingerprints")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    for label, key in (("model", "model_state_dict"),
                       ("ppo_optimizer", "optimizer_state_dict"),
                       ("student_optimizer", "optimizer_stu_enc_state_dict")):
        if state_sha256(saved[key], torch) != hashes[label]:
            raise ValueError(f"Zero-update preflight did not start from V23 {label}")
    receipt = dict(state="verified_not_training", mode=mode,
                   source_completed=source_completed, optimizer_updates=0,
                   source_sha256=digest(source),
                   finalized_sha256=digest(output / "FINALIZED.json"),
                   preflight=final)
    write_new(ORCH / f"{mode}_VERIFIED.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_OUTPUT), default="hip_only")
    args = parser.parse_args()
    print(json.dumps(verify(args.mode)["state"]))
