"""Independently verify the zero-update pit-traverse qualification receipt."""

import json
from pathlib import Path

from preflight import ENVS, HERE, STEPS, STAGE, digest, verify_frozen


def verify():
    frozen = verify_frozen()
    output = STAGE / "preflight_3000"
    final = json.loads((output / "FINALIZED.json").read_text())
    run = json.loads((output / "run.json").read_text())
    checks = run["resume_checks"]
    if (final["state"] != "complete" or final["optimizer_updates"] != 0
            or final["steps"] != STEPS or final["envs"] != ENVS
            or final["source_sha256"] != frozen["source_sha256"]
            or final["source_completed"] != frozen["source_completed_updates"]
            or final["model_and_dual_optimizer_hashes_before"]
            != final["model_and_dual_optimizer_hashes_after"]
            or set(final["model_and_dual_optimizer_hashes_after"])
            != {"model", "ppo_optimizer", "student_optimizer"}
            or not final["mutation_guard_restored"] or final["forbidden_calls"]
            or final["post_entry_control_samples"] <= 0
            or run["pit_training_profile"] != "platform_traverse_v1"
            or not checks.get("pit_course_gate_reset")
            or list(output.rglob("*.pt")) or digest(Path(frozen["source_path"]))
            != frozen["source_sha256"]):
        raise ValueError("Pit-traverse zero-update proof is incomplete or inconsistent")
    return dict(state="verified_zero_update", source=frozen["source_path"],
                source_sha256=frozen["source_sha256"],
                steps=STEPS, envs=ENVS,
                post_entry_control_samples=final["post_entry_control_samples"],
                post_entry_mean_abs_vx_error_mps=final["post_entry_mean_abs_vx_error_mps"],
                qualifier="runtime proof, not a learned-policy acceptance test")


if __name__ == "__main__":
    result = verify()
    path = HERE / "VERIFIED.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(path)
