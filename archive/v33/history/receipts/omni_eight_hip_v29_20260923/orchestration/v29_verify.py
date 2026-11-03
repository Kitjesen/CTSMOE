"""Independently verify the zero-update V29 preflight before any training."""

import json

from v29_common import ENVS, HERE, STAGE, STEPS, digest, frozen


def verify():
    receipt = frozen()
    path = STAGE / "preflight_3000"
    value = json.loads((path / "FINALIZED.json").read_text())
    run = json.loads((path / "run.json").read_text())
    if (value["state"] != "complete" or value["optimizer_updates"] != 0
            or value["steps"] != STEPS or value["envs"] != ENVS
            or value["source_sha256"] != receipt["source_sha256"]
            or value["model_and_dual_optimizer_before"]
            != value["model_and_dual_optimizer_after"]
            or value["forbidden_calls"] or not value["mutation_guard_restored"]
            or value["pit_active_final"] != 0
            or value["terrain_counts"].get("pit", 0) != 0
            or value["mixed_nonzero_yaw_steps"] <= 0
            or value["mixed_nonzero_side_steps"] <= 0
            or run["resume_sha256"] != receipt["source_sha256"]
            or run["additional_updates"] != 0
            or list(path.rglob("*.pt"))
            or digest(receipt["source_path"]) != receipt["source_sha256"]):
        raise RuntimeError("Independent V29 zero-update verification failed")
    return {"state": "verified_zero_update", "source_sha256": receipt["source_sha256"],
            "terrain_counts": value["terrain_counts"],
            "mixed_nonzero_yaw_steps": value["mixed_nonzero_yaw_steps"],
            "mixed_nonzero_side_steps": value["mixed_nonzero_side_steps"]}


if __name__ == "__main__":
    (HERE / "V29_VERIFIED.json").write_text(json.dumps(verify(), indent=2))
