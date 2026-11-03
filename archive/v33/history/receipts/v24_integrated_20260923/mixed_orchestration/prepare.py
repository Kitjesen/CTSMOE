"""CPU-only freeze after an explicit V24 source and behavior decision."""

from __future__ import annotations

import json
from pathlib import Path

from contract import (HERE, HIP, PROFILE, RESTORE, digest, load_json,
                      python_snapshot, source_options, validate_review,
                      verify_hip_complete, write_new)


def prepare() -> dict:
    import torch

    if (HERE / "FROZEN_MIXED.json").exists() or (HERE / "STARTED_MIXED.json").exists():
        raise FileExistsError("Mixed stage is already frozen or attempted")
    hip_frozen, status, proof, _ = verify_hip_complete()
    options = source_options(hip_frozen, status, proof)
    review = load_json(HERE / "SOURCE_REVIEW.json")
    source = validate_review(review, options, digest(HIP / "ASSESSMENT_PROOF.json"),
                             digest(HIP / "BEHAVIOR_COMPARISON.json"))
    path = Path(source["path"])
    if digest(path) != source["sha256"]:
        raise ValueError("Selected behavior-reviewed checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    hip_request = saved.get("hip_request_profile", "off")
    expected_hip = ("inward_unreachable_request_v1"
                    if review["selected_source"] == "v24_hip_candidate" else "off")
    if (saved.get("completed_updates") != source["completed_updates"]
            or saved.get("common_step_counter") != source["completed_updates"] * 24
            or saved.get("mixed_stair_profile", "off") != "off"
            or saved.get("hip_guard_profile") not in (None, "hip_default012_v23")
            or hip_request != expected_hip
            or not all(saved.get(key) for key in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("Source age, hip factor, model or dual optimizers differ")
    if not RESTORE or PROFILE != "mixed_stair_short_pulses_v1":
        raise RuntimeError("Unexpected mixed training contract")
    frozen = dict(
        stage="bounded_mixed_stair_v1", source=source,
        hip_request_profile=hip_request, source_review_sha256=digest(HERE / "SOURCE_REVIEW.json"),
        hip_frozen_sha256=digest(HIP / "FROZEN_V24.json"),
        hip_status_sha256=digest(HIP / "pipeline_status.json"),
        hip_assessment_proof_sha256=digest(HIP / "ASSESSMENT_PROOF.json"),
        hip_behavior_comparison_sha256=digest(HIP / "BEHAVIOR_COMPARISON.json"),
        mixed_code=python_snapshot(HERE), v23_assessment=hip_frozen["v23_assessment"],
        preflight_control_steps=3000, optimizer_update_segments=[16, 484],
        original_remaining_before_mixed=5000, original_remaining_after_mixed=4500,
        automatic_extension=False, automatic_reentry=False,
        real_robot_release=False,
    )
    write_new(HERE / "FROZEN_MIXED.json", frozen)
    return frozen


if __name__ == "__main__":
    print(json.dumps(prepare()["source"]))
