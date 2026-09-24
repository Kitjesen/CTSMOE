"""Freeze a reviewed V23 source and the V24 hip-only code before GPU use.

This script is CPU-only. It intentionally requires a separate, evidence-linked
human behavior review instead of treating 35721 completed updates as approval.
"""

from __future__ import annotations

from pathlib import Path
import re

from contract import (
    ALLOWED_SOURCE_COMPLETED, ORCH, ROOT, STAGE, V23, assessment_files,
    digest, files_snapshot, load_json, verify_v23_assessment, verify_v24_scope, write_new,
)


def validate_review(review: dict, source: Path, sha: str, completed: int, evaluation: Path) -> None:
    if (review.get("decision") != "eligible_for_isolated_hip_request_test"
            or review.get("source_path") != str(source)
            or review.get("source_sha256") != sha
            or review.get("source_completed_updates") != completed
            or review.get("v23_evaluation_path") != str(evaluation)
            or review.get("v23_evaluation_sha256") != digest(evaluation)
            or not isinstance(review.get("behavior_findings"), str)
            or len(review["behavior_findings"].strip()) < 30
            or review.get("real_robot_release") is not False):
        raise ValueError("V23 behavior review is missing, stale or not eligible")


def prepare() -> dict:
    import torch

    frozen_path = ORCH / "FROZEN_V24.json"
    if frozen_path.exists() or (ORCH / "STARTED_V24.json").exists():
        raise FileExistsError("A prepared/started stage already exists; no overwrite")
    status = load_json(V23 / "pipeline_status.json")
    if (status.get("stage") != "complete_pending_behavior_review"
            or status.get("completed_updates") != 35721
            or status.get("training_running") is not False):
        raise ValueError("V23 35721 training/assessment is not complete")
    final_v23 = Path(status["checkpoint"]).resolve()
    if final_v23.name != "model_35720.pt" or not final_v23.is_relative_to(V23.resolve()):
        raise ValueError("V23 final status points at an unexpected checkpoint")
    evaluation = V23 / "evaluation_35721/FINALIZED.json"
    evidence = load_json(evaluation)
    if (evidence.get("state") != "complete" or evidence.get("optimizer_updates") != 0
            or evidence.get("hashes_unchanged") is not True
            or evidence.get("smoke_only") is not False
            or evidence.get("policy_count") != 4):
        raise ValueError("V23 paired behavior evaluation is missing or only smoke")
    verify_v23_assessment()
    review_path = ORCH / "SOURCE_REVIEW.json"
    review = load_json(review_path)
    completed = review.get("source_completed_updates")
    if completed not in ALLOWED_SOURCE_COMPLETED:
        raise ValueError("Select one of the two assessed source checkpoint ages")
    source = Path(review["source_path"]).resolve()
    options = {
        34823: ROOT / "stair_coverage_v21/training_33881_to_34823/model_34822.pt",
        35721: final_v23,
    }
    if source != options[completed].resolve():
        raise ValueError("Selected checkpoint does not match its reviewed source age")
    sha = digest(source)
    original_hashes = load_json(V23 / "evaluation_35721/HASHES_BEFORE.json")
    if original_hashes.get(str(source)) != sha:
        raise ValueError("Selected checkpoint was not included in the V23 paired evaluation")
    validate_review(review, source, sha, completed, evaluation)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if (saved.get("completed_updates") != completed
            or saved.get("common_step_counter") != completed * 24
            or saved.get("hip_guard_profile") not in (None, "hip_default012_v23")
            or saved.get("hip_request_profile", "off") != "off"
            or saved.get("mixed_stair_profile", "off") != "off"
            or not all(saved.get(key) for key in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("V23 source model, dual optimizers or hip profile mismatch")
    code = STAGE / "code"
    if not all((code / name).is_file() for name in (
            "train.py", "hip_request_excess.py", "mixed_stair_runtime.py",
            "stair_platform_witness.py")):
        raise ValueError("Incomplete V24 candidate code")
    if not re.search(r"--hip-request-profile", (code / "train.py").read_text()):
        raise ValueError("Candidate training entry omits the explicit hip factor")
    candidate_files = files_snapshot(code)
    verify_v24_scope(candidate_files)
    frozen = dict(
        stage="hip_request_only_v24", optimizer_update_cap=1000,
        original_remaining_before_v24=6000,
        automatic_mixed_training=False, real_robot_release=False,
        source=dict(path=str(source), sha256=sha, completed_updates=completed,
                    common_step_counter=completed * 24),
        source_review_sha256=digest(review_path),
        v23_evaluation=dict(path=str(evaluation), sha256=digest(evaluation)),
        candidate_code=candidate_files, orchestration=files_snapshot(ORCH),
        v23_assessment=assessment_files(),
    )
    write_new(frozen_path, frozen)
    return frozen


if __name__ == "__main__":
    print(prepare()["source"])
