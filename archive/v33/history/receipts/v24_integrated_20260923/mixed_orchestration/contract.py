"""Fail-closed contract for a separate, finite mixed-stair experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


STAGE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
HIP = STAGE / "orchestration"
ROOT = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922")
BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
V23 = ROOT / "hip_default_v23_20260923"
PROFILE = "mixed_stair_short_pulses_v1"
PREFLIGHT_STEPS = 3000
NUM_ENVS = 4096
FIRST_UPDATES = 16
FINAL_UPDATES = 484
MIXED_UPDATES = FIRST_UPDATES + FINAL_UPDATES
TOTAL_BUDGET = 10000
SPENT_AFTER_HIP = 5000
RESTORE = (
    "model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
    "command_terrain_membership", "terrain_restored_exactly",
    "terrain_allocation", "stair_episode_totals",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def python_snapshot(folder: Path) -> dict[str, str]:
    return {p.name: digest(p) for p in sorted(folder.glob("*.py"))}


def source_options(hip_frozen: dict, hip_status: dict, proof: dict) -> dict[str, dict]:
    original = hip_frozen["source"]
    candidate = {
        "path": hip_status["candidate_checkpoint"],
        "sha256": proof["candidate_sha256"],
        "completed_updates": hip_status["completed_updates"],
    }
    if (proof["source_sha256"] != original["sha256"]
            or candidate["completed_updates"] != original["completed_updates"] + 1000):
        raise ValueError("V24 paired source/candidate identity or optimizer age differs")
    return {"v24_original": original, "v24_hip_candidate": candidate}


def verify_hip_complete() -> tuple[dict, dict, dict, dict]:
    frozen = load_json(HIP / "FROZEN_V24.json")
    status = load_json(HIP / "pipeline_status.json")
    proof = load_json(HIP / "ASSESSMENT_PROOF.json")
    comparison = load_json(HIP / "BEHAVIOR_COMPARISON.json")
    if (status.get("stage") != "complete_pending_behavior_review"
            or status.get("training_running") is not False
            or status.get("v24_optimizer_updates") != 1000
            or status.get("remaining_budget_updates") != 5000
            or status.get("spent_since_31831_including_discarded") != SPENT_AFTER_HIP
            or proof.get("state") != "complete_pending_behavior_review"
            or comparison.get("state") != "fixed_behavior_review_required"
            or proof.get("behavior_comparison_sha256") != digest(HIP / "BEHAVIOR_COMPARISON.json")
            or comparison.get("disposition", "").startswith("RELEASE")):
        raise ValueError("V24 hip training and full paired behavior evidence are not complete")
    evaluation = Path(status["assessment"])
    final = load_json(evaluation / "FINALIZED.json")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("hashes_unchanged") is not True
            or final.get("smoke_only") is not False or final.get("policy_count") != 4
            or set(proof.get("suites", {})) != {"terrain83", "flat25", "stairend", "stairend_15cm"}
            or digest(Path(frozen["source"]["path"])) != proof.get("source_sha256")
            or digest(Path(status["candidate_checkpoint"])) != proof.get("candidate_sha256")):
        raise ValueError("V24 all-mode paired assessment is absent or incomplete")
    expected_counts = {"terrain83": 332, "flat25": 80,
                       "stairend": 96, "stairend_15cm": 96}
    if set(comparison.get("evidence_sha256", {})) != set(expected_counts):
        raise ValueError("V24 behavior review has incomplete trajectory evidence")
    for suite, receipt in proof["suites"].items():
        folder = evaluation / suite
        evidence = comparison["evidence_sha256"][suite]
        if (receipt.get("fixed_cases") != expected_counts[suite]
                or digest(folder / "manifest.json") != receipt["manifest_sha256"]
                or digest(folder / "HIP_REVIEW_v2.json") != receipt["hip_review_sha256"]
                or digest(folder / "manifest.json") != evidence.get("manifest")
                or digest(folder / "traces.npz") != evidence.get("traces")
                or digest(folder / "HIP_REVIEW_v2.json") != evidence.get("per_joint")):
            raise ValueError(f"V24 paired suite changed or is missing: {suite}")
    code = STAGE / "code"
    expected = frozen["candidate_code"]
    actual = {p.relative_to(code).as_posix(): digest(p) for p in code.rglob("*.py")}
    if actual != expected:
        raise ValueError("Frozen V24 training code changed")
    return frozen, status, proof, comparison


def validate_review(review: dict, options: dict[str, dict], proof_sha: str,
                    comparison_sha: str) -> dict:
    selected = options.get(review.get("selected_source"))
    if (review.get("decision") != "eligible_for_bounded_mixed_stair_test"
            or selected is None
            or review.get("source_path") != selected["path"]
            or review.get("source_sha256") != selected["sha256"]
            or review.get("source_completed_updates") != selected["completed_updates"]
            or review.get("v24_assessment_proof_sha256") != proof_sha
            or review.get("v24_behavior_comparison_sha256") != comparison_sha
            or not isinstance(review.get("behavior_findings"), str)
            or len(review["behavior_findings"].strip()) < 80
            or review.get("real_robot_release") is not False):
        raise ValueError("Explicit V24 behavior review and exact selected source are required")
    return selected


def verify_frozen() -> dict:
    frozen = load_json(HERE / "FROZEN_MIXED.json")
    hip_frozen, status, proof, _ = verify_hip_complete()
    options = source_options(hip_frozen, status, proof)
    review = load_json(HERE / "SOURCE_REVIEW.json")
    selected = validate_review(review, options, digest(HIP / "ASSESSMENT_PROOF.json"),
                               digest(HIP / "BEHAVIOR_COMPARISON.json"))
    if (frozen.get("source") != selected
            or frozen.get("source_review_sha256") != digest(HERE / "SOURCE_REVIEW.json")
            or frozen.get("hip_frozen_sha256") != digest(HIP / "FROZEN_V24.json")
            or frozen.get("hip_status_sha256") != digest(HIP / "pipeline_status.json")
            or frozen.get("hip_assessment_proof_sha256") != digest(HIP / "ASSESSMENT_PROOF.json")
            or frozen.get("hip_behavior_comparison_sha256") != digest(HIP / "BEHAVIOR_COMPARISON.json")
            or frozen.get("mixed_code") != python_snapshot(HERE)
            or frozen.get("v23_assessment") != hip_frozen["v23_assessment"]
            or digest(Path(selected["path"])) != selected["sha256"]):
        raise ValueError("Frozen mixed stage, reviewed source or V24 evidence changed")
    for relative, sha in frozen["v23_assessment"].items():
        if digest(V23 / relative) != sha:
            raise ValueError(f"V23 fixed assessment changed: {relative}")
    return frozen


def training_args(source: Path, output: Path, updates: int, hip_request: str) -> list[str]:
    if updates not in (0, FIRST_UPDATES, FINAL_UPDATES):
        raise ValueError("Only 0, 16 and 484 optimizer-update segments are allowed")
    if hip_request not in ("off", "inward_unreachable_request_v1"):
        raise ValueError("Unreviewed hip reward factor")
    return [
        "--root", str(BASE), "--output", str(output), "--num-envs", str(NUM_ENVS),
        "--updates", str(updates), "--mixed-pit", "--resume", str(source),
        "--early-restart-profile", "early_phase_v14_no_reward",
        "--startup-repair-profile", "off", "--pit-posture-profile", "off",
        "--surface-posture-profile", "off", "--stance-strength-profile", "off",
        "--allocation-profile", "balanced_adaptive_v1",
        "--pit-training-profile", "strict_v1", "--pit-course-profile", "global_v1",
        "--actuator-profile", "rs0402_transient_v4", "--slope-profile", "slope45_v1",
        "--command-curriculum-start", "intermediate", "--base-height-penalty", "original",
        "--terrain-training-profile", "stairs80_v1",
        "--stair-command-profile", "straight_speed_v1",
        "--wave-reset-profile", "wave_surface_v1",
        "--hip-request-profile", hip_request, "--mixed-stair-profile", PROFILE,
    ]


def write_status(stage: str, source_age: int, completed: int, **extra) -> None:
    spent = SPENT_AFTER_HIP + completed - source_age
    if not source_age <= completed <= source_age + MIXED_UPDATES or spent > TOTAL_BUDGET:
        raise ValueError("Mixed-stage update boundary or original budget exceeded")
    record = dict(
        stage=stage, source_completed_updates=source_age, completed_updates=completed,
        mixed_optimizer_updates=completed - source_age,
        target_completed_updates=source_age + MIXED_UPDATES,
        spent_since_31831_including_discarded=spent,
        remaining_budget_updates=TOTAL_BUDGET - spent,
        automatic_extension=False, automatic_reentry=False,
        real_robot_release=False, utc=datetime.now(timezone.utc).isoformat(), **extra,
    )
    temporary = HERE / "pipeline_status.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(HERE / "pipeline_status.json")
