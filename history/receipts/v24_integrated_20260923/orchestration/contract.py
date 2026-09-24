"""V24 hip-only stage contracts. No module import starts simulation or training."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
ROOT = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922")
V23 = ROOT / "hip_default_v23_20260923"
STAGE = Path(__file__).resolve().parent.parent
ORCH = Path(__file__).resolve().parent
ALLOWED_SOURCE_COMPLETED = (34823, 35721)
V24_UPDATES = 1000
NUM_ENVS = 4096
PREFLIGHT_STEPS = 3000
TOTAL_BUDGET = 10000
GLOBAL_SPENT_BEFORE_V24 = 4000  # Includes V21, abandoned V22 and completed V23 updates.
RESTORE = (
    "model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
    "command_terrain_membership", "terrain_restored_exactly", "terrain_allocation",
    "stair_episode_totals",
)
ALLOWED_ADDITIONS = frozenset({
    "hip_request_excess.py", "mixed_stair_logic.py", "mixed_stair_runtime.py",
    "stair_platform_witness.py", "test_hip_request_excess.py",
    "test_mixed_stair_config.py", "test_mixed_stair_logic.py",
    "test_mixed_stair_resume.py", "test_stair_platform_witness.py",
})


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
        json.dump(value, stream, ensure_ascii=False, indent=2)


def write_status(stage: str, source_completed: int, completed: int, **extra) -> None:
    if source_completed not in ALLOWED_SOURCE_COMPLETED:
        raise ValueError("Unreviewed source age")
    if not source_completed <= completed <= source_completed + V24_UPDATES:
        raise ValueError("V24 hip-only update boundary exceeded")
    spent = GLOBAL_SPENT_BEFORE_V24 + completed - source_completed
    if spent > TOTAL_BUDGET:
        raise ValueError("Original 10000-update budget exceeded")
    status = dict(
        stage=stage, completed_updates=completed,
        source_completed_updates=source_completed,
        target_completed_updates=source_completed + V24_UPDATES,
        v24_optimizer_updates=completed - source_completed,
        total_budget_updates=TOTAL_BUDGET,
        spent_since_31831_including_discarded=spent,
        remaining_budget_updates=TOTAL_BUDGET - spent,
        automatic_mixed_training=False, automatic_extension=False,
        utc=datetime.now(timezone.utc).isoformat(), **extra,
    )
    temporary = ORCH / "pipeline_status.tmp"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(ORCH / "pipeline_status.json")


def command(source: Path, output: Path, updates: int, *, mixed: bool = False) -> list[str]:
    """Exact V23 continuation flags; select one V24 factor at a time."""
    if updates not in (0, 16, 984):
        raise ValueError("Only zero-update, 16-update, or 984-update segments are allowed")
    profile = "off" if mixed else "inward_unreachable_request_v1"
    mixed_profile = "mixed_stair_short_pulses_v1" if mixed else "off"
    if mixed and updates:
        raise ValueError("Mixed stairs have zero-update diagnostics only in this stage")
    return [
        "--root", str(BASE), "--output", str(output), "--num-envs", str(NUM_ENVS),
        "--updates", str(updates), "--mixed-pit", "--resume", str(source),
        "--early-restart-profile", "early_phase_v14_no_reward",
        "--startup-repair-profile", "off", "--pit-posture-profile", "off",
        "--surface-posture-profile", "off", "--stance-strength-profile", "off",
        "--allocation-profile", "balanced_adaptive_v1",
        "--pit-training-profile", "strict_v1", "--pit-course-profile", "global_v1",
        "--actuator-profile", "rs0402_transient_v4", "--slope-profile", "slope45_v1",
        "--command-curriculum-start", "intermediate",
        "--base-height-penalty", "original",
        "--terrain-training-profile", "stairs80_v1",
        "--stair-command-profile", "straight_speed_v1",
        "--wave-reset-profile", "wave_surface_v1",
        "--hip-request-profile", profile, "--mixed-stair-profile", mixed_profile,
    ]


def files_snapshot(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*.py"))}


def verify_snapshot(folder: Path, expected: dict[str, str]) -> None:
    if not expected or files_snapshot(folder) != expected:
        raise ValueError(f"Frozen Python source differs: {folder}")


def verify_v24_scope(candidate_files: dict[str, str]) -> None:
    original = load_json(V23 / "PREPARED.json")["files"]
    if (set(original) - set(candidate_files)
            or set(candidate_files) - set(original) != ALLOWED_ADDITIONS
            or {name for name in original if candidate_files[name] != original[name]} != {"train.py"}):
        raise ValueError("V24 scope exceeds train.py and the reviewed opt-in modules/tests")


def assessment_files() -> dict[str, str]:
    manifest = load_json(V23 / "ORCHESTRATION_V23.json")
    return {name: sha for name, sha in manifest["files"].items()
            if name.startswith("assessment/")}


def verify_v23_assessment() -> None:
    manifest = load_json(V23 / "ORCHESTRATION_V23.json")
    expected = assessment_files()
    if len(expected) != 4:
        raise ValueError("Incomplete reviewed V23 paired-assessment manifest")
    for name, sha in expected.items():
        if digest(V23 / name) != sha:
            raise ValueError(f"V23 assessment changed: {name}")
    for path, sha in manifest["external"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"V23 fixed-case protocol dependency changed: {path}")


def verify_review_evidence(frozen: dict) -> None:
    if digest(ORCH / "SOURCE_REVIEW.json") != frozen["source_review_sha256"]:
        raise ValueError("Frozen V23 source behavior review changed")
    evaluation = V23 / "evaluation_35721/FINALIZED.json"
    recorded = frozen["v23_evaluation"]
    if (recorded["path"] != str(evaluation)
            or digest(evaluation) != recorded["sha256"]):
        raise ValueError("Frozen V23 paired behavior evaluation changed")
    result = load_json(evaluation)
    if (result.get("state") != "complete" or result.get("optimizer_updates") != 0
            or result.get("hashes_unchanged") is not True
            or result.get("smoke_only") is not False
            or result.get("policy_count") != 4):
        raise ValueError("Frozen V23 paired behavior evaluation is incomplete")


def verify_frozen() -> dict:
    frozen = load_json(ORCH / "FROZEN_V24.json")
    verify_snapshot(STAGE / "code", frozen["candidate_code"])
    verify_v24_scope(frozen["candidate_code"])
    verify_snapshot(ORCH, frozen["orchestration"])
    verify_review_evidence(frozen)
    if frozen["v23_assessment"] != assessment_files():
        raise ValueError("V23 assessment manifest changed")
    verify_v23_assessment()
    source = Path(frozen["source"]["path"])
    if digest(source) != frozen["source"]["sha256"]:
        raise ValueError("Frozen V23 source checkpoint changed")
    return frozen
