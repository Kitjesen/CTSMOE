"""CPU-only identities and limits for the one-shot V25 qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
ROOT = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922")
SOURCE = ROOT / "stair_coverage_v21/training_33881_to_34823/model_34822.pt"
V23_ASSESSMENT = ROOT / "hip_default_v23_20260923/assessment"
V21_FIXED_PROTOCOL = ROOT / "stair_coverage_v21/evaluation_34823/terrain83/manifest.json"
SOURCE_SHA256 = "521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5"
SOURCE_COMPLETED = 34823
NUM_ENVS = 4096
ROUTE_ENVS = 205
PIT_ENVS = 409
PREFLIGHT_STEPS = 3000
MAX_V25_UPDATES = 500
ORIGINAL_BUDGET = 10000
# 2992 V21 + 102 discarded V22 + 8 lost + 898 V23 = 4000.
# Reserve both separately planned V24 caps until a signed budget ledger can
# establish a smaller actually spent amount; selecting old V21 returns none.
PRIOR_KNOWN_SPENT = 4000
V24_HIP_RESERVATION = 1000
V24_MIXED_RESERVATION = 500
PRIOR_SPENT_UPPER_BOUND = PRIOR_KNOWN_SPENT + V24_HIP_RESERVATION + V24_MIXED_RESERVATION
RESTORE_CHECKS = ("model", "ppo_optimizer", "student_optimizer",
                  "terrain_assignment", "command_terrain_membership",
                  "terrain_allocation", "stair_episode_totals")
STATE_KEYS = {"model": "model_state_dict", "ppo_optimizer": "optimizer_state_dict",
              "student_optimizer": "optimizer_stu_enc_state_dict"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def snapshot(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): digest(path)
            for path in sorted(folder.rglob("*.py")) if "__pycache__" not in path.parts}


def validate_source(saved: dict) -> None:
    allocation = saved.get("terrain_allocation")
    if (saved.get("completed_updates") != SOURCE_COMPLETED
            or saved.get("common_step_counter") != SOURCE_COMPLETED * 24
            or saved.get("mixed_profile") != "mixed9_v1"
            or saved.get("allocation_profile") != "balanced_adaptive_v1"
            or saved.get("stair_command_profile") != "straight_speed_v1"
            or saved.get("wave_reset_profile") != "wave_surface_v1"
            or saved.get("hip_request_profile", "off") != "off"
            or saved.get("mixed_stair_profile", "off") != "off"
            or saved.get("composite_route_profile") is not None
            or not isinstance(allocation, dict)
            or allocation.get("pending")
            or not all(saved.get(key) is not None for key in STATE_KEYS.values())):
        raise ValueError("Only the settled V21 34823 checkpoint with both V24 branches off is eligible")
    levels, columns = saved.get("terrain_levels"), saved.get("terrain_types")
    if (levels is None or columns is None or len(levels) != NUM_ENVS
            or len(columns) != NUM_ENVS or int(columns.min()) < 0
            or int(columns.max()) >= 40):
        raise ValueError("V21 source must carry 4096 original 40-column terrain assignments")


def training_args(source: Path, output: Path, updates: int) -> list[str]:
    if not isinstance(updates, int) or not 0 <= updates <= MAX_V25_UPDATES:
        raise ValueError("V25 entry permits only 0..500 optimizer updates")
    return [
        "--root", str(BASE), "--output", str(output), "--num-envs", str(NUM_ENVS),
        "--updates", str(updates), "--mixed-pit", "--composite-route",
        "--resume", str(source), "--early-restart-profile", "early_phase_v14_no_reward",
        "--startup-repair-profile", "off", "--pit-posture-profile", "off",
        "--surface-posture-profile", "off", "--stance-strength-profile", "off",
        "--allocation-profile", "balanced_adaptive_v1",
        "--pit-training-profile", "strict_v1", "--pit-course-profile", "global_v1",
        "--actuator-profile", "rs0402_transient_v4", "--slope-profile", "slope45_v1",
        "--command-curriculum-start", "intermediate", "--base-height-penalty", "original",
        "--terrain-training-profile", "stairs80_v1",
        "--stair-command-profile", "straight_speed_v1",
        "--wave-reset-profile", "wave_surface_v1",
    ]


def budget(updates: int) -> dict:
    if not isinstance(updates, int) or not 0 <= updates <= MAX_V25_UPDATES:
        raise ValueError("V25 update cap exceeded")
    spent_bound = PRIOR_SPENT_UPPER_BOUND + updates
    if spent_bound > ORIGINAL_BUDGET:
        raise ValueError("Original global update budget exceeded")
    return dict(original_total_updates=ORIGINAL_BUDGET,
                known_spent_before_v25=PRIOR_KNOWN_SPENT,
                v24_hip_reserved=V24_HIP_RESERVATION,
                v24_mixed_reserved=V24_MIXED_RESERVATION,
                prior_spent_upper_bound=PRIOR_SPENT_UPPER_BOUND,
                v25_max_updates=MAX_V25_UPDATES,
                v25_requested_updates=updates,
                remaining_under_conservative_reservation=ORIGINAL_BUDGET - spent_bound,
                selecting_older_source_refunds_updates=False)


def verify_frozen() -> dict:
    frozen = load_json(HERE / "FROZEN_V25.json")
    if (frozen.get("stage") != "composite_route_v25"
            or frozen.get("source") != dict(path=str(SOURCE), sha256=SOURCE_SHA256,
                                                completed_updates=SOURCE_COMPLETED)
            or frozen.get("candidate_code") != snapshot(STAGE / "code")
            or frozen.get("orchestration") != snapshot(HERE)
            or frozen.get("v23_assessment") != snapshot(V23_ASSESSMENT)
            or frozen.get("v21_fixed_protocol_sha256") != digest(V21_FIXED_PROTOCOL)
            or frozen.get("budget") != budget(0)
            or digest(SOURCE) != SOURCE_SHA256):
        raise ValueError("Frozen V25 source, code or budget changed")
    return frozen
