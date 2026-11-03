"""Prove that the 300-update run completed despite its final status writer error."""

import fcntl
import json
from pathlib import Path
import sys


STAGE = Path(__file__).resolve().parent.parent
ORCH = STAGE / "orchestration"
sys.path.insert(0, str(ORCH))

from driver import gpu_available, source_checkpoint  # noqa: E402
from preflight import SHARED_LOCK, digest, verify_frozen  # noqa: E402
from verify_preflight import verify as verify_preflight  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def segment(folder, source, old, new):
    run = read(folder / "run.json")
    success = read(folder / "SUCCESS.json")
    metrics = json.loads((folder / "metrics.jsonl").read_text().splitlines()[-1])
    required = (
        "model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
        "command_terrain_membership", "terrain_restored_exactly",
        "terrain_allocation", "stair_episode_totals",
    )
    if old == 34823:
        required += ("pit_course_gate_reset",)
    if (run["resume_sha256"] != digest(source)
            or run["additional_updates"] != new - old
            or run["target_completed_updates"] != new
            or run["pit_training_profile"] != "platform_traverse_v1"
            or not all(run["resume_checks"].get(item) is True for item in required)
            or run["mixed_checks"]["pit_rewards_zero_on_old"] is not True
            or run["pit"]["stop_protocol"]
            != "no platform stop; selected forward command persists until success/reset"
            or success["completed_updates"] != new
            or metrics["completed_updates"] != new):
        raise ValueError(f"Training segment {old} to {new} lacks final evidence")
    checkpoint = folder / f"model_{new - 1}.pt"
    source_checkpoint(checkpoint, new, "platform_traverse_v1")
    return checkpoint


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions must remain enabled")
    frozen = verify_frozen()
    verified = verify_preflight()
    if (frozen["source_completed_updates"] != 34823
            or verified["source_sha256"] != frozen["source_sha256"]):
        raise ValueError("Reviewed source or preflight identity changed")
    status = read(ORCH / "pipeline_status.json")
    if (status["stage"] != "technical_failure"
            or status["completed_updates"] != 35123
            or status["error"] != "dict() got multiple values for keyword argument 'automatic_extension'"):
        raise ValueError("Expected final-only status writer failure is absent")
    with SHARED_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        source = Path(frozen["source_path"])
        first = segment(ORCH / "training_34823_to_34839", source, 34823, 34839)
        final = segment(ORCH / "training_34839_to_35123", first, 34839, 35123)
        verify_frozen()
        result = dict(
            state="complete_pending_behavior_review",
            recovered_from="final status serialization error only",
            original_pipeline_status="technical_failure_preserved",
            source_completed_updates=34823,
            candidate_completed_updates=35123,
            optimizer_updates=300,
            automatic_extension=False,
            candidate_checkpoint=str(final),
            candidate_sha256=digest(final),
            model_and_dual_optimizers_saved=True,
            segment_success_receipts_both_verified=True,
            original_training_not_restarted=True,
            behavior_accepted=False,
        )
        destination = ORCH / "RECOVERED_COMPLETION.json"
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(destination)


if __name__ == "__main__":
    main()
