"""Finalize V14 early-restart preflight only after every invariant passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

EXPECTED_STEPS = 3000
PROFILE = "early_phase_v14"
RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"
ALLOWED_COMPLETED_UPDATES = {30000, 31831}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def live_audit_processes(output: Path) -> list[dict[str, Any]]:
    found, needle = [], str(output.resolve())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = [
                x.decode(errors="replace")
                for x in (entry / "cmdline").read_bytes().split(b"\0")
                if x
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in " ".join(command) and any(
            Path(x).name in {"launcher.py", "train.py"} for x in command
        ):
            found.append({"pid": int(entry.name), "command": command})
    return found


def validate_functional_evidence(stats: dict[str, Any]) -> None:
    expected_cfg = {
        "episode_seconds": 25.0,
        "native_resampling_seconds": 5.0,
        "early_restart_profile": PROFILE,
    }
    if stats.get("configuration") != expected_cfg:
        raise RuntimeError("25s/5s/profile configuration mismatch")
    if (
        stats.get("reward_nonfinite_count") != 0
        or stats.get("action_nonfinite_count") != 0
    ):
        raise RuntimeError("Non-finite reward or action observed")
    if stats.get("reset_target_receipt") != {
        "version": RESET_TARGETS_VERSION,
        "robot": "robot",
    }:
        raise RuntimeError("Reset target receipt mismatch")
    if float(stats.get("first_target_actual_max_abs_error", 1.0)) > 1e-6:
        raise RuntimeError("Reset target was not synchronized")
    counts = (stats.get("startup_reset_counts") or {}).get("counts")
    if not counts or any(
        int(v) <= 0 for learner in counts for kind in learner for v in kind
    ):
        raise RuntimeError("Command cohort reset coverage is incomplete")
    observed = stats.get("command_cohort_observations") or {}
    expected = {str((learner, kind)) for learner in range(2) for kind in range(10)}
    if not expected.issubset(observed):
        raise RuntimeError("Command learner/kind observations are incomplete")
    for learner in ("teacher", "student"):
        if float(stats["maximum_episode_age_seconds"][learner]) < 24.9:
            raise RuntimeError(f"{learner} did not reach a 25-second episode")
        interval = stats["native_resample_intervals_seconds"][learner]
        if (
            interval["count"] <= 0
            or interval["minimum"] < 4.9
            or interval["maximum"] > 5.1
        ):
            raise RuntimeError(
                f"{learner} command resampling did not remain at 5 seconds"
            )


def finalize(
    output: Path,
    checkpoint: Path,
    source_completed: int,
    process_check: Callable[[Path], list[dict[str, Any]]] = live_audit_processes,
) -> dict[str, Any]:
    output, checkpoint = output.resolve(), checkpoint.resolve()
    if source_completed not in ALLOWED_COMPLETED_UPDATES:
        raise ValueError("Unsupported source completed_updates")
    if process_check(output):
        raise RuntimeError("Audit process is still running")
    audit_path, stats_path = (
        output / "audit_status.json",
        output / "preflight_stats.json",
    )
    for path in (
        audit_path,
        stats_path,
        output / "run.json",
        output / "SUCCESS.json",
        output / "status.json",
        checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "state": "AUDIT_COMPLETE",
        "written_before_learn_return": True,
        "completed_updates": source_completed,
        "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS,
        "num_envs": 4096,
        "profile": PROFILE,
    }
    mismatch = {k: (audit.get(k), v) for k, v in expected.items() if audit.get(k) != v}
    if mismatch:
        raise RuntimeError(f"Audit status mismatch: {mismatch}")
    if (
        audit.get("state_hashes_before") != audit.get("state_hashes_after")
        or audit.get("forbidden_calls") != []
    ):
        raise RuntimeError(
            "Model/optimizer state changed or forbidden method was called"
        )
    checkpoint_hash = file_sha256(checkpoint)
    # Isaac Kit can end the interpreter in app.close(), after the authoritative
    # before-learn-return receipt but before launcher's optional epilogue. This
    # separate, post-exit verifier therefore recomputes the checkpoint hash;
    # it never trusts a missing epilogue flag as evidence of unchanged bytes.
    if (
        audit.get("checkpoint_sha256") != checkpoint_hash
        or audit.get("input_file_unchanged") is False
    ):
        raise RuntimeError("Checkpoint hash mismatch")
    if audit.get("checkpoint_artifacts_written") != []:
        raise RuntimeError("Preflight wrote a checkpoint")
    if list(output.glob("model_*.pt")) or list(output.glob("student_*.pt")):
        raise RuntimeError("Forbidden checkpoint artifact exists")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("steps") != EXPECTED_STEPS or stats.get("optimizer_steps") != 0:
        raise RuntimeError("Statistics are not a 3000-step zero-update run")
    if abs(float(stats.get("simulated_seconds", 0)) - 60.0) > 1e-6:
        raise RuntimeError("Statistics do not cover 60 simulated seconds")
    validate_functional_evidence(stats)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("resume_sha256") != checkpoint_hash
        or run.get("starting_completed_updates") != source_completed
    ):
        raise RuntimeError("run.json does not identify the explicit early source")
    if process_check(output):
        raise RuntimeError("Audit process appeared during finalization")
    result = {
        "state": "early_restart_v14_preflight_complete_optimizer_steps_0",
        "completed_updates": source_completed,
        "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS,
        "simulated_seconds": 60.0,
        "num_envs": 4096,
        "checkpoint_written": False,
        "checkpoint_sha256": checkpoint_hash,
        "input_file_unchanged": True,
        "input_hash_verified_by": "independent_post_exit_finalizer",
        "finite_rewards_actions_and_states": True,
        "command_cohort_coverage_valid": True,
        "reset_targets_valid": True,
        "episode_and_resampling_schedule_valid": True,
    }
    with (output / "FINALIZED.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-completed",
        type=int,
        choices=sorted(ALLOWED_COMPLETED_UPDATES),
        required=True,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(args.output, args.checkpoint, args.source_completed), indent=2
        )
    )


if __name__ == "__main__":
    main()
