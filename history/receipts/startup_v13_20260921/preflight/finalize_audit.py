"""Finalize V13 preflight evidence after the isolated Isaac process exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


EXPECTED_COMPLETED_UPDATES = 44_711
EXPECTED_STEPS = 3_000
PROFILE = "omni_full_episode_v13"
RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"
EXPECTED_COEFFICIENTS = {"main": 0.21093362834788681, "student": 0.0022221143221353456}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def live_audit_processes(output: Path) -> list[dict[str, Any]]:
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Process verification requires Linux /proc")
    needle = str(output.resolve())
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in " ".join(command) and any(Path(arg).name in {"launcher.py", "train.py"} for arg in command):
            found.append({"pid": int(entry.name), "command": command})
    return found


def write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def read_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return {"path": path.name, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "raw": raw}


def validate_functional_evidence(stats: dict[str, Any], v13: dict[str, Any]) -> None:
    expected_configuration = {
        "episode_seconds": 25.0,
        "native_resampling_seconds": 5.0,
        "evidence_scope": "flat normal cohort",
    }
    if stats.get("configuration") != expected_configuration:
        raise RuntimeError("Runtime did not retain 25 second episodes and 5 second native resampling")
    for learner in ("teacher", "student"):
        if float(stats["maximum_episode_age_seconds"][learner]) < 24.9:
            raise RuntimeError(f"{learner} did not reach a real 25 second episode")
        if int(stats["normal_timeouts_age_ge_24p9_seconds"][learner]) <= 0:
            raise RuntimeError(f"{learner} has no normal 25 second timeout")
        intervals = stats["native_resample_intervals_seconds"][learner]
        if intervals["count"] <= 0 or intervals["minimum"] < 4.9 or intervals["maximum"] > 5.1:
            raise RuntimeError(f"{learner} native resampling was not on the 5 second schedule")
    anchor = stats.get("startup_anchor") or {}
    if anchor.get("coefficients") != EXPECTED_COEFFICIENTS:
        raise RuntimeError("Runtime statistics do not retain fixed V11 anchor coefficients")
    samples = stats.get("algorithm_startup_anchor_counts") or {}
    if int(samples.get("teacher", 0)) <= 0 or int(samples.get("student", 0)) <= 0:
        raise RuntimeError("Runtime startup-anchor samples are missing")
    expected_receipt = {"version": RESET_TARGETS_VERSION, "robot": "robot"}
    if v13.get("reset_target_receipt") != expected_receipt:
        raise RuntimeError("Reset-target runtime receipt is missing or wrong")
    if float(v13.get("first_learn_target_actual_max_abs_error", 1.0)) > 1e-6:
        raise RuntimeError("First learn target did not match actual reset q")
    counts = (v13.get("startup_reset_counts") or {}).get("counts")
    if not counts or len(counts) != 2 or any(len(kinds) != 10 for kinds in counts):
        raise RuntimeError("Startup reset count shape is not [2,10,2]")
    if any(len(modes) != 2 or any(int(value) <= 0 for value in modes) for kinds in counts for modes in kinds):
        raise RuntimeError("Every teacher/student-kind-direct/ramp startup cell must be sampled")
    if v13.get("violations") != []:
        raise RuntimeError(f"V13 startup trajectory violations: {v13.get('violations')}")
    step_zero = v13.get("step_zero_observations") or {}
    ramp_checks = v13.get("ramp_first_ten_step_checks") or {}
    expected_step_zero = {str((learner, kind, mode)) for learner in range(2) for kind in range(10) for mode in range(2)}
    expected_ramp = {str((learner, kind, 1)) for learner in range(2) for kind in range(10)}
    if not expected_step_zero.issubset(step_zero) or not expected_ramp.issubset(ramp_checks):
        raise RuntimeError("Missing per-kind step-zero or first-ten-step startup observations")


def finalize(
    output: Path,
    checkpoint: Path,
    reference: Path,
    coefficient_donor: Path,
    process_check: Callable[[Path], list[dict[str, Any]]] = live_audit_processes,
) -> dict[str, Any]:
    output, checkpoint, reference = output.resolve(), checkpoint.resolve(), reference.resolve()
    coefficient_donor = coefficient_donor.resolve()
    live = process_check(output)
    if live:
        raise RuntimeError(f"Audit process still running: {live}")
    names = (
        "audit_status.json", "full_episode_stats.json", "v13_summary.json",
        "run.json", "SUCCESS.json", "status.json",
    )
    paths = {name: output / name for name in names}
    for path in (*paths.values(), checkpoint, reference, coefficient_donor):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(paths["audit_status.json"].read_text(encoding="utf-8"))
    expected = {
        "state": "AUDIT_COMPLETE", "smoke_only": False,
        "completed_updates": EXPECTED_COMPLETED_UPDATES, "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS, "num_envs": 4096,
        "profile": PROFILE, "written_before_learn_return": True,
    }
    mismatch = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Audit status mismatch: {mismatch}")
    if audit.get("state_hashes_before") != audit.get("state_hashes_after") or audit.get("forbidden_calls") != []:
        raise RuntimeError("Model/optimizer hashes changed or a forbidden mutation was called")
    hashes = {
        "checkpoint": file_sha256(checkpoint),
        "reference": file_sha256(reference),
        "coefficient_donor": file_sha256(coefficient_donor),
    }
    if any(audit.get(f"{name}_sha256") != value for name, value in hashes.items()):
        raise RuntimeError("Input hashes no longer match the launch receipt")
    if audit.get("checkpoint_artifacts_written") != []:
        raise RuntimeError("Launcher reported a model artifact")
    artifacts = [path for pattern in ("model_*.pt", "student_*.pt", "*.tmp") for path in output.glob(pattern)]
    if artifacts:
        raise RuntimeError(f"Preflight wrote forbidden artifacts: {artifacts}")
    stats = json.loads(paths["full_episode_stats.json"].read_text(encoding="utf-8"))
    v13 = json.loads(paths["v13_summary.json"].read_text(encoding="utf-8"))
    if stats.get("steps") != EXPECTED_STEPS or stats.get("optimizer_steps") != 0:
        raise RuntimeError("Statistics are not the 3000-step zero-update preflight")
    if not abs(float(stats.get("simulated_seconds", 0)) - 60.0) < 1e-6:
        raise RuntimeError("Statistics do not cover 60 simulated seconds")
    validate_functional_evidence(stats, v13)
    run = json.loads(paths["run.json"].read_text(encoding="utf-8"))
    if run.get("resume_sha256") != hashes["checkpoint"] or run.get("starting_completed_updates") != 44_711:
        raise RuntimeError("Training receipt does not identify V9 completed_updates=44711")
    anchor = run.get("startup_anchor") or {}
    if anchor.get("coefficients") != EXPECTED_COEFFICIENTS:
        raise RuntimeError("Training receipt does not contain fixed anchor coefficients")
    origin = anchor.get("coefficient_origin") or {}
    if origin.get("sha256") != hashes["coefficient_donor"] or origin.get("donor_completed_updates") != 44_759:
        raise RuntimeError("Training receipt does not identify the V11 metadata donor")
    if process_check(output):
        raise RuntimeError("Audit process appeared during finalization")
    preservation = output / "pre_finalization_training_receipts.json"
    write_json(preservation, {
        "reason": "Preserve generic train.py receipts; audit_status.json is authoritative",
        "SUCCESS.json": read_receipt(paths["SUCCESS.json"]),
        "status.json": read_receipt(paths["status.json"]),
    }, exclusive=True)
    result = {
        "state": "startup_v13_preflight_complete_optimizer_steps_0",
        "completed_updates": EXPECTED_COMPLETED_UPDATES,
        "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS,
        "simulated_seconds": 60.0,
        "num_envs": 4096,
        "checkpoint_written": False,
        **{f"{name}_sha256": value for name, value in hashes.items()},
        "authoritative_receipt": "audit_status.json",
        "v13_summary": "v13_summary.json",
        "preserved_training_receipts": preservation.name,
        "functional_sampling_valid": True,
        "startup_sampler_valid": True,
        "reset_targets_valid": True,
        "policy_quality_gate_applied": False,
    }
    audit.update(result)
    write_json(paths["audit_status.json"], audit)
    write_json(output / "FINALIZED.json", result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--coefficient-donor", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.output, args.checkpoint, args.reference, args.coefficient_donor)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
