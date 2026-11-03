"""Finalize a completed V12 preflight after Isaac has exited."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


EXPECTED_COMPLETED_UPDATES = 44_711
EXPECTED_STEPS = 3_000
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
            command = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
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


def validate_functional_evidence(stats: dict[str, Any]) -> None:
    """Require real long episodes and resampling, without gating policy quality."""
    groups = stats.get("groups") or {}
    required = {f"{learner}_{cohort}" for learner in ("teacher", "student") for cohort in ("special", "ordinary")}
    if not required.issubset(groups):
        raise RuntimeError(f"Missing flat audit groups: {sorted(required - set(groups))}")
    if max(float(groups[name]["maximum_episode_age_seconds"]) for name in required) < 24.9:
        raise RuntimeError("No flat environment reached a real >=24.9 second episode age")
    if sum(int(groups[name]["normal_timeouts_age_ge_24p9_seconds"]) for name in required) <= 0:
        raise RuntimeError("No normal flat 25 second timeout was observed")
    for name in sorted(required):
        sample = groups[name]["normal_translation_resample_intervals_seconds"]
        generations = sample["episode_generation_counts"]
        if sample["count"] <= 0 or generations["episode_0"] <= 0 or generations["episode_1_plus"] <= 0:
            raise RuntimeError(f"Missing episode-0 or subsequent normal resampling for {name}")
        if sample["minimum"] < 4.9 or sample["maximum"] > 5.1:
            raise RuntimeError(f"Normal translation resampling was not on the 5 second schedule for {name}")
    anchor = stats.get("startup_anchor") or {}
    if anchor.get("coefficients") != EXPECTED_COEFFICIENTS:
        raise RuntimeError("Runtime statistics do not retain fixed startup-anchor coefficients")
    counts = stats.get("algorithm_startup_anchor_counts") or {}
    if int(counts.get("teacher", 0)) <= 0 or int(counts.get("student", 0)) <= 0:
        raise RuntimeError("Runtime startup-anchor sample counts are missing")


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
    paths = {name: output / name for name in (
        "audit_status.json", "full_episode_stats.json", "run.json", "SUCCESS.json", "status.json"
    )}
    for path in (*paths.values(), checkpoint, reference, coefficient_donor):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(paths["audit_status.json"].read_text(encoding="utf-8"))
    expected = {
        "state": "AUDIT_COMPLETE", "smoke_only": False,
        "completed_updates": EXPECTED_COMPLETED_UPDATES, "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS, "num_envs": 4096,
        "profile": "omni_full_episode_v1", "written_before_learn_return": True,
    }
    mismatch = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Audit status mismatch: {mismatch}")
    if audit.get("state_hashes_before") != audit.get("state_hashes_after") or audit.get("forbidden_calls") != []:
        raise RuntimeError("Model/optimizer hashes changed or a forbidden mutation was called")
    checkpoint_hash, reference_hash = file_sha256(checkpoint), file_sha256(reference)
    coefficient_donor_hash = file_sha256(coefficient_donor)
    if checkpoint_hash != audit.get("checkpoint_sha256") or reference_hash != audit.get("reference_sha256"):
        raise RuntimeError("Input checkpoint hashes no longer match the launch receipt")
    if coefficient_donor_hash != audit.get("coefficient_donor_sha256"):
        raise RuntimeError("Coefficient donor hash no longer matches the launch receipt")
    if audit.get("checkpoint_artifacts_written") != []:
        raise RuntimeError("Launcher reported a model artifact")
    artifacts = [path for pattern in ("model_*.pt", "student_*.pt", "*.tmp") for path in output.glob(pattern)]
    if artifacts:
        raise RuntimeError(f"Preflight wrote forbidden artifacts: {artifacts}")
    stats = json.loads(paths["full_episode_stats.json"].read_text(encoding="utf-8"))
    if stats.get("steps") != EXPECTED_STEPS or stats.get("optimizer_steps") != 0:
        raise RuntimeError("Statistics are not the 3000-step zero-update preflight")
    if not abs(float(stats.get("simulated_seconds", 0)) - 60.0) < 1e-6:
        raise RuntimeError("Statistics do not cover 60 simulated seconds")
    validate_functional_evidence(stats)
    run = json.loads(paths["run.json"].read_text(encoding="utf-8"))
    if run.get("resume_sha256") != checkpoint_hash or run.get("starting_completed_updates") != EXPECTED_COMPLETED_UPDATES:
        raise RuntimeError("Training receipt does not identify the audited V9 checkpoint")
    anchor = run.get("startup_anchor") or {}
    if anchor.get("coefficients") != EXPECTED_COEFFICIENTS:
        raise RuntimeError("Training receipt does not contain the fixed audited anchor coefficients")
    origin = anchor.get("coefficient_origin") or {}
    if origin.get("sha256") != coefficient_donor_hash or origin.get("donor_completed_updates") != 44_759:
        raise RuntimeError("Training receipt does not identify the V11 coefficient donor")
    if process_check(output):
        raise RuntimeError("Audit process appeared during finalization")
    preservation = output / "pre_finalization_training_receipts.json"
    write_json(preservation, {
        "reason": "Preserve generic train.py receipts; audit_status.json is authoritative",
        "SUCCESS.json": read_receipt(paths["SUCCESS.json"]),
        "status.json": read_receipt(paths["status.json"]),
    }, exclusive=True)
    result = {
        "state": "full_episode_preflight_complete_optimizer_steps_0",
        "completed_updates": EXPECTED_COMPLETED_UPDATES,
        "optimizer_updates": 0,
        "control_steps": EXPECTED_STEPS,
        "simulated_seconds": 60.0,
        "num_envs": 4096,
        "isolated_audit_process_exited": True,
        "no_live_audit_process": True,
        "checkpoint_written": False,
        "checkpoint_sha256": checkpoint_hash,
        "reference_sha256": reference_hash,
        "coefficient_donor_sha256": coefficient_donor_hash,
        "authoritative_receipt": "audit_status.json",
        "preserved_training_receipts": preservation.name,
        "functional_sampling_valid": True,
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
    print(json.dumps(finalize(
        args.output, args.checkpoint, args.reference, args.coefficient_donor
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
