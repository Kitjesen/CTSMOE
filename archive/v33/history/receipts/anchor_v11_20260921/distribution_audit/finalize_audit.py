"""Validate and finalize the already completed 480-step zero-update audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


EXPECTED_COMPLETED_UPDATES = 44_759
EXPECTED_CHECKPOINT_SHA256 = "73d961f5855d36b43d86852845a31f9a27259805cea53a317787e0bf023e5c76"
EXPECTED_REFERENCE_SHA256 = "bdde0ca02d7e0d9db9ecdfd8c9b352cc1f511db49a9af936cc14c8046f6a77f8"
assert len(EXPECTED_CHECKPOINT_SHA256) == len(EXPECTED_REFERENCE_SHA256) == 64


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def live_audit_processes(output: Path) -> list[dict[str, Any]]:
    """Find a still-running launcher/train process for this exact audit output."""
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Process verification requires Linux /proc")
    output_text = str(output.resolve())
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            command = [value.decode(errors="replace") for value in arguments if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = " ".join(command)
        audit_entry = any(Path(value).name == "launcher.py" for value in command)
        training_entry = any(Path(value).name == "train.py" for value in command)
        if output_text in joined and (audit_entry or training_entry):
            found.append({"pid": int(entry.name), "command": command})
    return found


def read_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw": raw,
        "parsed": json.loads(raw),
    }


def write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def validate_audit_status(status: dict[str, Any]) -> None:
    expected = {
        "state": "AUDIT_COMPLETE",
        "smoke_only": False,
        "completed_updates": EXPECTED_COMPLETED_UPDATES,
        "optimizer_updates": 0,
        "horizons": 20,
        "control_steps": 480,
        "num_envs": 4096,
        "normalization_disabled": True,
        "storage_cleared_each_horizon": True,
        "mutation_patches_restored": True,
        "stats_environment_patch_restored": True,
    }
    mismatches = {key: (status.get(key), value) for key, value in expected.items() if status.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Audit status contract mismatch: {mismatches}")
    before = status.get("state_hashes_before")
    after = status.get("state_hashes_after")
    required = {"model", "ppo_optimizer", "student_optimizer"}
    if not isinstance(before, dict) or set(before) != required or before != after:
        raise RuntimeError("Model and optimizer hashes are absent or changed")
    if status.get("forbidden_calls") != []:
        raise RuntimeError(f"Forbidden mutation calls were recorded: {status.get('forbidden_calls')}")


def finalize(
    *,
    output: Path,
    checkpoint: Path,
    reference: Path,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    expected_reference_sha256: str = EXPECTED_REFERENCE_SHA256,
    process_check: Callable[[Path], list[dict[str, Any]]] = live_audit_processes,
) -> dict[str, Any]:
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    reference = reference.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    preservation = output / "pre_finalization_receipts.json"
    if preservation.exists():
        raise FileExistsError(f"Audit receipts were already finalized: {preservation}")
    live = process_check(output)
    if live:
        raise RuntimeError(f"Audit process is still running: {live}")
    status_path = output / "audit_status.json"
    distribution_path = output / "distribution.json"
    run_path = output / "run.json"
    success_path = output / "SUCCESS.json"
    train_status_path = output / "status.json"
    for path in (status_path, distribution_path, run_path, success_path, train_status_path, checkpoint, reference):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    validate_audit_status(status)
    checkpoint_hash = file_sha256(checkpoint)
    reference_hash = file_sha256(reference)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise RuntimeError(f"Unexpected V11 checkpoint hash: {checkpoint_hash}")
    if reference_hash != expected_reference_sha256:
        raise RuntimeError(f"Unexpected V6 reference hash: {reference_hash}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_expected = {
        "resume_sha256": checkpoint_hash,
        "starting_completed_updates": EXPECTED_COMPLETED_UPDATES,
        "additional_updates": 0,
        "target_completed_updates": EXPECTED_COMPLETED_UPDATES,
    }
    run_mismatches = {key: (run.get(key), value) for key, value in run_expected.items() if run.get(key) != value}
    if run_mismatches:
        raise RuntimeError(f"Training setup receipt does not identify this zero-update audit: {run_mismatches}")
    if (run.get("startup_anchor") or {}).get("reference_sha256") != reference_hash:
        raise RuntimeError("Training setup receipt does not identify the supplied V6 reference")
    artifacts = []
    for pattern in ("model_*.pt", "student_*.pt", "*.tmp"):
        artifacts.extend(sorted(output.glob(pattern)))
    if artifacts:
        raise RuntimeError(f"Audit wrote forbidden model artifacts: {artifacts}")
    distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    if distribution.get("steps") != 480 or distribution.get("optimizer_steps") != 0:
        raise RuntimeError("Distribution summary is not the completed zero-update 480-step audit")
    if distribution.get("coverage_gate") is not False:
        raise RuntimeError("This finalizer is scoped to the observed failed coverage gate")
    # Recheck immediately before the first mutation so a concurrent audit cannot race finalization.
    live = process_check(output)
    if live:
        raise RuntimeError(f"Audit process appeared before finalization: {live}")

    preserved = {
        "reason": "Preserve generic train.py completion receipts before zero-update audit correction",
        "SUCCESS.json": read_receipt(success_path),
        "status.json": read_receipt(train_status_path),
    }
    write_json(preservation, preserved, exclusive=True)
    terminal = {
        "state": "audit_complete_optimizer_steps_0",
        "outcome": "coverage_not_met",
        "completed_updates": EXPECTED_COMPLETED_UPDATES,
        "optimizer_updates": 0,
        "control_steps": 480,
        "num_envs": 4096,
        "coverage_gate": False,
        "next_stage_authorized": False,
        "isolated_audit_process_exited": True,
        "no_live_audit_process": True,
        "checkpoint_written": False,
        "checkpoint_sha256": checkpoint_hash,
        "reference_sha256": reference_hash,
        "authoritative_receipt": "audit_status.json",
        "prior_receipts": preservation.name,
    }
    status.update(
        checkpoint_sha256=checkpoint_hash,
        reference_sha256=reference_hash,
        input_files_unchanged=True,
        checkpoint_artifacts_written=[],
        coverage_gate=False,
        next_stage_authorized=False,
        isolated_audit_process_exited=True,
        no_live_audit_process=True,
        finalization="completed_after_source_process_exit",
        prior_receipts=preservation.name,
    )
    write_json(status_path, status)
    write_json(success_path, terminal)
    write_json(train_status_path, terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(output=args.output, checkpoint=args.checkpoint, reference=args.reference)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
