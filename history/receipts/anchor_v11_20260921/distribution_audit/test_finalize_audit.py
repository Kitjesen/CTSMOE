"""Contract tests for finalizing an already exited zero-update audit."""

import hashlib
import json
from finalize_audit import EXPECTED_CHECKPOINT_SHA256, EXPECTED_REFERENCE_SHA256, finalize


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_source_hashes_are_sha256_width():
    assert len(EXPECTED_CHECKPOINT_SHA256) == 64
    assert len(EXPECTED_REFERENCE_SHA256) == 64


def make_completed_audit(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    checkpoint = tmp_path / "model_44758.pt"
    reference = tmp_path / "model_43201.pt"
    checkpoint.write_bytes(b"v11")
    reference.write_bytes(b"v6")
    hashes = {
        "model": "a" * 64,
        "ppo_optimizer": "b" * 64,
        "student_optimizer": "c" * 64,
    }
    write_json(
        output / "audit_status.json",
        {
            "state": "AUDIT_COMPLETE",
            "smoke_only": False,
            "completed_updates": 44_759,
            "optimizer_updates": 0,
            "horizons": 20,
            "control_steps": 480,
            "num_envs": 4096,
            "normalization_disabled": True,
            "storage_cleared_each_horizon": True,
            "mutation_patches_restored": True,
            "stats_environment_patch_restored": True,
            "forbidden_calls": [],
            "state_hashes_before": hashes,
            "state_hashes_after": hashes,
        },
    )
    write_json(output / "distribution.json", {"steps": 480, "optimizer_steps": 0, "coverage_gate": False})
    write_json(
        output / "run.json",
        {
            "resume_sha256": digest(checkpoint),
            "starting_completed_updates": 44_759,
            "additional_updates": 0,
            "target_completed_updates": 44_759,
            "startup_anchor": {"reference_sha256": digest(reference)},
        },
    )
    write_json(output / "SUCCESS.json", {"state": "complete", "completed_updates": 44_760})
    write_json(output / "status.json", {"state": "complete", "completed_updates": 44_760})
    return output, checkpoint, reference


def test_finalize_preserves_prior_receipts_and_records_failed_coverage(tmp_path):
    output, checkpoint, reference = make_completed_audit(tmp_path)
    result = finalize(
        output=output,
        checkpoint=checkpoint,
        reference=reference,
        expected_checkpoint_sha256=digest(checkpoint),
        expected_reference_sha256=digest(reference),
        process_check=lambda _output: [],
    )
    assert result["completed_updates"] == 44_759
    assert result["optimizer_updates"] == 0
    assert result["coverage_gate"] is False
    assert result["next_stage_authorized"] is False
    assert result["isolated_audit_process_exited"] is True
    preserved = json.loads((output / "pre_finalization_receipts.json").read_text(encoding="utf-8"))
    assert preserved["SUCCESS.json"]["parsed"]["completed_updates"] == 44_760
    assert preserved["status.json"]["parsed"]["completed_updates"] == 44_760
    assert json.loads((output / "SUCCESS.json").read_text())["state"] == "audit_complete_optimizer_steps_0"
    status = json.loads((output / "audit_status.json").read_text())
    assert "launcher_patch_restored" not in status
    assert status["no_live_audit_process"] is True


def test_finalize_refuses_live_process_without_writing(tmp_path):
    output, checkpoint, reference = make_completed_audit(tmp_path)
    try:
        finalize(
            output=output,
            checkpoint=checkpoint,
            reference=reference,
            expected_checkpoint_sha256=digest(checkpoint),
            expected_reference_sha256=digest(reference),
            process_check=lambda _output: [{"pid": 123, "command": ["launcher.py"]}],
        )
    except RuntimeError as exc:
        assert "still running" in str(exc)
    else:
        raise AssertionError("live audit process was accepted")
    assert not (output / "pre_finalization_receipts.json").exists()
    assert json.loads((output / "status.json").read_text())["completed_updates"] == 44_760
