from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = load("audit_stats")
finalizer = load("finalize_audit")


def valid_stats() -> dict:
    return {
        "steps": 3000,
        "optimizer_steps": 0,
        "simulated_seconds": 60.0,
        "configuration": {
            "episode_seconds": 25.0,
            "native_resampling_seconds": 5.0,
            "evidence_scope": "flat normal cohort",
        },
        "maximum_episode_age_seconds": {"teacher": 24.98, "student": 24.98},
        "normal_timeouts_age_ge_24p9_seconds": {"teacher": 1, "student": 1},
        "native_resample_intervals_seconds": {
            "teacher": {"count": 2, "minimum": 5.0, "maximum": 5.02},
            "student": {"count": 2, "minimum": 5.0, "maximum": 5.02},
        },
        "startup_anchor": {"coefficients": finalizer.EXPECTED_COEFFICIENTS},
        "algorithm_startup_anchor_counts": {"teacher": 10, "student": 10},
    }


def valid_v13() -> dict:
    return {
        "reset_target_receipt": {
            "version": finalizer.RESET_TARGETS_VERSION,
            "robot": "robot",
        },
        "first_learn_target_actual_max_abs_error": 0.0,
        "startup_reset_counts": {
            "counts": [[ [1, 1] for _ in range(10)] for _ in range(2)]
        },
        "step_zero_observations": {
            str((learner, kind, mode)): 1
            for learner in range(2) for kind in range(10) for mode in range(2)
        },
        "ramp_first_ten_step_checks": {
            str((learner, kind, 1)): 11 for learner in range(2) for kind in range(10)
        },
        "violations": [],
    }


def test_expected_startup_command_has_step_direct_and_ten_step_ramp() -> None:
    goal = torch.tensor([[0.6, 0.0, 0.0]]).expand(4, -1)
    ramp = torch.tensor([False, True, True, True])
    age = torch.tensor([0, 0, 5, 10])
    actual = stats.expected_startup_command(goal, ramp, age)
    torch.testing.assert_close(actual[0], goal[0])
    torch.testing.assert_close(actual[1], torch.zeros(3))
    torch.testing.assert_close(actual[2], goal[2] * 0.5)
    torch.testing.assert_close(actual[3], goal[3])


def test_reset_counts_require_exact_long_learner_kind_mode_shape() -> None:
    result = stats.reset_count_payload(torch.ones(2, 10, 2, dtype=torch.long))
    assert result["all_cells_positive"] is True
    assert result["learner_labels"] == ["teacher", "student"]
    assert result["mode_labels"] == ["direct", "ramp"]
    with pytest.raises(RuntimeError, match="startup_reset_counts"):
        stats.reset_count_payload(torch.ones(2, 10, dtype=torch.long))
    with pytest.raises(RuntimeError, match="startup_reset_counts"):
        stats.reset_count_payload(torch.ones(2, 10, 2))


def test_functional_gate_requires_reset_sync_sampler_coverage_and_native_schedule() -> None:
    finalizer.validate_functional_evidence(valid_stats(), valid_v13())
    broken = valid_v13()
    broken["startup_reset_counts"]["counts"][1][9][1] = 0
    with pytest.raises(RuntimeError, match="Every teacher/student"):
        finalizer.validate_functional_evidence(valid_stats(), broken)
    broken = valid_v13()
    broken["first_learn_target_actual_max_abs_error"] = 0.1
    with pytest.raises(RuntimeError, match="First learn target"):
        finalizer.validate_functional_evidence(valid_stats(), broken)
    broken_stats = valid_stats()
    broken_stats["native_resample_intervals_seconds"]["student"]["minimum"] = 4.0
    with pytest.raises(RuntimeError, match="5 second schedule"):
        finalizer.validate_functional_evidence(broken_stats, valid_v13())


def test_finalizer_preserves_receipts_and_writes_v13_finalized(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    checkpoint, reference, donor = (tmp_path / name for name in ("checkpoint.pt", "reference.pt", "donor.pt"))
    checkpoint.write_bytes(b"checkpoint")
    reference.write_bytes(b"reference")
    donor.write_bytes(b"donor")
    hashes = {"model": "a", "ppo_optimizer": "b", "student_optimizer": "c"}
    audit = {
        "state": "AUDIT_COMPLETE", "smoke_only": False, "completed_updates": 44711,
        "optimizer_updates": 0, "control_steps": 3000, "num_envs": 4096,
        "profile": finalizer.PROFILE, "written_before_learn_return": True,
        "state_hashes_before": hashes, "state_hashes_after": hashes, "forbidden_calls": [],
        "checkpoint_sha256": finalizer.file_sha256(checkpoint),
        "reference_sha256": finalizer.file_sha256(reference),
        "coefficient_donor_sha256": finalizer.file_sha256(donor),
        "checkpoint_artifacts_written": [],
    }
    (output / "audit_status.json").write_text(json.dumps(audit))
    (output / "full_episode_stats.json").write_text(json.dumps(valid_stats()))
    (output / "v13_summary.json").write_text(json.dumps(valid_v13()))
    (output / "run.json").write_text(json.dumps({
        "resume_sha256": audit["checkpoint_sha256"], "starting_completed_updates": 44711,
        "startup_anchor": {
            "coefficients": finalizer.EXPECTED_COEFFICIENTS,
            "coefficient_origin": {
                "sha256": audit["coefficient_donor_sha256"], "donor_completed_updates": 44759,
            },
        },
    }))
    (output / "SUCCESS.json").write_text('{"state":"complete"}')
    (output / "status.json").write_text('{"state":"complete"}')

    result = finalizer.finalize(output, checkpoint, reference, donor, process_check=lambda _output: [])

    assert result["state"] == "startup_v13_preflight_complete_optimizer_steps_0"
    assert result["startup_sampler_valid"] is True
    assert result["reset_targets_valid"] is True
    assert (output / "FINALIZED.json").is_file()


def test_launcher_and_shell_are_fixed_to_v13_without_automatic_remote_start() -> None:
    launcher = (HERE / "launcher.py").read_text(encoding="utf-8")
    shell = (HERE / "run.sh").read_text(encoding="utf-8")
    for token in (
        'PROFILE = "omni_full_episode_v13"', "CONTROL_STEPS = 3_000", "SMOKE_STEPS = 3",
        'self._forbid(self.runner.alg, "update", "alg.update")',
        '"--startup-anchor-coefficients-from", str(inputs["coefficient_donor"])',
    ):
        assert token in launcher
    assert "RUN_SMOKE=${RUN_SMOKE:-0}" in shell
    assert "code_startup_v13_20260921" in shell
    assert 'full_episode_v12_pipeline.lock' in shell
    assert "ssh " not in shell
