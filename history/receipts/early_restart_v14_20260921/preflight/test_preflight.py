from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats_mod, finalizer = load("audit_stats"), load("finalize_audit")


def valid_stats():
    return {
        "steps": 3000,
        "optimizer_steps": 0,
        "simulated_seconds": 60.0,
        "configuration": {
            "episode_seconds": 25.0,
            "native_resampling_seconds": 5.0,
            "early_restart_profile": "early_phase_v14",
        },
        "maximum_episode_age_seconds": {"teacher": 24.98, "student": 24.98},
        "native_resample_intervals_seconds": {
            "teacher": {
                "count": 2,
                "minimum": 5.0,
                "maximum": 5.02,
                "short_intervals_involving_zero_translation": 0,
            },
            "student": {
                "count": 2,
                "minimum": 5.0,
                "maximum": 5.02,
                "short_intervals_involving_zero_translation": 3,
            },
        },
        "reward_nonfinite_count": 0,
        "action_nonfinite_count": 0,
        "reset_target_receipt": {
            "version": finalizer.RESET_TARGETS_VERSION,
            "robot": "robot",
        },
        "first_target_actual_max_abs_error": 0.0,
        "startup_reset_counts": {
            "counts": [[[1, 1] for _ in range(10)] for _ in range(2)]
        },
        "command_cohort_observations": {
            str((learner, kind)): 1 for learner in range(2) for kind in range(10)
        },
    }


def test_reset_counts_shape_and_dtype():
    assert stats_mod.reset_count_payload(torch.ones(2, 10, 2, dtype=torch.long))[
        "all_cells_positive"
    ]
    with pytest.raises(RuntimeError):
        stats_mod.reset_count_payload(torch.ones(2, 10, 2))


def test_functional_gate_rejects_nonfinite_and_missing_coverage():
    finalizer.validate_functional_evidence(valid_stats())
    broken = valid_stats()
    broken["reward_nonfinite_count"] = 1
    with pytest.raises(RuntimeError, match="Non-finite"):
        finalizer.validate_functional_evidence(broken)
    broken = valid_stats()
    broken["startup_reset_counts"]["counts"][1][9][1] = 0
    with pytest.raises(RuntimeError, match="coverage"):
        finalizer.validate_functional_evidence(broken)


@pytest.mark.parametrize("epilogue,mutated", [(True, False), (None, False), (False, False), (None, True)])
def test_finalized_only_after_all_evidence(tmp_path, epilogue, mutated):
    output = tmp_path / "out"
    output.mkdir()
    checkpoint = tmp_path / "early.pt"
    checkpoint.write_bytes(b"x")
    digest = finalizer.file_sha256(checkpoint)
    hashes = {"model": "a", "ppo_optimizer": "b", "student_optimizer": "c"}
    (output / "audit_status.json").write_text(
        json.dumps(
            {
                "state": "AUDIT_COMPLETE",
                "written_before_learn_return": True,
                "completed_updates": 30000,
                "optimizer_updates": 0,
                "control_steps": 3000,
                "num_envs": 4096,
                "profile": "early_phase_v14",
                "state_hashes_before": hashes,
                "state_hashes_after": hashes,
                "forbidden_calls": [],
                "checkpoint_sha256": digest,
                "input_file_unchanged": epilogue,
                "checkpoint_artifacts_written": [],
            }
        )
    )
    (output / "preflight_stats.json").write_text(json.dumps(valid_stats()))
    (output / "run.json").write_text(
        json.dumps({"resume_sha256": digest, "starting_completed_updates": 30000})
    )
    (output / "SUCCESS.json").write_text("{}")
    (output / "status.json").write_text("{}")
    if mutated:
        checkpoint.write_bytes(b"changed checkpoint")
    if mutated or epilogue is False:
        with pytest.raises(RuntimeError, match="Checkpoint hash mismatch"):
            finalizer.finalize(output, checkpoint, 30000, process_check=lambda _: [])
        assert not (output / "FINALIZED.json").exists()
        return
    result = finalizer.finalize(output, checkpoint, 30000, process_check=lambda _: [])
    assert result["optimizer_updates"] == 0 and (output / "FINALIZED.json").is_file()


def test_launcher_has_explicit_early_profiles_and_no_anchor():
    source = (HERE / "launcher.py").read_text(encoding="utf-8")
    for token in (
        'PROFILE = "early_phase_v14"',
        '"--startup-repair-profile"',
        '"--allocation-profile"',
        '"balanced_adaptive_v1"',
        '"--wave-reset-profile"',
        '"wave_surface_v1"',
    ):
        assert token in source
    assert "startup-anchor" not in source
