from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = load("audit_stats")
finalizer = load("finalize_audit")


def test_command_clock_tracks_continuous_nonzero_and_unchanged_segment() -> None:
    clock = stats.CommandClock(2, 0.02)
    episode = np.array([0, 0])
    first = np.array([[0.5, 0, 0], [0, 0, 0]], dtype=float)
    nonzero, segment, changed, reset = clock.update(first, episode)
    assert nonzero.tolist() == [0.0, 0.0]
    assert segment.tolist() == [0.0, 0.0]
    assert reset.tolist() == [True, True]
    nonzero, segment, changed, reset = clock.update(first, episode)
    assert nonzero.tolist() == pytest.approx([0.02, 0.0])
    assert segment.tolist() == pytest.approx([0.02, 0.0])
    assert not changed.any() and not reset.any()
    changed_command = np.array([[0.7, 0, 0], [0, 0, 0]], dtype=float)
    nonzero, segment, changed, _ = clock.update(changed_command, episode)
    assert nonzero[0] == pytest.approx(0.04)
    assert segment[0] == 0.0 and changed[0]


def test_vector_moments_are_per_joint() -> None:
    moments = stats.VectorMoments(2)
    moments.add(np.array([[1.0, -2.0], [3.0, 4.0]]))
    result = moments.summary(["a", "b"])
    assert result["states"] == 2
    assert result["joints"]["a"]["mean"] == 2.0
    assert result["joints"]["b"]["abs_max"] == 4.0


def test_timer_detects_repeat_resample_and_yaw_is_zero_translation() -> None:
    assert stats.timer_resampled(np.array([5.0]), np.array([0.02]), 0.02).tolist() == [True]
    commands = np.array([[0.0, 0.0, 0.8], [0.1, 0.0, 0.0]])
    assert stats.translation_nonzero(commands).tolist() == [False, True]


def test_finalizer_preserves_training_receipts_and_uses_launch_hashes(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    checkpoint, reference = tmp_path / "checkpoint.pt", tmp_path / "reference.pt"
    donor = tmp_path / "donor.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference.write_bytes(b"reference")
    donor.write_bytes(b"donor")
    hashes = {"model": "a", "ppo_optimizer": "b", "student_optimizer": "c"}
    audit = {
        "state": "AUDIT_COMPLETE", "smoke_only": False, "completed_updates": 44711,
        "optimizer_updates": 0, "control_steps": 3000, "num_envs": 4096,
        "profile": "omni_full_episode_v1", "written_before_learn_return": True,
        "state_hashes_before": hashes, "state_hashes_after": hashes, "forbidden_calls": [],
        "checkpoint_sha256": finalizer.file_sha256(checkpoint),
        "reference_sha256": finalizer.file_sha256(reference), "checkpoint_artifacts_written": [],
        "coefficient_donor_sha256": finalizer.file_sha256(donor),
    }
    (output / "audit_status.json").write_text(json.dumps(audit))
    groups = {}
    for learner in ("teacher", "student"):
        groups[f"{learner}_special"] = {
            "maximum_episode_age_seconds": 24.98,
            "normal_timeouts_age_ge_24p9_seconds": 1,
            "normal_translation_resample_intervals_seconds": {
                "count": 10, "minimum": 5.0, "maximum": 5.0,
                "episode_generation_counts": {"episode_0": 5, "episode_1_plus": 5},
            },
        }
        groups[f"{learner}_ordinary"] = {
            "maximum_episode_age_seconds": 24.98,
            "normal_timeouts_age_ge_24p9_seconds": 1,
            "normal_translation_resample_intervals_seconds": {
                "count": 10, "minimum": 5.0, "maximum": 5.0,
                "episode_generation_counts": {"episode_0": 5, "episode_1_plus": 5},
            },
        }
    (output / "full_episode_stats.json").write_text(json.dumps({
        "steps": 3000, "optimizer_steps": 0, "simulated_seconds": 60.0,
        "groups": groups,
        "startup_anchor": {"coefficients": finalizer.EXPECTED_COEFFICIENTS},
        "algorithm_startup_anchor_counts": {"teacher": 10, "student": 10},
    }))
    (output / "run.json").write_text(json.dumps({
        "resume_sha256": audit["checkpoint_sha256"], "starting_completed_updates": 44711,
        "startup_anchor": {
            "coefficients": finalizer.EXPECTED_COEFFICIENTS,
            "coefficient_origin": {
                "sha256": audit["coefficient_donor_sha256"], "donor_completed_updates": 44759
            },
        },
    }))
    (output / "SUCCESS.json").write_text('{"state":"complete"}')
    (output / "status.json").write_text('{"state":"complete"}')
    result = finalizer.finalize(output, checkpoint, reference, donor, process_check=lambda _output: [])
    assert result["state"] == "full_episode_preflight_complete_optimizer_steps_0"
    assert (output / "pre_finalization_training_receipts.json").is_file()
    assert (output / "FINALIZED.json").is_file()


def test_launcher_contract_is_fixed_to_requested_run() -> None:
    source = (HERE / "launcher.py").read_text(encoding="utf-8")
    for token in (
        "EXPECTED_COMPLETED_UPDATES = 44_711", "CONTROL_STEPS = 3_000",
        '"--startup-repair-profile", "omni_full_episode_v1"',
        '"--startup-anchor-coefficients-from", str(coefficient_donor)',
        'self._forbid(alg, "update", "alg.update")',
        'self._forbid(alg.optimizer, "step", "alg.optimizer.step")',
    ):
        assert token in source


def test_functional_gate_does_not_reject_physical_failures() -> None:
    groups = {}
    for learner in ("teacher", "student"):
        groups[f"{learner}_special"] = {
            "maximum_episode_age_seconds": 24.98,
            "normal_timeouts_age_ge_24p9_seconds": 1,
            "physical_ends": 999,
            "normal_translation_resample_intervals_seconds": {
                "count": 2, "minimum": 5.0, "maximum": 5.0,
                "episode_generation_counts": {"episode_0": 1, "episode_1_plus": 1},
            },
        }
        groups[f"{learner}_ordinary"] = {
            "maximum_episode_age_seconds": 24.98,
            "normal_timeouts_age_ge_24p9_seconds": 1,
            "physical_ends": 999,
            "normal_translation_resample_intervals_seconds": {
                "count": 2, "minimum": 5.0, "maximum": 5.0,
                "episode_generation_counts": {"episode_0": 1, "episode_1_plus": 1},
            },
        }
    finalizer.validate_functional_evidence({
        "groups": groups,
        "startup_anchor": {"coefficients": finalizer.EXPECTED_COEFFICIENTS},
        "algorithm_startup_anchor_counts": {"teacher": 1, "student": 1},
    })


def test_no_fixed_v11_checkpoint_hash_validator() -> None:
    source = (HERE / "finalize_audit.py").read_text(encoding="utf-8")
    assert "EXPECTED_CHECKPOINT_SHA256" not in source
    assert "73d961f5855d36b43d86852845a31f9a27259805cea53a317787e0bf023e5c76" not in source
