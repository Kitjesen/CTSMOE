import json
from pathlib import Path

import pytest

import run_pipeline as pipeline


def test_training_budget_stops_after_16_plus_284_updates() -> None:
    assert pipeline.BASE_COMPLETED == 44_711
    assert [item[1] for item in pipeline.TRAINING_STAGES] == [16, 284]
    assert sum(item[1] for item in pipeline.TRAINING_STAGES) == 300
    assert [item[2] for item in pipeline.TRAINING_STAGES] == [44_727, 45_011]
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "confirm1000" not in source
    assert "45_711" not in source
    assert "pilot300_passed_pending_review" in source
    assert "pilot300_rejected_pending_review" in source


def test_first_train_imports_only_v11_metadata_and_every_stage_uses_v6_reference() -> None:
    destination = pipeline.ROOT / "runs/example"
    first = pipeline.train_command(destination, pipeline.SOURCE, 16, first_stage=True)
    resumed = pipeline.train_command(destination, pipeline.SOURCE, 284, first_stage=False)
    assert first[first.index("--resume") + 1] == str(pipeline.SOURCE)
    assert first[first.index("--startup-anchor-coefficients-from") + 1] == str(pipeline.DONOR)
    assert "--startup-anchor-coefficients-from" not in resumed
    for command in (first, resumed):
        assert command[command.index("--startup-repair-profile") + 1] == "omni_full_episode_v13"
        assert command[command.index("--startup-anchor-reference") + 1] == str(pipeline.REFERENCE)


def test_evaluation_keeps_existing_332_and_100_case_budgets() -> None:
    checkpoint = pipeline.ROOT / "runs/candidate/model_45010.pt"
    normal = pipeline.selection(checkpoint, 45_011)
    long = pipeline.selection(checkpoint, 45_011, flat25=True)
    assert len(normal) * 83 == 332
    assert len(long) * 20 == 100
    assert all(item["label"] != "V6_43202" for item in normal)
    assert sum(item["label"] == "V6_43202" for item in long) == 1
    assert [item["label"] for item in normal] == [
        "V9_44711", "V9_44711_teacher", "V13_45011", "V13_45011_teacher",
    ]


def test_all_existing_gates_run_but_cannot_trigger_more_training() -> None:
    calls = []

    def normal_gate(analysis, baseline, candidate):
        calls.append(("normal", baseline, candidate))
        return {"accepted": candidate.endswith("_teacher")}

    def flat_gate(analysis, baseline, candidate):
        calls.append(("flat25", baseline, candidate))
        return {"accepted": True}

    result = pipeline.evaluate_gates({}, {}, 45_011, normal_gate, flat_gate)
    assert len(calls) == 4
    assert result["normal_student"]["accepted"] is False
    assert result["accepted"] is False


def test_preconditions_require_finalized_v13_zero_update_evidence() -> None:
    valid = {
        "state": "startup_v13_preflight_complete_optimizer_steps_0",
        "functional_sampling_valid": True,
        "startup_sampler_valid": True,
        "reset_targets_valid": True,
        "optimizer_updates": 0,
        "completed_updates": 44_711,
        "control_steps": 3000,
        "num_envs": 4096,
        "checkpoint_sha256": "source",
        "reference_sha256": "reference",
        "coefficient_donor_sha256": "donor",
    }
    artifacts = {
        "checkpoint_sha256": "source",
        "reference_sha256": "reference",
        "coefficient_donor_sha256": "donor",
    }
    pipeline.validate_preflight(valid, artifacts)
    for replacement in (
        {"startup_sampler_valid": False}, {"reset_targets_valid": False},
        {"optimizer_updates": 1}, {"state": "incomplete"},
    ):
        with pytest.raises(AssertionError):
            pipeline.validate_preflight({**valid, **replacement}, artifacts)


def test_evidence_files_are_exclusive_and_rejection_is_not_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    pipeline.write_json_new(existing / "evidence.json", {"state": "first"})
    with pytest.raises(FileExistsError):
        pipeline.write_json_new(existing / "evidence.json", {"state": "second"})
    assert json.loads((existing / "evidence.json").read_text()) == {"state": "first"}


def test_launcher_uses_shared_lock_preflight_gate_and_pinned_isaac_python() -> None:
    launcher = Path(pipeline.__file__).with_name("run_pipeline.sh").read_text(encoding="utf-8")
    assert "full_episode_v12_pipeline.lock" in launcher
    assert "flock -n 9" in launcher
    assert "refused_missing_preflight" in launcher
    assert "FINALIZED.json is required" in launcher
    assert "sleep 20" not in launcher
    assert "nvidia-smi --query-compute-apps=pid" in launcher
    assert "/root/autodl-tmp/conda/envs/isaaclab/bin/python" in launcher
