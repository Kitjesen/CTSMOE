from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import pipeline


def make_pipeline(tmp_path: Path) -> pipeline.Pipeline:
    root, new = tmp_path / "R", tmp_path / "N"
    source = root / pipeline.SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable")
    new.mkdir()
    selection = new / "source_selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "selected",
                "source": str(source.resolve()),
                "completed_updates": pipeline.SOURCE_COMPLETED,
                "source_sha256": pipeline.sha256(source),
            }
        )
    )
    return pipeline.Pipeline(root, new, selection)


def test_source_selection_is_absolute_exact_and_hash_bound(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    subject.verify_source_selection()
    payload = json.loads(subject.source_selection.read_text())
    payload["completed_updates"] = 30000
    subject.source_selection.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="completed_updates"):
        subject.verify_source_selection()


def test_train_command_explicitly_disables_late_profiles(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    command = subject.train_command(tmp_path / "out", subject.source, 16)
    joined = " ".join(command)
    for required in (
        "--early-restart-profile early_phase_v14",
        "--startup-repair-profile off",
        "--pit-posture-profile off",
        "--surface-posture-profile off",
        "--stance-strength-profile off",
        "--allocation-profile balanced_adaptive_v1",
        "--pit-training-profile strict_v1",
        "--wave-reset-profile wave_surface_v1",
    ):
        assert required in joined
    assert "anchor" not in joined
    assert "--fast-delay" not in command


def test_code_hash_change_is_fail_closed(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    code = subject.new_root / "one.py"
    code.write_text("before")
    subject.expected_hashes = {str(code): pipeline.sha256(code)}
    subject.verify_code()
    code.write_text("after")
    with pytest.raises(RuntimeError, match="Frozen code changed"):
        subject.verify_code()


def test_checkpoint_structure_requires_three_finite_nonempty_states(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "completed_updates": 31847,
            "model_state_dict": {"w": torch.ones(2)},
            "optimizer_state_dict": {"state": {0: {"v": torch.zeros(1)}}},
            "optimizer_stu_enc_state_dict": {"state": {0: {"v": torch.zeros(1)}}},
            "terrain_allocation": {
                "profile": "balanced_adaptive_v1",
                "course": {
                    "profile": "early_balanced_v14",
                    "weights": pipeline.EARLY_ALLOCATION,
                },
                "pending": {},
            },
        },
        checkpoint,
    )
    result = pipeline.Pipeline.checkpoint_structure(checkpoint, 31847)
    assert set(result) == {
        "model_state_dict",
        "optimizer_state_dict",
        "optimizer_stu_enc_state_dict",
    }
    saved = torch.load(checkpoint, weights_only=False)
    saved["model_state_dict"]["w"][0] = float("nan")
    torch.save(saved, checkpoint)
    with pytest.raises(RuntimeError, match="Non-finite"):
        pipeline.Pipeline.checkpoint_structure(checkpoint, 31847)


def test_existing_status_is_never_overwritten(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    subject.status_path.write_text('{"state":"complete"}')
    with pytest.raises(FileExistsError):
        subject.run()
    assert json.loads(subject.status_path.read_text()) == {"state": "complete"}


def test_real_preflight_run_receipt_matches_training_contract():
    receipt = json.loads(Path(__file__).with_name("preflight_run.json").read_text())
    pipeline.Pipeline.validate_run_contract(receipt, 0, 31831, 31831)


def test_allocation_snapshot_requires_exact_counts_weights_and_no_pending():
    allocation = {
        "profile": "balanced_adaptive_v1",
        "bounds_profile": "early_balanced_v14",
        "target_weights": dict(pipeline.EARLY_ALLOCATION),
        "actual_counts": dict(pipeline.EARLY_COUNTS),
        "pending_moves": 0,
    }
    pipeline.Pipeline.validate_allocation_snapshot(allocation)
    allocation["actual_counts"]["flat"] -= 1
    with pytest.raises(RuntimeError, match="allocation contract"):
        pipeline.Pipeline.validate_allocation_snapshot(allocation)


def test_continuation_requires_restored_allocation_resume_check():
    receipt = json.loads(Path(__file__).with_name("preflight_run.json").read_text())
    receipt["starting_completed_updates"] = 31847
    receipt["additional_updates"] = 284
    receipt["target_completed_updates"] = 32131
    receipt["resume_checks"]["terrain_allocation"] = False
    with pytest.raises(RuntimeError, match="restore terrain allocation"):
        pipeline.Pipeline.validate_run_contract(receipt, 284, 31847, 32131)


def test_env_contract_requires_exact_five_second_resampling(tmp_path: Path):
    env = tmp_path / "env.yaml"
    env.write_text("commands:\n  base_velocity:\n    resampling_time: 5.0\n")
    pipeline.Pipeline.validate_env_contract(env)
    env.write_text("commands:\n  base_velocity:\n    resampling_time: 2.0\n")
    with pytest.raises(RuntimeError, match="5-second"):
        pipeline.Pipeline.validate_env_contract(env)


def test_gate_schema_requires_four_exact_paired_rows(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    completed = 32131
    rows = [
        {
            "source_label": source,
            "candidate_label": candidate,
            "suite": suite,
            "passed": True,
            "reason_indexes": [],
        }
        for source, candidate in (
            ("early_31831", f"V14_{completed}"),
            ("early_31831_teacher", f"V14_{completed}_teacher"),
        )
        for suite in ("normal", "flat")
    ]
    result = {
        "accepted": True,
        "action": "continue_700",
        "next_updates": 700,
        "stop": False,
        "gates": rows,
    }

    def write_gate(*_args, **_kwargs):
        (subject.new_root / "gate300.json").write_text(json.dumps(result))
        return 0

    subject.command = write_gate
    outputs = {"terrain83": tmp_path / "normal", "flat25": tmp_path / "flat"}
    assert subject.gates(outputs, completed, 300)["accepted"] is True
    result["gates"] = rows[:-1]
    with pytest.raises(RuntimeError, match="four exact"):
        subject.gates(outputs, completed, 300)


def test_gate_exit_one_is_rejection_but_crash_output_is_failed(tmp_path: Path):
    subject = make_pipeline(tmp_path)
    completed = 32131
    rows = [
        {
            "source_label": source,
            "candidate_label": candidate,
            "suite": suite,
            "passed": not (source == "early_31831" and suite == "flat"),
            "reason_indexes": [0]
            if source == "early_31831" and suite == "flat"
            else [],
        }
        for source, candidate in (
            ("early_31831", f"V14_{completed}"),
            ("early_31831_teacher", f"V14_{completed}_teacher"),
        )
        for suite in ("normal", "flat")
    ]
    rejection = {
        "accepted": False,
        "action": "reject",
        "next_updates": 0,
        "stop": True,
        "gates": rows,
        "reasons": [{"code": "regression"}],
    }

    def write_rejection(*_args, **_kwargs):
        (subject.new_root / "gate300.json").write_text(json.dumps(rejection))
        return 1

    subject.command = write_rejection
    outputs = {"terrain83": tmp_path / "normal", "flat25": tmp_path / "flat"}
    assert subject.gates(outputs, completed, 300)["action"] == "reject"
    rejection.pop("reasons")
    with pytest.raises(RuntimeError, match="fail-closed rejection"):
        subject.gates(outputs, completed, 300)


class FakePipeline(pipeline.Pipeline):
    def __init__(self, root, new, selection, gate300, gate1000):
        super().__init__(root, new, selection)
        self.gate_results = {300: gate300, 1000: gate1000}
        self.trained = []

    def verify_preconditions(self):
        self.hash_manifest.write_text("{}")

    def train(self, name, resume, updates, expected_start, expected_completed):
        self.trained.append((name, updates, expected_completed))
        return self.new_root / f"model_{expected_completed - 1}.pt"

    def evaluate(self, checkpoint, completed, stage):
        return {"terrain83": self.new_root / "normal", "flat25": self.new_root / "flat"}

    def gates(self, outputs, completed, stage):
        return self.gate_results[stage]


def test_gate300_reject_stops_without_700_updates(tmp_path: Path):
    initial = make_pipeline(tmp_path)
    subject = FakePipeline(
        initial.root,
        initial.new_root,
        initial.source_selection,
        {"accepted": False, "action": "reject"},
        {},
    )
    subject.run()
    assert [row[1] for row in subject.trained] == [16, 284]
    assert json.loads(subject.status_path.read_text())["state"] == "behavior_rejected"


def test_acceptance_runs_exactly_1000_updates_then_stops(tmp_path: Path):
    initial = make_pipeline(tmp_path)
    subject = FakePipeline(
        initial.root,
        initial.new_root,
        initial.source_selection,
        {"accepted": True, "action": "continue_700"},
        {"accepted": True, "action": "stop_at_1000"},
    )
    subject.run()
    assert [row[1] for row in subject.trained] == [16, 284, 700]
    status = json.loads(subject.status_path.read_text())
    assert status["state"] == "complete" and status["trained_updates"] == 1000
