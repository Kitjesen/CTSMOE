"""CPU tests of V24 factor isolation, budget, and trajectory accounting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import contract
from behavior_review import compare, summarize_suite
from prepare import validate_review
from qualification import validate_mixed_initial


def test_hip_stage_never_trains_mixed_stairs(tmp_path):
    args = contract.command(Path("/source.pt"), tmp_path / "out", 16)
    assert args[args.index("--hip-request-profile") + 1] == "inward_unreachable_request_v1"
    assert args[args.index("--mixed-stair-profile") + 1] == "off"
    assert args[args.index("--updates") + 1] == "16"
    assert args[args.index("--num-envs") + 1] == "4096"
    assert args[args.index("--stair-command-profile") + 1] == "straight_speed_v1"
    assert contract.command(Path("/source.pt"), tmp_path / "diag", 0, mixed=True)[
        args.index("--mixed-stair-profile") + 1] == "mixed_stair_short_pulses_v1"
    with pytest.raises(ValueError, match="zero-update"):
        contract.command(Path("/source.pt"), tmp_path / "forbidden", 16, mixed=True)
    with pytest.raises(ValueError, match="Only zero-update"):
        contract.command(Path("/source.pt"), tmp_path / "forbidden", 1001)


@pytest.mark.parametrize("source", [34823, 35721])
def test_global_budget_keeps_discarded_branch_spent(tmp_path, monkeypatch, source):
    monkeypatch.setattr(contract, "ORCH", tmp_path)
    contract.write_status("start", source, source)
    assert json.loads((tmp_path / "pipeline_status.json").read_text())[
        "remaining_budget_updates"] == 6000
    contract.write_status("end", source, source + 1000)
    result = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert result["remaining_budget_updates"] == 5000
    assert result["automatic_mixed_training"] is False
    with pytest.raises(ValueError, match="boundary"):
        contract.write_status("over", source, source + 1001)


def test_source_review_requires_explicit_choice_and_evidence(tmp_path):
    source, evidence = tmp_path / "model.pt", tmp_path / "FINALIZED.json"
    source.write_bytes(b"checkpoint")
    evidence.write_text("{}")
    payload = dict(decision="eligible_for_isolated_hip_request_test",
                   source_path=str(source), source_sha256=contract.digest(source),
                   source_completed_updates=34823,
                   v23_evaluation_path=str(evidence),
                   v23_evaluation_sha256=contract.digest(evidence),
                   behavior_findings="The paired fixed cases were inspected with hip pose and stair survival.",
                   real_robot_release=False)
    validate_review(payload, source, contract.digest(source), 34823, evidence)
    payload["decision"] = "PENDING"
    with pytest.raises(ValueError, match="review"):
        validate_review(payload, source, contract.digest(source), 34823, evidence)


@pytest.mark.parametrize("changed", ["review", "evaluation"])
def test_frozen_review_and_evaluation_reject_later_changes(tmp_path, monkeypatch, changed):
    orchestration = tmp_path / "orchestration"
    evaluation = tmp_path / "v23/evaluation_35721/FINALIZED.json"
    orchestration.mkdir()
    evaluation.parent.mkdir(parents=True)
    review = orchestration / "SOURCE_REVIEW.json"
    review.write_text('{"decision":"reviewed"}', encoding="utf-8")
    final = dict(state="complete", optimizer_updates=0, hashes_unchanged=True,
                 smoke_only=False, policy_count=4)
    evaluation.write_text(json.dumps(final), encoding="utf-8")
    monkeypatch.setattr(contract, "ORCH", orchestration)
    monkeypatch.setattr(contract, "V23", tmp_path / "v23")
    frozen = dict(source_review_sha256=contract.digest(review),
                  v23_evaluation=dict(path=str(evaluation), sha256=contract.digest(evaluation)))
    contract.verify_review_evidence(frozen)
    target = review if changed == "review" else evaluation
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        contract.verify_review_evidence(frozen)


def test_frozen_evaluation_rejects_incomplete_result(tmp_path, monkeypatch):
    orchestration = tmp_path / "orchestration"
    evaluation = tmp_path / "v23/evaluation_35721/FINALIZED.json"
    orchestration.mkdir()
    evaluation.parent.mkdir(parents=True)
    review = orchestration / "SOURCE_REVIEW.json"
    review.write_text("{}", encoding="utf-8")
    evaluation.write_text(json.dumps(dict(state="complete", optimizer_updates=1,
                                          hashes_unchanged=True, smoke_only=False,
                                          policy_count=4)), encoding="utf-8")
    monkeypatch.setattr(contract, "ORCH", orchestration)
    monkeypatch.setattr(contract, "V23", tmp_path / "v23")
    frozen = dict(source_review_sha256=contract.digest(review),
                  v23_evaluation=dict(path=str(evaluation), sha256=contract.digest(evaluation)))
    with pytest.raises(ValueError, match="incomplete"):
        contract.verify_review_evidence(frozen)


def test_candidate_scope_is_single_existing_file_plus_opt_in_modules(monkeypatch):
    receipts = Path(__file__).resolve().parents[2]
    local_receipt = receipts / "hip_default_v23_20260923/candidate"
    # The local receipt stores a copied candidate; the server stores V23 at
    # the canonical absolute path from the orchestration contract.
    reviewed_v23 = local_receipt if (local_receipt / "PREPARED.json").is_file() else contract.V23
    assert (reviewed_v23 / "PREPARED.json").is_file()
    monkeypatch.setattr(contract, "V23", reviewed_v23)
    candidate = contract.files_snapshot(Path(__file__).resolve().parents[1] / "code")
    contract.verify_v24_scope(candidate)
    with pytest.raises(ValueError, match="scope"):
        contract.verify_v24_scope({**candidate, "unreviewed.py": "0" * 64})


def test_mixed_diagnostic_rejects_missing_first_reset_group():
    course = SimpleNamespace(active=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
                             group=torch.tensor([0, 1, 2, 0]))
    command = SimpleNamespace(mixed_stair_course=course,
                              learner=torch.tensor([0, 0, 1, 1]),
                              mixed_terminal_mask=lambda: torch.tensor([1, 1, 1, 0], dtype=torch.bool))
    raw = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    result = validate_mixed_initial(raw, torch)
    assert result["selected_first_reset"] == 3
    assert result["groups_first_reset"] == [1, 1, 1]
    course.active[2] = False
    with pytest.raises(RuntimeError, match="First-reset"):
        validate_mixed_initial(raw, torch)


def test_behavior_review_separates_request_target_and_actual(tmp_path):
    names = ["fr_hip_joint", "fl_hip_joint", "rr_hip_joint", "rl_hip_joint"]
    names += [f"leg_{i}_joint" for i in range(8)]
    names += [f"foot_{i}_foot_joint" for i in range(4)]
    labels = ["source_default012", "source_default012_teacher",
              "candidate_hip_request", "candidate_hip_request_teacher"]
    manifest = dict(joint_names=names, request_joint_ids=list(range(16)), dt=.02,
                    history=[dict(label=label) for label in labels],
                    cases=[dict(id=i, policy=i, terrain="flat", level=0, end_step=2,
                                terms=[], passed_4m=True) for i in range(4)])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    shape = (3, 4, 16)
    actual = np.zeros(shape)
    target = np.zeros(shape)
    requested = np.zeros(shape)
    for case in range(4):
        inward = .395 if case < 2 else .25
        req = .5 if case < 2 else .3
        tgt = .369 if case < 2 else .3
        for j, sign in enumerate((-1, 1, 1, -1)):
            actual[:, case, j] = inward * sign
            target[:, case, j] = tgt * sign
            requested[:, case, j] = req * sign
    np.savez_compressed(tmp_path / "traces.npz", actual=actual, joint_pos=actual,
                        target=target, requested=requested, torque_peak=np.zeros(shape),
                        speed_peak=np.zeros(shape), world_velocity=np.zeros((3, 4, 3)))
    rows = [dict(policy=i, terrain="flat", calf_contact_peak_N=0.,
                 startup_pose_rms_peak=0., xy_rmse=0., yaw_rmse=0.) for i in range(4)]
    (tmp_path / "HIP_REVIEW_v2.json").write_text(json.dumps(dict(cases=rows)))
    summary = summarize_suite(tmp_path)
    before = summary["source_default012/flat"]
    after = summary["candidate_hip_request/flat"]
    assert before["raw_excess_rad_seconds"] > 0
    assert before["target_near_soft_seconds"] > 0
    assert before["actual_near_hard_seconds"] > 0
    assert after["raw_excess_rad_seconds"] == 0
    assert after["target_near_soft_seconds"] == 0
    assert after["actual_near_hard_seconds"] == 0
    assert set(before["hip_by_joint"]) == set(names[:4])
    paired = compare({"flat25": summary})
    assert paired["flat25/student/flat"]["actual_inward_p95_rad"]["delta"] < 0
