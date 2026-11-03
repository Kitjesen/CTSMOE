"""CPU checks of the hard gates; importing these modules never starts Isaac."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import contract
from qualification import aligned_resets, reject_outside_before_reset
from route_fixed import fixed_groups_ready, planned_speed
import training_entry
import verify_diagnostics
from verify_diagnostics import verify_route


def source_stub():
    return dict(completed_updates=contract.SOURCE_COMPLETED,
                common_step_counter=contract.SOURCE_COMPLETED * 24,
                mixed_profile="mixed9_v1", allocation_profile="balanced_adaptive_v1",
                stair_command_profile="straight_speed_v1",
                wave_reset_profile="wave_surface_v1",
                terrain_allocation={"pending": []},
                terrain_levels=torch.zeros(contract.NUM_ENVS, dtype=torch.long),
                terrain_types=torch.zeros(contract.NUM_ENVS, dtype=torch.long),
                model_state_dict={}, optimizer_state_dict={},
                optimizer_stu_enc_state_dict={})


@pytest.mark.parametrize("field", ("hip_request_profile", "mixed_stair_profile"))
def test_v24_active_source_is_rejected_before_any_qualification(field):
    saved = source_stub()
    contract.validate_source(saved)
    saved[field] = "active_v24_branch"
    with pytest.raises(ValueError, match="both V24 branches off"):
        contract.validate_source(saved)


def test_only_v21_age_and_settled_allocation_are_eligible():
    saved = source_stub()
    saved["completed_updates"] += 1000
    with pytest.raises(ValueError, match="Only the settled V21"):
        contract.validate_source(saved)
    saved = source_stub()
    saved["terrain_allocation"]["pending"] = [1]
    with pytest.raises(ValueError, match="Only the settled V21"):
        contract.validate_source(saved)


def test_global_budget_keeps_v23_and_both_v24_reservations_spent():
    claim = contract.budget(500)
    assert claim["known_spent_before_v25"] == 4000
    assert claim["prior_spent_upper_bound"] == 5500
    assert claim["remaining_under_conservative_reservation"] == 4000
    assert claim["selecting_older_source_refunds_updates"] is False
    with pytest.raises(ValueError, match="cap"):
        contract.budget(501)
    with pytest.raises(ValueError, match="0..500"):
        contract.training_args(Path("/source.pt"), Path("/output"), 501)
    assert "--composite-route" in contract.training_args(
        Path("/source.pt"), Path("/output"), 0)


def test_fixed_route_plan_has_paired_students_teachers_speeds_and_seeds():
    command = SimpleNamespace(
        composite=SimpleNamespace(active=torch.ones(12, dtype=torch.bool)),
        learner=torch.tensor([0] * 6 + [1] * 6))
    plan = planned_speed(command, torch)
    assert len(plan) == 12
    assert [(row[1], row[2], row[3]) for row in plan[:6]] == [
        (0, 0, .2), (0, 0, .6), (0, 0, 1.),
        (0, 1, .2), (0, 1, .6), (0, 1, 1.)]
    assert [(row[2], row[3]) for row in plan[:6]] == [
        (row[2], row[3]) for row in plan[6:]]
    assert fixed_groups_ready(command)
    command.learner.zero_()
    assert not fixed_groups_ready(command)


def test_route_spawn_check_requires_forward_yaw_and_spawn_origin():
    data = SimpleNamespace(root_pos_w=torch.tensor([[10., 20., 0.]]),
                           root_quat_w=torch.tensor([[1., 0., 0., 0.]]))
    class Scene:
        env_origins = torch.tensor([[10., 20., 0.]])

        def __getitem__(self, _key):
            return SimpleNamespace(data=data)

    raw = SimpleNamespace(scene=Scene())
    ids = torch.tensor((0,))
    assert aligned_resets(raw, ids, torch)
    data.root_quat_w[0] = torch.tensor([.7071068, 0., 0., .7071068])
    assert not aligned_resets(raw, ids, torch)


def test_boundary_guard_sees_exit_before_episode_reset_clears_it():
    route = SimpleNamespace(active=torch.tensor([True, False]),
                            outside=torch.tensor([True, True]), refresh=lambda: None)
    resets = []

    def reset(ids):
        resets.append(ids)
        route.outside[ids] = False

    raw = SimpleNamespace(_reset_idx=reset)
    with reject_outside_before_reset(raw, route) as audit:
        with pytest.raises(RuntimeError, match="before reset"):
            raw._reset_idx(torch.tensor([0]))
        assert resets == []
        raw._reset_idx(torch.tensor([1]))
        assert len(resets) == 1
    assert audit == {"route_resets_checked": 1, "restored": True}
    assert raw._reset_idx is reset


def route_fixture(successes=0):
    cases = []
    for learner in ("student", "teacher"):
        for seed in (0, 1):
            for speed in (0.2, 0.6, 1.0):
                cases.append(dict(id=len(cases), learner=learner, seed=seed,
                                  speed_mps=speed, level=0, end_step=0,
                                  end_reason="time_budget", maximum_dwell_s=0.,
                                  minimum_leg_clearance_m=1., minimum_leg_clearance_step=0,
                                  final_progress_m=0.,
                                  time_feasible=speed > .2,
                                  seen_slope=bool(successes), seen_stair=bool(successes),
                                  full_body_platform=bool(successes),
                                  two_second_success=len(cases) < successes,
                                  physical=False, outside=False))
    traces = dict(command=np.zeros((1, 12, 3)), root_local=np.zeros((1, 12, 3)),
                  wheel_force_z=np.ones((1, 12, 4)) * 10,
                  joint_request=np.zeros((1, 12, 16)), joint_target=np.zeros((1, 12, 16)),
                  joint_actual=np.zeros((1, 12, 16)), joint_torque=np.zeros((1, 12, 16)),
                  joint_speed=np.zeros((1, 12, 16)), raw_action=np.zeros((1, 12, 16)),
                  leg_clearance_m=np.ones((1, 12)), dwell_s=np.zeros((1, 12)))
    traces["command"][0, :, 0] = [case["speed_mps"] for case in cases]
    for name in ("seen_slope", "seen_stair", "full_body_platform", "success",
                 "physical", "outside"):
        traces[name] = np.zeros((1, 12), dtype=bool)
    for index in range(successes):
        for name in ("seen_slope", "seen_stair", "full_body_platform", "success"):
            traces[name][0, index] = True
    manifest = dict(state="complete", source=str(contract.SOURCE),
                    source_sha256=contract.SOURCE_SHA256,
                    source_completed_updates=contract.SOURCE_COMPLETED,
                    optimizer_updates=0, route_level=0, fixed_cases=12,
                    speeds_mps=[.2, .6, 1.], seeds=[0, 1], observed_steps=1,
                    dt_s=.02,
                    paired_initial_state_and_observations=True,
                    state_hashes_before={"model": "0" * 64},
                    state_hashes_after={"model": "0" * 64}, forbidden_calls=[],
                    mutation_guard_restored=True, fixed_cases_only=True,
                    random_success_rate_inferred=False, cases=cases)
    return manifest, traces


def test_all_failed_feasible_l0_cases_are_recorded_but_block_training():
    manifest, traces = route_fixture()
    result = verify_route(manifest, traces)
    assert result["feasible_cases"] == 8
    assert result["feasible_successes"] == 0
    assert result["source_l0_eligible_for_behavior_review"] is False
    manifest["cases"][1]["two_second_success"] = True
    for name in ("seen_slope", "seen_stair", "full_body_platform", "success"):
        manifest["cases"][1][name if name != "success" else "two_second_success"] = True
        traces[name][0, 1] = True
    manifest["cases"][1]["end_reason"] = "success"
    manifest["cases"][1]["maximum_dwell_s"] = 2.
    traces["dwell_s"][0, 1] = 2.
    result = verify_route(manifest, traces)
    assert result["feasible_successes"] == 1
    traces["dwell_s"][0, 1] = .02
    manifest["cases"][1]["maximum_dwell_s"] = .02
    with pytest.raises(ValueError, match="two-second"):
        verify_route(manifest, traces)


def test_written_behavior_review_is_required_after_diagnostics(tmp_path, monkeypatch):
    qualification = tmp_path / "qualification.json"
    diagnostics = tmp_path / "diagnostics.json"
    qualification.write_text("{}")
    diagnostics.write_text("{}")
    monkeypatch.setattr(training_entry, "QUALIFIED", qualification)
    monkeypatch.setattr(training_entry, "DIAGNOSTICS", diagnostics)
    review = dict(decision="eligible_for_bounded_v25_training",
                  source_path=str(contract.SOURCE), source_sha256=contract.SOURCE_SHA256,
                  qualification_sha256=contract.digest(qualification),
                  diagnostics_sha256=contract.digest(diagnostics),
                  route_findings="L0 route observations and full support were reviewed in both learners. " * 2,
                  old_nine_findings="All original terrains, hip posture and physical terminations were reviewed. " * 2,
                  leg_clearance_reviewed=True, prior_global_spent_updates=5500,
                  prior_budget_evidence="Reviewed stage receipts and reserved all V24 caps conservatively.",
                  automatic_training=False, real_robot_release=False)
    training_entry.validate_eligibility(review, dict(route={"feasible_successes": 1}))
    review["decision"] = "HOLD"
    with pytest.raises(ValueError, match="Written"):
        training_entry.validate_eligibility(review, dict(route={"feasible_successes": 1}))
    review["decision"] = "eligible_for_bounded_v25_training"
    with pytest.raises(ValueError, match="Written"):
        training_entry.validate_eligibility(review, dict(route={"feasible_successes": 0}))


def test_template_starts_hold_and_never_auto_trains():
    template = json.loads((Path(__file__).parent / "BEHAVIOR_ELIGIBILITY.template.json").read_text())
    assert template["decision"] == "HOLD"
    assert template["automatic_training"] is False
    assert template["real_robot_release"] is False


def test_old_nine_verifier_binds_fixed_protocol_source_and_both_policies(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    protocol_file = tmp_path / "v21_manifest.json"
    protocol_file.write_text(json.dumps({"protocol_config": {"protocol_id": "fixed",
                                                           "sources": [{"path": "previous"}]}}))
    protocol = {"protocol_id": "fixed", "sources": [dict(
        path=str(contract.SOURCE), completed_updates=contract.SOURCE_COMPLETED)]}
    selection = [dict(path=str(contract.SOURCE), label="source_v21", teacher=False),
                 dict(path=str(contract.SOURCE), label="source_v21_teacher", teacher=True)]
    (old / "FINALIZED.json").write_text(json.dumps(dict(
        state="complete", optimizer_updates=0, hashes_unchanged=True,
        smoke_only=False, policy_count=2)))
    (old / "HASHES_BEFORE.json").write_text(json.dumps(
        {str(contract.SOURCE): contract.SOURCE_SHA256}))
    (old / "protocol.json").write_text(json.dumps(protocol))
    (old / "selection.json").write_text(json.dumps(selection))
    families = sorted(verify_diagnostics.OLD_TYPES)
    for suite, count in verify_diagnostics.SUITES.items():
        folder = old / suite
        folder.mkdir()
        cases = [dict(id=i, policy=i // (count // 2),
                      teacher=i >= count // 2, checkpoint=str(contract.SOURCE),
                      terrain=families[i % len(families)] if suite == "terrain83" else "flat")
                 for i in range(count)]
        manifest = dict(cases=cases, smoke_only=False, optimizer_steps=0,
                        training_updates=0, protocol_config=protocol,
                        checkpoint=str(contract.SOURCE),
                        completed_updates=contract.SOURCE_COMPLETED,
                        stairs15=suite == "stairend_15cm", history=selection)
        (folder / "manifest.json").write_text(json.dumps(manifest))
        (folder / "HIP_REVIEW_v2.json").write_text("{}")
        np.savez_compressed(folder / "traces.npz",
                            body_pos=np.zeros((1, count, 1, 3)),
                            joint_pos=np.zeros((1, count, 16)),
                            command=np.zeros((1, count, 3)))
    monkeypatch.setattr(verify_diagnostics, "OLD", old)
    monkeypatch.setattr(verify_diagnostics, "V21_FIXED_PROTOCOL", protocol_file)
    assert set(verify_diagnostics.verify_old_nine()) == set(verify_diagnostics.SUITES)
    (old / "protocol.json").write_text(json.dumps({**protocol, "protocol_id": "changed"}))
    with pytest.raises(ValueError, match="frozen V21 protocol"):
        verify_diagnostics.verify_old_nine()
    (old / "protocol.json").write_text(json.dumps(protocol))
    terrain_file = old / "terrain83/manifest.json"
    wrong = json.loads(terrain_file.read_text())
    wrong["cases"][0]["checkpoint"] = "/another/model.pt"
    terrain_file.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="Incomplete original"):
        verify_diagnostics.verify_old_nine()
