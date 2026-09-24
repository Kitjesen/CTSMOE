"""CPU contract tests; no Isaac, GPU, SSH or training is invoked."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from behavior_review import pulse_windows
from contract import (PROFILE, V23, digest, source_options, training_args, validate_review)
from driver import verify_assessment_identity, verify_signed_mixed
from evaluate_signed_mixed import build_cases
from preflight import validate_contact_provider


def test_explicit_review_and_exact_source_identity() -> None:
    options = {"v24_original": {"path": "/checked/model.pt", "sha256": "a" * 64,
                                "completed_updates": 34823}}
    review = dict(decision="eligible_for_bounded_mixed_stair_test",
                  selected_source="v24_original", source_path="/checked/model.pt",
                  source_sha256="a" * 64, source_completed_updates=34823,
                  v24_assessment_proof_sha256="b" * 64,
                  v24_behavior_comparison_sha256="c" * 64,
                  behavior_findings="Reviewed paired hips, posture, 15 cm route and dwell, all nine terrain types, torque, speed, and physical stops.",
                  real_robot_release=False)
    assert validate_review(review, options, "b" * 64, "c" * 64) == options["v24_original"]
    for key, wrong in (("decision", "HOLD"), ("source_sha256", "d" * 64),
                       ("v24_behavior_comparison_sha256", "d" * 64),
                       ("real_robot_release", True)):
        with pytest.raises(ValueError):
            validate_review({**review, key: wrong}, options, "b" * 64, "c" * 64)


def test_source_age_and_finite_mixed_command() -> None:
    frozen = {"source": {"path": "/original", "sha256": "a", "completed_updates": 34823}}
    status = {"candidate_checkpoint": "/candidate", "completed_updates": 35823}
    proof = {"source_sha256": "a", "candidate_sha256": "b"}
    assert source_options(frozen, status, proof)["v24_hip_candidate"]["completed_updates"] == 35823
    with pytest.raises(ValueError):
        source_options(frozen, {**status, "completed_updates": 35824}, proof)
    argv = training_args(Path("/candidate"), Path("/out"), 484,
                         "inward_unreachable_request_v1")
    assert argv[argv.index("--mixed-stair-profile") + 1] == PROFILE
    assert argv[argv.index("--hip-request-profile") + 1] == "inward_unreachable_request_v1"
    assert argv[argv.index("--updates") + 1] == "484"
    with pytest.raises(ValueError):
        training_args(Path("/candidate"), Path("/out"), 500, "off")


def test_signed_cases_are_pairwise_identical_except_policy() -> None:
    selection = [{"teacher": flag} for flag in (False, True, False, True)]
    cases = build_cases(selection)
    assert len(cases) == 128
    for offset in (32, 64, 96):
        for source, other in zip(cases[:32], cases[offset:offset + 32]):
            assert (source["terrain"], source["level"], source["command_group"],
                    source["pulse_sign"], source["speed_mps"]) == (
                        other["terrain"], other["level"], other["command_group"],
                        other["pulse_sign"], other["speed_mps"])


def test_live_contact_provider_is_mandatory(monkeypatch) -> None:
    mask = torch.tensor([True, True, False])
    command = SimpleNamespace(mixed_terminal_mask=lambda: mask,
                              mixed_stair_course=SimpleNamespace(
                                  active=mask, group=torch.tensor([1, 2, 0])))
    raw = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    provider = ModuleType("stair_platform_witness")
    witness = dict(kind="conservative_center_hull", validated_collision_geometry=True,
                   collision_geometry_id="urdf:checked", wheel_centers_world_xyz=torch.zeros(2, 4, 3),
                   terrain_forces_n=torch.ones(2, 4), platform_world_z_m=torch.zeros(2),
                   body_collision_hull_world_xy=torch.zeros(2, 8, 2))
    provider.conservative_contact_provider = lambda _raw, _ids: witness
    monkeypatch.setitem(sys.modules, "stair_platform_witness", provider)
    assert validate_contact_provider(raw, torch)["positive_terrain_force_samples"] == 8
    witness["validated_collision_geometry"] = False
    with pytest.raises(RuntimeError, match="validated geometry witness"):
        validate_contact_provider(raw, torch)


def test_signed_pulse_requires_both_phases_to_respond() -> None:
    case = {"end_step": 209, "command_group": "forward_yaw",
            "pulse_sign": 1, "speed_mps": .45}
    commands = np.zeros((210, 3))
    commands[:, 0] = .45
    actual = np.zeros((210, 3))
    angular = np.zeros((210, 3))
    commands[100:140, 2] = .12
    commands[140:180, 2] = -.12
    angular[:, 2] = commands[:, 2] * .6
    assert pulse_windows(case, commands, actual, angular, .02)["both_phases_followed"]
    angular[:, 2] = 0
    assert not pulse_windows(case, commands, actual, angular, .02)["both_phases_followed"]
    angular[:, 2] = commands[:, 2] * .3
    assert not pulse_windows(case, commands, actual, angular, .02)["both_phases_followed"]
    commands[140:180, 2] = .12
    with pytest.raises(ValueError, match="fixed case schedule"):
        pulse_windows(case, commands, actual, angular, .02)
    commands[140:180, 2] = -.12
    with pytest.raises(ValueError, match="fixed case schedule"):
        pulse_windows({**case, "pulse_sign": -1}, commands, actual, angular, .02)


def test_lateral_pulse_checks_negative_first_phase() -> None:
    case = {"end_step": 209, "command_group": "forward_lateral",
            "pulse_sign": -1, "speed_mps": .65}
    commands = np.zeros((210, 3))
    commands[:, 0] = .65
    commands[100:150, 1] = -.08
    commands[150:200, 1] = .08
    actual = commands * .6
    assert pulse_windows(case, commands, actual, np.zeros_like(actual), .02)[
        "both_phases_followed"]
    commands[100:150, 1] = .08
    with pytest.raises(ValueError, match="fixed case schedule"):
        pulse_windows(case, commands, actual, np.zeros_like(actual), .02)


def test_assessment_identity_binds_candidate_hash_and_teacher_labels(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    candidate = tmp_path / "candidate.pt"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    candidate_sha = digest(candidate)
    frozen = {"source": {"path": str(source), "sha256": digest(source),
                         "completed_updates": 34823}}
    sources = [{"path": str(source), "completed_updates": 34823},
               {"path": str(candidate), "completed_updates": 35323}]
    selection = [dict(path=item["path"], label=label + ("_teacher" if teacher else ""),
                      teacher=teacher)
                 for item, label in zip(sources, ("source_mixed", "candidate_mixed"))
                 for teacher in (False, True)]
    (tmp_path / "protocol.json").write_text(json.dumps({"sources": sources}))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "HASHES_BEFORE.json").write_text(json.dumps({
        str(source): digest(source), str(candidate): candidate_sha}))
    signed = tmp_path / "signed_mixed_stairs"
    signed.mkdir()
    (signed / "manifest.json").write_text(json.dumps({"history": selection}))
    verify_assessment_identity(tmp_path, frozen, candidate, 35323, candidate_sha, signed=True)
    (signed / "manifest.json").write_text(json.dumps({
        "history": [selection[1], selection[0], *selection[2:]]}))
    with pytest.raises(ValueError, match="history"):
        verify_assessment_identity(tmp_path, frozen, candidate, 35323, candidate_sha, signed=True)
    candidate.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        verify_assessment_identity(tmp_path, frozen, candidate, 35323, candidate_sha)


def test_signed_evaluation_requires_every_pair_and_four_wheel_witness(tmp_path: Path) -> None:
    folder = tmp_path / "signed_mixed_stairs"
    folder.mkdir()
    labels = ("source_mixed", "source_mixed_teacher", "candidate_mixed",
              "candidate_mixed_teacher")
    cases = build_cases([{"teacher": flag} for flag in (False, True, False, True)])
    for case in cases:
        case.update(end_step=4, terms=[], end_reason="time_budget", passed_4m=False,
                    straight_4m=False, stable=False, platform_hold_s=0.)
    names = [f"{side}_{joint}_joint" for side in ("fr", "fl", "rr", "rl")
             for joint in ("hip", "thigh", "calf")]
    manifest = dict(protocol_id="signed_mixed_stairs", optimizer_steps=0,
                    training_updates=0, smoke_only=False, dt=.02,
                    history=[{"label": label} for label in labels], cases=cases,
                    joint_names=names, request_joint_ids=list(range(12)),
                    default_q=[0.] * 12, hip_contract={"profile": "hip_default012_v23"},
                    local_platform_diagnostic=dict(
                        kind="conservative_center_hull", inferred_not_measured_contact_xy=True,
                        collision_geometry_id="urdf:checked", required_wheels=4, dwell_s=2.))
    joint = np.zeros((5, 128, 12))
    vector = np.zeros((5, 128, 3))
    np.savez_compressed(folder / "traces.npz", joint_pos=joint, target=joint,
                        requested=joint, torque_peak=joint, speed_peak=joint,
                        acceleration_peak=joint, velocity=vector, angular_velocity=vector,
                        command=vector, calf_contact_peak=np.zeros((5, 128, 4)))
    (folder / "manifest.json").write_text(json.dumps(manifest))
    metrics_script = (Path(__file__).resolve().parent.parent
                      / "reference/assessment/hip_metrics.py")
    if not metrics_script.is_file():
        # The local receipt contains a frozen copy; the server keeps V23 at
        # its canonical path. Exercise the same extractor in either layout.
        metrics_script = V23 / "assessment/hip_metrics.py"
    assert metrics_script.is_file()
    subprocess.run([sys.executable, str(metrics_script), str(folder)],
                   check=True, capture_output=True)
    assert verify_signed_mixed(folder, "urdf:checked")["fixed_cases"] == 128
    manifest["cases"][0]["pulse_sign"] = 1
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Both signed pulse directions"):
        verify_signed_mixed(folder, "urdf:checked")
