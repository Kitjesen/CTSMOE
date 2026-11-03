"""Tests for the read-only Thunder collision-envelope diagnostic."""

import hashlib
import json

import mujoco
import numpy as np
import pytest

from probe import (
    BulletProximity,
    DEFAULT_URDF,
    EXPECTED_URDF_SHA256,
    analyze,
    build_model,
    frame_distances,
)


MANIFEST = (
    DEFAULT_URDF.parent.parent
    / "v24_integrated_20260923"
    / "reference"
    / "v23_final_assessment"
    / "stairend_15cm_manifest.json"
)


def test_actual_thunder_urdf_matches_assessment_and_fk_distance() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(DEFAULT_URDF.read_bytes()).hexdigest() == EXPECTED_URDF_SHA256
    assert manifest["urdf_sha256"] == EXPECTED_URDF_SHA256
    model, pairs = build_model(DEFAULT_URDF)
    assert model.nq == 16
    assert model.ngeom == 38
    assert {key: len(value.links) for key, value in pairs.items()} == {
        "inter_leg": 384,
        "leg_body_hip": 20,
        "leg_body_distal": 140,
    }
    neutral = np.asarray(manifest["default_q"])
    inward = neutral.copy()
    inward[:4] = [0.4, -0.4, -0.4, 0.4]
    first = frame_distances(model, pairs, manifest["joint_names"], neutral)
    second = frame_distances(model, pairs, manifest["joint_names"], inward)
    assert first["inter_leg"]["distance_m"] > second["inter_leg"]["distance_m"] > 0
    assert all(np.isfinite(value["distance_m"]) for value in second.values())


def test_bounding_sphere_pruning_matches_exhaustive_geometry_pairs() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model, pairs = build_model(DEFAULT_URDF)
    q = np.asarray(manifest["default_q"])
    q[:4] = [0.37, -0.35, -0.39, 0.32]
    optimized = frame_distances(model, pairs, manifest["joint_names"], q)
    data = mujoco.MjData(model)
    for name, angle in zip(manifest["joint_names"], q, strict=True):
        data.qpos[model.joint(name).qposadr] = angle
    mujoco.mj_forward(model, data)
    proximity = BulletProximity(model)
    try:
        proximity.update(data)
        exhaustive = {
            category: min(
                proximity.distance(int(a), int(b))
                for a, b in zip(
                    pair_index.geometry_a, pair_index.geometry_b, strict=True
                )
            )
            for category, pair_index in pairs.items()
        }
    finally:
        proximity.close()
    assert optimized["inter_leg"]["distance_m"] == pytest.approx(
        exhaustive["inter_leg"], abs=1e-9
    )
    assert optimized["leg_body_distal"]["distance_m"] == pytest.approx(
        exhaustive["leg_body_distal"], abs=1e-9
    )
    assert optimized["leg_body"]["distance_m"] == pytest.approx(
        min(exhaustive["leg_body_hip"], exhaustive["leg_body_distal"]), abs=1e-9
    )


def test_fixed_case_analysis_pairs_only_equal_cases_and_rejects_updates(
    tmp_path,
) -> None:
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    q = np.asarray(source["default_q"])
    trace = np.repeat(q[None, None, :], 4, axis=0)
    trace = np.repeat(trace, 2, axis=1)
    trace[:, 1, :4] = [0.4, -0.4, -0.4, 0.4]
    trace_path = tmp_path / "traces.npz"
    np.savez_compressed(
        trace_path,
        joint_pos=trace,
        initial_q=np.repeat(q[None, :], 2, axis=0),
        initial_root=np.zeros((2, 13)),
    )
    case = {
        "terrain": "stairs_up",
        "level": 9,
        "mode": "direct",
        "direction": "forward",
        "velocity": [0.3, 0, 0],
        "teacher": False,
        "suite": "stairend",
        "landing_mode": "continue",
        "end_step": 2,
    }
    manifest = {
        "zero_update_evaluation": True,
        "initial_state_matched": True,
        "optimizer_steps": 0,
        "training_updates": 0,
        "urdf_sha256": EXPECTED_URDF_SHA256,
        "joint_names": source["joint_names"],
        "default_q": q.tolist(),
        "history": [{"label": "source"}, {"label": "candidate"}],
        "cases": [{**case, "id": 0, "policy": 0}, {**case, "id": 1, "policy": 1}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze(manifest_path, trace_path, DEFAULT_URDF)
    assert result["physx_self_contacts_measured"] is False
    assert len(result["paired"]) == 1
    assert result["cases"][0]["sampled_frames"] == 3
    assert (
        result["cases"][0]["categories"]["inter_leg"]["minimum"]["distance_m"]
        > result["cases"][1]["categories"]["inter_leg"]["minimum"]["distance_m"]
    )
    assert result["paired"][0]["candidate_minus_source_minimum_m"]["inter_leg"] < 0
    manifest["optimizer_steps"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="zero optimizer"):
        analyze(manifest_path, trace_path, DEFAULT_URDF)
