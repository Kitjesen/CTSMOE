import json

import numpy as np
import pytest

from analyze_hips import analyze_archive, analyze_case


NAMES = ["fl_hip_joint", "fr_hip_joint", "rl_hip_joint", "rr_hip_joint"]
REQUEST_IDS = [1, 0, 3, 2]
DEFAULT = [0.12, -0.12, -0.12, 0.12]


def make_case():
    # Request columns are FR, FL, RR, RL, deliberately unlike physics order.
    requested = np.array([
        [-0.12, 0.12, 0.12, -0.12],
        [-0.50, 0.40, 0.12, -0.12],
        [-0.55, 0.45, 0.12, -0.12],
        [-0.12, 0.12, 0.12, -0.12],
    ])
    target = np.array([
        [0.12, -0.12, -0.12, 0.12],
        [0.369, -0.369, -0.12, 0.12],
        [0.369, -0.369, -0.12, 0.12],
        [0.12, -0.12, -0.12, 0.12],
    ])
    actual = target.copy()
    actual[:, 0] = [0.12, 0.39, 0.40, 0.12]
    actual[:, 1] = [-0.12, -0.38, -0.42, -0.12]
    return {
        "joint_pos": actual,
        "target": target,
        "requested": requested,
        "stair_platform_hull": np.array([False, False, True, True]),
        "calf_contact_peak": np.array([[0, 0], [0, 2], [8, 1], [0, 0]]),
        "torque_peak": np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                                  [9, 10, 11, 12], [1, 2, 3, 4]]),
        "joint_vel": np.array([[0, 0, 0, 0], [0.1, -0.2, 0, 0],
                                [0.2, -0.3, 0, 0], [0, 0, 0, 0]]),
    }


def meta():
    return {"id": 28, "condition": "nominal", "joint_names": NAMES,
            "request_joint_ids": REQUEST_IDS, "default_q": DEFAULT,
            "dt": 0.02, "terrain": "stairs_up", "level": 6}


def test_mirrored_request_target_actual_and_durations():
    result = analyze_case(make_case(), meta())
    front_left = result["hips"]["fl_hip_joint"]
    front_right = result["hips"]["fr_hip_joint"]
    assert front_left["requested_inward"]["peak_rad"] == pytest.approx(0.45)
    assert front_right["requested_inward"]["peak_rad"] == pytest.approx(0.55)
    assert front_right["actual_inward"]["peak_rad"] == pytest.approx(0.42)
    assert front_left["durations"]["requested_beyond_soft_limit"] == {
        "total_s": 0.04, "longest_contiguous_s": 0.04}
    assert front_left["durations"]["actual_beyond_soft_limit"]["total_s"] == pytest.approx(0.04)
    assert front_right["durations"]["actual_near_hard_limit_within_001_rad"]["total_s"] == 0.02
    four = result["four_hip_request_excess_above_soft_limit"]
    assert four["mean_rad_per_step"] == pytest.approx(((0.40 - 0.369 + 0.50 - 0.369)
                                                       + (0.45 - 0.369 + 0.55 - 0.369)) / 4)
    assert four["any_excess_duration_s"] == 0.04
    assert front_left["at_actual_inward_peak"]["calf_terrain_contact_peak_N"] == 8.0
    assert front_left["stage_summary"]["before_platform"]["samples"] == 2
    assert front_left["stage_summary"]["on_or_after_platform"]["samples"] == 2
    assert front_left["torque_peak_at_control_samples"]["mean"] == 4.0
    assert front_left["limited_target_minus_actual_inward"]["mean_rad"] < 0
    assert front_right["signed_inward_joint_velocity_rad_s"]["p95"] > 0
    assert front_left["at_actual_inward_peak"]["self_collision_force_available"] is False


def test_prefixed_archive_and_batched_archive_agree(tmp_path):
    case = make_case()
    prefixed = tmp_path / "prefixed.npz"
    prefix = "nominal_28__"
    np.savez_compressed(prefixed, **{prefix + key: value for key, value in case.items()})
    selected_meta = tmp_path / "selected.json"
    selected_meta.write_text(json.dumps([meta()]), encoding="utf-8")
    selected = analyze_archive(prefixed, selected_meta)

    batched = tmp_path / "batched.npz"
    data = {key: value[:, None, ...] for key, value in case.items()}
    data["active"] = np.ones((4, 1), dtype=bool)
    data["case_ids"] = np.array([28])
    np.savez_compressed(batched, **data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({**meta(), "cases": [{"id": 28, "terrain": "stairs_up"}]}), encoding="utf-8")
    batched_result = analyze_archive(batched, manifest)
    assert selected["cases"] == batched_result["cases"]
    assert selected["input"]["trace_sha256"] != batched_result["input"]["trace_sha256"]


def test_rejects_nonprefix_active_or_incorrect_joint_order(tmp_path):
    case = make_case()
    broken = tmp_path / "broken.npz"
    data = {key: value[:, None, ...] for key, value in case.items()}
    data["active"] = np.array([[True], [False], [True], [False]])
    np.savez_compressed(broken, **data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({**meta(), "cases": [{"id": 0}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        analyze_archive(broken, manifest)
    bad_mapping = {**meta(), "request_joint_ids": [1, 1, 3, 2]}
    with pytest.raises(ValueError, match="duplicates"):
        analyze_case(case, bad_mapping)
