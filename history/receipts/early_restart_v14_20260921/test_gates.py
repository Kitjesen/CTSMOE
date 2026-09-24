import importlib.util
from collections import Counter
from pathlib import Path


spec = importlib.util.spec_from_file_location("early_v14_gates", Path(__file__).with_name("gates.py"))
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


DIRECTIONS = (
    "forward", "reverse", "left", "right", "yaw_left", "yaw_right",
    "diagonal", "diagonal_back", "forward_turn", "side_turn",
)


def case(label, teacher, case_id, terrain="flat", level=0, speed=.6,
         mode="direct", direction="forward"):
    return {
        "id": case_id, "label": label, "teacher": teacher, "terrain": terrain,
        "level": level, "velocity": [speed, 0.0, 0.0], "mode": mode,
        "direction": direction, "physical": False, "pass": True,
        "pose_pass": True, "tracking_pass": True, "completed_25s": True,
        "first_peak_rms": .20, "first_peak_joint_rad": .45, "max_rms": .28,
        "height_min": .40, "height_max": .52, "last_xy_error": .10,
        "last_yaw_error": .10, "leg_torque_peak_nm": 80.0,
        "wheel_torque_peak_nm": 14.0, "hard_position_excess_rad": 0.0,
        "straight_4m": True, "max_side_m": .20,
        "end_vz_abs_peak": .20,
    }


def policy_cases(label, teacher, suite):
    rows = []
    for mode in ("direct", "settle"):
        for direction in DIRECTIONS:
            rows.append(case(label, teacher, len(rows), mode=mode, direction=direction))
    if suite == "normal":
        for terrain in ("stairs_up", "stairs_down", "slope_up", "slope_down",
                        "wave", "rough_slope", "obstacles"):
            for level in (6, 9):
                for speed in (.3, .6, 1.2):
                    rows.append(case(label, teacher, len(rows), terrain, level, speed))
        for level in (0, 2, 4, 6, 9, 15, 21):
            for speed in (.3, .6, .9):
                rows.append(case(label, teacher, len(rows), "pit", level, speed))
    return rows


def summary(rows):
    grouped = Counter((row["label"], row["terrain"]) for row in rows)
    return {"forward": [], "pit": [], "flat_failures": [], "terrains": [
        {"label": label, "terrain": terrain, "count": count,
         "passed": count, "physical": 0, "straight": count}
        for (label, terrain), count in grouped.items()
    ]}


def joints(rows):
    return [
        {"label": row["label"], "case": row["id"], "joint": f"leg_{joint}_joint",
         "target_step_peak_rad": .20}
        for row in rows for joint in range(12)
    ]


def evidence():
    baseline_normal, baseline_flat, candidate_normal, candidate_flat = [], [], [], []
    pairs = [("source", "candidate"), ("source_teacher", "candidate_teacher")]
    for teacher, (source, candidate) in zip((False, True), pairs):
        old_normal = policy_cases(source, teacher, "normal")
        old_flat = policy_cases(source, teacher, "flat")
        new_normal = policy_cases(candidate, teacher, "normal")
        new_flat = policy_cases(candidate, teacher, "flat")
        for collection in (old_normal, new_normal):
            offset = len(baseline_normal) if collection is old_normal else len(candidate_normal)
            for row in collection:
                row["id"] += offset
        for collection in (old_flat, new_flat):
            offset = len(baseline_flat) if collection is old_flat else len(candidate_flat)
            for row in collection:
                row["id"] += offset
        baseline_normal += old_normal
        baseline_flat += old_flat
        candidate_normal += new_normal
        candidate_flat += new_flat
    return {
        "baseline_normal": {"cases": baseline_normal},
        "baseline_flat": {"cases": baseline_flat},
        "candidate_normal": {"cases": candidate_normal},
        "candidate_flat": {"cases": candidate_flat},
        "baseline_normal_summary": summary(baseline_normal),
        "baseline_flat_summary": summary(baseline_flat),
        "candidate_normal_summary": summary(candidate_normal),
        "candidate_flat_summary": summary(candidate_flat),
        "candidate_normal_joints": joints(candidate_normal),
        "candidate_flat_joints": joints(candidate_flat),
        "pairs": pairs,
        "stage": 300,
    }


def run(values):
    return gates.continuation_gate(**values)


def candidate_case(values, suite, label="candidate", **match):
    return next(row for row in values[f"candidate_{suite}"]["cases"]
                if row["label"] == label and all(row[key] == value for key, value in match.items()))


def baseline_case(values, suite, label="source", **match):
    return next(row for row in values[f"baseline_{suite}"]["cases"]
                if row["label"] == label and all(row[key] == value for key, value in match.items()))


def test_clean_300_gate_allows_only_700_more_and_1000_stops():
    values = evidence()
    result = run(values)
    assert result["accepted"] and result["action"] == "continue_700"
    assert len(result["gates"]) == 4 and all(gate["passed"] for gate in result["gates"])
    assert result["next_updates"] == 700 and not result["stop"]
    values["stage"] = 1000
    result = run(values)
    assert result["accepted"] and result["action"] == "stop_at_1000"
    assert result["next_updates"] == 0 and result["stop"]


def test_missing_data_and_teacher_coverage_are_rejected_with_both_labels():
    values = evidence()
    candidate_case(values, "flat").pop("height_min")
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] == "missing_flat_metric" for reason in result["reasons"])
    assert all("source_label" in reason and "candidate_label" in reason for reason in result["reasons"])

    values = evidence()
    for key in ("candidate_normal", "candidate_flat"):
        values[key]["cases"] = [row for row in values[key]["cases"]
                                  if row["label"] != "candidate_teacher"]
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] in {"candidate_case_count", "learner_coverage"}
               for reason in result["reasons"])


def test_single_joint_excursion_is_rejected():
    values = evidence()
    candidate_case(values, "flat", direction="left")["first_peak_joint_rad"] = .60001
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] == "absolute_first_peak_joint_rad" for reason in result["reasons"])


def test_new_speed_related_physical_failure_is_rejected():
    values = evidence()
    row = candidate_case(values, "normal", terrain="slope_up", level=9,
                         velocity=[1.2, 0.0, 0.0])
    row["physical"] = True
    row["violation_causes"] = {"speed": True, "position": False, "torque": False}
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] == "new_physical_failure" for reason in result["reasons"])


def test_previously_successful_high_platform_case_cannot_regress():
    values = evidence()
    row = candidate_case(values, "normal", terrain="pit", level=9,
                         velocity=[.6, 0.0, 0.0])
    row["pass"] = False
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] == "lost_skill_case" for reason in result["reasons"])


def test_previously_failed_pit_1p2_case_need_not_become_successful():
    values = evidence()
    before = baseline_case(values, "normal", terrain="pit", level=21,
                           velocity=[.9, 0.0, 0.0])
    after = candidate_case(values, "normal", terrain="pit", level=21,
                          velocity=[.9, 0.0, 0.0])
    before["pass"] = after["pass"] = False
    before["straight_4m"] = after["straight_4m"] = False
    assert run(values)["accepted"]


def test_missing_joint_delta_and_stair_vz_or_safety_excess_are_rejected():
    values = evidence()
    row = candidate_case(values, "normal", terrain="stairs_up", level=6,
                         velocity=[.3, 0.0, 0.0])
    row.pop("end_vz_abs_peak")
    values["candidate_normal_joints"] = [joint for joint in values["candidate_normal_joints"]
                                           if not (joint["label"] == row["label"]
                                                   and joint["case"] == row["id"]
                                                   and joint["joint"] == "leg_11_joint")]
    candidate_case(values, "flat", direction="right")["hard_position_excess_rad"] = .00101
    codes = {reason["code"] for reason in run(values)["reasons"]}
    assert {"missing_stair_vz", "missing_target_step", "hard_position_limit"} <= codes


def test_case_level_target_delta_is_sufficient_without_joint_assessment():
    values = evidence()
    for suite in ("normal", "flat"):
        for row in values[f"candidate_{suite}"]["cases"]:
            row["target_step_peak_rad"] = .2401
        values[f"candidate_{suite}_joints"] = None
    assert run(values)["accepted"]


def test_newly_successful_longer_case_is_not_rejected_for_relative_quality_only():
    values = evidence()
    before = baseline_case(values, "normal", terrain="stairs_up", level=9,
                           velocity=[1.2, 0.0, 0.0])
    after = candidate_case(values, "normal", terrain="stairs_up", level=9,
                          velocity=[1.2, 0.0, 0.0])
    before["pass"] = False
    before["straight_4m"] = False
    after["pass"] = True
    after["max_side_m"] = .4
    after["end_vz_abs_peak"] = .4
    result = run(values)
    assert result["accepted"]
    assert any(item["code"] == "quality_comparison_skipped_new_success"
               for item in result["observations"])

    after["physical"] = True
    result = run(values)
    assert not result["accepted"]
    assert any(reason["code"] == "new_physical_failure" for reason in result["reasons"])
