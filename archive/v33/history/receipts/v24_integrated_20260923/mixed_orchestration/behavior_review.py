"""CPU comparison of fixed source/candidate behavior; never releases hardware."""

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from contract import HERE, MIXED_UPDATES, digest, load_json, verify_frozen, write_new
from driver import LABELS, SUITES, verify_all_mode, verify_signed_mixed


def original_suite_summary(folder) -> dict:
    manifest = load_json(folder / "manifest.json")
    report = load_json(folder / "HIP_REVIEW_v2.json")
    labels = {i: entry["label"] for i, entry in enumerate(manifest["history"])}
    rows = defaultdict(list)
    joint_rows = defaultdict(list)
    hip_rows = defaultdict(lambda: defaultdict(list))
    hip_signs = {"fr_hip_joint": -1., "fl_hip_joint": 1.,
                 "rr_hip_joint": 1., "rl_hip_joint": -1.}
    names = manifest["joint_names"]
    request_ids = manifest["request_joint_ids"]
    dt = float(manifest["dt"])
    for case in report["cases"]:
        rows[(labels[case["policy"]], case["terrain"])].append(case)
    for joint in report["joints"]:
        joint_rows[(labels[joint["policy"]], joint["terrain"])].append(joint)
    with np.load(folder / "traces.npz") as data:
        requested, target, actual = (data[name] for name in
                                     ("requested", "target", "joint_pos"))
        for case in manifest["cases"]:
            key = (labels[case["policy"]], case["terrain"])
            count, case_id = case["end_step"] + 1, case["id"]
            for joint, sign in hip_signs.items():
                joint_id = names.index(joint)
                request_id = request_ids.index(joint_id)
                hip_rows[key]["raw"].append(requested[:count, case_id, request_id] * sign)
                hip_rows[key]["target"].append(target[:count, case_id, joint_id] * sign)
                hip_rows[key]["actual"].append(actual[:count, case_id, joint_id] * sign)
    output = {}
    for (label, terrain), cases in sorted(rows.items()):
        joints = joint_rows[(label, terrain)]
        hips = [row for row in joints if row["joint"].endswith("_hip_joint")]
        raw, target, actual = (np.concatenate(hip_rows[(label, terrain)][name])
                               for name in ("raw", "target", "actual"))
        if not all(np.isfinite(value).all() for value in (raw, target, actual)):
            raise ValueError("Nonfinite hip trace in all-mode regression")
        output[f"{label}/{terrain}"] = dict(
            fixed_cases=len(cases), physical_failures=sum(bool(c["physical"]) for c in cases),
            route_4m=sum(bool(c.get("straight_4m")) for c in cases),
            platform_dwell=sum(c.get("end_reason") == "supported_platform_dwell" for c in cases),
            pit_success=sum(bool(c.get("pit_success")) for c in cases),
            hip_actual_inward_peak_rad=max((h.get("actual_inward_peak", 0.) for h in hips),
                                            default=None),
            hip_raw_inward_p95_rad=float(np.quantile(raw, .95)),
            hip_target_inward_p95_rad=float(np.quantile(target, .95)),
            hip_actual_inward_p95_rad=float(np.quantile(actual, .95)),
            hip_raw_excess_rad_seconds=float(np.maximum(raw - .369, 0.).sum() * dt),
            hip_target_near_soft_seconds=float((target >= .36).sum() * dt),
            hip_actual_near_hard_seconds=float((actual >= .39).sum() * dt),
            leg_torque_peak_Nm=max(j["torque_peak_Nm"] for j in joints),
            leg_speed_peak_rad_s=max(j["speed_peak_rad_s"] for j in joints),
            startup_pose_rms_peak_mean_rad=float(np.mean([
                c["startup_pose_rms_peak"] for c in cases])),
            xy_tracking_rmse_mean_mps=float(np.mean([c["xy_rmse"] for c in cases])),
        )
    return output


def pulse_windows(case: dict, commands: np.ndarray, actual: np.ndarray,
                  angular: np.ndarray, dt: float) -> dict:
    count = case["end_step"] + 1
    time = np.arange(count) * dt
    group = case["command_group"]
    if group not in ("forward_yaw", "forward_lateral") or case["pulse_sign"] not in (-1, 1):
        raise ValueError("Unknown signed pulse case")
    axis = 2 if group == "forward_yaw" else 1
    actual_axis = angular[:count, 2] if axis == 2 else actual[:count, 1]
    split = 2.8 if axis == 2 else 3.0
    end = 3.6 if axis == 2 else 4.0
    amplitude = 0.12 if axis == 2 else 0.08
    rows = []
    for phase, (left, right) in enumerate(((2.0, split), (split, end))):
        mask = (time >= left - 1e-8) & (time < right - 1e-8)
        requested = commands[:count, axis][mask]
        observed = actual_axis[mask]
        expected_samples = round((right - left) / dt)
        if (len(requested) != expected_samples
                or not np.isfinite(requested).all() or not np.isfinite(observed).all()):
            rows.append(dict(complete=False, samples=len(requested),
                             signed_response_mean=None, mean_abs_error=None, followed=False))
            continue
        expected = case["pulse_sign"] * amplitude * (1 if phase == 0 else -1)
        other_axis = 1 if axis == 2 else 2
        if (not np.allclose(requested, expected, atol=1e-6, rtol=0)
                or not np.allclose(commands[:count, 0][mask], case["speed_mps"], atol=1e-6, rtol=0)
                or not np.allclose(commands[:count, other_axis][mask], 0., atol=1e-6, rtol=0)):
            raise ValueError("Signed pulse commands differ from the fixed case schedule")
        orientation = np.sign(expected)
        response = float((orientation * observed).mean())
        error = float(np.abs(observed - requested).mean())
        followed = response >= 0.40 * amplitude and error <= 0.75 * amplitude
        rows.append(dict(complete=True, samples=len(requested),
                         issued_mean=float(requested.mean()),
                         signed_response_mean=response, mean_abs_error=error,
                         followed=followed))
    return {"first": rows[0], "second": rows[1],
            "both_phases_followed": all(item["followed"] for item in rows)}


def signed_summary(folder) -> dict:
    manifest = load_json(folder / "manifest.json")
    cases = manifest["cases"]
    dt = float(manifest["dt"])
    if abs(dt - .02) > 1e-10:
        raise ValueError("Signed pulse control period differs")
    with np.load(folder / "traces.npz") as data:
        command = data["command"]
        velocity = data["velocity"]
        angular = data["angular_velocity"]
        force = data["wheel_ground_force_z"]
        support = data["four_wheel_platform_support"]
        torque = data["torque_peak"]
        speed = data["speed_peak"]
        hip = data["joint_pos"]
        target = data["target"]
        requested = data["requested"]
    labels = {i: entry["label"] for i, entry in enumerate(manifest["history"])}
    hip_signs = {"fr_hip_joint": -1., "fl_hip_joint": 1.,
                 "rr_hip_joint": 1., "rl_hip_joint": -1.}
    hip_indices = [(manifest["joint_names"].index(name), sign)
                   for name, sign in hip_signs.items()]
    request_indices = [manifest["request_joint_ids"].index(index)
                       for index, _ in hip_indices]
    rows = defaultdict(list)
    for case in cases:
        i, count = case["id"], case["end_step"] + 1
        if not all(np.isfinite(value[:count, i]).all() for value in (
                command, velocity, angular, force, torque, speed, hip, target,
                requested)):
            raise ValueError("Signed case has nonfinite physical trajectory")
        actual_hip = np.concatenate([hip[:count, i, index] * sign
                                     for index, sign in hip_indices])
        target_hip = np.concatenate([target[:count, i, index] * sign
                                     for index, sign in hip_indices])
        raw_hip = np.concatenate([requested[:count, i, request_id] * sign
                                  for request_id, (_, sign) in zip(request_indices, hip_indices)])
        response = pulse_windows(case, command[:, i], velocity[:, i], angular[:, i], dt)
        if not case["four_wheel_dwell_2s"] and case["end_reason"] == "supported_platform_dwell":
            raise ValueError("Platform completion was claimed without four-wheel dwell")
        if case["four_wheel_dwell_2s"] and (case["physical"] or
                support[:count, i].astype(int)[-100:].sum() != 100):
            raise ValueError("Claimed dwell lacks 100 consecutive physical support samples")
        key = (labels[case["policy"]], case["terrain"], case["level"],
               case["command_group"], case["pulse_sign"])
        rows[key].append(dict(
            case_id=i, speed_mps=case["speed_mps"], response=response,
            physical_failure=bool(case["physical"] or case["actuator_or_base_contact"]),
            violation_causes=case["violation_causes"],
            four_wheel_dwell_2s=bool(case["four_wheel_dwell_2s"]),
            platform_hold_s=case["platform_hold_s"],
            termination=case["end_reason"],
            wheel_force_peak_N=float(force[:count, i].max()),
            joint_torque_peak_Nm=float(torque[:count, i].max()),
            joint_speed_peak_rad_s=float(speed[:count, i].max()),
            hip_raw_inward_p95_rad=float(np.quantile(raw_hip, .95)),
            hip_target_inward_p95_rad=float(np.quantile(target_hip, .95)),
            hip_actual_inward_p95_rad=float(np.quantile(actual_hip, .95)),
            hip_actual_near_hard_seconds=float((actual_hip >= .39).sum() * dt),
        ))
    return {"/".join(map(str, key)): value for key, value in sorted(rows.items())}


def paired_all_mode(suites: dict) -> dict:
    paired = {}
    for suite, rows in suites.items():
        for key, source in rows.items():
            label, terrain = key.split("/", 1)
            if label not in ("source_mixed", "source_mixed_teacher"):
                continue
            candidate_label = label.replace("source_mixed", "candidate_mixed", 1)
            candidate = rows[f"{candidate_label}/{terrain}"]
            if source["fixed_cases"] != candidate["fixed_cases"]:
                raise ValueError("All-mode paired fixed-case counts differ")
            paired[f"{suite}/{label.removeprefix('source_mixed') or 'student'}/{terrain}"] = {
                field: dict(source=value, candidate=candidate[field],
                            delta=candidate[field] - value)
                for field, value in source.items()
                if field != "fixed_cases" and value is not None and candidate[field] is not None
            }
    return paired


def paired_signed(signed: dict) -> dict:
    paired = {}
    for key, sources in signed.items():
        label, rest = key.split("/", 1)
        if label not in ("source_mixed", "source_mixed_teacher"):
            continue
        candidate_label = label.replace("source_mixed", "candidate_mixed", 1)
        candidates = signed[f"{candidate_label}/{rest}"]
        if [row["speed_mps"] for row in sources] != [row["speed_mps"] for row in candidates]:
            raise ValueError("Signed source and candidate speeds differ")
        for source, candidate in zip(sources, candidates):
            name = f"{label.removeprefix('source_mixed') or 'student'}/{rest}/{source['speed_mps']}"
            paired[name] = dict(
                physical_failure=dict(source=source["physical_failure"],
                                      candidate=candidate["physical_failure"]),
                four_wheel_dwell_2s=dict(source=source["four_wheel_dwell_2s"],
                                         candidate=candidate["four_wheel_dwell_2s"]),
                both_phases_followed=dict(
                    source=source["response"]["both_phases_followed"],
                    candidate=candidate["response"]["both_phases_followed"]),
                first_signed_response_mean=dict(
                    source=source["response"]["first"]["signed_response_mean"],
                    candidate=candidate["response"]["first"]["signed_response_mean"]),
                second_signed_response_mean=dict(
                    source=source["response"]["second"]["signed_response_mean"],
                    candidate=candidate["response"]["second"]["signed_response_mean"]),
                hip_actual_inward_p95_rad=dict(
                    source=source["hip_actual_inward_p95_rad"],
                    candidate=candidate["hip_actual_inward_p95_rad"]),
                hip_actual_near_hard_seconds=dict(
                    source=source["hip_actual_near_hard_seconds"],
                    candidate=candidate["hip_actual_near_hard_seconds"]),
            )
    if len(paired) != 64:
        raise ValueError("Signed paired comparison omitted a fixed case")
    return paired


def main() -> dict:
    frozen = verify_frozen()
    source_age = frozen["source"]["completed_updates"]
    evaluation = HERE / f"evaluation_{source_age + MIXED_UPDATES}"
    preflight = load_json(HERE / "PREFLIGHT_VERIFIED.json")
    verify_all_mode(evaluation)
    verify_signed_mixed(evaluation / "signed_mixed_stairs", preflight["collision_geometry_id"])
    old = {suite: original_suite_summary(evaluation / suite) for suite in SUITES}
    signed = signed_summary(evaluation / "signed_mixed_stairs")
    candidate_cases = [case for key, rows in signed.items()
                       if key.split("/", 1)[0] in ("candidate_mixed", "candidate_mixed_teacher")
                       for case in rows]
    if len(candidate_cases) != 64 or not LABELS:
        raise ValueError("Candidate student/teacher signed cases missing")
    physical = sum(case["physical_failure"] for case in candidate_cases)
    dwell = sum(case["four_wheel_dwell_2s"] for case in candidate_cases)
    responsive = sum(case["response"]["both_phases_followed"] for case in candidate_cases)
    gate = dict(
        passes=physical == 0 and dwell == 64 and responsive == 64,
        candidate_cases=64, physical_violations=physical,
        four_wheel_2s_dwell=dwell, both_signed_phases_followed=responsive,
        acceptance="all 64 student/teacher cases must satisfy each criterion",
    )
    report = dict(
        state="fixed_behavior_review_required", candidate_signed_gate=gate,
        original_all_mode=old, signed_mixed_cases=signed,
        paired_all_mode_deltas=paired_all_mode(old),
        paired_signed_cases=paired_signed(signed),
        fixed_cases_only=True, no_random_success_rate_inferred=True,
        real_robot_release=False,
        contact_limitation="Wheel contact XY is conservatively inferred from validated collider centers and hull; it is not measured.",
        collision_limitation="Simulation self-collision is disabled; no inference about physical leg clearance.",
        evidence_sha256={
            name: {file: digest(evaluation / name / file)
                   for file in ("manifest.json", "traces.npz", "HIP_REVIEW_v2.json")}
            for name in (*SUITES, "signed_mixed_stairs")},
        disposition="HOLD for human behavior, random-initial-state and hardware-contract review; no deployment",
    )
    write_new(HERE / "BEHAVIOR_COMPARISON.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(main()["candidate_signed_gate"]))
