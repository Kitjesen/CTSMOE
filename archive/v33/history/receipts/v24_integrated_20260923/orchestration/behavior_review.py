"""CPU analysis of fixed paired traces; this never approves real-robot use."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from contract import ORCH, V24_UPDATES, digest, load_json, verify_frozen, write_new


SIGNS = {"fr_hip_joint": -1., "fl_hip_joint": 1.,
         "rr_hip_joint": 1., "rl_hip_joint": -1.}
SUITES = ("terrain83", "flat25", "stairend", "stairend_15cm")
POLICIES = (
    "source_default012", "source_default012_teacher",
    "candidate_hip_request", "candidate_hip_request_teacher",
)


def summarize_suite(folder: Path) -> dict:
    manifest = load_json(folder / "manifest.json")
    report = load_json(folder / "HIP_REVIEW_v2.json")
    names = manifest["joint_names"]
    ids = manifest["request_joint_ids"]
    dt = float(manifest["dt"])
    if not 0.005 <= dt <= 0.05 or set(SIGNS) - set(names):
        raise ValueError("Joint names or control period changed")
    entries = defaultdict(lambda: defaultdict(list))
    case_rows = defaultdict(list)
    with np.load(folder / "traces.npz") as archive:
        requested = archive["requested"]
        target = archive["target"]
        actual = archive["joint_pos"]
        torque = archive["torque_peak"]
        speed = archive["speed_peak"]
        world_velocity = archive["world_velocity"] if "world_velocity" in archive.files else None
        for case in manifest["cases"]:
            policy = manifest["history"][case["policy"]]["label"]
            if policy not in POLICIES:
                raise ValueError("Unpaired policy entered the fixed cases")
            terrain = case["terrain"]
            key = f"{policy}/{terrain}"
            case_id = int(case["id"])
            count = int(case["end_step"]) + 1
            if count <= 0 or count > actual.shape[0]:
                raise ValueError("Invalid fixed-case trajectory length")
            required = (requested[:count, case_id], target[:count, case_id],
                        actual[:count, case_id], torque[:count, case_id], speed[:count, case_id])
            if world_velocity is not None:
                required += (world_velocity[:count, case_id],)
            if not all(np.isfinite(value).all() for value in required):
                raise ValueError("Nonfinite valid trajectory in fixed comparison")
            for joint, sign in SIGNS.items():
                j = names.index(joint)
                request_j = ids.index(j)
                req = requested[:count, case_id, request_j] * sign
                tgt = target[:count, case_id, j] * sign
                act = actual[:count, case_id, j] * sign
                entries[key]["raw_inward"].append(req)
                entries[key]["target_inward"].append(tgt)
                entries[key]["actual_inward"].append(act)
                entries[key][f"{joint}/raw"].append(req)
                entries[key][f"{joint}/target"].append(tgt)
                entries[key][f"{joint}/actual"].append(act)
                entries[key]["raw_excess_rad_seconds"].append(float(
                    np.maximum(req - .369, 0).sum() * dt))
                entries[key]["target_near_soft_seconds"].append(float((tgt >= .36).sum() * dt))
                entries[key]["actual_near_hard_seconds"].append(float((act >= .39).sum() * dt))
            leg_ids = [i for i, name in enumerate(names) if not name.endswith("foot_joint")]
            wheel_ids = [i for i, name in enumerate(names) if name.endswith("foot_joint")]
            row = dict(case_id=case_id, terrain=terrain, level=int(case["level"]),
                       physical_failure=bool(set(case.get("terms", [])) & {
                           "illegal_contact", "actuator_violation"}),
                       termination_terms=case.get("terms", []),
                       route_4m=bool(case.get("passed_4m", False)),
                       stable_platform=case.get("end_reason") == "supported_platform_dwell",
                       pit_success=bool(case.get("stable", False)),
                       leg_torque_peak_Nm=float(torque[:count, case_id, leg_ids].max()),
                       wheel_torque_peak_Nm=float(torque[:count, case_id, wheel_ids].max()),
                       leg_speed_peak_rad_s=float(speed[:count, case_id, leg_ids].max()),
                       wheel_speed_peak_rad_s=float(speed[:count, case_id, wheel_ids].max()),
                       last_second_world_upward_speed_peak_mps=(
                           float(world_velocity[max(0, count - round(1 / dt)):count, case_id, 2].max())
                           if world_velocity is not None else None))
            case_rows[policy].append(row)
    report_cases = defaultdict(list)
    for case in report["cases"]:
        label = manifest["history"][case["policy"]]["label"]
        report_cases[label].append(case)
    result = {}
    for key, values in entries.items():
        policy = key.split("/", 1)[0]
        terrain = key.split("/", 1)[1]
        cases = [case for case in case_rows[policy] if case["terrain"] == terrain]
        detail = [case for case in report_cases[policy] if case["terrain"] == terrain]
        if len(cases) != len(detail) or not cases:
            raise ValueError("Case rows differ between physical trace and per-joint report")
        raw = np.concatenate(values["raw_inward"])
        tgt = np.concatenate(values["target_inward"])
        act = np.concatenate(values["actual_inward"])
        by_joint = {}
        for joint in SIGNS:
            req_j = np.concatenate(values[f"{joint}/raw"])
            tgt_j = np.concatenate(values[f"{joint}/target"])
            act_j = np.concatenate(values[f"{joint}/actual"])
            by_joint[joint] = dict(
                raw_inward_p95_rad=float(np.quantile(req_j, .95)),
                target_inward_p95_rad=float(np.quantile(tgt_j, .95)),
                actual_inward_p95_rad=float(np.quantile(act_j, .95)),
                actual_inward_max_rad=float(act_j.max()),
                raw_excess_rad_seconds=float(np.maximum(req_j - .369, 0).sum() * dt),
                target_near_soft_seconds=float((tgt_j >= .36).sum() * dt),
                actual_near_hard_seconds=float((act_j >= .39).sum() * dt),
            )
        result[key] = dict(
            fixed_cases=len(cases),
            hip_by_joint=by_joint,
            raw_inward_p95_rad=float(np.quantile(raw, .95)),
            target_inward_p95_rad=float(np.quantile(tgt, .95)),
            actual_inward_p95_rad=float(np.quantile(act, .95)),
            raw_inward_max_rad=float(raw.max()),
            target_inward_max_rad=float(tgt.max()),
            actual_inward_max_rad=float(act.max()),
            raw_excess_rad_seconds=float(sum(values["raw_excess_rad_seconds"])),
            target_near_soft_seconds=float(sum(values["target_near_soft_seconds"])),
            actual_near_hard_seconds=float(sum(values["actual_near_hard_seconds"])),
            raw_request_above_soft_fraction=float((raw > .369).mean()),
            target_near_soft_fraction=float((tgt >= .36).mean()),
            actual_near_hard_fraction=float((act >= .39).mean()),
            physical_failures=sum(case["physical_failure"] for case in cases),
            route_4m=sum(case["route_4m"] for case in cases),
            platform_dwell=sum(case["stable_platform"] for case in cases),
            pit_success=sum(case["pit_success"] for case in cases),
            leg_torque_peak_Nm=max(case["leg_torque_peak_Nm"] for case in cases),
            wheel_torque_peak_Nm=max(case["wheel_torque_peak_Nm"] for case in cases),
            leg_speed_peak_rad_s=max(case["leg_speed_peak_rad_s"] for case in cases),
            wheel_speed_peak_rad_s=max(case["wheel_speed_peak_rad_s"] for case in cases),
            calf_contact_peak_N=max(case["calf_contact_peak_N"] for case in detail),
            last_second_world_upward_speed_peak_mps=(max(
                case["last_second_world_upward_speed_peak_mps"] for case in cases)
                if cases[0]["last_second_world_upward_speed_peak_mps"] is not None else None),
            startup_pose_rms_peak_rad=max(case["startup_pose_rms_peak"] for case in detail),
            xy_tracking_rmse_mean_mps=float(np.mean([case["xy_rmse"] for case in detail])),
            yaw_tracking_rmse_mean_rad_s=float(np.mean([case["yaw_rmse"] for case in detail])),
            fixed_case_pit_success_heights_m=sorted({
                round(.15 + .05 * case["level"], 8) for case in cases
                if terrain == "pit" and case["pit_success"]}),
        )
    return result


def compare(results: dict) -> dict:
    paired = {}
    for suite, values in results.items():
        for learner in ("", "_teacher"):
            baseline = "source_default012" + learner
            candidate = "candidate_hip_request" + learner
            terrains = {key.split("/", 1)[1] for key in values if key.startswith(baseline + "/")}
            for terrain in terrains:
                source = values[baseline + "/" + terrain]
                new = values[candidate + "/" + terrain]
                if source["fixed_cases"] != new["fixed_cases"]:
                    raise ValueError("Baseline and candidate fixed-case counts differ")
                metrics = ("actual_inward_p95_rad", "actual_near_hard_seconds",
                           "raw_excess_rad_seconds", "target_near_soft_seconds",
                           "physical_failures", "route_4m", "platform_dwell", "pit_success",
                           "leg_torque_peak_Nm", "wheel_torque_peak_Nm",
                           "leg_speed_peak_rad_s", "wheel_speed_peak_rad_s",
                           "calf_contact_peak_N", "last_second_world_upward_speed_peak_mps",
                           "startup_pose_rms_peak_rad", "xy_tracking_rmse_mean_mps",
                           "yaw_tracking_rmse_mean_rad_s")
                paired[f"{suite}/{learner or 'student'}/{terrain}"] = {
                    metric: dict(source=source[metric], candidate=new[metric],
                                 delta=new[metric] - source[metric])
                    for metric in metrics if source[metric] is not None and new[metric] is not None}
    return paired


def main() -> dict:
    frozen = verify_frozen()
    evaluation = ORCH / f"evaluation_{frozen['source']['completed_updates'] + V24_UPDATES}"
    results = {suite: summarize_suite(evaluation / suite) for suite in SUITES}
    record = dict(
        state="fixed_behavior_review_required", fixed_cases_only=True,
        no_random_success_rate_inferred=True, no_real_robot_fault_log=True,
        collision_limitation=("The mixed simulation asset has self-collision disabled; "
                              "absence of a simulated self-contact cannot establish leg clearance."),
        source_reward_factor="hip-request off", candidate_reward_factor="inward excess -0.02",
        interpretation=("A lower raw request excess alone is insufficient. Review actual hip inward "
                        "p95 and time near .40 rad, target saturation, 15cm stair cases, terrain "
                        "success, torque/speed/contact, and high-platform fixed cases together."),
        high_platform_height_definition="15cm + 5cm per pit level; fixed-case successful heights only",
        suites=results, deltas=compare(results),
        evidence_sha256={suite: {
            "manifest": digest(evaluation / suite / "manifest.json"),
            "traces": digest(evaluation / suite / "traces.npz"),
            "per_joint": digest(evaluation / suite / "HIP_REVIEW_v2.json")
        } for suite in SUITES},
        disposition="HOLD for trace, random-initial-state, noise and hardware-contract review; no deployment",
    )
    write_new(ORCH / "BEHAVIOR_COMPARISON.json", record)
    return record


if __name__ == "__main__":
    print(json.dumps({"state": main()["state"]}))
