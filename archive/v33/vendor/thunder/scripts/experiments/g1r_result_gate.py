#!/usr/bin/env python3
"""Apply G1-R nominal or latency hard gates to evaluator JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MOVEMENT = (
    "forward_0.6",
    "backward_-0.6",
    "left_0.6",
    "right_-0.6",
    "yaw_left_0.6",
    "yaw_right_-0.6",
)
LATERAL = ("left_0.6", "right_-0.6")
YAW = ("yaw_left_0.6", "yaw_right_-0.6")
G1_TRACKING_RMSE = {
    "forward_0.6": 0.004349761,
    "backward_-0.6": 0.012185449,
    "left_0.6": 0.036139850,
    "right_-0.6": 0.044075493,
    "yaw_left_0.6": 0.093986981,
    "yaw_right_-0.6": 0.043312497,
}
G1_STANDING_LEG_RMS_NM = 7.486396799
G1_STANDING_HIP_RMS_NM = 2.247056484
MODEL_13500_STANDING_LEG_RMS_NM = 5.62
MODEL_13500_STANDING_HIP_RMS_NM = 1.07


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def movement_rmse(result: dict) -> float:
    return sum(result["scenarios"][name]["tracking_rmse"] for name in MOVEMENT) / len(MOVEMENT)


def safe(result: dict) -> tuple[bool, bool, float]:
    scenarios = result["scenarios"].values()
    finite = all(item["finite"] for item in scenarios)
    no_unexpected_done = all(item["unexpected_done_fraction"] == 0.0 for item in scenarios)
    contact_peak = max(item["contact"]["history_peak_n"] for item in scenarios)
    return finite, no_unexpected_done, contact_peak


def lateral_attitude(result: dict) -> tuple[float, float]:
    rms = 0.0
    peak = 0.0
    for name in LATERAL:
        attitude = result["scenarios"][name]["base_attitude"]
        rms = max(rms, math.degrees(attitude["roll_rms_rad"]), math.degrees(attitude["pitch_rms_rad"]))
        peak = max(
            peak,
            math.degrees(attitude["roll_peak_abs_rad"]),
            math.degrees(attitude["pitch_peak_abs_rad"]),
        )
    return rms, peak


def nominal(result: dict) -> dict:
    finite, no_done, contact_peak = safe(result)
    lateral_rms, lateral_peak = lateral_attitude(result)
    stand = result["scenarios"]["stand"]["actual_velocity_mean"]
    stand_metrics = result["scenarios"]["stand"]
    lateral_speed = min(abs(result["scenarios"][name]["actual_velocity_mean"][1]) for name in LATERAL)
    yaw_speed = min(abs(result["scenarios"][name]["actual_velocity_mean"][2]) for name in YAW)
    metrics = {
        "movement_tracking_rmse": movement_rmse(result),
        "minimum_lateral_speed_mps": lateral_speed,
        "minimum_yaw_speed_radps": yaw_speed,
        "standing_xy_speed_mps": math.hypot(stand[0], stand[1]),
        "lateral_attitude_rms_deg": lateral_rms,
        "lateral_attitude_peak_deg": lateral_peak,
        "contact_peak_n": contact_peak,
        "standing_leg_rms_nm": stand_metrics["leg_torque"]["rms"],
        "standing_hip_rms_nm": stand_metrics["hip_torque"]["rms"],
        "standing_leg_rms_vs_model_13500": (
            stand_metrics["leg_torque"]["rms"] / MODEL_13500_STANDING_LEG_RMS_NM
        ),
        "standing_hip_rms_vs_model_13500": (
            stand_metrics["hip_torque"]["rms"] / MODEL_13500_STANDING_HIP_RMS_NM
        ),
    }
    checks = {
        "movement_tracking_rmse_at_most_g1_x1_05": (
            metrics["movement_tracking_rmse"] <= 0.040958755425
        ),
        "lateral_speed_at_least_0_58": lateral_speed >= 0.58,
        "yaw_speed_at_least_0_57": yaw_speed >= 0.57,
        "standing_xy_speed_at_most_0_002": metrics["standing_xy_speed_mps"] <= 0.002,
        "lateral_attitude_rms_below_0_5deg": lateral_rms < 0.5,
        "lateral_attitude_peak_below_1deg": lateral_peak < 1.0,
        "contact_peak_below_450n": contact_peak < 450.0,
        "standing_leg_rms_at_most_g1": (
            metrics["standing_leg_rms_nm"] <= G1_STANDING_LEG_RMS_NM
        ),
        "standing_hip_rms_at_most_g1": (
            metrics["standing_hip_rms_nm"] <= G1_STANDING_HIP_RMS_NM
        ),
        "actuator_over_90pct_fraction_at_most_1pct": all(
            item[actuator]["over_90_limit_fraction"] <= 0.01
            for item in result["scenarios"].values()
            for actuator in ("leg_torque", "wheel_torque")
        ),
        "finite": finite,
        "no_unexpected_done": no_done,
    }
    checks.update(
        {
            f"{name}_rmse_at_most_g1_x1_05": (
                result["scenarios"][name]["tracking_rmse"] <= baseline * 1.05
            )
            for name, baseline in G1_TRACKING_RMSE.items()
        }
    )
    return {"mode": "nominal", "passed": all(checks.values()), "checks": checks, "metrics": metrics}


def latency(zero: dict, three_frames: dict, nominal_result: dict) -> dict:
    zero_finite, zero_no_done, _ = safe(zero)
    finite, no_done, contact_peak = safe(three_frames)
    _, lateral_peak = lateral_attitude(three_frames)
    zero_rmse = movement_rmse(zero)
    delayed_rmse = movement_rmse(three_frames)
    ratio = delayed_rmse / zero_rmse if zero_rmse > 0 else math.inf
    nominal_rmse = movement_rmse(nominal_result)
    zero_nominal_ratio = zero_rmse / nominal_rmse if nominal_rmse > 0 else math.inf
    metrics = {
        "zero_tracking_rmse": zero_rmse,
        "three_frame_tracking_rmse": delayed_rmse,
        "tracking_ratio": ratio,
        "zero_tracking_vs_nominal_ratio": zero_nominal_ratio,
        "lateral_attitude_peak_deg": lateral_peak,
        "contact_peak_n": contact_peak,
    }
    checks = {
        "tracking_ratio_at_most_1_20": ratio <= 1.20,
        "zero_tracking_vs_nominal_at_most_1_05": zero_nominal_ratio <= 1.05,
        "contact_peak_below_450n": contact_peak < 450.0,
        "lateral_attitude_peak_below_2deg": lateral_peak < 2.0,
        "zero_finite": zero_finite,
        "zero_no_unexpected_done": zero_no_done,
        "finite": finite,
        "no_unexpected_done": no_done,
    }
    return {"mode": "latency", "passed": all(checks.values()), "checks": checks, "metrics": metrics}


def smoke(result: dict) -> dict:
    finite, no_done, _ = safe(result)
    checks = {"finite": finite, "no_unexpected_done": no_done}
    return {"mode": "smoke", "passed": all(checks.values()), "checks": checks, "metrics": {}}


def select(results: list[dict], checkpoints: list[str]) -> dict:
    candidates = []
    for result, checkpoint in zip(results, checkpoints, strict=True):
        decision = nominal(result)
        lateral_rms, lateral_peak = lateral_attitude(result)
        candidates.append(
            {
                "checkpoint": checkpoint,
                "decision": decision,
                "rank": [
                    movement_rmse(result),
                    lateral_rms,
                    lateral_peak,
                    decision["metrics"]["contact_peak_n"],
                ],
            }
        )
    passed = [candidate for candidate in candidates if candidate["decision"]["passed"]]
    winner = min(passed, key=lambda candidate: candidate["rank"])["checkpoint"] if passed else None
    return {"mode": "select", "passed": winner is not None, "winner": winner, "candidates": candidates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("nominal", "latency", "smoke", "select"))
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs=2)
    parser.add_argument("--winner-file", type=Path)
    parser.add_argument("--nominal-result", type=Path)
    args = parser.parse_args()
    expected = 2 if args.mode in ("latency", "select") else 1
    if len(args.results) != expected:
        parser.error(f"{args.mode} requires {expected} result file(s)")
    if args.mode == "select" and (not args.checkpoints or not args.winner_file):
        parser.error("select requires --checkpoints A B and --winner-file")
    if args.mode == "latency" and not args.nominal_result:
        parser.error("latency requires --nominal-result")
    inputs = [load(path) for path in args.results]
    if args.mode == "latency":
        decision = latency(
            *inputs,
            nominal_result=load(args.nominal_result),
        )
    elif args.mode == "select":
        decision = select(inputs, args.checkpoints)
    else:
        decision = globals()[args.mode](inputs[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "select" and decision["passed"]:
        args.winner_file.parent.mkdir(parents=True, exist_ok=True)
        args.winner_file.write_text(decision["winner"] + "\n", encoding="utf-8")
    elif args.mode == "select":
        args.winner_file.unlink(missing_ok=True)
    print(f"G1R_{args.mode.upper()}_PASS={decision['passed']}")
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
