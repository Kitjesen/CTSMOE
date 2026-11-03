"""Fail-closed matched gates for the 300/1000-update V14 checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLAT_CASES = 20
NORMAL_CASES = 83
LEG_TORQUE_LIMIT = 120.0
WHEEL_TORQUE_LIMIT = 17.0
LIMIT_EPS = 1e-3
TARGET_STEP_LIMIT = 0.2401


def _case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    velocity = case.get("velocity")
    return (
        case.get("terrain"), case.get("level"), case.get("mode"),
        case.get("direction"), tuple(velocity) if isinstance(velocity, list) else velocity,
    )


def _finite(case: dict[str, Any], field: str) -> bool:
    value = case.get(field)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _allowed_increase(value: float, absolute: float) -> float:
    return max(absolute, 0.20 * abs(value))


def _index(analysis: dict[str, Any], label: str) -> tuple[dict[tuple[Any, ...], dict], str | None]:
    cases = analysis.get("cases")
    if not isinstance(cases, list):
        return {}, "analysis has no cases list"
    rows = [case for case in cases if isinstance(case, dict) and case.get("label") == label]
    indexed = {_case_key(case): case for case in rows}
    if len(indexed) != len(rows):
        return indexed, "duplicate matched case identity"
    return indexed, None


def _summary_error(summary: dict[str, Any] | None, label: str,
                   indexed: dict[tuple[Any, ...], dict]) -> str | None:
    if not isinstance(summary, dict) or not isinstance(summary.get("terrains"), list):
        return "missing summary terrains schema"
    expected: dict[str, int] = {}
    for case in indexed.values():
        terrain = case.get("terrain")
        expected[terrain] = expected.get(terrain, 0) + 1
    actual = {row.get("terrain"): row.get("count") for row in summary["terrains"]
              if isinstance(row, dict) and row.get("label") == label}
    if actual != expected:
        return f"summary terrain counts disagree: expected={expected}, actual={actual}"
    return None


def _target_steps(joints: list[dict[str, Any]] | None, label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(joints, list):
        return result
    for row in joints:
        if not isinstance(row, dict) or row.get("label") != label:
            continue
        case_id, value = row.get("case"), row.get("target_step_peak_rad")
        if (isinstance(case_id, int) and isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value)):
            item = result.setdefault(case_id, {"peak": 0.0, "joints": set()})
            item["peak"] = max(item["peak"], float(value))
            if isinstance(row.get("joint"), str):
                item["joints"].add(row["joint"])
    return result


def continuation_gate(
    baseline_normal: dict[str, Any],
    baseline_flat: dict[str, Any],
    candidate_normal: dict[str, Any],
    candidate_flat: dict[str, Any],
    *,
    pairs: list[tuple[str, str]],
    candidate_normal_joints: list[dict[str, Any]] | None,
    candidate_flat_joints: list[dict[str, Any]] | None,
    stage: int,
    baseline_normal_summary: dict[str, Any] | None = None,
    baseline_flat_summary: dict[str, Any] | None = None,
    candidate_normal_summary: dict[str, Any] | None = None,
    candidate_flat_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare exact source/candidate cases and return a structured decision.

    ``pairs`` must contain exactly one student and one teacher label pair.  A
    missing metric, case, summary row, or joint-assessment row is a rejection.
    Baseline failures define no instant capability requirement; only baseline
    successes are protected.
    """
    if stage not in (300, 1000):
        raise ValueError("stage must be 300 or 1000")
    reasons: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    def reject(source: str, candidate: str, code: str, message: str,
               case: dict[str, Any] | None = None) -> None:
        reason = dict(source_label=source, candidate_label=candidate,
                      code=code, message=message)
        if case is not None:
            reason["case"] = dict(terrain=case.get("terrain"), level=case.get("level"),
                                  mode=case.get("mode"), direction=case.get("direction"),
                                  velocity=case.get("velocity"), id=case.get("id"))
        reasons.append(reason)

    if len(pairs) != 2 or len(set(pairs)) != 2:
        for source, candidate in pairs or [("<missing>", "<missing>")]:
            reject(source, candidate, "pair_coverage", "exactly two distinct student/teacher pairs required")
    modes: set[bool] = set()
    bundles = (
        ("normal", baseline_normal, candidate_normal, baseline_normal_summary,
         candidate_normal_summary, candidate_normal_joints, NORMAL_CASES),
        ("flat", baseline_flat, candidate_flat, baseline_flat_summary,
         candidate_flat_summary, candidate_flat_joints, FLAT_CASES),
    )

    for source_label, candidate_label in pairs:
        pair_modes: set[bool] = set()
        for suite, baseline, candidate, source_summary, candidate_summary, joints, count in bundles:
            reason_start = len(reasons)
            old, old_error = _index(baseline, source_label)
            new, new_error = _index(candidate, candidate_label)
            if old_error:
                reject(source_label, candidate_label, "baseline_schema", f"{suite}: {old_error}")
            if new_error:
                reject(source_label, candidate_label, "candidate_schema", f"{suite}: {new_error}")
            if len(old) != count:
                reject(source_label, candidate_label, "baseline_case_count",
                       f"{suite}: expected {count} cases, got {len(old)}")
            if len(new) != count:
                reject(source_label, candidate_label, "candidate_case_count",
                       f"{suite}: expected {count} cases, got {len(new)}")
            for summary, label, indexed, role in (
                (source_summary, source_label, old, "baseline"),
                (candidate_summary, candidate_label, new, "candidate"),
            ):
                error = _summary_error(summary, label, indexed)
                if error:
                    reject(source_label, candidate_label, f"{role}_summary", f"{suite}: {error}")
            if set(old) != set(new):
                reject(source_label, candidate_label, "case_identity",
                       f"{suite}: baseline/candidate case identities differ")

            teacher_values = {case.get("teacher") for case in old.values()}
            candidate_teacher_values = {case.get("teacher") for case in new.values()}
            if len(teacher_values) != 1 or teacher_values - {False, True}:
                reject(source_label, candidate_label, "baseline_teacher_mode",
                       f"{suite}: source label does not have one boolean teacher mode")
            elif candidate_teacher_values != teacher_values:
                reject(source_label, candidate_label, "candidate_teacher_mode",
                       f"{suite}: candidate teacher mode differs from source")
            else:
                pair_modes |= teacher_values

            target_steps = _target_steps(joints, candidate_label)
            for key in sorted(set(old) & set(new), key=str):
                before, after = old[key], new[key]
                case_id = after.get("id")
                assessment = target_steps.get(case_id) if isinstance(case_id, int) else None
                if _finite(after, "target_step_peak_rad"):
                    target_step = float(after["target_step_peak_rad"])
                elif assessment is not None and len(assessment["joints"]) == 12:
                    target_step = assessment["peak"]
                else:
                    target_step = None
                if target_step is None:
                    reject(source_label, candidate_label, "missing_target_step",
                           f"{suite}: case lacks finite target delta for all 12 leg joints", after)
                elif target_step > TARGET_STEP_LIMIT:
                    reject(source_label, candidate_label, "target_step_limit",
                           f"target step {target_step:.6g} > {TARGET_STEP_LIMIT}", after)

                for field, limit, code in (
                    ("leg_torque_peak_nm", LEG_TORQUE_LIMIT + LIMIT_EPS, "leg_torque_limit"),
                    ("wheel_torque_peak_nm", WHEEL_TORQUE_LIMIT + LIMIT_EPS, "wheel_torque_limit"),
                    ("hard_position_excess_rad", LIMIT_EPS, "hard_position_limit"),
                ):
                    if not _finite(after, field):
                        reject(source_label, candidate_label, "missing_safety_metric",
                               f"missing or nonfinite {field}", after)
                    elif after[field] > limit:
                        reject(source_label, candidate_label, code,
                               f"{field}={after[field]:.6g} exceeds {limit:.6g}", after)

                if after.get("physical") is True and before.get("physical") is not True:
                    reject(source_label, candidate_label, "new_physical_failure",
                           "candidate adds a physical failure", after)

                if after.get("terrain") == "flat":
                    required = ("first_peak_rms", "first_peak_joint_rad", "max_rms",
                                "height_min", "height_max", "last_xy_error", "last_yaw_error")
                    missing = [field for field in required
                               if not _finite(before, field) or not _finite(after, field)]
                    if missing:
                        reject(source_label, candidate_label, "missing_flat_metric",
                               f"missing or nonfinite flat metrics: {missing}", after)
                        continue
                    if before.get("completed_25s") is True and after.get("completed_25s") is not True:
                        reject(source_label, candidate_label, "lost_25s",
                               "baseline completed 25 seconds but candidate did not", after)
                    for field, limit in (("first_peak_rms", .25),
                                         ("first_peak_joint_rad", .60),
                                         ("max_rms", .35)):
                        if after[field] > limit:
                            reject(source_label, candidate_label, f"absolute_{field}",
                                   f"{field}={after[field]:.6g} > {limit}", after)
                    if after["height_min"] < .34 or after["height_max"] > .60:
                        reject(source_label, candidate_label, "absolute_height",
                               f"height range [{after['height_min']:.6g}, {after['height_max']:.6g}] outside [.34, .60]",
                               after)
                    for field in ("pose_pass", "tracking_pass"):
                        if before.get(field) is True and after.get(field) is not True:
                            reject(source_label, candidate_label, f"lost_{field}",
                                   f"previously true {field} regressed", after)
                    for field, allowance, code in (
                        ("last_xy_error", .03, "xy_tail_regression"),
                        ("last_yaw_error", .05, "yaw_tail_regression"),
                    ):
                        limit = before[field] + _allowed_increase(before[field], allowance)
                        if after[field] > limit:
                            reject(source_label, candidate_label, code,
                                   f"{field}={after[field]:.6g} > baseline {before[field]:.6g} + allowance {limit-before[field]:.6g}",
                                   after)
                else:
                    if before.get("pass") is True and after.get("pass") is not True:
                        reject(source_label, candidate_label, "lost_skill_case",
                               "previously accepted terrain/level/speed case regressed", after)
                    if before.get("straight_4m") is True and after.get("straight_4m") is not True:
                        reject(source_label, candidate_label, "lost_straight_case",
                               "previously straight 4 m traversal regressed", after)
                    comparable_quality = before.get("pass") is True and after.get("pass") is True
                    if comparable_quality:
                        if _finite(before, "max_side_m"):
                            if not _finite(after, "max_side_m"):
                                reject(source_label, candidate_label, "missing_straight_metric",
                                       "missing or nonfinite max_side_m", after)
                            else:
                                limit = before["max_side_m"] + _allowed_increase(before["max_side_m"], .1)
                                if after["max_side_m"] > limit:
                                    reject(source_label, candidate_label, "straight_regression",
                                           f"max_side_m={after['max_side_m']:.6g} > baseline {before['max_side_m']:.6g} + allowance {limit-before['max_side_m']:.6g}",
                                           after)
                        if str(after.get("terrain", "")).startswith("stairs"):
                            field = "end_vz_abs_peak"
                            if not _finite(before, field) or not _finite(after, field):
                                reject(source_label, candidate_label, "missing_stair_vz",
                                       f"missing or nonfinite {field}", after)
                            else:
                                limit = before[field] + _allowed_increase(before[field], .1)
                                if after[field] > limit:
                                    reject(source_label, candidate_label, "stair_vz_regression",
                                           f"{field}={after[field]:.6g} > baseline {before[field]:.6g} + allowance {limit-before[field]:.6g}",
                                           after)
                    elif before.get("pass") is not True and after.get("pass") is True:
                        observations.append({
                            "source_label": source_label,
                            "candidate_label": candidate_label,
                            "code": "quality_comparison_skipped_new_success",
                            "case": dict(terrain=after.get("terrain"), level=after.get("level"),
                                         mode=after.get("mode"), direction=after.get("direction"),
                                         velocity=after.get("velocity"), id=after.get("id")),
                            "baseline_max_side_m": before.get("max_side_m"),
                            "candidate_max_side_m": after.get("max_side_m"),
                            "baseline_end_vz_abs_peak": before.get("end_vz_abs_peak"),
                            "candidate_end_vz_abs_peak": after.get("end_vz_abs_peak"),
                        })
            local_reasons = list(range(reason_start, len(reasons)))
            gate_results.append({
                "source_label": source_label,
                "candidate_label": candidate_label,
                "suite": suite,
                "passed": not local_reasons,
                "reason_indexes": local_reasons,
            })
        if len(pair_modes) == 1:
            modes |= pair_modes
    if modes != {False, True}:
        for source_label, candidate_label in pairs:
            reject(source_label, candidate_label, "learner_coverage",
                   f"student and teacher must both be evaluated; observed modes={sorted(modes)}")

    global_codes = {"pair_coverage", "learner_coverage"}
    for gate in gate_results:
        global_reasons = [index for index, reason in enumerate(reasons)
                          if reason["code"] in global_codes
                          and reason["source_label"] == gate["source_label"]
                          and reason["candidate_label"] == gate["candidate_label"]]
        gate["reason_indexes"] = sorted(set(gate["reason_indexes"] + global_reasons))
        gate["passed"] = not gate["reason_indexes"]

    accepted = not reasons
    action = "continue_700" if accepted and stage == 300 else (
        "stop_at_1000" if accepted else "reject"
    )
    return {
        "accepted": accepted,
        "stage": stage,
        "action": action,
        "next_updates": 700 if accepted and stage == 300 else 0,
        "stop": stage == 1000 or not accepted,
        "pairs": [{"source_label": source, "candidate_label": candidate}
                  for source, candidate in pairs],
        "gates": gate_results,
        "reasons": reasons,
        "observations": observations,
        "thresholds": {
            "first_second_rms_max": .25,
            "first_second_joint_max_rad": .60,
            "full_episode_rms_max": .35,
            "height_m": [.34, .60],
            "xy_tail_allowed_increase": "max(0.03, 20%)",
            "yaw_tail_allowed_increase": "max(0.05, 20%)",
            "straight_and_stair_vz_allowed_increase": "max(0.1, 20%)",
            "leg_torque_nm": LEG_TORQUE_LIMIT,
            "wheel_torque_nm": WHEEL_TORQUE_LIMIT,
            "target_step_rad": TARGET_STEP_LIMIT,
            "hard_limit_excess_rad": LIMIT_EPS,
        },
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path) -> Any | None:
    return _load(path) if path.exists() else None


def _sibling(path: Path, name: str) -> Path:
    return path.parent / name


def _analysis_path(path: Path) -> Path:
    return path / "analysis_v14.json" if path.is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("baseline-normal", "baseline-flat", "candidate-normal", "candidate-flat"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-student", required=True)
    parser.add_argument("--source-teacher", required=True)
    parser.add_argument("--candidate-student", required=True)
    parser.add_argument("--candidate-teacher", required=True)
    parser.add_argument("--stage", type=int, choices=(300, 1000), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {name.replace("_", "-"): _analysis_path(getattr(args, name))
             for name in ("baseline_normal", "baseline_flat", "candidate_normal", "candidate_flat")}
    analyses = {name: _load(path) for name, path in paths.items()}
    summaries = {name: _load(_sibling(path, "summary.json")) for name, path in paths.items()}
    joints = {name: _load_optional(_sibling(paths[name], "joint_assessment.json"))
              for name in ("candidate-normal", "candidate-flat")}
    result = continuation_gate(
        analyses["baseline-normal"], analyses["baseline-flat"],
        analyses["candidate-normal"], analyses["candidate-flat"],
        pairs=[(args.source_student, args.candidate_student),
               (args.source_teacher, args.candidate_teacher)],
        candidate_normal_joints=joints["candidate-normal"],
        candidate_flat_joints=joints["candidate-flat"],
        stage=args.stage,
        baseline_normal_summary=summaries["baseline-normal"],
        baseline_flat_summary=summaries["baseline-flat"],
        candidate_normal_summary=summaries["candidate-normal"],
        candidate_flat_summary=summaries["candidate-flat"],
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
