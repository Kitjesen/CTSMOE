"""Compare matched stair probes and describe the two slow high-stair failures."""

import argparse
import json
from pathlib import Path

import numpy as np


CASE_KEYS = ("learner", "terrain", "level", "speed", "repeat", "column", "start_lateral_m")
MATCH_KEYS = (
    "actuator_profile", "checkpoint_sha256", "completed_updates", "conditions", "diagnostic_mode",
    "heading_feedback", "physics_dt", "control_dt", "joint_names", "hard_limits_rad",
    "soft_limits_rad", "velocity_limits_rad_s", "effort_limits_nm",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_identity(case):
    return {key: case[key] for key in CASE_KEYS}


def require_matched_setup(reference, candidate):
    for key in MATCH_KEYS:
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"Cannot compare different setups: {key}")
    if len(reference["cases"]) != 162 or len(candidate["cases"]) != 162:
        raise ValueError("Expected the original 162 cases in both probes")
    for index, (expected, actual) in enumerate(zip(reference["cases"], candidate["cases"], strict=True)):
        if case_identity(expected) != case_identity(actual):
            raise ValueError(f"Case identity or ordering changed at index {index}")


def compare_reproduction(reference, guarded, atol, rtol):
    """Compare every original case field, allowing newly added observations."""
    require_matched_setup(reference, guarded)
    differences = []
    mismatch_count = 0
    changed_cases = set()
    max_abs_difference_by_field = {}

    def compare(expected, actual, path, case_index):
        nonlocal mismatch_count
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key, value in expected.items():
                compare(value, actual.get(key), f"{path}.{key}", case_index)
            return
        if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                compare(left, right, f"{path}[{index}]", case_index)
            return
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            absolute = abs(float(expected) - float(actual))
            field = path.split("[", 1)[0]
            max_abs_difference_by_field[field] = max(max_abs_difference_by_field.get(field, 0.0), absolute)
            matches = bool(np.isclose(expected, actual, rtol=rtol, atol=atol))
        else:
            matches = expected == actual
        if not matches:
            mismatch_count += 1
            changed_cases.add(case_index)
            if len(differences) < 100:
                differences.append(dict(case_index=case_index, field=path, reference=expected, guarded=actual))

    for index, (expected, actual) in enumerate(zip(reference["cases"], guarded["cases"], strict=True)):
        compare(expected, actual, "case", index)
    return dict(
        reproduced=mismatch_count == 0, absolute_tolerance=atol, relative_tolerance=rtol,
        compared_case_count=162, mismatched_value_count=mismatch_count,
        changed_case_indices=sorted(changed_cases), first_100_differences=differences,
        max_abs_difference_by_field=max_abs_difference_by_field,
    )


def outcome(case):
    return dict(
        forward_exit=bool(case["forward_exit"]), physical_failure=bool(case["physical_failure"]),
        duration_s=case["duration_s"], final_forward_m=case["final_forward_m"],
        termination_reasons=case["termination_reasons"], first_end=case["first_end"],
    )


def failure_comparisons(guarded, no_termination):
    comparisons = []
    joint_names = guarded["joint_names"]
    # V2's termination threshold is 102% of each configured velocity limit.
    speed_threshold = np.asarray(guarded["velocity_limits_rad_s"]) * 1.02
    for index, (before, after) in enumerate(zip(guarded["cases"], no_termination["cases"], strict=True)):
        if before["terrain"] not in ("stairs_up", "stairs_down") or before["forward_exit"]:
            continue
        speed = np.asarray(before["joints"]["max_speed_rad_s"])
        comparisons.append({
            "case_index": index, **case_identity(before),
            "guarded": outcome(before), "without_actuator_termination": outcome(after),
            "forward_progress_difference_m": after["final_forward_m"] - before["final_forward_m"],
            "passed_only_after_suppressing_actuator_termination": bool(after["forward_exit"]),
            "active_physics_peak_speed_exceeding_v2_termination_threshold": [
                dict(joint=joint_names[j], peak_rad_s=float(speed[j]), threshold_rad_s=float(speed_threshold[j]))
                for j in np.flatnonzero(speed > speed_threshold)
            ],
        })
    return comparisons


def absolute_stats(values):
    values = np.abs(values)
    return dict(mean=float(values.mean()), p95=float(np.quantile(values, 0.95)), max=float(values.max()))


def describe_focus_case(result, traces, case_index, contact_threshold):
    selected = np.flatnonzero(traces["case_indices"] == case_index)
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one trace for case {case_index}")
    slot = int(selected[0])
    valid = np.flatnonzero(traces["active_before_step"][:, slot])
    if len(valid) < 2 or not np.array_equal(valid, np.arange(len(valid))):
        raise ValueError(f"Case {case_index} has an empty or noncontiguous valid prefix")
    time = traces["time_s"][valid]
    case = result["cases"][case_index]
    if not np.isclose(time[-1], case["duration_s"], atol=1e-5, rtol=0):
        raise ValueError(f"Case {case_index} trace duration does not match results")
    window_start = max(float(time[0]), float(time[-1] - 5.0))
    window = time >= window_start
    window_time = time[window]

    def series(name):
        values = traces[name][valid, slot]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite trace: {name}, case {case_index}")
        return values

    root = series("root_pos_relative_w_m")[window]
    body_velocity = series("root_lin_vel_b_mps")[window]
    world_velocity = series("root_link_lin_vel_w_mps")[window]
    wheel_position = series("wheel_pos_relative_w_m")[window]
    wheel_root_position = series("wheel_pos_root_relative_w_m")[window]
    contact_force = series("wheel_contact_force_norm_n")[window]
    wheel_names = traces["wheel_body_names"].tolist()
    front = [i for i, name in enumerate(wheel_names) if name.startswith(("fl_", "fr_"))]
    rear = [i for i, name in enumerate(wheel_names) if name.startswith(("rl_", "rr_"))]
    if len(front) != 2 or len(rear) != 2:
        raise ValueError("Cannot identify two front and two rear wheels")
    wheel_statistics = {}
    for index, name in enumerate(wheel_names):
        position = wheel_position[:, index]
        wheel_statistics[name] = dict(
            xyz_start_m=position[0].tolist(), xyz_end_m=position[-1].tolist(),
            xyz_change_m=(position[-1] - position[0]).tolist(),
            height_min_m=float(position[:, 2].min()), height_max_m=float(position[:, 2].max()),
            height_range_m=float(np.ptp(position[:, 2])),
            root_relative_xyz_start_m=wheel_root_position[0, index].tolist(),
            root_relative_xyz_end_m=wheel_root_position[-1, index].tolist(),
            contact_sample_fraction=float(np.mean(contact_force[:, index] > contact_threshold)),
            contact_force_norm_n=absolute_stats(contact_force[:, index]),
        )
    grouped_wheels = {}
    for name, indices in (("front", front), ("rear", rear)):
        positions = wheel_position[:, indices].mean(axis=1)
        grouped_wheels[name] = dict(
            mean_xyz_change_m=(positions[-1] - positions[0]).tolist(),
            mean_height_start_m=float(positions[0, 2]), mean_height_end_m=float(positions[-1, 2]),
            both_in_contact_sample_fraction=float(np.mean((contact_force[:, indices] > contact_threshold).all(axis=1))),
            either_in_contact_sample_fraction=float(np.mean((contact_force[:, indices] > contact_threshold).any(axis=1))),
        )

    q, qd = series("joint_pos_rad")[window], series("joint_vel_rad_s")[window]
    target_all = series("joint_pos_target_rad")
    velocity_target_all = series("joint_vel_target_rad_s")
    target = target_all[window]
    applied = series("joint_applied_torque_nm")[window]
    computed = series("joint_computed_torque_nm")[window]
    rate_window = time[1:] >= window_start
    position_target_rate = (np.diff(target_all, axis=0) / np.diff(time)[:, None])[rate_window]
    velocity_target_rate = (np.diff(velocity_target_all, axis=0) / np.diff(time)[:, None])[rate_window]
    joint_statistics = {}
    for index, name in enumerate(result["joint_names"]):
        wheel = name.endswith("foot_joint")
        joint_statistics[name] = dict(
            position_start_rad=float(q[0, index]), position_end_rad=float(q[-1, index]),
            velocity_abs_rad_s=absolute_stats(qd[:, index]),
            position_target_error_abs_rad=None if wheel else absolute_stats(target[:, index] - q[:, index]),
            position_target_rate_abs_rad_s=None if wheel else absolute_stats(position_target_rate[:, index]),
            wheel_velocity_target_error_abs_rad_s=absolute_stats(
                velocity_target_all[window, index] - qd[:, index]) if wheel else None,
            wheel_velocity_target_rate_abs_rad_s2=absolute_stats(velocity_target_rate[:, index]) if wheel else None,
            applied_torque_abs_nm=absolute_stats(applied[:, index]),
            applied_torque_rms_nm=float(np.sqrt(np.mean(applied[:, index] ** 2))),
            computed_torque_abs_nm=absolute_stats(computed[:, index]),
            computed_minus_applied_torque_abs_nm=absolute_stats(computed[:, index] - applied[:, index]),
            clipped_control_sample_fraction=float(np.mean(np.abs(computed[:, index] - applied[:, index]) > 0.001)),
        )
    return {
        "case_index": case_index, **case_identity(case), "outcome": outcome(case),
        "window": dict(start_s=float(window_time[0]), end_s=float(window_time[-1]),
                       sample_count=int(window.sum()), duration_s=float(window_time[-1] - window_time[0])),
        "root": dict(
            xyz_start_m=root[0].tolist(), xyz_end_m=root[-1].tolist(), xyz_change_m=(root[-1] - root[0]).tolist(),
            world_forward_progress_rate_mps=float((root[-1, 0] - root[0, 0]) / (window_time[-1] - window_time[0])),
            world_forward_velocity_mean_mps=float(world_velocity[:, 0].mean()),
            body_forward_velocity_mean_mps=float(body_velocity[:, 0].mean()),
            body_forward_velocity_min_mps=float(body_velocity[:, 0].min()),
            body_forward_velocity_max_mps=float(body_velocity[:, 0].max()),
            body_forward_tracking_mae_mps=float(np.mean(np.abs(body_velocity[:, 0] - case["speed"]))),
        ),
        "wheel_contact_threshold_n": contact_threshold,
        "wheels": wheel_statistics, "front_rear_wheel_groups": grouped_wheels,
        "joints": joint_statistics,
    }


def plot_focus(result, traces, focus, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(focus), 2, figsize=(12, 3.6 * len(focus)), squeeze=False)
    for row, summary in enumerate(focus):
        index = summary["case_index"]
        slot = int(np.flatnonzero(traces["case_indices"] == index)[0])
        valid = traces["active_before_step"][:, slot]
        times = traces["time_s"][valid]
        progress = traces["root_pos_relative_w_m"][valid, slot, 0]
        wheel_z = traces["wheel_pos_relative_w_m"][valid, slot, :, 2]
        axes[row, 0].plot(times, progress, label="Root forward displacement")
        axes[row, 0].axhline(4.0, color="0.5", linestyle="--", label="Probe exit at 4 m")
        for wheel, name in enumerate(traces["wheel_body_names"]):
            axes[row, 1].plot(times, wheel_z[:, wheel], label=str(name))
        axes[row, 0].set_ylabel("Root forward displacement (m)")
        axes[row, 1].set_ylabel("Wheel origin height from initial root (m)")
        case = result["cases"][index]
        for axis in axes[row]:
            axis.axvspan(summary["window"]["start_s"], times[-1], color="orange", alpha=0.12)
            axis.set_xlabel("Time (s)")
            axis.set_title(f"Student stairs up L9, {case['speed']} m/s, repeat {case['repeat']}")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded-results", type=Path, default=directory / "guarded_results.json")
    parser.add_argument("--guarded-traces", type=Path, default=directory / "guarded_traces.npz")
    parser.add_argument("--no-actuator-termination-results", type=Path,
                        default=directory / "no_actuator_termination_results.json")
    parser.add_argument("--reference", type=Path,
                        default=directory.parent / "actuator_peak_20260918/heading19001_peak.json")
    parser.add_argument("--output", type=Path, default=directory / "failure_summary.json")
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    guarded = read_json(args.guarded_results)
    no_termination = read_json(args.no_actuator_termination_results)
    reference = read_json(args.reference)
    if guarded.get("diagnostic_no_actuator_termination") is not False:
        raise ValueError("Guarded input must explicitly retain actuator termination")
    if no_termination.get("diagnostic_no_actuator_termination") is not True:
        raise ValueError("Comparison input must explicitly suppress actuator termination")
    require_matched_setup(guarded, no_termination)
    if guarded["actuator_profile"] != "rs0402_peak_v2":
        raise ValueError("This bounded analysis expects the rs0402_peak_v2 profile")
    reproduction = compare_reproduction(reference, guarded, args.atol, args.rtol)
    focus_indices = [i for i, case in enumerate(guarded["cases"]) if case["learner"] == "student"
                     and case["terrain"] == "stairs_up" and case["level"] == 9
                     and case["speed"] == 0.3 and case["repeat"] in (0, 1)]
    if len(focus_indices) != 2:
        raise ValueError("Expected exactly two student up-L9 / 0.3 m/s repeat 0/1 cases")
    with np.load(args.guarded_traces, allow_pickle=False) as archive:
        traces = {name: archive[name] for name in archive.files}
    if traces["joint_names"].tolist() != guarded["joint_names"]:
        raise ValueError("Trace joint order differs from results")
    focus = [describe_focus_case(guarded, traces, index, args.contact_threshold_n) for index in focus_indices]
    summary = dict(
        inputs={name: str(getattr(args, name)) for name in (
            "guarded_results", "guarded_traces", "no_actuator_termination_results", "reference")},
        reproduction=reproduction, stair_failures=failure_comparisons(guarded, no_termination),
        student_high_stair_slow_case_last_five_seconds=focus,
        interpretation_limits=[
            "Guarded reproduction must pass before treating the comparisons as a matched replay.",
            "Passing after suppressing actuator termination establishes a threshold-censoring contribution; it does not validate hardware safety.",
            "4 m forward exit is the probe criterion, not completion of every riser with all wheels on the platform.",
            "Wheel positions are body origins in world axes relative to initial root; they are not exact collision points or terrain clearance.",
            "Contact proportions and torque RMS here use 50 Hz final-substep samples; original results retain 200 Hz peak metrics.",
            "Continuous wheel position target minus angle is not meaningful; wheel velocity target tracking is reported instead.",
            "Position/force/torque amplitudes alone do not identify collision causality or prove which wheel is mechanically blocked.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if args.plot:
        plot_focus(guarded, traces, focus, args.output.with_suffix(".png"))
    print(json.dumps(dict(output=str(args.output), reproduced=reproduction["reproduced"],
                          guarded_stair_failure_count=len(summary["stair_failures"]),
                          focus_case_count=len(focus))))


if __name__ == "__main__":
    main()
