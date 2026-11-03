"""Validate and summarize the evaluation-only V11 late-failure probe."""

import argparse
import json
from pathlib import Path

import numpy as np


WINDOWS = (
    ("0_1s", 0.0, 1.0),
    ("1_2s", 1.0, 2.0),
    ("2p2_2p6s", 2.2, 2.6),
    ("2p6_3p3s", 2.6, 3.3),
    ("3p3_4s", 3.3, 4.0),
    ("4_6s", 4.0, 6.0),
)


def scalar(value):
    """Convert a finite NumPy scalar to JSON, preserving missing windows as null."""
    value = float(value)
    return value if np.isfinite(value) else None


def stats(values):
    """Return mean, RMS, and extrema for a non-empty vector."""
    if not len(values):
        return None
    return {
        "mean": scalar(np.mean(values)),
        "rms": scalar(np.sqrt(np.mean(np.square(values)))),
        "max": scalar(np.max(values)),
        "min": scalar(np.min(values)),
    }


def interval_mask(times, lo, hi):
    """Use half-open intervals so adjacent diagnostic windows do not overlap."""
    return (times >= lo) & (times < hi)


def top1_summary(weights):
    if not len(weights):
        return None
    top = np.argmax(weights, axis=-1)
    counts = np.bincount(top, minlength=weights.shape[-1])
    return {
        "mode": int(np.argmax(counts)),
        "fractions": (counts / counts.sum()).tolist(),
    }


def summarize_window(mask, signals):
    if not mask.any():
        return None
    weights = signals["gate"][mask]
    sorted_weights = np.sort(weights, axis=-1)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=-1)
    gate_tv = np.abs(np.diff(weights, axis=0)).sum(axis=-1)
    return {
        "actual_pose_rms_rad": stats(signals["actual_rms"][mask]),
        "requested_pose_rms_rad": stats(signals["requested_rms"][mask]),
        "target_pose_rms_rad": stats(signals["target_rms"][mask]),
        "xy_tracking_error_mps": stats(signals["xy_error"][mask]),
        "yaw_tracking_error_radps": stats(signals["yaw_error"][mask]),
        "v11_student_teacher_action_l2": stats(signals["action_st"][mask]),
        "v11_student_v6_action_l2": stats(signals["action_sv6"][mask]),
        "v11_student_v9_action_l2": stats(signals["action_sv9"][mask]),
        "v9_student_teacher_action_l2": stats(signals["action_v9st"][mask]),
        "v11_latent_l2": stats(signals["latent_l2"][mask]),
        "v11_latent_cosine": stats(signals["latent_cos"][mask]),
        "gate_entropy": stats(entropy),
        "gate_top1": top1_summary(weights),
        "gate_top1_margin": stats(sorted_weights[:, -1] - sorted_weights[:, -2]),
        "gate_step_tv": stats(gate_tv),
    }


def peak(times, values):
    if not len(values):
        return None
    index = int(np.argmax(values))
    return {"time_s": scalar(times[index]), "value": scalar(values[index])}


parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
if manifest["smoke_only"]:
    raise ValueError("Smoke traces are not a six-second diagnosis")
with np.load(args.directory / "traces.npz") as loaded:
    data = {key: loaded[key] for key in loaded.files}

assert len(manifest["cases"]) == 64
assert manifest["initial_state_matched"]
assert len({case["id"] for case in manifest["cases"]}) == 64
core_counts = {}
for case in manifest["cases"]:
    core_counts.setdefault(case["direction"], 0)
    core_counts[case["direction"]] += case["intervention_time"] not in (2.6, 3.0)
assert sorted(core_counts.values()) == [10] * 6, core_counts

joint_names = manifest["joint_names"]
leg_ids = np.array([i for i, name in enumerate(joint_names) if not name.endswith("_foot_joint")])
assert len(leg_ids) == 12, [joint_names[i] for i in leg_ids]
request_ids = np.asarray(manifest["request_joint_ids"], dtype=int)
assert sorted(request_ids.tolist()) == sorted(leg_ids.tolist())
default_q = np.asarray(manifest["default_q"])
controller_ids = manifest["controller_ids"]
controller_names = {value: key for key, value in controller_ids.items()}

essential = (
    "joint_pos",
    "joint_vel",
    "target",
    "requested",
    "raw_action",
    "command",
    "velocity",
    "angular_velocity",
    "root_clearance",
    "action_v6_student",
    "action_v9_student",
    "action_v9_teacher",
    "action_v11_student",
    "action_v11_teacher",
    "latent_v11_student",
    "latent_v11_teacher",
    "latent_v11_delta",
    "gate_v11",
    "actual_controller_id",
)
assert set(manifest["observation_traces"].values()) <= set(data)

baseline = {
    case["direction"]: case
    for case in manifest["cases"]
    if case["controller_source"] == "v11_student" and case["intervention_target"] is None
}
rows = []
for case in manifest["cases"]:
    index = case["id"]
    count = case["end_step"] + 1
    times = (np.arange(count) + 1) * manifest["dt"]
    for key in essential:
        assert np.isfinite(data[key][:count, index]).all(), (case["id"], key)
    assert data["active"][:count, index].all(), case["id"]
    expected_command = np.broadcast_to(np.asarray(case["command"]), data["command"][:count, index].shape)
    np.testing.assert_allclose(data["command"][:count, index], expected_command, atol=1e-6)

    source_id = controller_ids[case["controller_source"]]
    expected_ids = np.full(count, source_id)
    if case["intervention_time"] is not None:
        expected_ids[times - manifest["dt"] + 1e-8 >= case["intervention_time"]] = controller_ids[
            case["intervention_target"]
        ]
    actual_ids = data["actual_controller_id"][:count, index].astype(int)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    action_arrays = {
        controller_ids["v6_student"]: data["action_v6_student"][:count, index],
        controller_ids["v9_student"]: data["action_v9_student"][:count, index],
        controller_ids["v11_student"]: data["action_v11_student"][:count, index],
        controller_ids["v11_teacher"]: data["action_v11_teacher"][:count, index],
    }
    expected_action = np.empty_like(data["raw_action"][:count, index])
    for controller_id, candidate in action_arrays.items():
        expected_action[actual_ids == controller_id] = candidate[actual_ids == controller_id]
    np.testing.assert_allclose(data["raw_action"][:count, index], expected_action, atol=1e-6)

    gate = data["gate_v11"][:count, index]
    np.testing.assert_allclose(gate.sum(axis=-1), 1.0, atol=1e-5)
    assert (gate >= 0).all(), case["id"]
    actual_q = data["joint_pos"][:count, index][:, leg_ids] - default_q[leg_ids]
    requested_q = data["requested"][:count, index] - default_q[request_ids]
    target_q = data["target"][:count, index][:, leg_ids] - default_q[leg_ids]
    actual_rms = np.sqrt(np.mean(np.square(actual_q), axis=-1))
    requested_rms = np.sqrt(np.mean(np.square(requested_q), axis=-1))
    target_rms = np.sqrt(np.mean(np.square(target_q), axis=-1))
    command = data["command"][:count, index]
    xy_error = np.linalg.norm(data["velocity"][:count, index, :2] - command[:, :2], axis=-1)
    yaw_error = np.abs(data["angular_velocity"][:count, index, 2] - command[:, 2])

    def action_gap(left, right):
        return np.linalg.norm(data[left][:count, index] - data[right][:count, index], axis=-1)

    latent_s = data["latent_v11_student"][:count, index]
    latent_t = data["latent_v11_teacher"][:count, index]
    np.testing.assert_allclose(data["latent_v11_delta"][:count, index], latent_s - latent_t, atol=1e-6)
    latent_l2 = np.linalg.norm(latent_s - latent_t, axis=-1)
    latent_cos = np.sum(latent_s * latent_t, axis=-1) / np.clip(
        np.linalg.norm(latent_s, axis=-1) * np.linalg.norm(latent_t, axis=-1), 1e-12, None
    )
    signals = {
        "actual_rms": actual_rms,
        "requested_rms": requested_rms,
        "target_rms": target_rms,
        "xy_error": xy_error,
        "yaw_error": yaw_error,
        "action_st": action_gap("action_v11_student", "action_v11_teacher"),
        "action_sv6": action_gap("action_v11_student", "action_v6_student"),
        "action_sv9": action_gap("action_v11_student", "action_v9_student"),
        "action_v9st": action_gap("action_v9_student", "action_v9_teacher"),
        "latent_l2": latent_l2,
        "latent_cos": latent_cos,
        "gate": gate,
    }
    report_windows = {
        name: summarize_window(interval_mask(times, lo, hi), signals) for name, lo, hi in WINDOWS
    }
    report_windows["startup_0_1s"] = summarize_window(times <= 1.0, signals)
    report_windows["middle_1_4s"] = summarize_window((times > 1.0) & (times <= 4.0), signals)
    report_windows["last_second"] = summarize_window(times > max(0.0, times[-1] - 1.0), signals)

    pre_switch = None
    post_switch = None
    if case["intervention_time"] is not None:
        reference = baseline[case["direction"]]
        reference_count = reference["end_step"] + 1
        before = min(int(round(case["intervention_time"] / manifest["dt"])), count, reference_count)
        if before:
            joint_delta = data["joint_pos"][:before, index][:, leg_ids] - data["joint_pos"][
                :before, reference["id"]
            ][:, leg_ids]
            velocity_delta = data["velocity"][:before, index] - data["velocity"][:before, reference["id"]]
            clearance_delta = data["root_clearance"][:before, index] - data["root_clearance"][
                :before, reference["id"]
            ]
            pre_switch = {
                "reference_case_id": reference["id"],
                "samples": before,
                "joint_rms_difference_rad": scalar(np.sqrt(np.mean(np.square(joint_delta)))),
                "joint_max_difference_rad": scalar(np.max(np.abs(joint_delta))),
                "velocity_rms_difference_mps": scalar(np.sqrt(np.mean(np.square(velocity_delta)))),
                "clearance_max_difference_m": scalar(np.max(np.abs(clearance_delta))),
                "interpretation": "Independent matched environments; this measures agreement and does not assume exact identity.",
            }
        post_mask = (times >= case["intervention_time"] + 0.5) & (
            times < case["intervention_time"] + 1.5
        )
        post_switch = summarize_window(post_mask, signals)

    rows.append(
        {
            "case_id": index,
            "direction": case["direction"],
            "command": case["command"],
            "controller_source": case["controller_source"],
            "intervention_target": case["intervention_target"],
            "intervention_time_s": case["intervention_time"],
            "duration_s": scalar(times[-1]),
            "physical": case["physical"],
            "terms": case["terms"],
            "controller_sequence": [
                controller_names[value]
                for position, value in enumerate(actual_ids)
                if position == 0 or value != actual_ids[position - 1]
            ],
            "windows": report_windows,
            "peaks": {
                "actual_pose_rms_rad": peak(times, actual_rms),
                "xy_tracking_error_mps": peak(times, xy_error),
                "yaw_tracking_error_radps": peak(times, yaw_error),
                "v11_student_teacher_action_l2": peak(times, signals["action_st"]),
                "v11_latent_l2": peak(times, latent_l2),
            },
            "pre_switch_match": pre_switch,
            "post_switch_after_0p5s": post_switch,
        }
    )

analysis = {
    "purpose": "evaluation-only diagnosis; no training acceptance decision is produced",
    "window_definition": "post-step samples at t=(step+1)*dt; diagnostic windows are half-open",
    "core_commands": {direction: [row for row in rows if row["direction"] == direction][:10] for direction in baseline},
    "extra_forward_0p6_switches": [
        row for row in rows if row["intervention_time_s"] in (2.6, 3.0)
    ],
    "cases": rows,
}
(args.directory / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
validation = {
    "valid": True,
    "cases": len(rows),
    "core_cases_per_command": core_counts,
    "checks": [
        "matched-initial-state manifest assertion",
        "finite essential telemetry through each first-episode truncation",
        "constant requested command",
        "executed controller timeline and exact action-source equality",
        "V11 MoE gate probability simplex",
        "six commands with ten core cases each plus four forward extras",
    ],
    "training_pass_inferred": False,
}
(args.directory / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
print(json.dumps(validation, indent=2))
