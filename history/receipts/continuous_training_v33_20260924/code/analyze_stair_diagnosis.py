"""Measure first-episode stair execution and command-path counterfactuals."""
import argparse
import json
from pathlib import Path

import numpy as np


def yaw_of(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def commanded_path(cmd, start_xy, start_yaw, dt, observed_yaw=None):
    """Planar midpoint integration; a reference, not physical feasibility."""
    yaw = start_yaw + np.cumsum(cmd[:, 2]) * dt - cmd[:, 2] * dt / 2
    if observed_yaw is not None:
        yaw = observed_yaw
    c, s = np.cos(yaw), np.sin(yaw)
    velocity = np.column_stack((c * cmd[:, 0] - s * cmd[:, 1],
                                s * cmd[:, 0] + c * cmd[:, 1]))
    return start_xy + np.cumsum(velocity * dt, axis=0)


def analyze(directory):
    meta = json.loads((directory / "manifest.json").read_text())
    data = np.load(directory / "traces.npz")
    dt = meta["dt"]
    radius = meta["max_radius_threshold_m"]
    bodies = meta["body_names"]
    joints = meta["joint_names"]
    legs = [i for i, name in enumerate(joints) if any(part in name.lower() for part in ("hip", "thigh", "calf"))]
    wheels = [i for i, name in enumerate(joints) if any(part in name.lower() for part in ("wheel", "foot"))]
    assert len(legs) == 12 and len(wheels) == 4
    calves = [i for i, name in enumerate(bodies) if "calf" in name.lower()]
    assert len(calves) == 4
    history = data["history"]
    rows = []
    for i in range(meta["num_envs"]):
        h_full = history[:, i, :].astype(np.float64)
        n = int(data["end_step"][i])
        h = h_full[:n]
        origin = data["origin_xy"][i]
        start = data["initial_position"][i]
        q = data["initial_quaternion"][i]
        actual_radii = np.linalg.norm(h[:, 3:5] - origin, axis=1)
        crossing = np.flatnonzero(actual_radii > radius)
        passed = bool(crossing.size)
        # State and failure statistics stop at first 4 m crossing. Otherwise
        # post-success travel can leave this terrain patch and confound failure.
        first_pass_step = int(crossing[0]) + 1 if passed else None
        observed_steps = first_pass_step or n
        state = h[:observed_steps]
        cmd_ref = commanded_path(h_full[:, :3], start[:2], yaw_of(q), dt)
        yaw_ref = commanded_path(h[:, :3], start[:2], yaw_of(q), dt, yaw_of(h[:, 6:10]))
        ideal_radius = np.linalg.norm(cmd_ref - origin, axis=1)
        observed_yaw_radius = np.linalg.norm(yaw_ref - origin, axis=1)
        ideal_full_pass = bool(np.any(ideal_radius > radius))
        physical_failure = bool(data["terminated"][i]) and not passed
        category = "crossed_4m" if passed else (
            "terminated_before_4m" if physical_failure else (
                "survived_but_command_reference_below_4m" if not ideal_full_pass
                else "survived_command_reference_above_4m_execution_below_4m"))
        joint_clip = np.abs(state[:, 14:30] - state[:, 30:46]) > 0.001
        tracking_error = state[:, 10:13] - state[:, :3]
        row = dict(
            env=i, category=category, crossed_4m=passed,
            first_pass_seconds=first_pass_step * dt if passed else None,
            first_terminal_seconds=n * dt, terminated_before_4m=physical_failure,
            terminated_at_any_time=bool(data["terminated"][i]),
            timeout=bool(data["timeout"][i]),
            actual_max_radius_m=float(actual_radii[:observed_steps].max()),
            actual_path_m=float(np.linalg.norm(np.diff(np.vstack((start[:2], state[:, 3:5])), axis=0), axis=1).sum()),
            initial_xy_from_origin=(start[:2] - origin).tolist(),
            initial_yaw_rad=float(yaw_of(q)),
            height_gain_m=float(state[-1, 5] - start[2]),
            command_reference_max_radius_25s_m=float(ideal_radius.max()),
            command_reference_max_radius_first_life_m=float(ideal_radius[:n].max()),
            command_reference_crosses_4m_25s=ideal_full_pass,
            observed_yaw_reference_max_radius_first_life_m=float(observed_yaw_radius.max()),
            observed_yaw_reference_crosses_4m_first_life=bool(np.any(observed_yaw_radius > radius)),
            command_path_25s_m=float(np.linalg.norm(h_full[:, :2], axis=1).sum() * dt),
            tracking_rmse_vx_vy_yaw=np.sqrt(np.square(tracking_error).mean(axis=0)).tolist(),
            leg_clip_fraction=float(joint_clip[:, legs].mean()),
            wheel_clip_fraction=float(joint_clip[:, wheels].mean()),
            leg_peak_applied_torque_nm=float(np.abs(state[:, 30:46][:, legs]).max()),
            calf_contact_over_5n_fraction=float((state[:, 46:46 + len(bodies)][:, calves] > 5).any(axis=1).mean()),
            maximum_tilt_deg=float(np.rad2deg(np.arccos(np.clip(state[:, 13], -1, 1))).max()),
        )
        rows.append(row)
    categories = {name: sum(r["category"] == name for r in rows) for name in sorted({r["category"] for r in rows})}
    passed_times = [r["first_pass_seconds"] for r in rows if r["crossed_4m"]]
    zones = {}
    for name, low, high in (("central_root_square", 0., 1.3), ("outer_stair_region", 1.8, 100.)):
        samples = []
        for i, row in enumerate(rows):
            n = round(row["first_pass_seconds"] / dt) if row["crossed_4m"] else int(data["end_step"][i])
            h = history[:n, i, :]
            square_radius = np.max(np.abs(h[:, 3:5] - data["origin_xy"][i]), axis=1)
            samples.append(h[(square_radius >= low) & (square_radius < high)])
        h = np.concatenate(samples)
        assert len(h) > 0
        component_response = []
        for j in range(3):
            selected = np.abs(h[:, j]) > 0.15
            component_response.append(dict(
                samples=int(selected.sum()),
                mean_abs_command=float(np.abs(h[selected, j]).mean()) if selected.any() else None,
                mean_velocity_along_command_sign=float((h[selected, 10 + j] * np.sign(h[selected, j])).mean()) if selected.any() else None,
            ))
        zones[name] = dict(samples=len(h), root_square_radius_range_m=[low, high],
                           tracking_rmse_vx_vy_yaw=np.sqrt(np.square(h[:, 10:13] - h[:, :3]).mean(axis=0)).tolist(),
                           calf_contact_over_5n_fraction=float((h[:, 46:46 + len(bodies)][:, calves] > 5).any(axis=1).mean()),
                           component_response=component_response)
    result = dict(
        commands=meta["commands"], randomization=meta["randomization"],
        num_envs=meta["num_envs"], level=meta["level"], seed=meta["seed"],
        policy_sha256=meta["policy_sha256"], categories=categories,
        crossed_4m=sum(r["crossed_4m"] for r in rows),
        command_reference_crosses_4m_25s=sum(r["command_reference_crosses_4m_25s"] for r in rows),
        terminated_before_4m=sum(r["terminated_before_4m"] for r in rows),
        pass_seconds_mean=float(np.mean(passed_times)) if passed_times else None,
        mean_actual_max_radius_m=float(np.mean([r["actual_max_radius_m"] for r in rows])),
        mean_actual_path_m=float(np.mean([r["actual_path_m"] for r in rows])),
        mean_command_reference_max_radius_25s_m=float(np.mean([r["command_reference_max_radius_25s_m"] for r in rows])),
        mean_tracking_rmse_vx_vy_yaw=np.mean([r["tracking_rmse_vx_vy_yaw"] for r in rows], axis=0).tolist(),
        mean_leg_clip_fraction=float(np.mean([r["leg_clip_fraction"] for r in rows])),
        mean_wheel_clip_fraction=float(np.mean([r["wheel_clip_fraction"] for r in rows])),
        mean_calf_contact_over_5n_fraction=float(np.mean([r["calf_contact_over_5n_fraction"] for r in rows])),
        mean_maximum_tilt_deg=float(np.mean([r["maximum_tilt_deg"] for r in rows])),
        zones=zones,
        caveat="One fixed level/terrain and seed, deterministic student actions. Command paths are planar kinematic references, not physically guaranteed targets. All state/failure metrics stop at first 4 m crossing or first termination; reference paths retain the full 25 s command schedule. Contact and torque are sampled at the last physics substep of each 20 ms control step.",
        episodes=rows,
    )
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return {k: v for k, v in result.items() if k != "episodes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([analyze(p) for p in args.directories], indent=2))


if __name__ == "__main__":
    main()
