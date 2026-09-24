"""Recompute the first 25 cm pit-stop transition from frozen V21 probes.

This is read-only trace analysis. It never constructs an Isaac environment or
updates a policy. Its output is descriptive; it cannot estimate a ramp's effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parent.parent / "stair_coverage_v21_20260922"
OUT = Path(__file__).with_name("PIT_STOP_TIMING.json")
DT = 0.02
STUDENT_TRACE_START = 16  # The probe selects 16 teachers, then 16 students.


def first_student_episode(finalized: dict, env_id: int) -> dict | None:
    return next(
        (episode for episode in finalized["episodes"]
         if episode["env"] == env_id and episode["learner"] == 1),
        None,
    )


def examine(mode: str) -> dict:
    finalized = json.loads((SOURCE / f"pit_probe_{mode}_FINALIZED.json").read_text())
    if finalized["optimizer_updates"] != 0 or finalized["hold_mismatches"] != 0:
        raise ValueError(f"{mode} probe is not a valid zero-update timing receipt")
    with np.load(SOURCE / f"pit_probe_{mode}_traces.npz") as traces:
        values = traces["values"]
        command = traces["command"]
        speed = traces["xy"]
        angular = traces["angular"]
        hold = traces["hold"]
        env_ids = traces["env_ids"]
        age = traces["episode_age"]
        if env_ids.tolist() != finalized["selected_envs"]:
            raise ValueError(f"{mode} trace IDs differ from the finalized receipt")
        rows = []
        for column in range(STUDENT_TRACE_START, len(env_ids)):
            env_id = int(env_ids[column])
            episode = first_student_episode(finalized, env_id)
            if episode is None or episode["level"] != 2:
                continue
            stop = values[:, column, 7]
            first_stops = np.flatnonzero(stop & ~np.r_[False, stop[:-1]])
            if not len(first_stops):
                raise ValueError(f"{mode} env {env_id} never received a stop latch")
            step = int(first_stops[0])
            if age[step, column] < 1:
                raise ValueError(f"{mode} env {env_id} stopped outside an episode")
            end = min(step + 100, len(values))
            local_speed = speed[step:end, column]
            below = np.flatnonzero(local_speed < 0.15)
            zero = np.flatnonzero(np.all(command[step:end, column] == 0, axis=1))
            if not len(below) or not len(zero):
                raise ValueError(f"{mode} env {env_id} lacks a braking event")
            first_low = int(below[0])
            lost_support = np.flatnonzero(~values[step:end, column, 0])
            half_second = slice(step, min(step + 25, len(values)))
            rows.append({
                "env_id": env_id,
                "height_m": 0.25,
                "held_command_mps": float(episode["speed"]),
                "speed_at_latch_mps": float(local_speed[0]),
                "angular_at_latch_rad_s": float(angular[step, column]),
                "zero_command_lag_steps": int(zero[0]),
                "first_below_0p15_s": first_low * DT,
                "path_length_until_low_m": float(np.sum(local_speed[: first_low + 1]) * DT),
                "first_support_loss_s": float(lost_support[0] * DT) if len(lost_support) else None,
                "support_lost_before_low": bool(len(lost_support) and lost_support[0] < first_low),
                "support_fraction_first_0p5s": float(np.mean(values[half_second, column, 0])),
                "quiet_angular_fraction_first_0p5s": float(np.mean(values[half_second, column, 4])),
                "max_hold_in_trace_s": float(np.max(hold[:, column]) * DT),
            })
    fields = (
        "held_command_mps", "speed_at_latch_mps", "angular_at_latch_rad_s",
        "first_below_0p15_s", "path_length_until_low_m",
        "support_fraction_first_0p5s", "quiet_angular_fraction_first_0p5s",
        "max_hold_in_trace_s",
    )
    return {
        "source_sha256": finalized["source_sha256"],
        "count": len(rows),
        "medians": {field: float(np.median([row[field] for row in rows])) for field in fields},
        "zero_command_lag_steps": sorted({row["zero_command_lag_steps"] for row in rows}),
        "support_lost_before_low_count": sum(row["support_lost_before_low"] for row in rows),
        "support_lost_within_0p5s_count": sum(
            row["first_support_loss_s"] is not None and row["first_support_loss_s"] < 0.5
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    modes = {mode: examine(mode) for mode in ("stochastic", "mean")}
    if modes["stochastic"]["source_sha256"] != modes["mean"]["source_sha256"]:
        raise ValueError("The probes use different checkpoints")
    if [row["env_id"] for row in modes["stochastic"]["rows"]] != [
        row["env_id"] for row in modes["mean"]["rows"]
    ]:
        raise ValueError("The 25 cm student cohorts do not match")
    result = {
        "source": "34823 frozen policy; zero-optimizer paired diagnostics",
        "scope": "14 selected 25 cm student first stop events; not population success rates",
        "time_step_s": DT,
        "modes": modes,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({mode: {k: v for k, v in data.items() if k != "rows"}
                      for mode, data in modes.items()}, indent=2))


if __name__ == "__main__":
    main()
