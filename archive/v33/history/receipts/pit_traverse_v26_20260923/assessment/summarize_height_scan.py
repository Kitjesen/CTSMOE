"""Summarize observed mixed-height simulation episodes by learner and height."""

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_height_scan" / "FINALIZED.json"
LEVELS = (0, 2, 5, 10, 15, 21)


def average(values):
    return mean(values) if values else None


def main():
    report = json.loads(SOURCE.read_text())
    if (report["state"] != "complete" or report["optimizer_updates"] != 0
            or report["completed_updates"] != 34823
            or tuple(report["forced_levels"]) != LEVELS
            or not report["model_and_dual_optimizer_unchanged"]):
        raise ValueError("Height-scan receipt does not identify the frozen source")
    rows = report["episodes"]
    result = dict(
        source_completed_updates=34823,
        action="training stochastic",
        fixed_cases=False,
        independent_seeded_rollout=True,
        by_learner_height={},
    )
    for learner in ("student", "teacher"):
        for level in LEVELS:
            items = [row for row in rows if row["learner"] == learner and row["level"] == level]
            if not items:
                raise ValueError(f"No episodes for {learner} level {level}")
            entered = [row for row in items if row["entered"]]
            result["by_learner_height"][f"{learner}:{level}"] = dict(
                height_m=round(.15 + .05 * level, 8), episodes=len(items),
                success=sum(row["success"] for row in items),
                entered=len(entered), physical=sum(row["physical"] for row in items),
                overspeed=sum(row["overspeed"] for row in items),
                outside=sum(row["outside"] for row in items),
                outside_x=sum(row["outside_x"] for row in items),
                outside_y=sum(row["outside_y"] for row in items),
                mean_top_speed_error_mps=average([
                    row["top_mean_abs_speed_error_mps"] for row in entered
                    if row["top_mean_abs_speed_error_mps"] is not None]),
                near_hard_hip_seconds_per_episode=average([
                    row["near_hard_hip_seconds"] for row in items]),
            )
    destination = HERE / "SOURCE_HEIGHT_SUMMARY.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
    for name, group in result["by_learner_height"].items():
        print(name, group)


if __name__ == "__main__":
    main()
