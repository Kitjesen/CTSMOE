"""Summarize the physical direction of candidate high-platform lane exits."""

import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
REPORT = HERE / "candidate_boundary_25cm" / "FINALIZED.json"


def middle(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return median(values) if values else None


def main():
    report = json.loads(REPORT.read_text())
    if (report["state"] != "complete" or report["optimizer_updates"] != 0
            or report["completed_updates"] != 35123
            or report["forced_height_m"] != .25
            or not report["model_and_dual_optimizer_unchanged"]):
        raise ValueError("Diagnosis has no valid zero-update receipt")
    result = {}
    for learner in ("student", "teacher"):
        rows = [row for row in report["episodes"] if row["learner"] == learner]
        exits = [row for row in rows if row["outside"]]
        wins = [row for row in rows if row["success"]]
        result[learner] = dict(
            episodes=len(rows), success=len(wins), outside=len(exits),
            outside_x=sum(row["outside_x"] for row in exits),
            outside_y=sum(row["outside_y"] for row in exits),
            outside_both=sum(row["outside_x"] and row["outside_y"] for row in exits),
            outside_after_entry=sum(row["entered"] for row in exits),
            entry_x_outside_median_m=middle(exits, "entry_x_m"),
            entry_x_success_median_m=middle(wins, "entry_x_m"),
            terminal_x_outside_median_m=middle(exits, "terminal_root_x_m"),
            terminal_y_outside_median_m=middle(exits, "terminal_root_y_m"),
            top_travel_outside_median_m=middle(exits, "top_travel_m"),
            outside_front_rear_supported=sum(
                row["front_supported"] and row["rear_supported"] for row in exits),
            outside_body_clear=sum(row["body_clear_at_end"] for row in exits),
            outside_track_s_median=middle(exits, "supported_tracking_s"),
            outside_speed_error_mps_median=middle(exits, "top_mean_abs_speed_error_mps"),
        )
    path = HERE / "BOUNDARY_DIAGNOSIS.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    for learner, values in result.items():
        print(learner, values)


if __name__ == "__main__":
    main()
