"""Classify lateral exits by measured heading and body-side velocity contributions."""

import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
INPUT = HERE / "candidate_lateral_25cm" / "FINALIZED.json"


def middle_abs(rows, key):
    return median(abs(row[key]) for row in rows) if rows else None


def main():
    report = json.loads(INPUT.read_text())
    if (report["state"] != "complete" or report["optimizer_updates"] != 0
            or report["completed_updates"] != 35123
            or not report["model_and_dual_optimizer_unchanged"]):
        raise ValueError("No frozen candidate lateral diagnosis")
    result = {}
    for learner in ("student", "teacher"):
        rows = [row for row in report["episodes"] if row["learner"] == learner]
        result[learner] = {}
        for group_name, selected in (
                ("outside", [row for row in rows if row["outside"] and row["entered"]]),
                ("success", [row for row in rows if row["success"]])):
            result[learner][group_name] = dict(
                episodes=len(selected),
                post_entry_s_median=median(row["post_entry_s"] for row in selected)
                if selected else None,
                side_displacement_abs_m_median=middle_abs(
                    selected, "post_entry_side_displacement_m"),
                heading_component_abs_m_median=middle_abs(selected, "heading_side_component_m"),
                body_side_component_abs_m_median=middle_abs(selected, "body_side_component_m"),
                world_side_integral_abs_m_median=middle_abs(
                    selected, "world_side_velocity_integral_m"),
                heading_dominant_count=sum(
                    abs(row["heading_side_component_m"]) >
                    abs(row["body_side_component_m"]) for row in selected),
                body_vy_abs_integral_m_median=median(
                    row["abs_body_vy_integral_m"] for row in selected) if selected else None,
                yaw_rate_abs_integral_rad_median=median(
                    row["abs_yaw_rate_integral_rad"] for row in selected) if selected else None,
            )
    path = HERE / "LATERAL_DECOMPOSITION.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    for learner, groups in result.items():
        for group_name, values in groups.items():
            print(learner, group_name, values)


if __name__ == "__main__":
    main()
