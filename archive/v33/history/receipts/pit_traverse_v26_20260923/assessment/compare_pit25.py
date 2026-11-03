"""Compare independent 25-cm rollouts; never interpret them as paired cases."""

import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_25cm_stochastic" / "FINALIZED.json"
CANDIDATE = HERE / "candidate_25cm_stochastic" / "FINALIZED.json"


def average(values):
    return mean(values) if values else None


def cohort(rows):
    entered = [row for row in rows if row["entered"]]
    outside = [row for row in rows if row["outside"]]
    return dict(
        episodes=len(rows), success=sum(row["success"] for row in rows),
        entered=len(entered), outside=len(outside),
        outside_after_entry=sum(row["entered"] for row in outside),
        physical=sum(row["physical"] for row in rows),
        overspeed=sum(row["overspeed"] for row in rows),
        timeout=sum(row["timeout"] for row in rows),
        command_vx_mean_mps=average([row["command_vx_mps"] for row in rows]),
        entered_top_speed_error_mean_mps=average([
            row["top_mean_abs_speed_error_mps"] for row in entered
            if row["top_mean_abs_speed_error_mps"] is not None]),
        outside_top_speed_error_mean_mps=average([
            row["top_mean_abs_speed_error_mps"] for row in outside
            if row["top_mean_abs_speed_error_mps"] is not None]),
        outside_top_travel_m_median=median([row["top_travel_m"] for row in outside])
        if outside else None,
        outside_supported_tracking_s_median=median([
            row["supported_tracking_s"] for row in outside]) if outside else None,
        near_hard_hip_seconds_per_episode=average([
            row["near_hard_hip_seconds"] for row in rows]),
        max_actual_hip_inward_rad=max(row["max_actual_hip_inward_rad"] for row in rows),
    )


def main():
    files = {"source": SOURCE, "candidate": CANDIDATE}
    reports = {name: json.loads(path.read_text()) for name, path in files.items()}
    for name, report in reports.items():
        if (report["state"] != "complete" or report["optimizer_updates"] != 0
                or not report["model_and_dual_optimizer_unchanged"]
                or report["forced_height_m"] != .25
                or report["evaluated_episodes"] != len(report["episodes"])
                or any(row["level"] != 2 for row in report["episodes"])):
            raise ValueError(f"Invalid zero-update 25-cm report: {name}")
    if (reports["source"]["completed_updates"] != 34823
            or reports["candidate"]["completed_updates"] != 35123):
        raise ValueError("Unexpected policy ages")
    result = dict(
        fixed_height_m=.25,
        action_sampling="training stochastic",
        comparison="independent batched rollouts; not paired-case causal estimate",
        source_updates=34823, candidate_updates=35123,
        by_checkpoint={},
    )
    for name, report in reports.items():
        rows = report["episodes"]
        result["by_checkpoint"][name] = {
            learner: {
                "all": cohort([row for row in rows if row["learner"] == learner]),
                "speed_bins": {
                    str(speed_bin): cohort([
                        row for row in rows if row["learner"] == learner
                        and row["speed_bin"] == speed_bin])
                    for speed_bin in range(4)
                },
            }
            for learner in ("student", "teacher")
        }
    path = HERE / "PIT25_COMPARISON.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    for name, groups in result["by_checkpoint"].items():
        for learner, groups_by_bin in groups.items():
            summary = groups_by_bin["all"]
            print(name, learner, {key: summary[key] for key in (
                "episodes", "success", "entered", "outside", "outside_after_entry",
                "physical", "overspeed", "entered_top_speed_error_mean_mps",
                "outside_top_speed_error_mean_mps", "outside_top_travel_m_median",
                "outside_supported_tracking_s_median", "near_hard_hip_seconds_per_episode")})
            print("speed_bins", {key: {metric: row[metric] for metric in (
                "episodes", "success", "outside", "physical")}
                for key, row in groups_by_bin["speed_bins"].items()})


if __name__ == "__main__":
    main()
