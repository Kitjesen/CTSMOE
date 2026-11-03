"""Summarize exact audit counts; do not treat adjacent states as independent trials."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(rows):
    count = sum(row["states"] for row in rows)
    if not count:
        return {"states": 0}
    signals = rows[0]["means"]
    return {
        "states": count,
        "means": {key: sum(row["means"][key] * row["states"] for row in rows) / count
                  for key in signals},
        "maxima": {key: max(row["maxima"][key] for row in rows) for key in signals},
        "episode_ends": sum(row["episode_ends"] for row in rows),
        "physical_ends": sum(row["physical_ends"] for row in rows),
    }


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "distribution.json").read_text(encoding="utf-8"))
    groups = data["groups"]
    special = [row for row in groups if row["cell"][:2] == [1, 1]]
    ordinary = [row for row in groups if row["cell"][:2] == [1, 0]]
    by_age = []
    for age, label in enumerate(data["age_bins"]):
        selected = [row for row in special if row["cell"][2] == age]
        by_age.append({
            "age": label,
            "all": aggregate(selected),
            "zero_command": aggregate([row for row in selected if row["cell"][4] == 0]),
            "moving_phase": aggregate([row for row in selected if row["cell"][3] == 1]),
            "forward_0.45_to_0.75": aggregate([
                row for row in selected if row["cell"][4:6] == [1, 1]]),
        })
    summary = {
        "scope": "480 rollout steps; frozen V11; correlated state counts, not success rates",
        "student_special": aggregate(special),
        "student_ordinary": aggregate(ordinary),
        "student_special_command_segment_at_least_3s": aggregate([
            row for row in special if row["cell"][7] >= 4]),
        "student_ordinary_forward_0.45_to_0.75": aggregate([
            row for row in ordinary if row["cell"][4:6] == [1, 1]]),
        "by_episode_age": by_age,
        "candidate": data["candidate"],
        "critical": data["critical_coverage"],
        "coverage_gate": data["coverage_gate"],
    }
    (directory / "count_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = by_age[:-1]
    labels = [row["age"] for row in rows]
    counts = np.array([row["all"]["states"] for row in rows])
    zeros = np.array([row["zero_command"]["states"] for row in rows])
    forward = np.array([row["forward_0.45_to_0.75"]["states"] for row in rows])
    other = counts - zeros - forward
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for values, bottom, name, color in [
        (zeros, np.zeros(len(rows)), "Zero command", "#aeb9c3"),
        (other, zeros, "Other nonzero commands", "#5c8ba1"),
        (forward, zeros + other, "Straight forward 0.45-0.75 m/s", "#e08d42"),
    ]:
        axes[0].bar(x, 100 * values / counts, bottom=100 * bottom / counts,
                    label=name, color=color)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 100),
                xlabel="Episode age (s)", ylabel="Share of sampled states (%)",
                title="Student startup cohort: command coverage")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].bar([0, 1], [data["critical_coverage"]["distinct_envs"], 32],
                color=["#e08d42", "#8797a6"])
    axes[1].set(xticks=[0, 1], xticklabels=["Observed", "Pre-registered minimum"],
                ylabel="Distinct student environments",
                title="Critical candidate coverage at episode age 2.5-3.0 s", ylim=(0, 38))
    for position, value in enumerate([data["critical_coverage"]["distinct_envs"], 32]):
        axes[1].text(position, value + .7, str(value), ha="center")
    fig.suptitle("Zero optimizer updates | 4096 environments x 480 control steps", fontsize=14)
    fig.savefig(directory / "coverage.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"critical_envs": data["critical_coverage"]["distinct_envs"],
                      "critical_age_zero_fraction": zeros[4] / counts[4],
                      "coverage_gate": data["coverage_gate"]}))


if __name__ == "__main__":
    main()
