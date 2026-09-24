"""Gate the finite V27 trial on independently simulated 25 cm pit episodes."""

import argparse
import json
from pathlib import Path


def load(path):
    value = json.loads(Path(path).read_text())
    if (value["state"] != "complete" or value["optimizer_updates"] != 0
            or value["forced_height_m"] != .25
            or not value["model_and_dual_optimizer_unchanged"]
            or value["steps"] < 750):
        raise ValueError(f"Incomplete zero-update 25 cm receipt: {path}")
    return value


def summarize(value, learner):
    rows = [r for r in value["episodes"] if r["learner"] == learner]
    if len(rows) < (200 if learner == "student" else 600):
        raise ValueError(f"Too few {learner} episodes: {len(rows)}")
    n = len(rows)
    top = [r["top_mean_abs_speed_error_mps"] for r in rows
           if r["top_mean_abs_speed_error_mps"] is not None]
    def mean(key):
        return sum(r[key] for r in rows) / n
    return {
        "episodes": n,
        "complete": sum(r["success"] for r in rows),
        "success_rate": mean("success"),
        "outside": sum(r["outside"] for r in rows),
        "outside_rate": mean("outside"),
        "physical": sum(r["physical"] for r in rows),
        "physical_rate": mean("physical"),
        "near_hard_hip_s_per_episode": mean("near_hard_hip_seconds"),
        "near_hard_pre_entry_s_per_episode": mean("near_hard_hip_pre_entry_seconds"),
        "near_hard_post_entry_s_per_episode": mean("near_hard_hip_post_entry_seconds"),
        "top_vx_mae_mps": sum(top) / len(top) if top else None,
        "max_leg_torque_Nm": max(r["max_leg_torque_Nm"] for r in rows),
        "max_wheel_torque_Nm": max(r["max_wheel_torque_Nm"] for r in rows),
        "max_leg_speed_rad_s": max(r["max_leg_speed_rad_s"] for r in rows),
        "max_wheel_speed_rad_s": max(r["max_wheel_speed_rad_s"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source, candidate = load(args.source), load(args.candidate)
    if (source["completed_updates"] != 34823
            or candidate["completed_updates"] != 35123
            or source["steps"] != candidate["steps"]):
        raise ValueError("Mismatched checkpoint ages or rollout lengths")
    result = {name: {learner: summarize(value, learner)
                     for learner in ("student", "teacher")}
              for name, value in (("source", source), ("candidate", candidate))}
    checks = {}
    for learner in ("student", "teacher"):
        src = result["source"][learner]
        dst = result["candidate"][learner]
        checks[learner] = {
            "success": dst["success_rate"] >= src["success_rate"] - .02,
            "outside": dst["outside_rate"] <= max(.01, src["outside_rate"] + .005),
            "physical": dst["physical_rate"] <= src["physical_rate"] + .02,
            "near_hard_total": dst["near_hard_hip_s_per_episode"] <= max(
                .25, src["near_hard_hip_s_per_episode"] * 1.5),
            "near_hard_post_entry": dst["near_hard_post_entry_s_per_episode"] <= max(
                .05, src["near_hard_post_entry_s_per_episode"] * 1.5),
            "top_vx": dst["top_vx_mae_mps"] is not None
            and src["top_vx_mae_mps"] is not None
            and dst["top_vx_mae_mps"] <= src["top_vx_mae_mps"] + .03,
        }
    result["checks"] = checks
    result["pass"] = all(all(row.values()) for row in checks.values())
    result["scope"] = "Independent seeded simulator cohorts, not matched episodes or real hardware"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
