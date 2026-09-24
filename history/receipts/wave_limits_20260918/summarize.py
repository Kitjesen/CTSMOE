"""Summarize bounded wave/limits probe evidence without simulation dependencies."""
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(path):
    data = json.loads(path.read_text())
    names = data["joint_names"]
    groups = {}
    for learner in ("student", "teacher"):
        for terrain in ("wave", "stairs_up", "stairs_down"):
            cases = [c for c in data["cases"] if c["learner"] == learner and c["terrain"] == terrain]
            assert len(cases) == 27
            stats = dict(trials=len(cases), forward_exits=sum(c["forward_exit"] for c in cases),
                         physical_failures=sum(c["physical_failure"] for c in cases),
                         mean_vx_mae=statistics.mean(c["vx_mae_mps"] for c in cases),
                         median_peak_cross_m=statistics.median(c["max_cross_m"] for c in cases),
                         worst_peak_cross_m=max(c["max_cross_m"] for c in cases),
                         worst_tilt_deg=max(c["max_roll_or_pitch_deg"] for c in cases))
            stats["by_level"] = {}
            for level in (3, 6, 9):
                rows = [c for c in cases if c["level"] == level]
                stats["by_level"][level] = dict(trials=len(rows), exits=sum(c["forward_exit"] for c in rows),
                    failures=sum(c["physical_failure"] for c in rows),
                    mean_vx_mae=statistics.mean(c["vx_mae_mps"] for c in rows),
                    median_peak_cross_m=statistics.median(c["max_cross_m"] for c in rows),
                    worst_peak_cross_m=max(c["max_cross_m"] for c in rows))
            stats["joint_groups"] = {}
            for kind in ("hip", "thigh", "calf", "foot"):
                indices = [i for i, n in enumerate(names) if n.endswith(kind + "_joint")]
                denom = sum(c["physics_samples"] for c in cases) * len(indices)
                row = {}
                for key in cases[0]["joints"]:
                    values = [c["joints"][key][i] for c in cases for i in indices]
                    assert all(math.isfinite(v) for v in values)
                    if key.startswith("max_"):
                        row[key] = max(values)
                    else:
                        row[key] = sum(values)
                        row[key + "_pct"] = 100 * sum(values) / denom
                        row[key + "_trials"] = sum(any(c["joints"][key][i] > 0 for i in indices) for c in cases)
                row["position_excess_degrees"] = {k: row[k] * 180 / math.pi for k in (
                    "max_hard_excess_rad", "max_soft_excess_rad", "max_target_hard_excess_rad")}
                stats["joint_groups"][kind] = row
            groups[learner + "_" + terrain] = stats
    result = dict(completed_updates=data["completed_updates"], groups=groups)
    path.with_name("summary.json").write_text(json.dumps(result, indent=2))
    for name, row in groups.items():
        print(json.dumps({"group": name, **{k: v for k, v in row.items() if k not in ("joint_groups", "by_level")}}))
        for kind, joint in row["joint_groups"].items():
            print(json.dumps({"joint_group": kind, **{k: joint[k] for k in (
                "position_excess_degrees", "max_speed_rad_s", "max_applied_torque_nm",
                "hard_excess_samples_pct", "soft_excess_samples_pct", "speed_excess_samples_pct",
                "torque_excess_samples_pct", "target_excess_samples_pct", "torque_clipped_samples_pct")}}))


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
