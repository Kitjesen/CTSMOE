"""Summarize signed trajectory components; do not treat cancellation as percentages."""
import json
import math
from pathlib import Path
import statistics
import sys


def analyze(path):
    data = json.loads(path.read_text())
    result = {"mode": data["diagnostic_mode"], "completed_updates": data["completed_updates"], "groups": {}}
    residual = max(abs(c["peak_integration_residual_m"]) for c in data["cases"])
    assert residual < 0.02, f"Position/velocity integration discrepancy: {residual} m"
    result["peak_integration_max_residual_m"] = residual
    for learner in ("student", "teacher"):
        for terrain in ("wave", "stairs_up", "stairs_down"):
            rows = [c for c in data["cases"] if c["learner"] == learner and c["terrain"] == terrain]
            peak = max(rows, key=lambda c: abs(c["peak_cross_signed_m"]))
            d = peak["decomposition_at_peak"]
            worst = {k: peak[k] for k in ("level", "speed", "start_lateral_m", "column", "peak_cross_signed_m",
                                         "peak_time_s", "peak_integration_residual_m")}
            worst.update(heading_at_peak_deg=math.degrees(peak["heading_at_peak_rad"]),
                         max_heading_deg=math.degrees(peak["max_heading_rad"]),
                         heading_from_command_m=d["heading_command_m"],
                         heading_from_speed_error_m=d["heading_speed_error_m"],
                         heading_actual_motion_m=d["heading_command_m"] + d["heading_speed_error_m"],
                         lateral_motion_m=d["lateral_motion_m"],
                         body_vx_mae=d["body_vx_abs_error_integral"] / peak["peak_time_s"],
                         body_vy_mae=d["body_vy_abs_integral"] / peak["peak_time_s"],
                         body_wz_mae=d["body_wz_abs_error_integral"] / peak["peak_time_s"],
                         body_wz_signed_mean=d["body_wz_signed_integral"] / peak["peak_time_s"])
            group = dict(trials=len(rows), median_peak_cross_m=statistics.median(abs(c["peak_cross_signed_m"]) for c in rows),
                         worst_peak_cross_m=abs(peak["peak_cross_signed_m"]),
                         failures=sum(c["physical_failure"] for c in rows),
                         exits=sum(c["forward_exit"] for c in rows),
                         median_max_heading_deg=statistics.median(math.degrees(c["max_heading_rad"]) for c in rows),
                         mean_body_vx_mae=statistics.mean(c["decomposition_final"]["body_vx_abs_error_integral"] / c["duration_s"] for c in rows),
                         mean_body_vy_mae=statistics.mean(c["decomposition_final"]["body_vy_abs_integral"] / c["duration_s"] for c in rows),
                         mean_body_wz_mae=statistics.mean(c["decomposition_final"]["body_wz_abs_error_integral"] / c["duration_s"] for c in rows),
                         heading_dominates_count=sum(abs(c["decomposition_at_peak"]["heading_command_m"] + c["decomposition_at_peak"]["heading_speed_error_m"])
                                                    > abs(c["decomposition_at_peak"]["lateral_motion_m"]) for c in rows), worst_case=worst)
            result["groups"][learner + "_" + terrain] = group
    path.with_name(path.stem + "_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
