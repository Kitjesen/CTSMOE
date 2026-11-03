"""Wait for the finite 700-update extension, then assess it once on the GPU."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from v29_common import LOCK, digest, frozen  # noqa: E402

STATUS = HERE / "POST700_STATUS.json"


def write(state, **extra):
    STATUS.write_text(json.dumps({"state": state, **extra}, indent=2))


def main():
    receipt = frozen()
    assessor = HERE / "assess_v29_v3.py"
    contract = json.loads((HERE / "ASSESSMENT_CONTRACT_ATTEMPT03.json").read_text())
    if (digest(assessor) != contract["assessor_sha256"]
            or contract["source_sha256"] != receipt["source_sha256"]
            or not (HERE / "source34823_attempt03/FINALIZED.json").is_file()):
        raise RuntimeError("The already evaluated source/assessor changed")
    if STATUS.exists() or (HERE / "candidate35823").exists():
        raise FileExistsError("Post-700 assessment already attempted")
    write("waiting_for_finite_training", target_completed_updates=35823)
    for _ in range(360):  # 3-hour limit; no infinite background waiter.
        extension_path = STAGE / "orchestration/V29_EXTENSION_STATUS.json"
        if extension_path.is_file():
            extension = json.loads(extension_path.read_text())
            if extension["state"] == "technical_failure":
                write("training_failed_no_assessment", detail=extension)
                return
            if extension["state"] == "complete_pending_evaluation":
                break
        time.sleep(30)
    else:
        write("timed_out_waiting_for_training")
        return
    if (extension["completed_updates"] != 35823
            or digest(extension["checkpoint"]) != extension["checkpoint_sha256"]):
        raise RuntimeError("Completed training checkpoint identity mismatch")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU occupied; cannot run post-training assessment")
        write("evaluating_candidate", checkpoint=extension["checkpoint"])
        argv = [sys.executable, "-u", str(assessor),
                "--checkpoint", extension["checkpoint"],
                "--output", str(HERE / "candidate35823")]
        with (HERE / "candidate35823.log").open("x") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if digest(assessor) != contract["assessor_sha256"]:
            raise RuntimeError("Assessor changed during final evaluation")
        source = json.loads((HERE / "source34823_attempt03/FINALIZED.json").read_text())
        middle = json.loads((HERE / "candidate35123_attempt03/FINALIZED.json").read_text())
        candidate = json.loads((HERE / "candidate35823/FINALIZED.json").read_text())
        if (not candidate["state_hashes_unchanged"] or candidate["optimizer_updates"] != 0
                or candidate["selected_sha256"] != extension["checkpoint_sha256"]
                or candidate["common_terrain_source_sha256"] != source["selected_sha256"]
                or set(candidate["terrain"]) != set(source["terrain"])):
            raise RuntimeError("Post-700 zero-update result is not comparable")
        comparisons = {}
        for name in source["terrain"]:
            comparisons[name] = {
                metric: [source["terrain"][name][metric],
                         middle["terrain"][name][metric],
                         candidate["terrain"][name][metric]]
                for metric in ("physical_termination_events", "any_hip_inward_ge_039_fraction",
                               "xy_tracking_mae_mps", "yaw_tracking_mae_radps",
                               "first_second_posture_rms_rad", "posture_rms_rad")}
        (HERE / "COMPARISON_35823.json").write_text(json.dumps({
            "state": "complete_pending_behavior_review", "policies": [34823, 35123, 35823],
            "student_environments": 1024, "steps_each": 1250,
            "matched_initial_terrain": True,
            "terrain": comparisons,
            "wheel_speed_peak_radps": [x["max_student_wheel_speed_radps"] for x in (source, middle, candidate)],
            "leg_torque_peak_nm": [x["max_student_leg_torque_nm"] for x in (source, middle, candidate)],
            "actual_hip_peak_rad": [x["actual_inward_peak_rad"] for x in (source, middle, candidate)],
            "requested_hip_peak_rad": [x["requested_inward_peak_rad"] for x in (source, middle, candidate)],
            "interpretation": "fixed matched student mean-action simulation, not random success rate or real robot clearance",
        }, indent=2))
        write("complete_pending_behavior_review", comparison=str(HERE / "COMPARISON_35823.json"),
              checkpoint=extension["checkpoint"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write("technical_failure", error=str(exc))
        raise
