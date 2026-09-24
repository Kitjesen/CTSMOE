"""One bounded 700-update continuation after the 300-update behavior review."""

import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

from v29_common import LOCK, STAGE, args, digest, frozen

HERE = Path(__file__).resolve().parent
INCREMENT = 700
SOURCE_COMPLETED = 35123
TARGET = SOURCE_COMPLETED + INCREMENT


def main():
    receipt = frozen()
    stage = json.loads((HERE / "V29_PIPELINE.json").read_text())
    assessment = json.loads((STAGE / "assessment/COMPARISON_ATTEMPT03.json").read_text())
    if (stage["stage"] != "complete_pending_behavior_review"
            or stage["completed_updates"] != SOURCE_COMPLETED
            or assessment["state"] != "complete_pending_human_behavior_review"
            or assessment["source_sha256"] != receipt["source_sha256"]
            or assessment["candidate_sha256"] != stage["candidate_sha256"]):
        raise RuntimeError("Missing reviewed 300-update matched evaluation")
    physical_source = sum(row["source_physical"] for row in assessment["terrain"].values())
    physical_candidate = sum(row["candidate_physical"] for row in assessment["terrain"].values())
    for stair in ("stairs_up", "stairs_down"):
        row = assessment["terrain"][stair]
        if row["candidate_hip_ge039_fraction"] > row["source_hip_ge039_fraction"] / 2:
            raise RuntimeError("Hip exposure did not improve on both stair directions")
    if (physical_candidate > physical_source
            or assessment["terrain"]["flat"]["candidate_start_pose_rms"]
            > assessment["terrain"]["flat"]["source_start_pose_rms"] + .02):
        raise RuntimeError("Physical events or flat initial posture regressed")
    candidate_eval = json.loads((STAGE / "assessment/candidate35123_attempt03/FINALIZED.json").read_text())
    if (candidate_eval["max_student_wheel_speed_radps"] >= 45.
            or not candidate_eval["state_hashes_unchanged"]):
        raise RuntimeError("Wheel-speed or zero-update assessment gate failed")
    source = Path(stage["candidate_checkpoint"])
    if digest(source) != stage["candidate_sha256"] or TARGET > 41823:
        raise RuntimeError("Checkpoint changed or original training budget exceeded")
    output = STAGE / f"training_{SOURCE_COMPLETED}_to_{TARGET}"
    log = HERE / f"train_{TARGET}.log"
    state = HERE / "V29_EXTENSION_STATUS.json"
    if output.exists() or log.exists() or state.exists():
        raise FileExistsError("Extension already attempted; no duplicate updates")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU is occupied")
        if shutil.disk_usage(STAGE).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free")
        frozen()
        argv = [sys.executable, "-u", str(STAGE / "code/train.py"),
                *args(source, output, INCREMENT)]
        with (HERE / "V29_EXTENSION_LAUNCH.json").open("x") as stream:
            stream.write(json.dumps({
                "source_sha256": stage["candidate_sha256"],
                "source_completed_updates": SOURCE_COMPLETED,
                "target_completed_updates": TARGET,
                "optimizer_update_budget": INCREMENT,
                "source_student_hip_stair_exposure_reduced": True,
                "source_and_candidate_physical_events": [physical_source, physical_candidate],
                "args": argv}, indent=2))
        with log.open("x") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE)
            state.write_text(json.dumps({"state": "running", "pid": process.pid,
                "starting_completed_updates": SOURCE_COMPLETED, "target": TARGET}, indent=2))
            code = process.wait()
        if code:
            state.write_text(json.dumps({"state": "technical_failure", "exit_code": code,
                "note": "Inspect checkpoint and metrics; no automatic retry"}, indent=2))
            raise RuntimeError(f"700-update segment exited {code}")
        run = json.loads((output / "run.json").read_text())
        if (run["resume_sha256"] != stage["candidate_sha256"]
                or run["additional_updates"] != INCREMENT
                or run["target_completed_updates"] != TARGET
                or run["mixed_checks"]["pit_count"] != 0
                or run["terrain_allocation"]["actual_counts"].get("pit", 0) != 0
                or run["hip_margin"]["profile"] != "actual_prehard_v1"
                or run["mixed_stair_profile"] != "mixed_stair_short_pulses_v1"
                or json.loads((output / "SUCCESS.json").read_text())["completed_updates"] != TARGET):
            raise RuntimeError("Finished extension violates frozen training contract")
        checkpoint = output / f"model_{TARGET - 1}.pt"
        frozen()
        state.write_text(json.dumps({"state": "complete_pending_evaluation",
            "completed_updates": TARGET, "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint), "optimizer_update_budget": INCREMENT}, indent=2))


if __name__ == "__main__":
    main()
