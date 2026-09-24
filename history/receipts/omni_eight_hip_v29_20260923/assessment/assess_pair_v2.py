"""Serialized source/candidate V29 comparison on identical eight-terrain states."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from v29_common import LOCK, digest, frozen  # noqa: E402


def main():
    receipt = frozen()
    status = json.loads((STAGE / "orchestration/V29_PIPELINE.json").read_text())
    if status["stage"] != "complete_pending_behavior_review":
        raise RuntimeError("Wait for the bounded 300-update checkpoint")
    candidate = Path(status["candidate_checkpoint"])
    if digest(candidate) != status["candidate_sha256"]:
        raise RuntimeError("Candidate checkpoint changed")
    assessment_sha = digest(HERE / "assess_v29_v2.py")
    with (HERE / "ASSESSMENT_CONTRACT_ATTEMPT02.json").open("x") as stream:
        stream.write(json.dumps({
            "source_sha256": receipt["source_sha256"],
            "candidate_sha256": status["candidate_sha256"],
            "assessor_sha256": assessment_sha,
            "protocol": "1250 control steps, same source-derived eight-terrain placement and seed; student mean action",
            "optimizer_updates": 0}, indent=2))
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if gpu.strip():
            raise RuntimeError("GPU occupied; assessment must be serialized")
        for label, checkpoint in (("source34823_attempt02", Path(receipt["source_path"])),
                                  ("candidate35123_attempt02", candidate)):
            output = HERE / label
            if output.exists() or (HERE / f"{label}.log").exists():
                raise FileExistsError("Preserve existing assessment attempts")
            argv = [sys.executable, "-u", str(HERE / "assess_v29_v2.py"),
                    "--checkpoint", str(checkpoint), "--output", str(output)]
            with (HERE / f"{label}.log").open("x") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
            if digest(HERE / "assess_v29_v2.py") != assessment_sha:
                raise RuntimeError("Assessment code changed during evaluation")
            result = json.loads((output / "FINALIZED.json").read_text())
            if (result["optimizer_updates"] != 0
                    or not result["state_hashes_unchanged"]
                    or result["selected_sha256"] != digest(checkpoint)
                    or result["student_environments"] != 1024
                    or set(result["terrain"]) != {"flat", "stairs_up", "stairs_down", "wave",
                                                  "slope_up", "slope_down", "rough_slope", "obstacles"}):
                raise RuntimeError("Incomplete zero-update terrain assessment")
        a = json.loads((HERE / "source34823_attempt02/FINALIZED.json").read_text())
        b = json.loads((HERE / "candidate35123_attempt02/FINALIZED.json").read_text())
        summary = {}
        for name in a["terrain"]:
            x, y = a["terrain"][name], b["terrain"][name]
            summary[name] = dict(
                source_physical=x["physical_termination_events"],
                candidate_physical=y["physical_termination_events"],
                source_hip_ge039_fraction=x["any_hip_inward_ge_039_fraction"],
                candidate_hip_ge039_fraction=y["any_hip_inward_ge_039_fraction"],
                source_xy_error=x["xy_tracking_mae_mps"],
                candidate_xy_error=y["xy_tracking_mae_mps"],
                source_start_pose_rms=x["first_second_posture_rms_rad"],
                candidate_start_pose_rms=y["first_second_posture_rms_rad"])
        (HERE / "COMPARISON_ATTEMPT02.json").write_text(json.dumps({
            "state": "complete_pending_human_behavior_review", "optimizer_updates": 0,
            "source_sha256": receipt["source_sha256"],
            "candidate_sha256": status["candidate_sha256"],
            "matched_student_cases": 1024,
            "terrain": summary,
            "note": "No automatic continuation or hardware release; inspect full trajectories and grade", 
        }, indent=2))


if __name__ == "__main__":
    main()
