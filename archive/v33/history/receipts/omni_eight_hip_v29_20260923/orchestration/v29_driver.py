"""Locked 16+284 V29 start; behavior review precedes any longer continuation."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from v29_common import HERE, LOCK, STAGE, args, digest, frozen
from v29_verify import verify


def status(stage, source_completed, completed, **extra):
    value = dict(stage=stage, source_completed_updates=source_completed,
                 completed_updates=completed, first_review_updates=source_completed + 300,
                 maximum_automatic_updates=300, training_running=False,
                 timestamp_utc=datetime.now(timezone.utc).isoformat())
    value.update(extra)
    temporary = HERE / "V29_PIPELINE.tmp"
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, HERE / "V29_PIPELINE.json")


def available():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if gpu:
        raise RuntimeError("GPU occupied; no concurrent training")
    if shutil.disk_usage(STAGE).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB disk free")


def checkpoint(path, completed, no_pit):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("completed_updates") != completed
            or saved.get("common_step_counter") != completed * 24
            or (saved.get("no_pit_profile") or "off") != no_pit
            or not all(saved.get(name) for name in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("Unexpected source checkpoint or missing dual optimizer")


def segment(source, completed, increment, receipt):
    target = completed + increment
    output = STAGE / f"training_{completed}_to_{target}"
    logpath = HERE / f"train_{target}.log"
    if output.exists() or logpath.exists():
        raise FileExistsError("Training segment already attempted")
    available()
    checkpoint(source, completed, "off" if completed == receipt["source_completed_updates"] else "omni_eight_v1")
    sha = digest(source)
    argv = [sys.executable, "-u", str(STAGE / "code/train.py"),
            *args(source, output, increment)]
    (HERE / f"launch_{target}.json").write_text(json.dumps({
        "args": argv, "resume_sha256": sha, "target_completed_updates": target,
        "increment": increment}, indent=2))
    frozen()
    with logpath.open("x") as log:
        process = subprocess.Popen(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status(f"training_to_{target}", receipt["source_completed_updates"], completed,
               child_pid=process.pid, training_running=True)
        code = process.wait()
    if code:
        status("technical_failure", receipt["source_completed_updates"], completed,
               child_exit_code=code, log=str(logpath),
               note="Inspect any saved and unsaved updates before recovery")
        raise RuntimeError(f"Segment exited {code}")
    run = json.loads((output / "run.json").read_text())
    expected = {"model", "ppo_optimizer", "student_optimizer",
                "terrain_assignment", "command_terrain_membership",
                "terrain_allocation", "stair_episode_totals"}
    if completed != receipt["source_completed_updates"]:
        expected |= {"terrain_restored_exactly", "mixed_stair_course"}
    if (run["resume_sha256"] != sha or digest(source) != sha
            or run["additional_updates"] != increment
            or run["target_completed_updates"] != target
            or run["no_pit_profile"] != "omni_eight_v1"
            or run["mixed_stair_profile"] != "mixed_stair_short_pulses_v1"
            or run["hip_margin"]["profile"] != "actual_prehard_v1"
            or run["mixed_checks"]["pit_count"] != 0
            or run["terrain_allocation"]["actual_counts"].get("pit", 0) != 0
            or not all(run["resume_checks"].get(name) is True for name in expected)
            or json.loads((output / "SUCCESS.json").read_text())["completed_updates"] != target):
        raise RuntimeError("Finished segment violates eight-terrain contract")
    result = output / f"model_{target - 1}.pt"
    checkpoint(result, target, "omni_eight_v1")
    frozen()
    return result


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions must remain enabled")
    receipt = frozen()
    verified = verify()
    if verified["source_sha256"] != receipt["source_sha256"]:
        raise RuntimeError("Wrong preflight source")
    source = Path(receipt["source_path"])
    completed = receipt["source_completed_updates"]
    if completed != 34823 or completed + 300 > 41823:
        raise RuntimeError("Unreviewed source age or training budget")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available()
        with (HERE / "V29_STARTED.json").open("x") as stream:
            stream.write(json.dumps({
                "source": str(source), "sha256": receipt["source_sha256"],
                "max_new_updates": 300, "high_platform_environments": 0,
                "behavior_review_required_before_continuation": True}, indent=2))
        try:
            first = segment(source, completed, 16, receipt)
            status("integration_16_complete", completed, completed + 16,
                   checkpoint=str(first))
            candidate = segment(first, completed + 16, 284, receipt)
            status("complete_pending_behavior_review", completed, completed + 300,
                   candidate_checkpoint=str(candidate), candidate_sha256=digest(candidate))
        except Exception as exc:
            observed = completed
            for metrics in STAGE.glob("training_*/metrics.jsonl"):
                for line in metrics.read_text().splitlines():
                    observed = max(observed, int(json.loads(line)["completed_updates"]))
            status("technical_failure", completed, observed, error=str(exc),
                   note="No automatic retry or budget extension")
            raise


def recover_status_writer_failure():
    """The first child outlived a status-writer TypeError; consume no repeated updates."""
    receipt = frozen()
    verify()
    completed = receipt["source_completed_updates"]
    failed = json.loads((HERE / "V29_PIPELINE.json").read_text())
    started = json.loads((HERE / "V29_STARTED.json").read_text())
    if (failed["stage"] != "technical_failure"
            or "multiple values for keyword argument 'training_running'" not in failed["error"]
            or started["sha256"] != receipt["source_sha256"]
            or started["max_new_updates"] != 300):
        raise RuntimeError("Recovery is restricted to the recorded status-writer failure")
    first_output = STAGE / f"training_{completed}_to_{completed + 16}"
    first_checkpoint = first_output / f"model_{completed + 15}.pt"
    if (not first_checkpoint.is_file()
            or json.loads((first_output / "SUCCESS.json").read_text())["completed_updates"]
            != completed + 16):
        raise RuntimeError("Wait for the already-running first segment to save")
    checkpoint(first_checkpoint, completed + 16, "omni_eight_v1")
    first_run = json.loads((first_output / "run.json").read_text())
    if (first_run["resume_sha256"] != receipt["source_sha256"]
            or first_run["additional_updates"] != 16
            or first_run["no_pit_profile"] != "omni_eight_v1"
            or first_run["mixed_checks"]["pit_count"] != 0
            or first_run["terrain_allocation"]["actual_counts"].get("pit", 0) != 0
            or not all(first_run["resume_checks"].get(name) is True for name in (
                "model", "ppo_optimizer", "student_optimizer", "terrain_assignment",
                "command_terrain_membership", "terrain_allocation", "stair_episode_totals"))):
        raise RuntimeError("The saved first segment is not the reviewed V29 continuation")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available()
        status("integration_16_recovered", completed, completed + 16,
               checkpoint=str(first_checkpoint),
               recovery="status writer only; all 16 optimizer updates preserved")
        try:
            candidate = segment(first_checkpoint, completed + 16, 284, receipt)
            status("complete_pending_behavior_review", completed, completed + 300,
                   candidate_checkpoint=str(candidate), candidate_sha256=digest(candidate))
        except Exception as exc:
            observed = completed + 16
            for metrics in STAGE.glob("training_*/metrics.jsonl"):
                for line in metrics.read_text().splitlines():
                    observed = max(observed, int(json.loads(line)["completed_updates"]))
            status("technical_failure", completed, observed, error=str(exc),
                   note="Recovery stopped without restarting the first segment")
            raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--recover-status-writer"]:
        recover_status_writer_failure()
    elif len(sys.argv) == 1:
        main()
    else:
        raise ValueError("Unknown V29 driver invocation")
