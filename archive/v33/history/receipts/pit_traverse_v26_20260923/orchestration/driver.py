"""Finite 16+284-update pit-traverse trial; stops before behavioral review."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from preflight import HERE, SHARED_LOCK, STAGE, digest, training_args, verify_frozen
from verify_preflight import verify as verify_preflight


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def status(stage, source_completed, completed, **extra):
    value = dict(stage=stage, source_completed_updates=source_completed,
                 completed_updates=completed, target_completed_updates=source_completed+300,
                 maximum_new_optimizer_updates=300, automatic_extension=False,
                 timestamp_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = HERE / "pipeline_status.tmp"
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, HERE / "pipeline_status.json")


def gpu_available():
    current = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if current:
        raise RuntimeError("GPU is occupied; no duplicate or parallel training")


def source_checkpoint(path, completed, profile):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("completed_updates") != completed
            or saved.get("common_step_counter") != completed*24
            or saved.get("pit_training_profile", "strict_v1") != profile
            or saved.get("mixed_stair_profile", "off") != "off"
            or not all(saved.get(key) for key in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("Checkpoint model, dual optimizers or pit protocol mismatch")


def train_segment(source, source_completed, completed, increment, frozen):
    target = completed + increment
    if (increment, target) not in (
            (16, source_completed+16), (284, source_completed+300)):
        raise ValueError("Only the reviewed 16+284 optimizer-update segments are allowed")
    output = HERE / f"training_{completed}_to_{target}"
    if output.exists() or (HERE / f"train_{target}.log").exists():
        raise FileExistsError("Segment already attempted; preserve its update accounting")
    gpu_available()
    if shutil.disk_usage(STAGE).free < 10*1024**3:
        raise RuntimeError("Less than 10 GiB free on the training disk")
    expected_profile = frozen["source_pit_profile"] if completed == source_completed else "platform_traverse_v1"
    source_checkpoint(source, completed, expected_profile)
    previous_hash = digest(source)
    argv = [sys.executable, "-u", str(STAGE / "code/train.py"),
            *training_args(source, output, increment)]
    write_new(HERE / f"launch_{target}.json", dict(
        args=argv, source_sha256=previous_hash, increment=increment,
        target_completed_updates=target, protocol="platform_traverse_v1",
        automatic_extension=False))
    verify_frozen()
    with (HERE / f"train_{target}.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status(f"train_{target}", source_completed, completed,
               child_pid=process.pid, training_running=True)
        exit_code = process.wait()
    if exit_code:
        status("technical_failure", source_completed, completed,
               training_running=False, child_exit_code=exit_code,
               note="Read metrics.jsonl for possibly spent unsaved optimizer updates; no auto retry")
        raise RuntimeError(f"Training segment exited {exit_code}; no automatic retry")
    verify_frozen()
    run = json.loads((output / "run.json").read_text())
    required = ["model", "ppo_optimizer", "student_optimizer",
                "terrain_assignment", "command_terrain_membership",
                "terrain_restored_exactly", "terrain_allocation", "stair_episode_totals"]
    if completed == source_completed:
        required.append("pit_course_gate_reset")
    if (run["resume_sha256"] != previous_hash or digest(source) != previous_hash
            or run["additional_updates"] != increment
            or run["target_completed_updates"] != target
            or run["pit_training_profile"] != "platform_traverse_v1"
            or not all(run["resume_checks"].get(name) is True for name in required)
            or run["mixed_checks"]["pit_rewards_zero_on_old"] is not True
            or run["pit"]["stop_protocol"]
            != "no platform stop; selected forward command persists until success/reset"
            or json.loads((output / "SUCCESS.json").read_text())["completed_updates"] != target):
        raise RuntimeError("Segment did not retain source state or the new pit-only contract")
    checkpoint = output / f"model_{target-1}.pt"
    source_checkpoint(checkpoint, target, "platform_traverse_v1")
    return checkpoint


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions must remain enabled")
    frozen = verify_frozen()
    checked = verify_preflight()
    if (checked["source_sha256"] != frozen["source_sha256"]
            or not (HERE / "VERIFIED.json").is_file()):
        raise RuntimeError("Independent zero-update verifier has not signed this source")
    source = Path(frozen["source_path"])
    completed = frozen["source_completed_updates"]
    if completed not in (34823, 35823) or completed+300 > 41823:
        raise ValueError("Unreviewed training age or exceeded original update budget")
    source_checkpoint(source, completed, frozen["source_pit_profile"])
    with SHARED_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        write_new(HERE / "STARTED.json", dict(
            source_path=str(source), source_sha256=frozen["source_sha256"],
            source_completed_updates=completed, max_new_optimizer_updates=300,
            protocol="platform_traverse_v1", automatic_extension=False,
            user_goal="keep tracking the selected nonzero forward speed on the upper platform"))
        try:
            source = train_segment(source, completed, completed, 16, frozen)
            status("train_16_complete", completed, completed+16, training_running=False)
            source = train_segment(source, completed, completed+16, 284, frozen)
            status("complete_pending_behavior_review", completed, completed+300,
                   candidate_checkpoint=str(source), candidate_sha256=digest(source),
                   training_running=False, automatic_extension=False)
        except Exception as exc:
            observed = completed
            for metrics in HERE.glob("training_*/metrics.jsonl"):
                for line in metrics.read_text().splitlines():
                    observed = max(observed, int(json.loads(line)["completed_updates"]))
            status("technical_failure", completed, observed, error=str(exc),
                   training_running=False, automatic_retry=False,
                   note="An in-flight unsaved update may also have consumed budget")
            raise


if __name__ == "__main__":
    main()
