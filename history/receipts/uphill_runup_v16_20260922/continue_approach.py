"""Adopt one running training segment, then serial diagnostics and a bounded uphill trial."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = Path("/root/autodl-tmp/cts-v15-long-10000-20260922")
FROZEN = Path("/root/cts-early-ablation-v15-20260921")
AUDIT = Path("/root/autodl-tmp/cts-v15-evaluation-audit-20260922")
BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
PARENT, CHILD = 8153, 8285
FIRST_TARGET = 32831
FINAL_TARGET = 41831


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_args(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]


def remaining_stages():
    # 1000 already completed from 31831 when adoption ends. 1000 trial + 8000 continuation.
    result = (16, 984, 2000, 2000, 2000, 2000)
    if FIRST_TARGET + sum(result) != FINAL_TARGET:
        raise ValueError("Authorized 10000-update total changed")
    return result


def gpu_pids():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def load_legacy():
    spec = importlib.util.spec_from_file_location("bounded_v15_continuation", OLD / "continue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module


def main():
    if (HERE / "pipeline_status.json").exists():
        raise FileExistsError("Existing V16 pipeline preserved; no automatic duplicate")
    manifest = read(HERE / "code_manifest.json")
    def verify_new():
        changed = [name for name, expected in manifest.items() if digest(HERE / name) != expected]
        if changed:
            raise RuntimeError(f"New frozen code changed: {changed}")
    verify_new()
    legacy = load_legacy()
    runner = legacy.Continuation()
    original_verify = runner.verify_code
    def verify_all():
        original_verify()
        verify_new()
    runner.verify_code = verify_all
    runner.verify_code()
    old_state = read(OLD / "pipeline_status.json")
    if old_state.get("stage") != "train_32831" or old_state.get("child_pid") != CHILD:
        raise RuntimeError("Old coordinator advanced; refusing stale handoff")
    parent_args, child_args = process_args(PARENT), process_args(CHILD)
    if str(OLD / "continue.py").encode() not in parent_args:
        raise RuntimeError("Parent PID identity mismatch")
    if str(OLD / "runs/completed_32831").encode() not in child_args:
        raise RuntimeError("Training child identity mismatch")
    if gpu_pids() != {CHILD}:
        raise RuntimeError("Unexpected GPU workload at handoff")
    handoff = dict(parent_pid=PARENT, continuing_training_pid=CHILD, prior_status=old_state,
                   parent_args=[x.decode() for x in parent_args], child_args=[x.decode() for x in child_args],
                   update_budget_unchanged=10000, final_completed_updates=FINAL_TARGET,
                   action="terminate coordinator only; adopt existing training child without restart", unix=time.time())
    with (HERE / "handoff.json").open("x") as f:
        json.dump(handoff, f, indent=2)
    # SIGTERM targets only the verified coordinator, never its process group or training child.
    os.kill(PARENT, signal.SIGTERM)
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Old coordinator did not release exclusive pipeline lock")
                time.sleep(.5)
        try:
            runner.record(state="waiting_current_training", stage="adopt_train_32831", child_pid=CHILD,
                          final_completed_updates=FINAL_TARGET, additional_budget=9000,
                          current_child_unchanged=True, code_manifest=str(HERE / "code_manifest.json"))
            while Path(f"/proc/{CHILD}/cmdline").exists():
                args = process_args(CHILD)
                if not args:  # Reaped/finished zombie: completion receipts below remain authoritative.
                    break
                if args != child_args:
                    raise RuntimeError("Adopted training PID was reused")
                legacy.disk_check()
                time.sleep(10)
            if gpu_pids():
                raise RuntimeError("GPU did not become idle after adopted training")
            completed = FIRST_TARGET
            checkpoint = runner.pipeline.validate_training(
                OLD / "runs/completed_32831", FROZEN / "runs/pilot300/model_32130.pt", 700, 32131, completed)
            runner.state.pop("child_pid", None)
            runner.record(completed_updates=completed, last_checkpoint=str(checkpoint))
            counts = runner.evaluate(checkpoint, completed)
            streak = legacy.update_collapse_streak({"student": 0, "teacher": 0}, counts)

            def runup_test(candidate, number):
                config_dir = HERE / f"runup_config_{number}"
                config_dir.mkdir(exist_ok=False)
                protocol = read(AUDIT / "protocol.json")
                protocol["sources"] = [dict(path=str(candidate), completed_updates=number)]
                protocol["purpose"] = "Matched uphill speeds and 0.1/1.5m geometric front gaps; zero updates"
                selection = [dict(path=str(candidate), label=f"candidate_{number}" + ("_teacher" if teacher else ""), teacher=teacher)
                             for teacher in (False, True)]
                legacy.write_new(config_dir / "protocol.json", protocol)
                legacy.write_new(config_dir / "selection.json", selection)
                output = HERE / f"runup_{number}"
                source_hash = digest(candidate)
                runner.command([sys.executable, "-u", str(HERE / "evaluate_runup.py"),
                                "--root", str(BASE), "--code", str(FROZEN / "code"), "--suite", "terrain83",
                                "--output", str(output), "--protocol", str(config_dir / "protocol.json"),
                                "--selection", str(config_dir / "selection.json")], HERE, f"runup_{number}")
                manifest = read(output / "manifest.json")
                if len(manifest["cases"]) != 180 or manifest["optimizer_steps"] != 0 or digest(candidate) != source_hash:
                    raise RuntimeError("Runup diagnostic completeness/hash mismatch")

            # Reproduce the original 332-case batch and capture every 5ms physics sample.
            runner.command([sys.executable, "-u", str(HERE / "evaluate_physics.py"),
                            "--root", str(BASE), "--code", str(FROZEN / "code"), "--suite", "terrain83",
                            "--output", str(HERE / "overspeed_replay"),
                            "--protocol", str(AUDIT / "protocol.json"), "--selection", str(AUDIT / "selection.json")],
                           HERE, "overspeed_replay")
            runup_test(checkpoint, completed)
            preflight_out = HERE / "preflight_32831"
            command = runner.pipeline.train_command(preflight_out, checkpoint, 0)
            command[2] = str(HERE / "train_runup.py")
            command.append("--runup-preflight")
            runner.command(command, HERE, "preflight_32831")
            receipt = read(preflight_out / "FINALIZED.json")
            if (receipt.get("state") != "complete" or receipt.get("optimizer_updates") != 0
                    or receipt.get("control_steps") != 3000 or receipt.get("num_envs") != 4096):
                raise RuntimeError("Uphill preflight not complete")
            (HERE / "runs").mkdir(exist_ok=False)
            for amount in remaining_stages():
                target = completed + amount
                output = HERE / "runs" / f"completed_{target}"
                command = runner.pipeline.train_command(output, checkpoint, amount)
                command[2] = str(HERE / "train_runup.py")
                runner.command(command, HERE, f"train_{target}", training_output=output)
                checkpoint = runner.pipeline.validate_training(output, checkpoint, amount, completed, target)
                contract = read(output / "run.json")["early_restart"]["uphill_approach"]
                if contract["profile"] != "uphill_approach_v16" or contract["reward_changes"] or contract["actuator_limit_changes"]:
                    raise RuntimeError("Uphill training contract changed")
                completed = target
                runner.record(completed_updates=completed, last_checkpoint=str(checkpoint),
                              experiment_updates_completed=completed - 31831)
                if amount == 16:
                    continue
                counts = runner.evaluate(checkpoint, completed)
                runup_test(checkpoint, completed)
                streak = legacy.update_collapse_streak(streak, counts)
                runner.record(flat25_pass=counts, collapse_streak=streak, learning_reviews=runner.reviews)
                if any(v >= 2 for v in streak.values()):
                    runner.record(state="stopped_for_review", reason="two_consecutive_flat_below_10_of_20")
                    return
            runner.record(state="complete", stage="budget_complete", final_policy_accepted=False)
        except Exception as error:
            runner.record(state="failed", error=str(error))
            raise


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
