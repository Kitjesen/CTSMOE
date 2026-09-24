"""Bounded continuation using frozen V15 training and corrected paired evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
V15 = Path("/root/cts-early-ablation-v15-20260921")
AUDIT = Path("/root/autodl-tmp/cts-v15-evaluation-audit-20260922")
HERE = Path(__file__).resolve().parent
STAGES = (700, 2000, 2000, 2000, 3000)
START = 32131
TOTAL = 9700
MIN_FREE = 10 * 1024**3


def planned_completions(start=START):
    result = []
    for amount in STAGES:
        start += amount
        result.append(start)
    return result


def update_collapse_streak(streak, counts):
    if set(counts) != {"student", "teacher"}:
        raise ValueError("Both student and teacher results are required")
    return {role: streak.get(role, 0) + 1 if count < 10 else 0
            for role, count in counts.items()}


def flat_counts(rows, candidate):
    output = {}
    for role, teacher in (("student", False), ("teacher", True)):
        selected = [row for row in rows if row["checkpoint"] == str(candidate)
                    and row["teacher"] is teacher]
        if len(selected) != 20:
            raise ValueError(f"Expected 20 matched flat cases for {role}")
        output[role] = sum(bool(row["pass"]) for row in selected)
    return output


def validate_evaluation(manifest, analysis, count, checkpoint, completed):
    if (len(manifest.get("cases", [])) != count or len(analysis.get("cases", [])) != count
            or manifest.get("optimizer_steps") != 0 or manifest.get("training_updates") != 0
            or manifest.get("zero_update_evaluation") is not True
            or manifest.get("checkpoint") != str(checkpoint)
            or manifest.get("completed_updates") != completed):
        raise ValueError("Incomplete or mismatched zero-update evaluation")
    mapping = manifest.get("terrain_mapping", {})
    if not all(mapping.get(key) is True for key in (
            "membership_asserted", "pit_active_asserted", "origins_match_atlas_asserted")):
        raise ValueError("Missing verified terrain mapping")
    for row in analysis["cases"]:
        for field in ("target_step_peak_rad", "leg_torque_peak_nm", "wheel_torque_peak_nm", "xy_mae"):
            if field not in row or not math.isfinite(row[field]):
                raise ValueError(f"Missing or nonfinite evaluation metric: {field}")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def disk_check():
    if shutil.disk_usage(HERE).free < MIN_FREE:
        raise RuntimeError("Data disk has less than 10 GiB free")


class Continuation:
    def __init__(self):
        sys.path.insert(0, str(V15))
        from pipeline import Pipeline
        self.pipeline = Pipeline(BASE, V15, V15 / "source_selection.json")
        self.pipeline.expected_hashes = read(V15 / "code_hash_manifest.json")["files"]
        self.eval_hashes = {str(AUDIT / name): digest(AUDIT / name)
                           for name in ("evaluate_corrected.py", "audit_helpers.py", "early_protocol.py")}
        self.state = dict(state="initializing", completed_updates=START,
                          already_used_updates=300, additional_budget=TOTAL,
                          experiment_update_budget=10000, final_completed_updates=41831,
                          source_checkpoint=str(V15 / "runs/pilot300/model_32130.pt"))
        self.reviews = []
        self.source_hashes = read(AUDIT / "source_hashes.json")

    def record(self, **fields):
        self.state.update(fields, updated_unix=time.time())
        temporary = HERE / "pipeline_status.tmp"
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(HERE / "pipeline_status.json")
        print(json.dumps(self.state), flush=True)

    def verify_code(self):
        self.pipeline.verify_code()
        for path, expected in self.eval_hashes.items():
            if digest(path) != expected:
                raise RuntimeError(f"Corrected evaluation code changed: {path}")
        for path, expected in self.source_hashes.items():
            if digest(path) != expected:
                raise RuntimeError(f"Preserved source checkpoint changed: {path}")

    def command(self, args, cwd, name, training_output=None):
        self.verify_code()
        disk_check()
        stop_for_disk = False
        with (HERE / f"{name}.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            self.record(state="running", stage=name, child_pid=child.pid,
                        log=str(HERE / f"{name}.log"))
            while child.poll() is None:
                if training_output is not None and shutil.disk_usage(HERE).free < MIN_FREE:
                    if training_output.is_dir():
                        (training_output / "STOP").touch(exist_ok=True)
                        stop_for_disk = True
                        self.record(state="stopping_for_disk", stage=name)
                time.sleep(10)
            code = child.returncode
        self.state.pop("child_pid", None)
        self.verify_code()
        if stop_for_disk:
            raise RuntimeError("Disk reserve reached; STOP requested checkpoint save")
        if code != 0:
            raise RuntimeError(f"{name} exited {code}; output preserved; no automatic retry")

    def evaluate(self, checkpoint, completed):
        destination = HERE / f"evaluation_{completed}"
        destination.mkdir(exist_ok=False)
        protocol = read(AUDIT / "protocol.json")
        protocol["purpose"] = "Bounded learning review, not a deployment acceptance gate."
        protocol["sources"] = [dict(path=str(self.pipeline.source), completed_updates=31831),
                               dict(path=str(checkpoint), completed_updates=completed)]
        selection = [dict(path=str(path), label=label + ("_teacher" if teacher else ""), teacher=teacher)
                     for path, label in ((self.pipeline.source, "early_31831"),
                                         (checkpoint, f"V15_{completed}"))
                     for teacher in (False, True)]
        write_new(destination / "protocol.json", protocol)
        write_new(destination / "selection.json", selection)
        summaries = {}
        candidate_hash = digest(checkpoint)
        for suite in ("terrain83", "flat25"):
            output = destination / suite
            self.command([sys.executable, "-u", str(AUDIT / "evaluate_corrected.py"),
                          "--root", str(BASE), "--code", str(V15 / "code"),
                          "--output", str(output), "--selection", str(destination / "selection.json"),
                          "--protocol", str(destination / "protocol.json"), "--suite", suite],
                         V15 / "code", f"eval_{completed}_{suite}")
            self.command([sys.executable, str(V15 / "analyze.py"), str(output)],
                         V15, f"analyze_{completed}_{suite}")
            self.command([sys.executable, str(V15 / "enrich_analysis.py"), str(output)],
                         V15, f"enrich_{completed}_{suite}")
            manifest = read(output / "manifest.json")
            analysis = read(output / "analysis_v14.json")
            validate_evaluation(manifest, analysis, 332 if suite == "terrain83" else 80,
                                checkpoint, completed)
            summaries[suite] = {
                label: {terrain: {"passed": sum(bool(r["pass"]) for r in analysis["cases"]
                                               if r["label"] == label and r["terrain"] == terrain),
                                  "physical": sum(bool(r["physical"]) for r in analysis["cases"]
                                                 if r["label"] == label and r["terrain"] == terrain)}
                        for terrain in dict.fromkeys(r["terrain"] for r in analysis["cases"])}
                for label in dict.fromkeys(r["label"] for r in analysis["cases"])}
        if digest(checkpoint) != candidate_hash:
            raise RuntimeError("Evaluation changed the candidate checkpoint")
        rows = read(destination / "flat25/analysis_v14.json")["cases"]
        counts = flat_counts(rows, checkpoint)
        review = dict(completed_updates=completed, flat25_pass=counts,
                      per_terrain=summaries, candidate_sha256=candidate_hash,
                      fixed_cases_not_random_success_rates=True,
                      final_policy_accepted=False, evaluation=str(destination))
        write_new(destination / "learning_review.json", review)
        self.reviews.append(review)
        return counts

    def run(self):
        audit = read(AUDIT / "status.json")
        if (audit.get("state") != "complete" or audit.get("checkpoint_hashes_unchanged") is not True
                or audit.get("optimizer_updates") != 0):
            raise RuntimeError("Corrected paired audit has not completed with unchanged checkpoints")
        for path, expected in read(AUDIT / "source_hashes.json").items():
            if digest(path) != expected:
                raise RuntimeError("Audit source checkpoint changed")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU is occupied")
        assert sum(STAGES) == TOTAL and planned_completions()[-1] == 41831
        self.verify_code()
        disk_check()
        write_new(HERE / "code_hash_manifest.json",
                  dict(training=self.pipeline.expected_hashes, evaluation=self.eval_hashes))
        resume = V15 / "runs/pilot300/model_32130.pt"
        self.pipeline.checkpoint_structure(resume, START)
        completed = START
        streak = {"student": 0, "teacher": 0}
        (HERE / "runs").mkdir(exist_ok=False)
        self.record(state="ready", source_sha256=digest(resume))
        for updates in STAGES:
            target = completed + updates
            output = HERE / "runs" / f"completed_{target}"
            self.command(self.pipeline.train_command(output, resume, updates), V15 / "code_train",
                         f"train_{target}", training_output=output)
            checkpoint = self.pipeline.validate_training(output, resume, updates, completed, target)
            self.record(completed_updates=target, last_checkpoint=str(checkpoint),
                        additional_updates_completed=target - START)
            counts = self.evaluate(checkpoint, target)
            streak = update_collapse_streak(streak, counts)
            self.record(flat25_pass=counts, collapse_streak=streak, learning_reviews=self.reviews)
            if any(value >= 2 for value in streak.values()):
                self.record(state="stopped_for_review", reason="two_consecutive_flat_below_10_of_20")
                return
            completed, resume = target, checkpoint
        self.record(state="complete", stage="budget_complete", final_policy_accepted=False)


def main():
    import fcntl
    if (HERE / "pipeline_status.json").exists():
        raise SystemExit("Existing run preserved; automatic duplicate or retry is forbidden")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Continuation()
        try:
            runner.run()
        except Exception as error:
            traceback.print_exc()
            runner.record(state="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
