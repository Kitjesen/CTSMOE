"""One-shot V24 hip-only 16+984 training and V23-protocol paired assessment.

Requires a frozen source review. A process failure or behavior result never
starts mixed-stair training, extends the budget, or deploys a model.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from contract import (
    BASE, ORCH, RESTORE, ROOT, STAGE, V23, V24_UPDATES,
    command, digest, load_json, verify_frozen,
    write_new, write_status,
)


def gpu_available() -> None:
    current = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if current:
        raise RuntimeError(f"GPU already occupied; no parallel training: {current}")


def source_checkpoint(path: Path, completed: int, *, candidate: bool) -> None:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("completed_updates") != completed
            or saved.get("common_step_counter") != completed * 24
            or saved.get("hip_guard_profile") not in (
                ("hip_default012_v23",) if candidate else (None, "hip_default012_v23"))
            or saved.get("hip_request_profile", "off") != (
                "inward_unreachable_request_v1" if candidate else "off")
            or saved.get("mixed_stair_profile", "off") != "off"
            or not all(saved.get(key) for key in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("Checkpoint model/dual optimizers/default/age/factor mismatch")


def execute(stage: str, argv: list[str], source_completed: int, completed: int) -> None:
    verify_frozen()
    with (ORCH / f"{stage}.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=ORCH, stdout=log, stderr=subprocess.STDOUT)
        write_status(stage, source_completed, completed,
                     child_pid=child.pid, training_running=stage.startswith("train"))
        exit_code = child.wait()
    if exit_code:
        raise RuntimeError(f"{stage} exited {exit_code}; preserve logs, no automatic retry")
    verify_frozen()


def train_segment(source: Path, source_completed: int, completed: int, increment: int) -> Path:
    import fcntl

    target = completed + increment
    if target > source_completed + V24_UPDATES or (increment, target) not in (
            (16, source_completed + 16), (984, source_completed + V24_UPDATES)):
        raise ValueError("Only the reviewed 16+984 hip-only segments are allowed")
    output = ORCH / f"training_{completed}_to_{target}"
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        if shutil.disk_usage(STAGE).free < 15 * 1024**3:
            raise RuntimeError("Less than 15 GiB free on the training disk")
        if output.exists() or (ORCH / f"train_{target}.log").exists():
            raise FileExistsError("Segment already attempted; cannot overwrite or repeat updates")
        source_checkpoint(source, completed, candidate=completed > source_completed)
        previous_hash = digest(source)
        args = [sys.executable, "-u", str(STAGE / "code/train.py"),
                *command(source, output, increment)]
        write_new(ORCH / f"launch_{target}.json", dict(
            args=args, source_sha256=previous_hash, increment=increment,
            target_completed_updates=target, factor="hip_request_only",
            mixed_stair_profile="off", automatic_extension=False))
        execute(f"train_{target}", args, source_completed, completed)
        run = load_json(output / "run.json")
        if (run.get("resume_sha256") != previous_hash or digest(source) != previous_hash
                or run.get("additional_updates") != increment
                or run.get("target_completed_updates") != target
                or run.get("hip_guard_profile") != "hip_default012_v23"
                or run.get("hip_request", {}).get("profile") != "inward_unreachable_request_v1"
                or run.get("hip_request", {}).get("weight") != -0.02
                or run.get("mixed_stair_profile") != "off"
                or not all(run.get("resume_checks", {}).get(key) is True for key in RESTORE)):
            raise RuntimeError("Segment did not preserve model, dual optimizer, terrain or reward factor")
        if load_json(output / "SUCCESS.json").get("completed_updates") != target:
            raise RuntimeError("Training segment did not end at the exact target")
        result = output / f"model_{target-1}.pt"
        source_checkpoint(result, target, candidate=True)
        return result


def verify_paired_evaluation(folder: Path) -> dict:
    final = load_json(folder / "FINALIZED.json")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("hashes_unchanged") is not True
            or final.get("smoke_only") is not False or final.get("policy_count") != 4):
        raise ValueError("Paired assessment is incomplete or smoke-only")
    suites = {"terrain83": 332, "flat25": 80, "stairend": 96, "stairend_15cm": 96}
    proof = {}
    for name, expected_cases in suites.items():
        destination = folder / name
        manifest = load_json(destination / "manifest.json")
        metrics = load_json(destination / "HIP_REVIEW_v2.json")
        if (len(manifest.get("cases", [])) != expected_cases
                or len(metrics.get("cases", [])) != expected_cases
                or len(metrics.get("joints", [])) != expected_cases * 12
                or metrics.get("fixed_cases_only") is not True
                or metrics.get("optimizer_updates") != 0
                or not (destination / "traces.npz").is_file()
                or not (destination / "HIP_PER_JOINT_v2.csv").is_file()):
            raise ValueError(f"Incomplete fixed-case or 12-joint evidence in {name}")
        labels = {item["label"] for item in manifest["history"]}
        expected_labels = {
            "source_default012", "source_default012_teacher",
            "candidate_hip_request", "candidate_hip_request_teacher",
        }
        if labels != expected_labels:
            raise ValueError(f"Missing paired student/teacher policies in {name}")
        if name == "terrain83":
            for label in labels:
                types = {key.split("/", 1)[1] for key in metrics["totals"]
                         if key.startswith(label + "/")}
                if len(types) != 9:
                    raise ValueError(f"Nine-terrain coverage missing for {label}")
        proof[name] = dict(fixed_cases=expected_cases, per_joint_rows=expected_cases * 12,
                           labels=sorted(labels),
                           manifest_sha256=digest(destination / "manifest.json"),
                           hip_review_sha256=digest(destination / "HIP_REVIEW_v2.json"))
    return proof


def observed_updates(source_completed: int) -> int:
    latest = source_completed
    for metrics in ORCH.glob("training_*/metrics.jsonl"):
        for row in metrics.read_text(encoding="utf-8").splitlines():
            try:
                latest = max(latest, int(json.loads(row)["completed_updates"]))
            except (ValueError, KeyError, TypeError):
                raise RuntimeError(f"Unparseable optimizer accounting: {metrics}") from None
    return latest


def main() -> None:
    import fcntl

    frozen = verify_frozen()
    source = Path(frozen["source"]["path"])
    source_completed = frozen["source"]["completed_updates"]
    target_completed = source_completed + V24_UPDATES
    source_checkpoint(source, source_completed, candidate=False)
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        write_new(ORCH / "STARTED_V24.json", dict(
            source=frozen["source"], factor="hip_request_only",
            max_new_optimizer_updates=1000, remaining_original_budget_after=5000,
            automatic_mixed_training=False, automatic_extension=False))
        completed = source_completed
        try:
            execute("hip_qualification", [sys.executable, "-u", str(ORCH / "qualification.py")],
                    source_completed, completed)
            execute("independent_verification", [sys.executable, "-u",
                                                  str(ORCH / "verify_qualification.py")],
                    source_completed, completed)
            ready = load_json(ORCH / "hip_only_VERIFIED.json")
            if (ready.get("state") != "verified_not_training"
                    or ready.get("source_completed") != source_completed
                    or ready.get("optimizer_updates") != 0):
                raise RuntimeError("Independent zero-update qualification did not pass")
            source = train_segment(source, source_completed, completed, 16)
            completed = source_completed + 16
            source = train_segment(source, source_completed, completed, 984)
            completed = target_completed
            evaluation = ORCH / f"evaluation_{target_completed}"
            execute("paired_assessment", [sys.executable, "-u", str(V23 / "assessment/run_assessment.py"),
                    "--output", str(evaluation), "--sources",
                    f"source_default012:{source_completed}:{frozen['source']['path']}",
                    f"candidate_hip_request:{completed}:{source}"], source_completed, completed)
            proof = verify_paired_evaluation(evaluation)
            execute("behavior_review", [sys.executable, "-u",
                                        str(ORCH / "behavior_review.py")], source_completed, completed)
            write_new(ORCH / "ASSESSMENT_PROOF.json", dict(
                state="complete_pending_behavior_review", fixed_cases_only=True,
                not_random_success_rate=True, not_real_hardware_validation=True,
                source_sha256=frozen["source"]["sha256"], candidate_sha256=digest(source),
                suites=proof,
                behavior_comparison_sha256=digest(ORCH / "BEHAVIOR_COMPARISON.json")))
            write_status("complete_pending_behavior_review", source_completed, completed,
                         candidate_checkpoint=str(source), training_running=False,
                         assessment=str(evaluation), qualified_for_mixed_training=False)
        except Exception as exc:
            # Completed optimizer updates remain spent even if no checkpoint was saved.
            write_status("technical_failure", source_completed, observed_updates(source_completed), error=str(exc),
                         training_running=False, automatic_retry=False,
                         budget_may_include_unsaved_inflight_update=True)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
