"""One-shot mixed-stair 3000-step preflight, 16+484 training and paired tests."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from contract import (BASE, FINAL_UPDATES, FIRST_UPDATES, HERE, MIXED_UPDATES,
                      PROFILE, RESTORE, ROOT, STAGE, V23, digest, load_json,
                      training_args, verify_frozen, write_new, write_status)


LABELS = {"source_mixed", "source_mixed_teacher",
          "candidate_mixed", "candidate_mixed_teacher"}
SUITES = {"terrain83": 332, "flat25": 80, "stairend": 96,
          "stairend_15cm": 96}


def gpu_available() -> None:
    current = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if current:
        raise RuntimeError(f"GPU occupied: {current}")


def checkpoint(path: Path, completed: int, hip_request: str, *, mixed: bool) -> None:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("completed_updates") != completed
            or saved.get("common_step_counter") != completed * 24
            or saved.get("hip_guard_profile") not in (None, "hip_default012_v23")
            or saved.get("hip_request_profile", "off") != hip_request
            or saved.get("mixed_stair_profile", "off") != (PROFILE if mixed else "off")
            or not all(saved.get(key) for key in (
                "model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"))):
        raise ValueError("Checkpoint age, model, dual optimizers or factor mismatch")
    if mixed and ("mixed_stair_course" not in saved
                  or not isinstance(saved.get("mixed_stair_support_contract"), dict)
                  or saved["mixed_stair_support_contract"].get("kind") != "conservative_center_hull"
                  or not saved["mixed_stair_support_contract"].get("collision_geometry_id")):
        raise ValueError("Trained checkpoint has no mixed-course or live platform witness")


def execute(name: str, argv: list[str], source_age: int, completed: int) -> None:
    verify_frozen()
    with (HERE / f"{name}.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        write_status(name, source_age, completed, child_pid=child.pid,
                     training_running=name.startswith("train"))
        code = child.wait()
    if code:
        raise RuntimeError(f"{name} exited {code}; preserve artifacts and do not auto-retry")
    verify_frozen()


def segment(source: Path, source_age: int, completed: int, increment: int,
            hip_request: str) -> Path:
    import fcntl

    target = completed + increment
    if (increment, target) not in ((FIRST_UPDATES, source_age + FIRST_UPDATES),
                                  (FINAL_UPDATES, source_age + MIXED_UPDATES)):
        raise ValueError("Only exact 16+484 mixed updates are permitted")
    output = HERE / f"training_{completed}_to_{target}"
    if output.exists() or (HERE / f"train_{target}.log").exists():
        raise FileExistsError("Training segment already attempted")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        if shutil.disk_usage(STAGE).free < 15 * 1024**3:
            raise RuntimeError("Less than 15 GiB free on training disk")
        checkpoint(source, completed, hip_request, mixed=completed > source_age)
        previous_hash = digest(source)
        argv = [sys.executable, "-u", str(STAGE / "code/train.py"),
                *training_args(source, output, increment, hip_request)]
        write_new(HERE / f"launch_{target}.json", dict(
            argv=argv, source_sha256=previous_hash, increment=increment,
            target_completed_updates=target, mixed_stair_profile=PROFILE,
            inherited_hip_request_profile=hip_request, automatic_extension=False))
        execute(f"train_{target}", argv, source_age, completed)
        run = load_json(output / "run.json")
        if (run.get("resume_sha256") != previous_hash or digest(source) != previous_hash
                or run.get("additional_updates") != increment
                or run.get("target_completed_updates") != target
                or run.get("hip_guard_profile") != "hip_default012_v23"
                or run.get("hip_request", {}).get("profile") != hip_request
                or run.get("mixed_stair_profile") != PROFILE
                or not all(run.get("resume_checks", {}).get(key) is True for key in RESTORE)
                or (completed > source_age
                    and run.get("resume_checks", {}).get("mixed_stair_course") is not True)):
            raise RuntimeError("Segment changed factors, missed exact restore, or overshot updates")
        if load_json(output / "SUCCESS.json").get("completed_updates") != target:
            raise RuntimeError("Training segment did not stop at exact target")
        result = output / f"model_{target - 1}.pt"
        checkpoint(result, target, hip_request, mixed=True)
        return result


def verify_all_mode(folder: Path) -> dict:
    final = load_json(folder / "FINALIZED.json")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("hashes_unchanged") is not True or final.get("smoke_only") is not False
            or final.get("policy_count") != 4):
        raise ValueError("V23 all-mode paired evaluation incomplete")
    proof = {}
    for suite, count in SUITES.items():
        destination = folder / suite
        manifest = load_json(destination / "manifest.json")
        metrics = load_json(destination / "HIP_REVIEW_v2.json")
        labels = {item["label"] for item in manifest["history"]}
        if (len(manifest.get("cases", [])) != count
                or len(metrics.get("cases", [])) != count
                or len(metrics.get("joints", [])) != count * 12
                or labels != LABELS or metrics.get("fixed_cases_only") is not True
                or metrics.get("optimizer_updates") != 0
                or not (destination / "traces.npz").is_file()
                or not (destination / "HIP_PER_JOINT_v2.csv").is_file()):
            raise ValueError(f"All-mode paired suite incomplete: {suite}")
        if suite == "terrain83":
            for label in labels:
                types = {key.split("/", 1)[1] for key in metrics["totals"]
                         if key.startswith(label + "/")}
                if len(types) != 9:
                    raise ValueError(f"Nine-terrain coverage missing: {label}")
        proof[suite] = dict(fixed_cases=count, labels=sorted(labels),
                            manifest_sha256=digest(destination / "manifest.json"),
                            traces_sha256=digest(destination / "traces.npz"),
                            review_sha256=digest(destination / "HIP_REVIEW_v2.json"))
    return proof


def verify_signed_mixed(folder: Path, expected_collision_id: str) -> dict:
    manifest = load_json(folder / "manifest.json")
    metrics = load_json(folder / "HIP_REVIEW_v2.json")
    cases = manifest.get("cases", [])
    labels = {item["label"] for item in manifest["history"]}
    if (manifest.get("protocol_id") != "signed_mixed_stairs"
            or manifest.get("optimizer_steps") != 0 or manifest.get("training_updates") != 0
            or manifest.get("smoke_only") is not False
            or abs(float(manifest.get("dt", 0.)) - .02) > 1e-10
            or labels != LABELS or len(cases) != 128
            or metrics.get("optimizer_updates") != 0
            or len(metrics.get("cases", [])) != 128
            or len(metrics.get("joints", [])) != 128 * 12
            or not (folder / "traces.npz").is_file()
            or not (folder / "HIP_PER_JOINT_v2.csv").is_file()):
        raise ValueError("Signed mixed-stair paired assessment incomplete")
    witness = manifest.get("local_platform_diagnostic", {})
    if (witness.get("kind") != "conservative_center_hull"
            or witness.get("inferred_not_measured_contact_xy") is not True
            or witness.get("collision_geometry_id") != expected_collision_id
            or witness.get("required_wheels") != 4 or witness.get("dwell_s") != 2.):
        raise ValueError("Signed evaluation did not preserve four-wheel support witness")
    expected = {(policy, terrain, level, group, sign, speed)
                for policy in range(4)
                for terrain in ("stairs_up", "stairs_down")
                for level in (0, 9)
                for group in ("forward_yaw", "forward_lateral")
                for sign in (-1, 1)
                for speed in (0.45, 0.65)}
    actual = Counter((case["policy"], case["terrain"], case["level"],
                      case["command_group"], case["pulse_sign"], case["speed_mps"])
                     for case in cases)
    if set(actual) != expected or any(count != 1 for count in actual.values()):
        raise ValueError("Both signed pulse directions, learners or stair levels missing")
    return dict(fixed_cases=128, labels=sorted(labels),
                 collision_geometry_id=expected_collision_id,
                 manifest_sha256=digest(folder / "manifest.json"),
                 traces_sha256=digest(folder / "traces.npz"),
                 review_sha256=digest(folder / "HIP_REVIEW_v2.json"))


def verify_assessment_identity(folder: Path, frozen: dict, candidate: Path,
                               candidate_age: int, candidate_sha: str,
                               *, signed: bool = False) -> None:
    source = frozen["source"]
    sources = [dict(path=source["path"], completed_updates=source["completed_updates"]),
               dict(path=str(candidate), completed_updates=candidate_age)]
    selection = [dict(path=item["path"], label=label + ("_teacher" if teacher else ""),
                      teacher=teacher)
                 for item, label in zip(sources, ("source_mixed", "candidate_mixed"))
                 for teacher in (False, True)]
    expected_hashes = {source["path"]: source["sha256"], str(candidate): candidate_sha}
    if (load_json(folder / "protocol.json").get("sources") != sources
            or load_json(folder / "selection.json") != selection
            or load_json(folder / "HASHES_BEFORE.json") != expected_hashes
            or digest(Path(source["path"])) != source["sha256"]
            or digest(candidate) != candidate_sha):
        raise ValueError("Paired assessment source, candidate or teacher identity changed")
    if signed:
        history = load_json(folder / "signed_mixed_stairs/manifest.json")["history"]
        actual = [{key: item[key] for key in ("path", "label", "teacher")}
                  for item in history]
        if actual != selection:
            raise ValueError("Signed assessment history differs from the paired selection")


def observed_updates(source_age: int) -> int:
    latest = source_age
    for metrics in HERE.glob("training_*/metrics.jsonl"):
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                latest = max(latest, int(json.loads(line)["completed_updates"]))
            except (ValueError, KeyError, TypeError):
                raise RuntimeError(f"Cannot account for optimizer updates: {metrics}") from None
    return latest


def main() -> None:
    import fcntl

    frozen = verify_frozen()
    source = Path(frozen["source"]["path"])
    source_age = frozen["source"]["completed_updates"]
    hip_request = frozen["hip_request_profile"]
    checkpoint(source, source_age, hip_request, mixed=False)
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_available()
        write_new(HERE / "STARTED_MIXED.json", dict(
            source=frozen["source"], hip_request_profile=hip_request,
            mixed_stair_profile=PROFILE, max_new_optimizer_updates=MIXED_UPDATES,
            remaining_original_budget_after=4500, automatic_reentry=False,
            real_robot_release=False))
        completed = source_age
        try:
            execute("mixed_preflight", [sys.executable, "-u", str(HERE / "preflight.py")],
                    source_age, completed)
            execute("verify_preflight", [sys.executable, "-u", str(HERE / "verify_preflight.py")],
                    source_age, completed)
            preflight = load_json(HERE / "PREFLIGHT_VERIFIED.json")
            if (preflight.get("state") != "verified_not_training"
                    or preflight.get("optimizer_updates") != 0
                    or preflight.get("source_sha256") != frozen["source"]["sha256"]
                    or not preflight.get("collision_geometry_id")):
                raise RuntimeError("Live Isaac contact witness has not passed zero-update preflight")
            source = segment(source, source_age, completed, FIRST_UPDATES, hip_request)
            completed = source_age + FIRST_UPDATES
            source = segment(source, source_age, completed, FINAL_UPDATES, hip_request)
            completed = source_age + MIXED_UPDATES
            candidate_sha = digest(source)
            evaluation = HERE / f"evaluation_{completed}"
            execute("all_mode_paired", [
                sys.executable, "-u", str(V23 / "assessment/run_assessment.py"),
                "--output", str(evaluation), "--sources",
                f"source_mixed:{source_age}:{frozen['source']['path']}",
                f"candidate_mixed:{completed}:{source}"], source_age, completed)
            old_proof = verify_all_mode(evaluation)
            verify_assessment_identity(evaluation, frozen, source, completed, candidate_sha)
            signed = evaluation / "signed_mixed_stairs"
            with (BASE / "full_episode_v12_pipeline.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu_available()
                execute("signed_mixed_paired", [
                    sys.executable, "-u", str(HERE / "evaluate_signed_mixed.py"),
                    "--root", str(BASE), "--code", str(STAGE / "code"),
                    "--output", str(signed), "--protocol", str(evaluation / "protocol.json"),
                    "--selection", str(evaluation / "selection.json")], source_age, completed)
            execute("signed_mixed_metrics", [
                sys.executable, "-u", str(V23 / "assessment/hip_metrics.py"), str(signed)],
                source_age, completed)
            mixed_proof = verify_signed_mixed(signed, preflight["collision_geometry_id"])
            verify_assessment_identity(evaluation, frozen, source, completed, candidate_sha,
                                       signed=True)
            execute("behavior_review", [sys.executable, "-u", str(HERE / "behavior_review.py")],
                     source_age, completed)
            verify_assessment_identity(evaluation, frozen, source, completed, candidate_sha,
                                       signed=True)
            review = load_json(HERE / "BEHAVIOR_COMPARISON.json")
            write_new(HERE / "ASSESSMENT_PROOF.json", dict(
                state="complete_pending_behavior_review", optimizer_updates=0,
                fixed_cases_only=True, not_random_success_rate=True,
                not_real_hardware_validation=True, source_sha256=frozen["source"]["sha256"],
                candidate_sha256=candidate_sha, all_mode_suites=old_proof,
                signed_mixed_stairs=mixed_proof,
                behavior_comparison_sha256=digest(HERE / "BEHAVIOR_COMPARISON.json")))
            write_status("complete_pending_behavior_review", source_age, completed,
                         candidate_checkpoint=str(source), training_running=False,
                         assessment=str(evaluation), qualified_for_real_robot=False,
                         signed_mixed_candidate_gate=review["candidate_signed_gate"])
        except Exception as exc:
            write_status("technical_failure", source_age, observed_updates(source_age),
                         error=str(exc), training_running=False, automatic_retry=False,
                         budget_may_include_unsaved_inflight_update=True)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
