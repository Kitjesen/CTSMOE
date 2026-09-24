"""Run the bounded V13 startup experiment and stop after the 300-update gates."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
STAGE = ROOT / "startup_v13_stage_20260921"
CODE = ROOT / "code_startup_v13_20260921"
SOURCE = ROOT / "runs/flat_v9_1000_20260921/model_44710.pt"
REFERENCE = ROOT / "runs/startup_repair_flat40_96_20260920/model_43201.pt"
DONOR = ROOT / "runs/anchor_v11_pilot_20260921/model_44758.pt"
PREFLIGHT = ROOT / "recording/startup_v13_preflight_3000_20260921/FINALIZED.json"
PROFILE = "omni_full_episode_v13"
RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"
BASE_COMPLETED = 44_711
COEFFICIENTS = {"main": 0.21093362834788681, "student": 0.0022221143221353456}
TRAINING_STAGES = (("smoke16", 16, 44_727), ("pilot300", 284, 45_011))
RESUME_CHECKS = (
    "model", "ppo_optimizer", "student_optimizer", "terrain_restored_exactly",
    "terrain_assignment", "command_terrain_membership", "pit_climb_totals",
    "pit_individual", "stair_episode_totals", "terrain_allocation",
)


def write_json_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def state(value: str, **details) -> None:
    data = dict(state=value, updated_unix=time.time(), **details)
    temporary = STAGE / "pipeline_status.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(STAGE / "pipeline_status.json")
    print(json.dumps(data), flush=True)


def run(script: str, *args) -> None:
    subprocess.run([sys.executable, "-u", str(STAGE / script), *map(str, args)], cwd=CODE, check=True)


def train_command(destination: Path, resume: Path, updates: int, first_stage: bool = False) -> list[str]:
    command = [
        sys.executable, "-u", "train.py", "--root", str(ROOT), "--output", str(destination),
        "--num-envs", "4096", "--updates", str(updates), "--mixed-pit", "--resume", str(resume),
        "--startup-repair-profile", PROFILE, "--startup-anchor-reference", str(REFERENCE),
    ]
    if first_stage:
        command.extend(("--startup-anchor-coefficients-from", str(DONOR)))
    return command


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_anchor(anchor: dict, require_samples: bool = False) -> None:
    assert anchor["coefficients"] == COEFFICIENTS
    if require_samples:
        assert anchor["samples"]["teacher"] > 0
        assert anchor["samples"]["student"] > 0


def train(
    name: str,
    resume: Path,
    updates: int,
    expected_start: int,
    expected_completed: int,
    first_stage: bool = False,
) -> Path:
    destination = ROOT / "runs" / f"startup_v13_{name}_20260921"
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run(train_command(destination, resume, updates, first_stage), cwd=CODE, check=True)
    success = json.loads((destination / "SUCCESS.json").read_text(encoding="utf-8"))
    receipt = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert success["completed_updates"] == expected_completed
    assert receipt["starting_completed_updates"] == expected_start
    assert receipt["additional_updates"] == updates
    assert receipt["target_completed_updates"] == expected_completed
    assert receipt["startup_repair_profile"] == PROFILE
    assert receipt["episode_length_s"] == 25.0
    assert receipt["reset_targets"] == {"version": RESET_TARGETS_VERSION, "robot": "robot"}
    for check in RESUME_CHECKS:
        assert receipt["resume_checks"][check], check
    assert not receipt["resume_checks"].get("explicit_pit_foundation_migration", False)
    assert not (destination / "phase_foundation_migration.json").exists()
    rows = [json.loads(line) for line in (destination / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == updates
    assert all(finite_numbers(row) for row in rows)
    validate_anchor(receipt["startup_anchor"])
    validate_anchor(rows[-1]["startup_anchor"], require_samples=True)
    assert rows[-1]["anchor_main_coefficient"] == COEFFICIENTS["main"]
    assert rows[-1]["anchor_student_coefficient"] == COEFFICIENTS["student"]
    return destination / f"model_{expected_completed - 1}.pt"


def selection(checkpoint: Path, completed: int, flat25: bool = False) -> list[dict]:
    policies = [
        {"path": str(SOURCE.relative_to(ROOT)), "label": "V9_44711", "teacher": False},
        {"path": str(SOURCE.relative_to(ROOT)), "label": "V9_44711_teacher", "teacher": True},
    ]
    if flat25:
        policies.append({"path": str(REFERENCE.relative_to(ROOT)), "label": "V6_43202", "teacher": False})
    policies.extend((
        {"path": str(checkpoint.relative_to(ROOT)), "label": f"V13_{completed}", "teacher": False},
        {"path": str(checkpoint.relative_to(ROOT)), "label": f"V13_{completed}_teacher", "teacher": True},
    ))
    return policies


def evaluate(checkpoint: Path, completed: int, suffix: str, flat25: bool = False) -> tuple[Path, dict]:
    policies = selection(checkpoint, completed, flat25)
    expected_cases = (20 if flat25 else 83) * len(policies)
    selection_path = STAGE / f"selection_{suffix}.json"
    write_json_new(selection_path, policies)
    destination = ROOT / "recording" / f"startup_v13_{suffix}_20260921"
    if destination.exists():
        raise FileExistsError(destination)
    args = ["--root", ROOT, "--code", CODE, "--selection", selection_path, "--output", destination]
    if flat25:
        args.append("--flat25")
    run("evaluate.py", *args)
    run("analyze.py", destination)
    run("plot_comparison.py", destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((destination / "analysis.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == expected_cases
    assert len(analysis["cases"]) == expected_cases
    return destination, analysis


def evaluate_gates(normal: dict, long: dict, completed: int, normal_gate, flat_gate) -> dict:
    candidate = f"V13_{completed}"
    result = {
        "normal_student": normal_gate(normal, "V9_44711", candidate),
        "normal_teacher": normal_gate(normal, "V9_44711_teacher", candidate + "_teacher"),
        "flat25_student": flat_gate(long, "V9_44711", candidate),
        "flat25_teacher": flat_gate(long, "V9_44711_teacher", candidate + "_teacher"),
    }
    result["accepted"] = all(item["accepted"] for item in result.values())
    return result


def validate_verification(hashes: dict, integration: dict) -> None:
    assert integration["passed"]
    if "source_hashes" in integration:
        assert integration["source_hashes"] == hashes


def validate_preflight(preflight: dict, artifact_hashes: dict | None = None) -> None:
    assert preflight["state"] == "startup_v13_preflight_complete_optimizer_steps_0"
    assert preflight["functional_sampling_valid"] is True
    assert preflight["startup_sampler_valid"] is True
    assert preflight["reset_targets_valid"] is True
    assert preflight["optimizer_updates"] == 0
    assert preflight["completed_updates"] == BASE_COMPLETED
    assert preflight["control_steps"] == 3000
    assert preflight["num_envs"] == 4096
    if artifact_hashes is not None:
        for name, actual in artifact_hashes.items():
            assert preflight[name] == actual


def verify_preconditions() -> None:
    hashes = json.loads((STAGE / "source_hashes.json").read_text(encoding="utf-8"))
    integration = json.loads((STAGE / "integration_verification.json").read_text(encoding="utf-8"))
    validate_verification(hashes, integration)
    for name, expected_hash in hashes.items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Unreviewed candidate source: {name}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    validate_preflight(preflight, {
        "checkpoint_sha256": sha256(SOURCE),
        "reference_sha256": sha256(REFERENCE),
        "coefficient_donor_sha256": sha256(DONOR),
    })
    for checkpoint, completed in ((SOURCE, 44_711), (REFERENCE, 43_202), (DONOR, 44_759)):
        assert checkpoint.is_file()
        success = json.loads((checkpoint.parent / "SUCCESS.json").read_text(encoding="utf-8"))
        assert success["completed_updates"] == completed
    donor = json.loads((DONOR.parent / "run.json").read_text(encoding="utf-8"))
    assert donor["startup_anchor"]["coefficients"] == COEFFICIENTS


def main() -> None:
    try:
        from gates import continuation_gate, flat25_gate

        verify_preconditions()
        state("startup_v13_smoke_training", updates=16, automatic_further_training=False)
        checkpoint = train("smoke16", SOURCE, 16, 44_711, 44_727, first_stage=True)
        state("startup_v13_pilot_training", additional_updates=284, target_completed_updates=45_011)
        checkpoint = train("pilot300", checkpoint, 284, 44_727, 45_011)
        state("startup_v13_pilot_evaluation", completed_updates=45_011, normal_cases=332, flat25_cases=100)
        normal_dir, normal = evaluate(checkpoint, 45_011, "pilot300_normal")
        long_dir, long = evaluate(checkpoint, 45_011, "pilot300_flat25", flat25=True)
        gates = evaluate_gates(normal, long, 45_011, continuation_gate, flat25_gate)
        write_json_new(STAGE / "gate300.json", gates)
        outcome = "pilot300_passed_pending_review" if gates["accepted"] else "pilot300_rejected_pending_review"
        state(
            outcome,
            completed_updates=45_011,
            trained_updates=300,
            normal_evidence=str(normal_dir),
            flat25_evidence=str(long_dir),
            gates=gates,
            automatic_further_training=False,
            deployed_to_hardware=False,
        )
    except BaseException as exc:
        state("failed", error=repr(exc), automatic_further_training=False, deployed_to_hardware=False)
        raise


if __name__ == "__main__":
    main()
