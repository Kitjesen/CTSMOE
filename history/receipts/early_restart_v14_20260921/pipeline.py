"""Bounded V14 early-restart training, evaluation, and continuation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any


DEFAULT_ROOT = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
DEFAULT_NEW_ROOT = Path("/root/cts-early-restart-v14-20260921")
SOURCE_RELATIVE = Path("runs/mixed9_4096_60000_relax45_20260919/model_31830.pt")
SOURCE_COMPLETED = 31_831
RESET_TARGETS = {"version": "thunder_cts_v13_reset_targets_v1", "robot": "robot"}
TRAINING_STAGES = (
    ("smoke16", 16, 31_831, 31_847),
    ("pilot300", 284, 31_847, 32_131),
    ("continuation1000", 700, 32_131, 32_831),
)
RUNTIME = {
    "actuator_profile": "rs0402_transient_v4",
    "slope_profile": "slope45_v1",
    "stair_command_profile": "straight_speed_v1",
    "wave_reset_profile": "wave_surface_v1",
    "pit_training_profile": "strict_v1",
    "base_height_penalty": "off",
    "pit_posture_profile": "off",
    "surface_posture_profile": "off",
    "stance_strength_profile": "off",
    "startup_repair_profile": "off",
    "command_curriculum_start": "intermediate",
    "common_step_counter": 0,
}
EARLY_WEIGHTS = {"hip": -0.20, "thigh_calf": -0.08, "height": -15.0, "vertical": -0.5}
EARLY_ALLOCATION = {
    "pit": 0.25,
    "stairs_up": 0.15,
    "stairs_down": 0.10,
    "wave": 0.05,
    "slope_up": 0.05,
    "slope_down": 0.05,
    "rough_slope": 0.05,
    "obstacles": 0.05,
    "flat": 0.25,
}
EARLY_COUNTS = {
    "pit": 1024,
    "flat": 1024,
    "stairs_up": 615,
    "stairs_down": 410,
    "wave": 205,
    "slope_up": 205,
    "slope_down": 205,
    "rough_slope": 204,
    "obstacles": 204,
}
RESUME_CHECKS = (
    "model",
    "ppo_optimizer",
    "student_optimizer",
    "terrain_assignment",
    "command_terrain_membership",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numbers(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class Pipeline:
    def __init__(self, root: Path, new_root: Path, source_selection: Path):
        self.root = root.resolve()
        self.new_root = new_root.resolve()
        self.source = (self.root / SOURCE_RELATIVE).resolve()
        self.source_selection = source_selection.resolve()
        self.code_train = self.new_root / "code_train"
        self.code_eval = self.new_root / "code"
        self.status_path = self.new_root / "pipeline_status.json"
        self.hash_manifest = self.new_root / "code_hash_manifest.json"
        self.started = time.time()
        self.status: dict[str, Any] = {
            "state": "initializing",
            "stage": "preconditions",
            "pid": os.getpid(),
            "started_unix": self.started,
            "source": str(self.source),
            "source_completed_updates": SOURCE_COMPLETED,
        }
        self.expected_hashes: dict[str, str] = {}

    def update(self, state: str, stage: str, **details: Any) -> None:
        self.status.update(
            state=state, stage=stage, updated_unix=time.time(), **details
        )
        atomic_json(self.status_path, self.status)
        print(json.dumps(self.status), flush=True)

    def output_paths(self) -> list[Path]:
        paths = [self.new_root / "runs" / name for name, *_ in TRAINING_STAGES]
        paths += [
            self.new_root / f"evaluation{stage}" / suite
            for stage in (300, 1000)
            for suite in ("terrain83", "flat25")
        ]
        paths += [
            self.new_root / f"{kind}_{stage}.json"
            for stage in (300, 1000)
            for kind in ("selection", "protocol")
        ]
        paths += [self.new_root / "gate300.json", self.new_root / "gate1000.json"]
        paths += [self.new_root / f"train_{name}.log" for name, *_ in TRAINING_STAGES]
        paths += [
            self.new_root / f"{kind}_{stage}_{suite}.log"
            for stage in (300, 1000)
            for suite in ("terrain83", "flat25")
            for kind in ("evaluate", "analyze", "enrich")
        ]
        paths += [self.new_root / f"gates_{stage}.log" for stage in (300, 1000)]
        return paths

    def verify_source_selection(self) -> None:
        selection = json.loads(self.source_selection.read_text(encoding="utf-8"))
        if selection.get("state") != "selected":
            raise RuntimeError("Source selection manifest is not finalized")
        selected_path = selection.get("source", selection.get("path"))
        if Path(selected_path).resolve() != self.source:
            raise RuntimeError(
                "Source selection does not identify the fixed 31831 checkpoint"
            )
        if selection.get("completed_updates") != SOURCE_COMPLETED:
            raise RuntimeError("Source selection completed_updates mismatch")
        if selection.get("source_sha256") != sha256(self.source):
            raise RuntimeError("Source selection hash mismatch")

    def tracked_code_files(self) -> list[Path]:
        files = []
        for directory in (self.code_train, self.code_eval):
            files.extend(sorted(directory.rglob("*.py")))
        files.extend(
            self.new_root / name
            for name in (
                "evaluate_early.py",
                "early_protocol.py",
                "analyze.py",
                "joint_assessment.py",
                "enrich_analysis.py",
                "gates.py",
                "pipeline.py",
            )
        )
        if any(not path.is_file() for path in files):
            missing = [str(path) for path in files if not path.is_file()]
            raise FileNotFoundError(f"Missing frozen code files: {missing}")
        return sorted(set(files))

    def create_code_manifest(self) -> None:
        hashes = {str(path): sha256(path) for path in self.tracked_code_files()}
        payload = {"created_unix": time.time(), "algorithm": "sha256", "files": hashes}
        with self.hash_manifest.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        self.expected_hashes = hashes

    def verify_code(self) -> None:
        changed = {
            path: (expected, sha256(Path(path)) if Path(path).is_file() else None)
            for path, expected in self.expected_hashes.items()
            if not Path(path).is_file() or sha256(Path(path)) != expected
        }
        if changed:
            raise RuntimeError(f"Frozen code changed during pipeline: {changed}")

    def verify_preconditions(self) -> None:
        if self.status_path.exists():
            raise FileExistsError(
                "pipeline_status.json already exists; automatic retry is forbidden"
            )
        if self.hash_manifest.exists():
            raise FileExistsError("code_hash_manifest.json already exists")
        existing = [str(path) for path in self.output_paths() if path.exists()]
        if existing:
            raise FileExistsError(f"Pipeline outputs already exist: {existing}")
        self.verify_source_selection()
        baseline = json.loads(
            (self.new_root / "baseline_status.json").read_text(encoding="utf-8")
        )
        if baseline.get("state") != "complete":
            raise RuntimeError("Baseline is incomplete")
        preflight = json.loads(
            (self.new_root / "preflight_31831/FINALIZED.json").read_text(
                encoding="utf-8"
            )
        )
        expected_preflight = {
            "state": "early_restart_v14_preflight_complete_optimizer_steps_0",
            "completed_updates": SOURCE_COMPLETED,
            "optimizer_updates": 0,
            "control_steps": 3000,
            "num_envs": 4096,
            "reset_targets_valid": True,
            "episode_and_resampling_schedule_valid": True,
            "command_cohort_coverage_valid": True,
            "finite_rewards_actions_and_states": True,
        }
        mismatch = {
            key: (preflight.get(key), value)
            for key, value in expected_preflight.items()
            if preflight.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"Preflight mismatch: {mismatch}")
        if preflight.get("checkpoint_sha256") != sha256(self.source):
            raise RuntimeError("Preflight source hash mismatch")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip():
            raise RuntimeError("GPU occupied")
        self.create_code_manifest()
        self.verify_code()

    def command(
        self,
        args: list[str],
        cwd: Path,
        log_name: str,
        stage: str,
        allowed_returncodes: tuple[int, ...] = (0,),
    ) -> int:
        self.verify_code()
        log_path = self.new_root / log_name
        with log_path.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(
                args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
            )
            self.update(
                "running",
                stage,
                child_pid=child.pid,
                log=str(log_path),
                stage_started_unix=time.time(),
            )
            return_code = child.wait()
        self.status.pop("child_pid", None)
        self.verify_code()
        if return_code not in allowed_returncodes:
            raise RuntimeError(
                f"{stage} failed with exit code {return_code}; inspect {log_path}"
            )
        return return_code

    def train_command(self, destination: Path, resume: Path, updates: int) -> list[str]:
        return [
            sys.executable,
            "-u",
            "train.py",
            "--root",
            str(self.root),
            "--output",
            str(destination),
            "--num-envs",
            "4096",
            "--updates",
            str(updates),
            "--mixed-pit",
            "--resume",
            str(resume),
            "--early-restart-profile",
            "early_phase_v14",
            "--startup-repair-profile",
            "off",
            "--pit-posture-profile",
            "off",
            "--surface-posture-profile",
            "off",
            "--stance-strength-profile",
            "off",
            "--allocation-profile",
            "balanced_adaptive_v1",
            "--pit-training-profile",
            "strict_v1",
            "--pit-course-profile",
            "global_v1",
            "--actuator-profile",
            "rs0402_transient_v4",
            "--slope-profile",
            "slope45_v1",
            "--command-curriculum-start",
            "intermediate",
            "--base-height-penalty",
            "off",
            "--terrain-training-profile",
            "stairs80_v1",
            "--stair-command-profile",
            "straight_speed_v1",
            "--wave-reset-profile",
            "wave_surface_v1",
        ]

    @staticmethod
    def checkpoint_structure(
        checkpoint: Path, expected_completed: int
    ) -> dict[str, Any]:
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("completed_updates") != expected_completed:
            raise RuntimeError("Checkpoint completed_updates mismatch")
        result = {}
        allocation = saved.get("terrain_allocation") or {}
        course = allocation.get("course") or {}
        if (
            allocation.get("profile") != "balanced_adaptive_v1"
            or course.get("profile") != "early_balanced_v14"
            or course.get("weights") != EARLY_ALLOCATION
            or allocation.get("pending") not in ({}, [])
        ):
            raise RuntimeError("Checkpoint early allocation state mismatch")
        for name in (
            "model_state_dict",
            "optimizer_state_dict",
            "optimizer_stu_enc_state_dict",
        ):
            state = saved.get(name)
            if not isinstance(state, dict) or not state:
                raise RuntimeError(f"Missing non-empty {name}")
            tensors = []

            def visit(value: Any) -> None:
                if torch.is_tensor(value):
                    if value.is_floating_point() and not bool(
                        torch.isfinite(value).all()
                    ):
                        raise RuntimeError(f"Non-finite tensor in {name}")
                    tensors.append(
                        (str(value.dtype), list(value.shape), int(value.numel()))
                    )
                elif isinstance(value, dict):
                    for child in value.values():
                        visit(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        visit(child)
                elif isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError(f"Non-finite scalar in {name}")

            visit(state)
            if not tensors:
                raise RuntimeError(f"{name} contains no tensors")
            result[name] = {
                "tensor_count": len(tensors),
                "element_count": sum(x[2] for x in tensors),
            }
        return result

    @staticmethod
    def validate_run_contract(
        run: dict[str, Any], updates: int, expected_start: int, expected_completed: int
    ) -> None:
        expected_run = {
            "starting_completed_updates": expected_start,
            "additional_updates": updates,
            "target_completed_updates": expected_completed,
            "early_restart_profile": "early_phase_v14",
            "startup_repair_profile": "off",
            "pit_posture_profile": "off",
            "surface_posture_profile": "off",
            "stance_strength_profile": "off",
            "allocation_profile": "balanced_adaptive_v1",
            "pit_training_profile": "strict_v1",
            "actuator_profile": "rs0402_transient_v4",
            "slope_profile": "slope45_v1",
            "command_curriculum_start": "intermediate",
            "base_height_penalty": "off",
            "terrain_training_profile": "stairs80_v1",
            "stair_command_profile": "straight_speed_v1",
            "wave_reset_profile": "wave_surface_v1",
            "episode_length_s": 25.0,
            "reset_targets": RESET_TARGETS,
        }
        mismatch = {
            key: (run.get(key), value)
            for key, value in expected_run.items()
            if run.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"Training contract mismatch: {mismatch}")
        early = run.get("early_restart") or {}
        if (
            early.get("profile") != "early_phase_v14"
            or early.get("weights") != EARLY_WEIGHTS
        ):
            raise RuntimeError("Early reward contract mismatch")
        sampling = early.get("sampling") or {}
        if (
            sampling.get("version") != "explicit_full_episode_v13"
            or sampling.get("ramp_seconds") != 0.2
            or sampling.get("explicit_fraction_of_normal_cohort") != 0.5
            or sampling.get("ramp_probability") != 0.5
            or len(sampling.get("kind_names") or []) != 10
            or any(
                int(value) <= 0 for value in sampling.get("selected_by_learner") or []
            )
            or len(sampling.get("selected_by_learner") or []) != 2
        ):
            raise RuntimeError("Early command cohort contract mismatch")
        Pipeline.validate_allocation_snapshot(run.get("terrain_allocation"))
        for check in RESUME_CHECKS:
            if run.get("resume_checks", {}).get(check) is not True:
                raise RuntimeError(f"Resume check failed: {check}")
        if (
            expected_start != SOURCE_COMPLETED
            and run.get("resume_checks", {}).get("terrain_allocation") is not True
        ):
            raise RuntimeError("Continuation did not restore terrain allocation state")

    @staticmethod
    def validate_allocation_snapshot(allocation: Any) -> None:
        allocation = allocation or {}
        if (
            allocation.get("profile") != "balanced_adaptive_v1"
            or allocation.get("bounds_profile") != "early_balanced_v14"
            or allocation.get("target_weights") != EARLY_ALLOCATION
            or allocation.get("actual_counts") != EARLY_COUNTS
            or allocation.get("pending_moves") != 0
        ):
            raise RuntimeError("Early allocation contract mismatch")

    @staticmethod
    def validate_env_contract(path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        values = [
            float(value)
            for value in re.findall(
                r"^\s*resampling_time:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE
            )
        ]
        if values != [5.0]:
            raise RuntimeError(
                f"Expected one 5-second command resampling term, got {values}"
            )

    def validate_training(
        self,
        directory: Path,
        resume: Path,
        updates: int,
        expected_start: int,
        expected_completed: int,
    ) -> Path:
        success = json.loads((directory / "SUCCESS.json").read_text(encoding="utf-8"))
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if success.get("completed_updates") != expected_completed:
            raise RuntimeError("SUCCESS completed_updates mismatch")
        self.validate_run_contract(run, updates, expected_start, expected_completed)
        if run.get("resume_sha256") != sha256(resume):
            raise RuntimeError("Training resume hash mismatch")
        self.validate_env_contract(directory / "env.yaml")
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if len(metrics) != updates or not all(finite_numbers(row) for row in metrics):
            raise RuntimeError("Metrics count or finiteness mismatch")
        final = metrics[-1]
        if (
            final.get("early_restart_profile") != "early_phase_v14"
            or final.get("allocation_profile") != "balanced_adaptive_v1"
        ):
            raise RuntimeError("Final metric profile mismatch")
        self.validate_allocation_snapshot(final.get("terrain_allocation"))
        checkpoint = directory / f"model_{expected_completed - 1}.pt"
        structure = self.checkpoint_structure(checkpoint, expected_completed)
        atomic_json(
            directory / "pipeline_validation.json",
            {
                "validated": True,
                "checkpoint_sha256": sha256(checkpoint),
                "structure": structure,
                "source_immutable": sha256(self.source),
                "episode_seconds": 25.0,
                "command_resampling_seconds": 5.0,
                "validated_unix": time.time(),
            },
        )
        return checkpoint

    def train(
        self,
        name: str,
        resume: Path,
        updates: int,
        expected_start: int,
        expected_completed: int,
    ) -> Path:
        destination = self.new_root / "runs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        source_hash = sha256(self.source)
        self.command(
            self.train_command(destination, resume, updates),
            self.code_train,
            f"train_{name}.log",
            f"train_{name}",
        )
        if sha256(self.source) != source_hash:
            raise RuntimeError("Immutable source checkpoint changed")
        return self.validate_training(
            destination, resume, updates, expected_start, expected_completed
        )

    def write_evaluation_inputs(
        self, checkpoint: Path, completed: int, stage: int
    ) -> tuple[Path, Path]:
        selection_path = self.new_root / f"selection_{stage}.json"
        protocol_path = self.new_root / f"protocol_{stage}.json"
        selection = [
            {"path": str(path), "label": label, "teacher": teacher}
            for path, base in (
                (self.source, "early_31831"),
                (checkpoint, f"V14_{completed}"),
            )
            for label, teacher in ((base, False), (base + "_teacher", True))
        ]
        protocol = {
            "schema_version": 1,
            "protocol_id": "early_restart_v14_zero_update_v1",
            "purpose": "Matched source/candidate behavior gate; not historical training reconstruction.",
            "runtime": RUNTIME,
            "sources": [
                {"path": str(self.source), "completed_updates": SOURCE_COMPLETED},
                {"path": str(checkpoint), "completed_updates": completed},
            ],
        }
        with selection_path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2)
        return selection_path, protocol_path

    def evaluate(self, checkpoint: Path, completed: int, stage: int) -> dict[str, Path]:
        selection, protocol = self.write_evaluation_inputs(checkpoint, completed, stage)
        outputs = {}
        for suite, count in (("terrain83", 332), ("flat25", 80)):
            destination = self.new_root / f"evaluation{stage}" / suite
            destination.parent.mkdir(parents=True, exist_ok=True)
            args = [
                sys.executable,
                "-u",
                str(self.new_root / "evaluate_early.py"),
                "--root",
                str(self.root),
                "--code",
                str(self.code_eval),
                "--output",
                str(destination),
                "--selection",
                str(selection),
                "--protocol",
                str(protocol),
                "--suite",
                suite,
            ]
            self.command(
                args,
                self.code_eval,
                f"evaluate_{stage}_{suite}.log",
                f"evaluate_{stage}_{suite}",
            )
            self.command(
                [sys.executable, str(self.new_root / "analyze.py"), str(destination)],
                self.new_root,
                f"analyze_{stage}_{suite}.log",
                f"analyze_{stage}_{suite}",
            )
            self.command(
                [
                    sys.executable,
                    str(self.new_root / "enrich_analysis.py"),
                    str(destination),
                ],
                self.new_root,
                f"enrich_{stage}_{suite}.log",
                f"enrich_{stage}_{suite}",
            )
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            analysis = json.loads(
                (destination / "analysis_v14.json").read_text(encoding="utf-8")
            )
            if (
                len(manifest.get("cases", [])) != count
                or len(analysis.get("cases", [])) != count
            ):
                raise RuntimeError(f"{suite} case count mismatch")
            outputs[suite] = destination
        return outputs

    def gates(
        self, outputs: dict[str, Path], completed: int, stage: int
    ) -> dict[str, Any]:
        gate_path = self.new_root / f"gate{stage}.json"
        args = [
            sys.executable,
            str(self.new_root / "gates.py"),
            # Both labels come from the same physics batch, avoiding batch-size/RNG drift.
            "--baseline-normal",
            str(outputs["terrain83"]),
            "--baseline-flat",
            str(outputs["flat25"]),
            "--candidate-normal",
            str(outputs["terrain83"]),
            "--candidate-flat",
            str(outputs["flat25"]),
            "--source-student",
            "early_31831",
            "--source-teacher",
            "early_31831_teacher",
            "--candidate-student",
            f"V14_{completed}",
            "--candidate-teacher",
            f"V14_{completed}_teacher",
            "--stage",
            str(stage),
            "--output",
            str(gate_path),
        ]
        return_code = self.command(
            args, self.new_root, f"gates_{stage}.log", f"gates_{stage}", (0, 1)
        )
        result = json.loads(gate_path.read_text(encoding="utf-8"))
        gates = result.get("gates") or {}
        expected = {
            ("early_31831", f"V14_{completed}", "normal"),
            ("early_31831", f"V14_{completed}", "flat"),
            ("early_31831_teacher", f"V14_{completed}_teacher", "normal"),
            ("early_31831_teacher", f"V14_{completed}_teacher", "flat"),
        }
        actual = (
            {
                (
                    item.get("source_label"),
                    item.get("candidate_label"),
                    item.get("suite"),
                )
                for item in gates
                if isinstance(item, dict)
            }
            if isinstance(gates, list)
            else set()
        )
        exact_rows = len(gates) == 4 and actual == expected
        all_passed = exact_rows and all(item.get("passed") is True for item in gates)
        if result.get("accepted") is not True:
            if (
                return_code != 1
                or not exact_rows
                or all_passed
                or result.get("action") != "reject"
                or result.get("next_updates") != 0
                or result.get("stop") is not True
                or not result.get("reasons")
            ):
                raise RuntimeError(
                    "Gate exit 1 did not contain a complete fail-closed rejection"
                )
            return result
        if return_code != 0 or not all_passed:
            raise RuntimeError(
                "Accepted gate result lacks four exact passed label/suite rows"
            )
        expected_action = "continue_700" if stage == 300 else "stop_at_1000"
        expected_next, expected_stop = (700, False) if stage == 300 else (0, True)
        if (
            result.get("action") != expected_action
            or result.get("next_updates") != expected_next
            or result.get("stop") is not expected_stop
        ):
            raise RuntimeError(
                f"Unexpected accepted gate action: {result.get('action')}"
            )
        return result

    def run(self) -> None:
        if self.status_path.exists():
            raise FileExistsError(
                "pipeline_status.json already exists; automatic retry is forbidden"
            )
        try:
            self.verify_preconditions()
            self.update(
                "running", "train_smoke16", code_hash_manifest=str(self.hash_manifest)
            )
            smoke = self.train("smoke16", self.source, 16, SOURCE_COMPLETED, 31_847)
            self.update(
                "running",
                "train_pilot300",
                last_checkpoint=str(smoke),
                completed_updates=31_847,
            )
            pilot = self.train("pilot300", smoke, 284, 31_847, 32_131)
            self.update(
                "running",
                "evaluate_300",
                last_checkpoint=str(pilot),
                completed_updates=32_131,
            )
            evidence300 = self.evaluate(pilot, 32_131, 300)
            gate300 = self.gates(evidence300, 32_131, 300)
            if (
                gate300.get("accepted") is not True
                or gate300.get("action") != "continue_700"
            ):
                self.update(
                    "behavior_rejected",
                    "gate_300",
                    completed_updates=32_131,
                    trained_updates=300,
                    gate=str(self.new_root / "gate300.json"),
                )
                return
            self.update(
                "running",
                "train_continuation1000",
                last_checkpoint=str(pilot),
                completed_updates=32_131,
                gate300=str(self.new_root / "gate300.json"),
            )
            final = self.train("continuation1000", pilot, 700, 32_131, 32_831)
            self.update(
                "running",
                "evaluate_1000",
                last_checkpoint=str(final),
                completed_updates=32_831,
            )
            evidence1000 = self.evaluate(final, 32_831, 1000)
            gate1000 = self.gates(evidence1000, 32_831, 1000)
            if gate1000.get("accepted") is not True:
                self.update(
                    "behavior_rejected",
                    "gate_1000",
                    completed_updates=32_831,
                    trained_updates=1000,
                    gate=str(self.new_root / "gate1000.json"),
                )
                return
            self.update(
                "complete",
                "stopped_at_1000",
                completed_updates=32_831,
                trained_updates=1000,
                final_checkpoint=str(final),
                gate1000=str(self.new_root / "gate1000.json"),
                finished_unix=time.time(),
            )
        except BaseException as exc:
            self.update(
                "failed",
                self.status.get("stage", "unknown"),
                error=repr(exc),
                traceback=traceback.format_exc(),
                failed_unix=time.time(),
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--source-selection", type=Path)
    return parser.parse_args()


def main() -> None:
    import fcntl

    args = parse_args()
    new_root = args.new_root.resolve()
    source_selection = args.source_selection or new_root / "source_selection.json"
    with (args.root.resolve() / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Pipeline(args.root, new_root, source_selection).run()


if __name__ == "__main__":
    main()
