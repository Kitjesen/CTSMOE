"""Evaluate the pre-registered gradient gate on a completed distribution audit.

This is an offline diagnostic.  It neither starts Isaac nor performs an optimizer
step.  The reported gradients are the V6 action-anchor and CTS latent-distillation
directions on retained audit observations; they exclude PPO and load balancing and
therefore are not a causal training-gradient reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import torch


EXPECTED_STUDENT_COEFFICIENT = 0.0022221143221353456
OBS_DIMS = {"policy": 530, "single_obs": 53, "critic": 291}
NUM_ACTIONS = 16
STATE_SAMPLE_CAP = 4096
EPISODE_SAMPLE_CAP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--gradient-module",
        type=Path,
        required=True,
        help="Pinned gradient_audit_v2.py used to evaluate audit_window.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_gradient_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("pinned_gradient_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gradient module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "assert_no_accumulated_gradients",
        "audit_window",
        "configured_call_scale",
        "model_config",
        "resolve_checkpoint",
        "state_hash",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise TypeError(f"gradient module lacks required callables: {missing}")
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_completed_audit(audit_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    status = read_json(audit_output / "audit_status.json")
    if status.get("state") != "AUDIT_COMPLETE":
        raise ValueError(f"distribution audit is not complete: {status.get('state')!r}")
    if status.get("authoritative") is not True:
        raise ValueError("distribution audit is not marked authoritative")
    if status.get("smoke_only") is not False:
        raise ValueError("smoke audit cannot be used for the gradient gate")
    if status.get("optimizer_updates") != 0:
        raise ValueError(f"audit performed optimizer updates: {status.get('optimizer_updates')!r}")
    before = status.get("state_hashes_before")
    after = status.get("state_hashes_after")
    if not isinstance(before, dict) or before != after:
        raise ValueError("audit model/optimizer hashes are missing or changed")
    if status.get("mutation_patches_restored") is not True:
        raise ValueError("audit mutation guard was not restored")
    if status.get("stats_environment_patch_restored") is not True:
        raise ValueError("audit environment statistics hook was not restored")
    cleanup_proved = status.get("launcher_patch_restored") is True or (
        status.get("isolated_audit_process_exited") is True
        and status.get("no_live_audit_process") is True
    )
    if not cleanup_proved:
        raise ValueError(
            "audit cleanup is unproved: require in-process patch restoration or "
            "verified isolated-process exit"
        )
    if status.get("input_files_unchanged") is not True:
        raise ValueError("audit has no complete proof that input files remained unchanged")
    for name in ("checkpoint_sha256", "reference_sha256"):
        value = status.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"audit lacks a valid {name}")
    distribution = read_json(audit_output / "distribution.json")
    return status, distribution


def coverage_gate(distribution: dict[str, Any]) -> dict[str, Any]:
    critical = distribution.get("critical_coverage")
    if not isinstance(critical, dict):
        return {
            "passed": False,
            "states": 0,
            "distinct_envs": 0,
            "episode_ordinals": 0,
            "reason": "critical coverage is absent",
        }
    states = int(critical.get("states", 0))
    envs = int(critical.get("distinct_envs", 0))
    generations = critical.get("episode_generations")
    if not isinstance(generations, dict):
        raise TypeError("critical episode_generations must be an object")
    ordinals = len(generations)
    passed = states >= 256 and envs >= 32 and ordinals >= 2
    if bool(distribution.get("coverage_gate")) != passed:
        raise ValueError("distribution coverage_gate disagrees with exact critical counts")
    return {
        "passed": passed,
        "states": states,
        "distinct_envs": envs,
        "episode_ordinals": ordinals,
        "thresholds": {"states": 256, "distinct_envs": 32, "episode_ordinals": 2},
    }


def validate_record(record: Any, sample_name: str, index: int) -> tuple[int, int]:
    if not isinstance(record, dict):
        raise TypeError(f"{sample_name}[{index}] is not a record")
    cell = record.get("cell")
    if not isinstance(cell, (list, tuple)) or len(cell) < 4:
        raise ValueError(f"{sample_name}[{index}] has an invalid cell key")
    if int(cell[0]) != 1 or int(cell[1]) != 1 or int(cell[3]) != 1:
        raise ValueError(
            f"{sample_name}[{index}] is not a student/special/moving sample: {cell[:4]}"
        )
    age = float(record.get("age", math.nan))
    command = np.asarray(record.get("command"), dtype=np.float64)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError(f"{sample_name}[{index}] has an invalid command")
    if not (
        math.isfinite(age)
        and 1.0 <= age < 3.0
        and 0.45 <= command[0] <= 0.75
        and abs(command[1]) <= 0.05
        and abs(command[2]) <= 0.05
    ):
        raise ValueError(
            f"{sample_name}[{index}] is outside the registered candidate mask: "
            f"age={age}, command={command.tolist()}"
        )
    for key, dimension in OBS_DIMS.items():
        value = np.asarray(record.get(f"obs_{key}"))
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(
                f"{sample_name}[{index}].obs_{key} has shape {value.shape}; "
                f"expected ({dimension},) with finite values"
            )
    try:
        identity = (int(record["env_id"]), int(record["episode"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{sample_name}[{index}] has no valid env/episode identity") from exc
    return identity


def load_samples(
    audit_output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(audit_output / "samples.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("samples.pt must contain a dictionary")
    state_weighted = payload.get("candidate_state_weighted")
    episode_balanced = payload.get("candidate_episode_balanced")
    if not isinstance(state_weighted, list) or not isinstance(episode_balanced, list):
        raise TypeError("samples.pt lacks the two candidate sample lists")
    if len(state_weighted) > STATE_SAMPLE_CAP or len(episode_balanced) > EPISODE_SAMPLE_CAP:
        raise ValueError(
            "candidate sample cap exceeded: "
            f"state={len(state_weighted)}, episode={len(episode_balanced)}"
        )
    for index, record in enumerate(state_weighted):
        validate_record(record, "candidate_state_weighted", index)
    identities = [
        validate_record(record, "candidate_episode_balanced", index)
        for index, record in enumerate(episode_balanced)
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("episode-balanced samples contain duplicate (env, episode) identities")
    return state_weighted, episode_balanced


def observation_batch(records: list[dict[str, Any]], device: torch.device, tensor_dict_type: type):
    tensors = {}
    for key, dimension in OBS_DIMS.items():
        array = np.stack([np.asarray(record[f"obs_{key}"]) for record in records])
        if array.shape != (len(records), dimension):
            raise AssertionError(f"unexpected stacked {key} shape: {array.shape}")
        tensors[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
    return tensor_dict_type(tensors, batch_size=[len(records)])


def load_models(
    *,
    args: argparse.Namespace,
    gradient: ModuleType,
    actor_critic_type: type,
    tensor_dict_type: type,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]], float, torch.Tensor]:
    runner_config = (
        args.root / "upstream/source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py"
    )
    env_config = runner_config.parent / "env_cfg.py"
    position_scale = gradient.configured_call_scale(env_config, "JointPositionActionCfg")
    velocity_scale = gradient.configured_call_scale(args.code / "config.py", "JointVelocityActionCfg")
    if position_scale != 0.25 or velocity_scale != 5.0:
        raise ValueError(
            f"unexpected action scales: joint_pos={position_scale}, joint_vel={velocity_scale}"
        )
    config = gradient.model_config(actor_critic_type, runner_config)
    template_obs = tensor_dict_type(
        {key: torch.zeros(1, dimension) for key, dimension in OBS_DIMS.items()},
        batch_size=[1],
    )
    template = actor_critic_type(
        template_obs,
        {"policy": ["policy"], "critic": ["critic"]},
        NUM_ACTIONS,
        **config,
    )
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not isinstance(selection, list):
        raise TypeError("selection must be a list")
    labels = [item.get("label") for item in selection if isinstance(item, dict)]
    if len(selection) != 3 or labels[:2] != ["V6_43202", "V9_44711"]:
        raise ValueError(f"expected V6/V9/V11 selection, got {labels}")
    if not isinstance(labels[2], str) or not labels[2].startswith("V11_"):
        raise ValueError(f"third selection must be V11, got {labels[2]!r}")

    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    coefficient: float | None = None
    for short_label, item in zip(("V6", "V9", "V11"), selection, strict=True):
        path = gradient.resolve_checkpoint(args.root, item["path"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = copy.deepcopy(template)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device).eval()
        models[short_label] = model
        checkpoints[short_label] = {
            "label": item["label"],
            "path": str(path),
            "completed_updates": state.get("completed_updates"),
        }
        if short_label in ("V6", "V9"):
            model.requires_grad_(False)
        else:
            startup_anchor = state.get("startup_anchor")
            if not isinstance(startup_anchor, dict):
                raise ValueError("V11 checkpoint lacks startup_anchor provenance")
            try:
                coefficient = float(startup_anchor["coefficients"]["student"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("V11 checkpoint lacks a valid student coefficient") from exc
        del state
    if coefficient is None or not math.isclose(
        coefficient,
        EXPECTED_STUDENT_COEFFICIENT,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"unexpected saved student coefficient: {coefficient!r}")
    action_scales = torch.tensor(
        [position_scale] * 12 + [velocity_scale] * 4,
        dtype=torch.float32,
        device=device,
    )
    return models, checkpoints, coefficient, action_scales


def gradient_decision(
    coverage: dict[str, Any],
    state_result: dict[str, Any],
    balanced_result: dict[str, Any],
) -> dict[str, Any]:
    state = state_result["student_encoder"]
    balanced = balanced_result["student_encoder"]
    state_cosine = state.get("gradient_cosine")
    balanced_cosine = balanced.get("gradient_cosine")
    ratio = state.get("weighted_anchor_to_distill_norm_ratio")
    finite = all(
        value is not None and math.isfinite(float(value))
        for value in (state_cosine, balanced_cosine, ratio)
    )
    state_direction = finite and float(state_cosine) >= 0.5
    state_scale = finite and 0.03 <= float(ratio) <= 0.15
    same_sign = finite and float(state_cosine) * float(balanced_cosine) >= 0.0
    sensitivity = finite and abs(float(state_cosine) - float(balanced_cosine)) <= 0.2
    checks = {
        "coverage": bool(coverage["passed"]),
        "state_weighted_cosine_at_least_0.5": state_direction,
        "state_weighted_ratio_in_0.03_to_0.15": state_scale,
        "balanced_cosine_not_opposite": same_sign,
        "balanced_cosine_absolute_difference_at_most_0.2": sensitivity,
        "finite_nonzero_gradient_metrics": finite,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "state_weighted_cosine": state_cosine,
            "state_weighted_weighted_norm_ratio": ratio,
            "episode_balanced_cosine": balanced_cosine,
            "cosine_absolute_difference": None
            if not finite
            else abs(float(state_cosine) - float(balanced_cosine)),
        },
        "decision": "eligible_for_separate_finite_mask_experiment"
        if all(checks.values())
        else "reject_without_starting_training",
    }


def main() -> None:
    args = parse_args()
    output = args.audit_output / "gradient_gate.json"
    if output.exists():
        raise FileExistsError(output)
    status, distribution = validate_completed_audit(args.audit_output)
    coverage = coverage_gate(distribution)
    state_records, episode_records = load_samples(args.audit_output)
    if not state_records or not episode_records:
        empty_sets = [
            name
            for name, records in (
                ("state_weighted", state_records),
                ("episode_balanced", episode_records),
            )
            if not records
        ]
        payload = {
            "scope": {
                "offline_fixed_audit_observations": True,
                "optimizer_steps": 0,
                "does_not_start_training": True,
            },
            "sources": {"audit_output": str(args.audit_output)},
            "coverage": coverage,
            "sample_sets": {
                "state_weighted": {"count": len(state_records), "cap": STATE_SAMPLE_CAP},
                "episode_balanced": {"count": len(episode_records), "cap": EPISODE_SAMPLE_CAP},
            },
            "results": {},
            "gate": {
                "passed": False,
                "checks": {"nonempty_candidate_sample_sets": False},
                "decision": "reject_without_starting_training",
                "reason": f"empty candidate sample sets: {empty_sets}",
            },
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "gate": payload["gate"]}, indent=2))
        return
    gradient = load_gradient_module(args.gradient_module)

    upstream_source = args.root / "upstream/source/rsl_rl"
    if not upstream_source.is_dir():
        raise FileNotFoundError(upstream_source)
    if not (args.code / "anchor_objective.py").is_file():
        raise FileNotFoundError(args.code / "anchor_objective.py")
    sys.path[:0] = [str(upstream_source), str(args.code)]
    from anchor_objective import deterministic_mean, masked_action_loss
    from rsl_rl.modules import ActorCriticMoECTS
    from tensordict import TensorDict

    device = torch.device(args.device)
    models, checkpoints, coefficient, action_scales = load_models(
        args=args,
        gradient=gradient,
        actor_critic_type=ActorCriticMoECTS,
        tensor_dict_type=TensorDict,
        device=device,
    )
    checkpoint_hash = file_sha256(Path(checkpoints["V11"]["path"]))
    reference_hash = file_sha256(Path(checkpoints["V6"]["path"]))
    if checkpoint_hash != status["checkpoint_sha256"]:
        raise ValueError("selected V11 does not match the completed distribution audit")
    if reference_hash != status["reference_sha256"]:
        raise ValueError("selected V6 does not match the completed distribution audit")
    if checkpoints["V11"]["completed_updates"] != status.get("completed_updates"):
        raise ValueError("selected V11 completed_updates disagrees with the audit")
    hashes_before = {label: gradient.state_hash(model) for label, model in models.items()}
    gradient.assert_no_accumulated_gradients(models)
    results = {}
    for name, records in (
        ("state_weighted", state_records),
        ("episode_balanced", episode_records),
    ):
        observations = observation_batch(records, device, TensorDict)
        results[name] = gradient.audit_window(
            observations,
            models,
            coefficient,
            action_scales,
            deterministic_mean,
            masked_action_loss,
        )
        gradient.assert_no_accumulated_gradients(models)
    hashes_after = {label: gradient.state_hash(model) for label, model in models.items()}
    if hashes_after != hashes_before:
        raise AssertionError("model state changed during read-only gradient gate")
    decision = gradient_decision(
        coverage,
        results["state_weighted"],
        results["episode_balanced"],
    )
    payload = {
        "scope": {
            "offline_fixed_audit_observations": True,
            "not_training_minibatch_gradients": True,
            "latent_gradient_excludes_load_balance": True,
            "no_ppo_gradient_or_direction": True,
            "not_causal_evidence": True,
            "optimizer_steps": 0,
            "does_not_start_training": True,
        },
        "sources": {
            "audit_output": str(args.audit_output),
            "selection": str(args.selection),
            "gradient_module": str(args.gradient_module),
            "code": str(args.code),
        },
        "audit_status": {
            "state": status["state"],
            "optimizer_updates": status["optimizer_updates"],
            "state_hashes_before": status["state_hashes_before"],
            "state_hashes_after": status["state_hashes_after"],
            "launcher_patch_restored": status.get("launcher_patch_restored"),
            "isolated_audit_process_exited": status.get(
                "isolated_audit_process_exited"
            ),
            "no_live_audit_process": status.get("no_live_audit_process"),
            "input_files_unchanged": status["input_files_unchanged"],
        },
        "coverage": coverage,
        "sample_sets": {
            "state_weighted": {"count": len(state_records), "cap": STATE_SAMPLE_CAP},
            "episode_balanced": {"count": len(episode_records), "cap": EPISODE_SAMPLE_CAP},
        },
        "checkpoints": checkpoints,
        "checkpoint_sha256": checkpoint_hash,
        "reference_sha256": reference_hash,
        "saved_anchor_student_coefficient": coefficient,
        "model_state_sha256_before": hashes_before,
        "model_state_sha256_after": hashes_after,
        "parameters_unchanged": True,
        "gradients_accumulated": False,
        "results": results,
        "gate": decision,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
