"""Offline gradient audit on fixed V11-student rollout observations.

This script never starts Isaac, steps an optimizer, or reconstructs PPO gradients.
It compares the frozen-V6 action anchor with the existing CTS latent-distillation
loss on observations already recorded by ``probe.py``.  Results describe gradient
directions on these fixed trajectories only; they are neither training-minibatch
gradients nor causal evidence for the observed behavior.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


OBS_DIMS = {"policy": 530, "single_obs": 53, "critic": 291}
NUM_ACTIONS = 16
WINDOWS_S = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0))
CONFIG_CLASS = "RslRlMoeCtsActorCriticCfg"
CONFIG_KEYS = (
    "init_noise_std",
    "actor_obs_normalization",
    "critic_obs_normalization",
    "actor_hidden_dims",
    "critic_hidden_dims",
    "teacher_encoder_hidden_dims",
    "student_encoder_hidden_dims",
    "expert_num",
    "activation",
    "noise_std_type",
    "state_dependent_std",
    "latent_dim",
    "norm_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path.cwd(),
        help="Root for relative checkpoint paths in selection JSON.",
    )
    parser.add_argument(
        "--upstream-source",
        type=Path,
        required=True,
        help="Directory containing the rsl_rl package (normally source/rsl_rl).",
    )
    parser.add_argument(
        "--runner-config",
        type=Path,
        required=True,
        help="Actual robot_lab/tasks/go2/rsl_rl_cfg.py used by training.",
    )
    parser.add_argument(
        "--code",
        type=Path,
        required=True,
        help="Verified training-code directory containing anchor_objective.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def class_literal_assignments(path: Path, class_name: str) -> dict[str, Any]:
    """Read literal class settings without importing the Isaac-dependent config."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {class_name} in {path}, found {len(matches)}")
    values: dict[str, Any] = {}
    for node in matches[0].body:
        name = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if name in CONFIG_KEYS and value_node is not None:
            values[name] = ast.literal_eval(value_node)
    return values


def class_literal_value(path: Path, class_name: str, key: str) -> Any:
    """Read one literal class assignment from the pinned runner config."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"expected one {class_name} in {path}, found {len(classes)}")
    matches = []
    for node in classes[0].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == key:
                matches.append(ast.literal_eval(node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == key and node.value is not None:
                matches.append(ast.literal_eval(node.value))
    if len(matches) != 1:
        raise ValueError(f"expected one literal {class_name}.{key} in {path}")
    return matches[0]


def model_config(actor_critic_type: type, config_path: Path) -> dict[str, Any]:
    """Merge explicit runner config with actual constructor defaults."""
    explicit = class_literal_assignments(config_path, CONFIG_CLASS)
    signature = inspect.signature(actor_critic_type.__init__)
    config: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        if key in explicit:
            config[key] = explicit[key]
            continue
        parameter = signature.parameters[key]
        if parameter.default is inspect.Parameter.empty:
            raise ValueError(f"{key} is absent from config and has no constructor default")
        config[key] = copy.deepcopy(parameter.default)
    if config["expert_num"] != 8:
        raise ValueError(f"expected the configured eight experts, got {config['expert_num']}")
    return config


def configured_call_scale(path: Path, call_name: str) -> float:
    """Read and validate one literal action scale without importing Isaac."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scales = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name != call_name:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if "scale" not in keywords:
            continue
        value = ast.literal_eval(keywords["scale"])
        values = value.values() if isinstance(value, dict) else (value,)
        unique = {float(item) for item in values}
        if len(unique) != 1:
            raise ValueError(f"{path} has nonuniform {call_name} scales: {unique}")
        scales.append(unique.pop())
    if len(scales) != 1:
        raise ValueError(f"expected one literal {call_name} scale in {path}, got {scales}")
    return scales[0]


def resolve_checkpoint(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_no_accumulated_gradients(models: dict[str, torch.nn.Module]) -> None:
    dirty = [
        f"{label}:{name}"
        for label, model in models.items()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if dirty:
        raise AssertionError(f"diagnostic accumulated .grad tensors: {dirty[:8]}")


def gradient_vector(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    parts = [
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None
        else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    vector = torch.cat(parts)
    if not torch.isfinite(vector).all():
        raise ValueError("non-finite gradient in offline audit")
    return vector


def vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.double()))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left.double()) * torch.linalg.vector_norm(
        right.double()
    )
    if denominator == 0.0:
        return None
    value = torch.dot(left.double(), right.double()) / denominator
    return float(value)


def action_mse(
    current: torch.Tensor,
    reference: torch.Tensor,
    masked_action_loss,
    action_scales: torch.Tensor,
) -> torch.Tensor:
    """Use production physical normalization with equal action/physical scales."""
    return masked_action_loss(
        current,
        reference,
        torch.ones(current.shape[0], dtype=torch.bool, device=current.device),
        action_scales,
        action_scales,
    )


def select_continuous_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        case
        for case in manifest["cases"]
        if case.get("controller_source") == "v11_student"
        and case.get("intervention_target") is None
    ]
    if len(cases) != 6:
        raise ValueError(f"expected six continuous V11-student cases, found {len(cases)}")
    directions = [case["direction"] for case in cases]
    if len(set(directions)) != 6:
        raise ValueError(f"continuous commands are not unique: {directions}")
    return cases


def samples_for_window(
    traces: dict[str, np.ndarray],
    cases: list[dict[str, Any]],
    dt: float,
    start_s: float,
    end_s: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in OBS_DIMS}
    counts: dict[str, int] = {}
    total_steps = traces["obs_policy"].shape[0]
    start_step = int(round(start_s / dt))
    end_step = int(round(end_s / dt))
    if abs(start_step * dt - start_s) > 1e-7 or abs(end_step * dt - end_s) > 1e-7:
        raise ValueError(f"window {start_s}-{end_s}s does not align with dt={dt}")
    for case in cases:
        case_id = int(case["id"])
        valid_end = min(end_step, int(case["end_step"]) + 1, total_steps)
        count = max(0, valid_end - start_step)
        counts[case["direction"]] = count
        if count == 0:
            continue
        source_keys = {
            "policy": "obs_policy",
            "single_obs": "obs_single_obs",
            "critic": "obs_critic",
        }
        for destination, source in source_keys.items():
            arrays[destination].append(traces[source][start_step:valid_end, case_id])
    if not arrays["policy"]:
        raise ValueError(f"window {start_s}-{end_s}s contains no valid observations")
    observations = {
        key: torch.as_tensor(np.concatenate(values), dtype=torch.float32, device=device)
        for key, values in arrays.items()
    }
    return observations, counts


def audit_window(
    observations: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    anchor_student_coefficient: float,
    action_scales: torch.Tensor,
    deterministic_mean,
    masked_action_loss,
) -> dict[str, Any]:
    v6 = models["V6"]
    v9 = models["V9"]
    v11 = models["V11"]
    with torch.no_grad():
        v6_action = deterministic_mean(v6, observations, is_teacher=False)
        v9_action = deterministic_mean(v9, observations, is_teacher=False)
        v11_student_value = deterministic_mean(v11, observations, is_teacher=False)
        v11_teacher_value = deterministic_mean(v11, observations, is_teacher=True)

    student_parameters = tuple(v11.student_moe_encoder.parameters())
    student_action = deterministic_mean(
        v11,
        observations,
        is_teacher=False,
        freeze_actor=True,
    )
    student_anchor_loss = action_mse(
        student_action,
        v6_action,
        masked_action_loss,
        action_scales,
    )
    anchor_gradient = gradient_vector(student_anchor_loss, student_parameters)

    actor_obs = v11.actor_obs_normalizer(v11.get_actor_obs(observations))
    student_latent, gating_weights = v11.student_moe_encoder(actor_obs)
    with torch.no_grad():
        critic_obs = v11.critic_obs_normalizer(v11.get_critic_obs(observations))
        teacher_latent = v11.teacher_encoder(critic_obs)
    latent_distill_loss = (teacher_latent - student_latent).square().mean()
    distill_gradient = gradient_vector(latent_distill_loss, student_parameters)

    anchor_norm = vector_norm(anchor_gradient)
    distill_norm = vector_norm(distill_gradient)
    weighted_anchor_norm = anchor_student_coefficient * anchor_norm
    weighted_ratio = None if distill_norm == 0.0 else weighted_anchor_norm / distill_norm

    student_main = deterministic_mean(
        v11,
        observations,
        is_teacher=False,
        detach_student=True,
    )
    teacher_main = deterministic_mean(v11, observations, is_teacher=True)
    student_main_loss = action_mse(
        student_main,
        v6_action,
        masked_action_loss,
        action_scales,
    )
    teacher_main_loss = action_mse(
        teacher_main,
        v6_action,
        masked_action_loss,
        action_scales,
    )
    main_anchor_loss = 0.5 * (student_main_loss + teacher_main_loss)
    actor_parameters = tuple(v11.actor.parameters())
    teacher_parameters = tuple(v11.teacher_encoder.parameters())
    student_actor_gradient = gradient_vector(student_main_loss, actor_parameters)
    teacher_branch_gradient = gradient_vector(
        teacher_main_loss,
        actor_parameters + teacher_parameters,
    )
    actor_size = sum(parameter.numel() for parameter in actor_parameters)
    teacher_actor_gradient = teacher_branch_gradient[:actor_size]
    teacher_encoder_gradient = teacher_branch_gradient[actor_size:]
    combined_actor_gradient = 0.5 * (student_actor_gradient + teacher_actor_gradient)

    def rmse(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.detach() - right.detach()).square().mean().sqrt())

    return {
        "sample_count": int(observations["policy"].shape[0]),
        "student_encoder": {
            "v6_action_anchor_loss": float(student_anchor_loss.detach()),
            "cts_latent_distill_loss": float(latent_distill_loss.detach()),
            "moe_mean_usage": gating_weights.detach().mean(dim=0).cpu().tolist(),
            "v6_action_anchor_gradient_norm": anchor_norm,
            "cts_latent_distill_gradient_norm": distill_norm,
            "gradient_cosine": cosine(anchor_gradient, distill_gradient),
            "anchor_student_coefficient": anchor_student_coefficient,
            "weighted_anchor_gradient_norm": weighted_anchor_norm,
            "weighted_anchor_to_distill_norm_ratio": weighted_ratio,
        },
        "actions": {
            "v11_student_vs_v6_rmse": rmse(v11_student_value, v6_action),
            "v11_student_vs_v9_rmse": rmse(v11_student_value, v9_action),
            "v11_teacher_vs_v6_rmse": rmse(v11_teacher_value, v6_action),
            "v11_teacher_vs_v9_rmse": rmse(v11_teacher_value, v9_action),
            "v6_vs_v9_rmse": rmse(v6_action, v9_action),
        },
        "main_action_anchor": {
            "student_action_loss": float(student_main_loss.detach()),
            "teacher_action_loss": float(teacher_main_loss.detach()),
            "combined_half_sum_loss": float(main_anchor_loss.detach()),
            "student_branch_actor_gradient_norm": vector_norm(student_actor_gradient),
            "teacher_branch_actor_gradient_norm": vector_norm(teacher_actor_gradient),
            "combined_actor_gradient_norm": vector_norm(combined_actor_gradient),
            "teacher_encoder_gradient_norm": 0.5
            * vector_norm(teacher_encoder_gradient),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.upstream_source.is_dir():
        raise FileNotFoundError(args.upstream_source)
    if not (args.code / "anchor_objective.py").is_file():
        raise FileNotFoundError(args.code / "anchor_objective.py")
    env_config = args.runner_config.parent / "env_cfg.py"
    thunder_config = args.code / "config.py"
    position_scale = configured_call_scale(env_config, "JointPositionActionCfg")
    velocity_scale = configured_call_scale(thunder_config, "JointVelocityActionCfg")
    if position_scale != 0.25 or velocity_scale != 5.0:
        raise ValueError(
            f"unexpected action scales: joint_pos={position_scale}, joint_vel={velocity_scale}"
        )
    sys.path.insert(0, str(args.upstream_source))
    sys.path.insert(0, str(args.code))
    from anchor_objective import deterministic_mean, masked_action_loss
    from rsl_rl.modules import ActorCriticMoECTS
    from tensordict import TensorDict

    device = torch.device(args.device)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    labels = [item["label"] for item in selection]
    if len(selection) != 3 or labels[:2] != ["V6_43202", "V9_44711"]:
        raise ValueError(f"expected V6/V9/V11 selection, got {labels}")
    if not labels[2].startswith("V11_"):
        raise ValueError(f"third selection must be V11, got {labels[2]}")

    config = model_config(ActorCriticMoECTS, args.runner_config)
    load_balance_coefficient = float(
        class_literal_value(
            args.runner_config,
            "RslRlMoeCtsAlgorithmCfg",
            "load_balance_coef",
        )
    )
    if not np.isfinite(load_balance_coefficient) or load_balance_coefficient < 0.0:
        raise ValueError(f"invalid load-balance coefficient: {load_balance_coefficient}")
    template_obs = TensorDict(
        {key: torch.zeros(1, dimension) for key, dimension in OBS_DIMS.items()},
        batch_size=[1],
    )
    template = ActorCriticMoECTS(
        template_obs,
        {"policy": ["policy"], "critic": ["critic"]},
        NUM_ACTIONS,
        **config,
    )
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for short_label, item in zip(("V6", "V9", "V11"), selection, strict=True):
        path = resolve_checkpoint(args.checkpoint_root, item["path"])
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
        if short_label == "V11":
            saved_anchor = state.get("startup_anchor")
            if not isinstance(saved_anchor, dict):
                raise ValueError("V11 checkpoint has no saved startup_anchor state")
            coefficient = saved_anchor["coefficients"]["student"]
            if coefficient is None or not np.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"invalid saved anchor student coefficient: {coefficient}")
            anchor_student_coefficient = float(coefficient)
        del state

    manifest = json.loads((args.traces_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("smoke_only"):
        raise ValueError("smoke traces are not a behavior diagnosis")
    dt = float(manifest["dt"])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"invalid manifest dt: {dt}")
    cases = select_continuous_cases(manifest)
    with np.load(args.traces_dir / "traces.npz") as archive:
        required = {"obs_policy", "obs_single_obs", "obs_critic"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trace archive lacks observation keys: {sorted(missing)}")
        traces = {key: archive[key] for key in required}
    shape_expectations = {
        "obs_policy": OBS_DIMS["policy"],
        "obs_single_obs": OBS_DIMS["single_obs"],
        "obs_critic": OBS_DIMS["critic"],
    }
    common_shape = traces["obs_policy"].shape[:2]
    for key, dimension in shape_expectations.items():
        value = traces[key]
        if value.ndim != 3 or value.shape[:2] != common_shape or value.shape[2] != dimension:
            raise ValueError(f"unexpected {key} shape {value.shape}; expected [T,C,{dimension}]")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")

    hashes_before = {label: state_hash(model) for label, model in models.items()}
    assert_no_accumulated_gradients(models)
    action_scales = torch.tensor(
        [position_scale] * 12 + [velocity_scale] * 4,
        dtype=torch.float32,
        device=device,
    )
    if action_scales.shape != (NUM_ACTIONS,):
        raise AssertionError(action_scales.shape)
    windows = []
    for start_s, end_s in WINDOWS_S:
        observations, counts = samples_for_window(
            traces,
            cases,
            dt,
            start_s,
            end_s,
            device,
        )
        result = audit_window(
            observations,
            models,
            anchor_student_coefficient,
            action_scales,
            deterministic_mean,
            masked_action_loss,
        )
        by_command = {}
        for case in cases:
            command_observations, command_counts = samples_for_window(
                traces,
                [case],
                dt,
                start_s,
                end_s,
                device,
            )
            command_result = audit_window(
                command_observations,
                models,
                anchor_student_coefficient,
                action_scales,
                deterministic_mean,
                masked_action_loss,
            )
            command_result["samples_per_command"] = command_counts
            by_command[case["direction"]] = command_result
        result.update(
            window_s=[start_s, end_s],
            samples_per_command=counts,
            anchor_active_during_training=bool(start_s >= 0.0 and end_s <= 1.0),
            by_command=by_command,
        )
        windows.append(result)
        assert_no_accumulated_gradients(models)

    hashes_after = {label: state_hash(model) for label, model in models.items()}
    if hashes_after != hashes_before:
        raise AssertionError("model state changed during read-only gradient audit")
    output = {
        "scope": {
            "description": "offline gradients on six fixed continuous V11-student trajectories",
            "not_training_minibatch_gradients": True,
            "no_ppo_direction_without_advantage_and_logprob": True,
            "not_causal_evidence": True,
            "post_1s_anchor_gradients_are_counterfactual_not_applied_in_training": True,
            "latent_gradient_excludes_load_balance": True,
            "actual_student_primary_also_includes_load_balance": True,
            "actual_student_primary_load_balance_coefficient": load_balance_coefficient,
            "optimizer_steps": 0,
        },
        "sources": {
            "selection": str(args.selection),
            "traces_dir": str(args.traces_dir),
            "upstream_source": str(args.upstream_source),
            "runner_config": str(args.runner_config),
            "anchor_objective": str(args.code / "anchor_objective.py"),
            "position_action_config": str(env_config),
            "velocity_action_config": str(thunder_config),
        },
        "architecture": {**config, "observation_dims": OBS_DIMS, "num_actions": NUM_ACTIONS},
        "action_normalization": {
            "joint_position_actions": 12,
            "joint_velocity_actions": 4,
            "joint_position_scale": position_scale,
            "joint_velocity_scale": velocity_scale,
            "physical_scales_equal_action_scales": True,
            "normalized_loss_equals_raw_action_mse": True,
        },
        "checkpoints": checkpoints,
        "model_state_sha256_before": hashes_before,
        "model_state_sha256_after": hashes_after,
        "parameters_unchanged": True,
        "gradients_accumulated": False,
        "manifest_dt": dt,
        "continuous_cases": [
            {
                "id": case["id"],
                "direction": case["direction"],
                "command": case["command"],
                "end_step": case["end_step"],
                "controller_source": case["controller_source"],
            }
            for case in cases
        ],
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "windows": len(windows)}, indent=2))


if __name__ == "__main__":
    main()
