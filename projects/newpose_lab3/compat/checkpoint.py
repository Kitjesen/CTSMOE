"""Validate and rename checkpoint containers for the upstream lab3 runner.

    Tensor names are unchanged. Newton's teacher-column migration is delegated
    to the independently audited teacher.checkpoint module, never duplicated.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from teacher.checkpoint import SOURCE_SHA256, migrate_newton_checkpoint
from teacher.privileged import FEATURE_NAMES
from . import HISTORY_LAYOUT, UPSTREAM_COMMIT
from .environment_state import retain_legacy_state, validate_retained_state

FORMAT = "thunder_go2_lab3_v1"
FRESH_FORMAT = "thunder_newton_fresh_v1"


def fresh_signature(critic_dim=309):
    return {"frame": 53, "history": 10, "actions": 16, "experts": 8,
            "latent": 32, "critic": critic_dim,
            "privileged_extension": list(FEATURE_NAMES),
            "material_schema": ["newton_shared_mu", "restitution"]}


def validate_signature(signature, critic_dim):
    expected = {"frame": 53, "history": 10, "actions": 16, "experts": 8,
                "latent": 32, "critic": critic_dim}
    if any(signature.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint observation/network signature differs from Thunder")
    if critic_dim == 309:
        if signature.get("privileged_extension") != list(FEATURE_NAMES):
            raise ValueError("Unknown privileged feature order")
        if signature.get("material_schema") != ["newton_shared_mu", "restitution"]:
            raise ValueError("Unknown Newton material schema")
    elif critic_dim != 274:
        raise ValueError("Only the original 274 and migrated Newton 309 layouts are supported")


def normalize_checkpoint(saved, source_sha256, critic_dim=309):
    """Distinguish native fresh runs from the separately verified legacy lineage."""
    native = saved.get("compat_format") == FRESH_FORMAT
    if native:
        result = dict(saved)
        provenance = saved.get("compat_provenance", {})
        if provenance.get("initialization") != "random" or not provenance.get("run_id"):
            raise ValueError("Missing fresh-run ancestry")
        if saved.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("Native checkpoint uses a different upstream algorithm revision")
        if saved.get("history_layout") != HISTORY_LAYOUT or saved.get("timeout_bootstrap") != "terminal_observation":
            raise ValueError("Native checkpoint observation/timeout contract differs")
        if saved.get("iter") != saved.get("completed_updates"):
            raise ValueError("Native checkpoint iteration must equal completed updates")
        if not saved.get("training_contract") or not saved.get("environment_state"):
            raise ValueError("Native checkpoint has no task/course provenance")
    elif saved.get("compat_format") == FORMAT:
        result = dict(saved)
        if saved.get("compat_provenance", {}).get("formal_source_sha256") != SOURCE_SHA256:
            raise ValueError("Missing verified formal checkpoint ancestry")
        if saved.get("history_layout") != HISTORY_LAYOUT:
            raise ValueError("Incompatible observation-history layout")
        if saved.get("timeout_bootstrap") != "terminal_observation":
            raise ValueError("Unknown timeout-bootstrap semantics")
        if saved.get("iter") != saved.get("completed_updates"):
            raise ValueError("Checkpoint iteration index is not its completed-update count")
        if saved.get("environment_state") is not None:
            validate_retained_state(saved["environment_state"])
    elif "model" in saved and "signature" in saved:
        signature = saved["signature"]
        if signature.get("critic") == 274:
            if source_sha256 != SOURCE_SHA256 or saved.get("completed_updates") != 19538:
                raise ValueError("Original checkpoint must match the verified frozen 19538 SHA256")
            if critic_dim == 309:
                saved = migrate_newton_checkpoint(saved)
            elif critic_dim != 274:
                raise ValueError("Unsupported destination critic schema")
        elif signature.get("critic") == 309:
            migration = saved.get("privileged_migration", {})
            if not migration.get("source_hash_verified") or migration.get("source_sha256") != SOURCE_SHA256:
                raise ValueError("Extended legacy checkpoint has no verified source migration")
        else:
            raise ValueError("Unknown legacy checkpoint schema")
        result = {
            "model_state_dict": saved["model"], "optimizer_state_dict": saved["ppo_optimizer"],
            "optimizer_stu_enc_state_dict": saved["student_optimizer"],
            "learning_rate": saved["learning_rate"], "completed_updates": saved["completed_updates"],
            "iter": saved["completed_updates"], "signature": dict(saved["signature"]),
            "history_layout": HISTORY_LAYOUT, "compat_format": FORMAT,
            "timeout_bootstrap": "terminal_observation",
            "environment_state": retain_legacy_state(saved),
            "compat_provenance": {"formal_source_sha256": SOURCE_SHA256,
                                  "input_checkpoint_sha256": source_sha256,
                                  "upstream_commit": UPSTREAM_COMMIT,
                                  "environment_state_restored": False,
                                  "teacher_critic_forward_preserved": critic_dim == 274,
                                  "note": "Model and both Adam states retained; cross-backend warm start. "
                                          "Old terrain/RNG/simulator state is not restored."},
            "infos": {"legacy_curriculum_provenance": saved.get("curriculum_provenance")},
        }
    else:
        raise ValueError("Unknown checkpoint container; do not infer counts from filenames or raw iter")
    validate_signature(result["signature"], critic_dim)
    completed = result.get("completed_updates")
    minimum = 0 if native else 19538
    if isinstance(completed, bool) or not isinstance(completed, int) or not minimum <= completed <= 20000:
        raise ValueError("Completed update count outside the experiment contract")
    rate = float(result["learning_rate"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid saved adaptive learning rate")
    if any(float(group["lr"]) != rate for group in result["optimizer_state_dict"]["param_groups"]):
        raise ValueError("Saved algorithm learning rate differs from PPO optimizer")
    return result


def validate_for_algorithm(saved, algorithm):
    """Validate every model tensor and Adam assignment before mutating the runner."""
    current = algorithm.policy.state_dict()
    incoming = saved["model_state_dict"]
    if list(current) != list(incoming):
        raise ValueError("Model tensor names/order differ from the installed upstream model")
    for name, target in current.items():
        source = incoming[name]
        if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
            raise ValueError(f"Incompatible/nonfinite model tensor: {name}")
    for key, optimizer in (("optimizer_state_dict", algorithm.optimizer),
                           ("optimizer_stu_enc_state_dict", algorithm.optimizer_stu_enc)):
        source = saved[key]
        if len(source["param_groups"]) != len(optimizer.param_groups):
            raise ValueError(f"Optimizer group layout differs: {key}")
        all_ids = []
        initial_optimizer = (saved.get("compat_format") == FRESH_FORMAT
                             and saved["completed_updates"] == 0 and not source["state"])
        for src, dst in zip(source["param_groups"], optimizer.param_groups):
            if len(src["params"]) != len(dst["params"]):
                raise ValueError(f"Optimizer parameter count differs: {key}")
            all_ids.extend(src["params"])
            for identifier, parameter in zip(src["params"], dst["params"]):
                moment = source["state"].get(identifier)
                if initial_optimizer:
                    continue
                if moment is None or not {"step", "exp_avg", "exp_avg_sq"} <= moment.keys():
                    raise ValueError(f"Incomplete Adam state: {key}")
                for field, tensor in moment.items():
                    if isinstance(tensor, torch.Tensor):
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f"Nonfinite Adam state: {key}/{field}")
                        if field == "step":
                            if tensor.numel() != 1:
                                raise ValueError("Adam step must be scalar")
                        elif tensor.shape != parameter.shape:
                            raise ValueError(f"Adam moment shape differs: {key}/{field}")
        if len(set(all_ids)) != len(all_ids) or (not initial_optimizer and set(all_ids) != set(source["state"])):
            raise ValueError(f"Unexpected/repeated optimizer IDs: {key}")


def load_checkpoint(path, critic_dim=309, map_location="cpu"):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, map_location=map_location, weights_only=False)
    return normalize_checkpoint(saved, digest, critic_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--critic-dim", type=int, choices=(274, 309), default=309)
    args = parser.parse_args()
    saved = load_checkpoint(args.source, args.critic_dim)
    with args.output.open("xb") as stream:
        torch.save(saved, stream)
    print(json.dumps({"output": str(args.output), "completed_updates": saved["completed_updates"],
                      "critic": args.critic_dim, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "physics_state_continuity": False}, indent=2))


if __name__ == "__main__":
    main()
