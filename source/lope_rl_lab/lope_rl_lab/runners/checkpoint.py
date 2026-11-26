"""Lope checkpoint container. Thunder formats are rejected."""
import hashlib
import math
from pathlib import Path

import torch

from lope_rl_lab.assets.lope_cfg import ACTION_DIM
from lope_rl_lab.tasks.locomotion.config import (
    CRITIC_DIM,
    CUMULATIVE_UPDATES,
    EXPERTS,
    FEATURE_NAMES,
    FRAME_DIM,
    HISTORY_LENGTH,
    LATENT_DIM,
)
from . import FRESH_FORMAT, HISTORY_LAYOUT, UPSTREAM_COMMIT


def fresh_signature(critic_dim=CRITIC_DIM):
    return {
        "frame": FRAME_DIM,
        "history": HISTORY_LENGTH,
        "actions": ACTION_DIM,
        "experts": EXPERTS,
        "latent": LATENT_DIM,
        "critic": critic_dim,
        "robot": "lope_rs02_12dof",
        "privileged_extension": list(FEATURE_NAMES),
    }


def _reject_foreign(saved):
    fmt = str(saved.get("compat_format", ""))
    blob = fmt + " " + str(saved.get("signature", ""))
    if "thunder" in blob.lower() or "newpose" in blob.lower():
        raise ValueError("Thunder checkpoints cannot be loaded for Lope")
    signature = saved.get("signature") or {}
    if signature.get("actions") not in (None, ACTION_DIM):
        raise ValueError("Refusing a checkpoint whose action width is not 12")


def normalize_checkpoint(saved, critic_dim=CRITIC_DIM):
    _reject_foreign(saved)
    if saved.get("compat_format") != FRESH_FORMAT:
        raise ValueError("Lope only loads lope_newton_fresh_v1 checkpoints")
    provenance = saved.get("compat_provenance", {})
    if provenance.get("initialization") != "random" or provenance.get("robot") != "lope_rs02_12dof":
        raise ValueError("Missing Lope from-scratch ancestry")
    if saved.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Native checkpoint uses a different upstream algorithm revision")
    if saved.get("history_layout") != HISTORY_LAYOUT or saved.get("timeout_bootstrap") != "terminal_observation":
        raise ValueError("Native checkpoint observation/timeout contract differs")
    if saved.get("iter") != saved.get("completed_updates"):
        raise ValueError("Native checkpoint iteration must equal completed updates")
    if not saved.get("training_contract") or not saved.get("environment_state"):
        raise ValueError("Native checkpoint has no task/course provenance")
    signature = saved.get("signature") or {}
    expected = fresh_signature(critic_dim)
    if any(signature.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint observation/network signature differs from Lope")
    completed = saved.get("completed_updates")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= CUMULATIVE_UPDATES:
        raise ValueError("Completed update count outside the Lope contract")
    rate = float(saved["learning_rate"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid saved adaptive learning rate")
    return dict(saved)


def validate_for_algorithm(saved, algorithm):
    current = algorithm.policy.state_dict()
    incoming = saved["model_state_dict"]
    if list(current) != list(incoming):
        raise ValueError("Model tensor names/order differ from the installed upstream model")
    for name, target in current.items():
        source = incoming[name]
        if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
            raise ValueError(f"Incompatible/nonfinite model tensor: {name}")


def load_checkpoint(path, critic_dim=CRITIC_DIM, map_location="cpu"):
    path = Path(path)
    saved = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint container is not a dictionary")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = normalize_checkpoint(saved, critic_dim)
    result.setdefault("compat_provenance", {})["input_checkpoint_sha256"] = digest
    return result
