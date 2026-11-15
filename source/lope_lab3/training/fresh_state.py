"""Course resume for a Lope run. Episodes restart; physics trajectories do not."""
import hashlib
import json
import random
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from . import COURSE_FORMAT


SOURCE_FILES = (
    "spec.py",
    "assets/contract.py",
    "assets/robot.py",
    "tasks/lope/env_cfg.py",
    "tasks/lope/commands.py",
    "tasks/lope/terrain.py",
    "tasks/lope/scene_cfg.py",
    "tasks/lope/mdp/observations.py",
    "tasks/lope/mdp/symmetry.py",
    "symmetry_math.py",
    "tasks/lope/mdp/terms.py",
    "tasks/lope/mdp/randomization.py",
    "tasks/lope/mdp/privileged.py",
    "tasks/lope/rsl_rl_cfg.py",
    "training/runner.py",
    "training/checkpoint.py",
    "training/fresh_state.py",
    "training/history.py",
    "training/timeout.py",
)


def canonical_configuration(value):
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: canonical_configuration(item) for key, item in value.items()}
        pairs = [[canonical_configuration(key), canonical_configuration(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, allow_nan=False))
        return {"__mapping__": pairs}
    if isinstance(value, tuple):
        return {"__tuple__": [canonical_configuration(item) for item in value]}
    if isinstance(value, list):
        return [canonical_configuration(item) for item in value]
    if isinstance(value, slice):
        return {"__slice__": canonical_configuration([value.start, value.stop, value.step])}
    if isinstance(value, (torch.device, torch.dtype)):
        return {"__" + type(value).__name__ + "__": str(value)}
    if isinstance(value, np.ndarray):
        return {"__numpy_array__": {"dtype": str(value.dtype), "values": canonical_configuration(value.tolist())}}
    if isinstance(value, np.generic):
        return canonical_configuration(value.item())
    if isinstance(value, Enum):
        return {"__enum__": type(value).__module__ + ":" + type(value).__qualname__, "value": canonical_configuration(value.value)}
    if callable(value):
        return {"__callable__": value.__module__ + ":" + value.__qualname__}
    if hasattr(value, "to_dict"):
        return canonical_configuration(value.to_dict())
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unserializable training configuration: " + type(value).__name__)


def training_contract(env):
    cfg = env.cfg
    values = cfg.to_dict()
    selected = {key: values.get(key) for key in (
        "seed", "decimation", "episode_length_s", "actions", "observations", "commands",
        "events", "rewards", "terminations", "curriculum", "lope_terrain_schema",
        "lope_physics_schema", "lope_action_dim", "lope_robot")}
    scene = values["scene"]
    selected["terrain"] = scene["terrain"]
    selected["height_scanner"] = scene["height_scanner"]
    selected["contact_forces"] = scene["contact_forces"]
    selected["robot"] = {key: scene["robot"][key] for key in ("init_state", "actuators", "soft_joint_pos_limit_factor")}
    selected["sim"] = {key: values["sim"][key] for key in ("dt", "physics", "use_newton_actuators")}
    asset = Path(scene["robot"]["spawn"]["asset_path"])
    selected["urdf_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    package = Path(__file__).resolve().parents[1]
    selected["task_source_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCE_FILES
    }
    canonical = json.dumps(canonical_configuration(selected), sort_keys=True, allow_nan=False)
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "terrain_schema": cfg.lope_terrain_schema,
        "physics_schema": cfg.lope_physics_schema,
        "urdf_sha256": selected["urdf_sha256"],
        "robot": "lope_rs02_12dof",
    }


def capture_course(env, completed_updates):
    terrain = env.scene.terrain
    return {
        "format": COURSE_FORMAT,
        "num_envs": env.num_envs,
        "completed_updates": completed_updates,
        "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
        "terrain_types": terrain.terrain_types.detach().cpu().clone(),
        "terrain_origins": terrain.terrain_origins.detach().cpu().clone(),
        "common_step_counter": int(env.common_step_counter),
        "sim_step_counter": int(getattr(env, "_sim_step_counter", 0)),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def validate_course(env, state, completed_updates):
    if state.get("format") != COURSE_FORMAT:
        raise ValueError("Missing Lope Newton course state")
    if state["num_envs"] != env.num_envs or state["completed_updates"] != completed_updates:
        raise ValueError("Resume requires the saved environment count and update count")
    origins = env.scene.terrain.terrain_origins.detach().cpu()
    if not torch.equal(origins, state["terrain_origins"].cpu()):
        raise ValueError("Generated terrain geometry layout differs; use the saved task/seed")


def restore_course(env, state, completed_updates):
    validate_course(env, state, completed_updates)
    terrain = env.scene.terrain
    terrain.terrain_levels.copy_(state["terrain_levels"].to(terrain.terrain_levels.device))
    terrain.terrain_types.copy_(state["terrain_types"].to(terrain.terrain_types.device))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.common_step_counter = state["common_step_counter"]
    env._sim_step_counter = state["sim_step_counter"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    compute = env.curriculum_manager.compute
    env.curriculum_manager.compute = lambda **kwargs: None
    try:
        env.reset()
    finally:
        env.curriculum_manager.compute = compute
    return {"mode": "lope_resume_episode_reset", "course_restored": True, "physics_state_restored": False}
