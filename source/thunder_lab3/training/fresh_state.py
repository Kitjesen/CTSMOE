"""Same-backend course/RNG resume; episodes restart, physics trajectories do not."""

import hashlib
import json
import random
from enum import Enum
from pathlib import Path

import numpy as np
import torch


def canonical_configuration(value):
    """Retain resolved Lab configuration types without repr addresses or key coercion."""
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
        return {"__enum__": type(value).__module__ + ":" + type(value).__qualname__,
                "value": canonical_configuration(value.value)}
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
    """Bind a fresh run to the actual task, asset and control/reward settings."""
    cfg = env.cfg
    if not hasattr(cfg, "to_dict"):
        # Synthetic runner tests supply an explicit, small contract.
        return dict(cfg.newpose_test_contract)
    values = cfg.to_dict()
    selected = {key: values.get(key) for key in (
        "seed", "decimation", "episode_length_s", "actions", "observations", "commands",
        "events", "rewards", "terminations", "curriculum", "newpose_terrain_schema",
        "newpose_physics_schema")}
    scene = values["scene"]
    selected["terrain"] = scene["terrain"]
    selected["robot"] = {key: scene["robot"][key] for key in ("init_state", "actuators", "soft_joint_pos_limit_factor")}
    selected["sim"] = {key: values["sim"][key] for key in ("dt", "physics", "use_newton_actuators")}
    asset = Path(scene["robot"]["spawn"]["asset_path"])
    selected["urdf_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    project = Path(__file__).resolve().parents[1]
    selected["task_source_sha256"] = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in ("tasks/thunder/env_cfg.py", "tasks/thunder/mdp/terms.py", "tasks/thunder/terrain.py",
                     "tasks/thunder/scene_cfg.py", "tasks/thunder/mdp/observations.py", "tasks/thunder/mdp/privileged.py",
                     "assets/contract.py", "assets/robot.py", "tasks/thunder/mdp/height.py",
                     "tasks/thunder/mdp/isaac_height.py", "tasks/thunder/mdp/height_config.py", "tasks/thunder/boxes.py")}

    canonical = json.dumps(canonical_configuration(selected), sort_keys=True, allow_nan=False)
    return {"sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "terrain_schema": cfg.newpose_terrain_schema,
            "physics_schema": cfg.newpose_physics_schema,
            "urdf_sha256": selected["urdf_sha256"]}


def capture_course(env, completed_updates):
    terrain = env.scene.terrain
    return {"format": "thunder_newton_course_v1", "num_envs": env.num_envs,
            "completed_updates": completed_updates,
            "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
            "terrain_types": terrain.terrain_types.detach().cpu().clone(),
            "terrain_origins": terrain.terrain_origins.detach().cpu().clone(),
            "common_step_counter": int(env.common_step_counter),
            "sim_step_counter": int(getattr(env, "_sim_step_counter", 0)),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def validate_course(env, state, completed_updates):
    if state.get("format") != "thunder_newton_course_v1":
        raise ValueError("Missing native Newton course state")
    if state["num_envs"] != env.num_envs or state["completed_updates"] != completed_updates:
        raise ValueError("Resume requires the saved environment count and update count")
    terrain = env.scene.terrain
    origins = terrain.terrain_origins.detach().cpu()
    if not torch.equal(origins, state["terrain_origins"].cpu()):
        raise ValueError("Generated terrain geometry layout differs; use the saved task/seed")
    for key, maximum in (("terrain_levels", origins.shape[0]), ("terrain_types", origins.shape[1])):
        value = state[key]
        if value.shape != (env.num_envs,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Invalid saved " + key)
        if (value < 0).any() or (value >= maximum).any():
            raise ValueError("Saved course index is out of bounds")
    if int(state["common_step_counter"]) < 0 or int(state["sim_step_counter"]) < 0:
        raise ValueError("Invalid saved simulation counters")
    if len(state["cuda_rng"]) != (torch.cuda.device_count() if torch.cuda.is_initialized() else 0):
        raise ValueError("Resume requires the saved CUDA device count")


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
    # A resume reset must not count the abandoned partial episode as a curriculum failure.
    compute = env.curriculum_manager.compute
    env.curriculum_manager.compute = lambda **kwargs: None
    try:
        env.reset()
    finally:
        env.curriculum_manager.compute = compute
    return {"mode": "native_resume_episode_reset", "course_restored": True,
            "rng_restored_before_reset": True, "physics_state_restored": False}
