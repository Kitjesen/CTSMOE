"""Retain the original course/RNG state and refuse unverified continuation.

This module does not pretend that PhysX RNG or terrain level numbers identify
the same difficulty/physics distribution in Newton. Restoration needs an
explicit schema migration and native acceptance before a production learn.
"""

import torch


STATE_KEYS = ("terrain_levels", "terrain_types", "common_step_counter",
              "python_rng", "numpy_rng", "torch_rng", "cuda_rng")


def retain_legacy_state(saved):
    missing = [key for key in STATE_KEYS if key not in saved]
    if missing:
        raise ValueError("Legacy checkpoint is missing continuation state: " + ", ".join(missing))
    result = {key: saved[key] for key in STATE_KEYS}
    result.update(format="newpose_legacy_environment_v1",
                  terrain_schema=saved["signature"]["terrain_schema"],
                  physics_schema="legacy_isaaclab_physx",
                  captured_at_updates=int(saved["completed_updates"]),
                  num_envs=int(saved["terrain_levels"].numel()))
    validate_retained_state(result)
    return result


def validate_retained_state(state):
    if state.get("format") != "newpose_legacy_environment_v1":
        raise ValueError("Unknown retained environment-state format")
    count = state["num_envs"]
    for key, maximum in (("terrain_levels", 10), ("terrain_types", 20)):
        value = state[key]
        if value.shape != (count,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Invalid retained " + key)
        if (value < 0).any() or (value >= maximum).any():
            raise ValueError("Retained course indices are outside the old 10 x 20 schema")
    if not isinstance(state["torch_rng"], torch.Tensor) or not isinstance(state["cuda_rng"], (list, tuple)):
        raise ValueError("Unrecognized saved Torch/CUDA RNG representation")


def describe_state(state, *, optimizer_loaded):
    """Report state retention and replay boundaries without exposing RNG values."""
    if state is None:
        return {"mode": "diagnostic_fresh_course", "retained": False,
                "course_restored": False, "rng_restored": False,
                "formal_continuation_allowed": False, "reason": "Missing source environment state"}
    validate_retained_state(state)
    cuda_states = state["cuda_rng"]
    current_count = torch.cuda.device_count()
    cuda_format_supported = all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                                and value.ndim == 1 for value in cuda_states)
    cuda_initialized = torch.cuda.is_initialized()
    if cuda_format_supported and current_count == len(cuda_states) and current_count and cuda_initialized:
        cuda_format_supported = all(value.numel() == torch.cuda.get_rng_state(index).numel()
                                    for index, value in enumerate(cuda_states))
    else:
        cuda_format_supported = False
    return {"mode": "unverified_continuation" if optimizer_loaded else "diagnostic_fresh_course",
            "retained": True, "captured_at_updates": state["captured_at_updates"],
            "source_num_envs": state["num_envs"], "terrain_schema": state["terrain_schema"],
            "physics_schema": state["physics_schema"], "source_cuda_device_count": len(cuda_states),
            "current_cuda_device_count": current_count,
            "cuda_rng_format_status": "structurally_compatible_not_restored" if cuda_format_supported
                                      else ("cuda_not_initialized_format_unverified_not_restored"
                                            if current_count == len(cuda_states) and not cuda_initialized
                                            else "unsupported_or_different_device_state_format_not_restored"),
            "course_restored": False, "rng_restored": False,
            "formal_continuation_allowed": False,
            "reason": "Cross-backend course/RNG mapping and native acceptance are not established"}


def require_formal_continuation(runner):
    state = runner._environment_state
    if state is None:
        raise ValueError("Formal continuation requires retained terrain and RNG state")
    validate_retained_state(state)
    raw_env = getattr(runner.env, "unwrapped", runner.env)
    if raw_env.num_envs != state["num_envs"]:
        raise ValueError(f"Formal course has {state['num_envs']} environments; got {raw_env.num_envs}. "
                         "Use load_optimizer=False for a fresh-course diagnostic")
    if getattr(raw_env.cfg, "newpose_terrain_schema", None) != state["terrain_schema"]:
        raise ValueError("Formal terrain schema differs or is undeclared; an explicit accepted migration is required")
    if getattr(raw_env.cfg, "newpose_physics_schema", None) != state["physics_schema"]:
        raise ValueError("Formal physics schema differs; old RNG state must not be assumed Newton-compatible")
    terrain = raw_env.scene.terrain
    if not torch.equal(terrain.terrain_types.detach().cpu(), state["terrain_types"].detach().cpu()):
        raise ValueError("Formal terrain column assignments differ; do not silently remap the course")
    if state["captured_at_updates"] != runner.completed_updates:
        raise ValueError("Retained course state is older than the checkpoint update count")
    # No configuration flag bypasses this gate. An actual restoration/mapping
    # implementation and native acceptance must precede enabling formal learn.
    status = runner.environment_resume_status
    if not status["course_restored"] or not status["rng_restored"] or not status["formal_continuation_allowed"]:
        raise ValueError("Course/RNG restoration and native acceptance are unverified; formal learn is blocked")
