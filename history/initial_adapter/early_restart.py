"""Independent early-phase prevention profile for a normal early checkpoint.

The profile adds continuous actual-state costs without changing the official
all-terrain hip/joint L1 terms. Geometry alone releases the added costs during
stairs or pit climbing; flat is always active and unrelated terrain is inert.
"""
from __future__ import annotations

import torch

from startup_repair import learner_cohort_mask


PROFILE = "early_phase_v14"
RESET_FRACTION = 0.75
HIP_WEIGHT = -0.20
THIGH_CALF_WEIGHT = -0.08
HEIGHT_WEIGHT = -15.0
VERTICAL_WEIGHT = -0.5
STANDING_SCALE = 3.0
ZERO_COMMAND_THRESHOLD = 0.05
TARGET_HEIGHT_M = 0.46


def special_ids(env, env_ids) -> torch.Tensor:
    command = env.command_manager.get_term("base_velocity")
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    flat = command.terrain_idxs[ids] == command.terrain_type2idx["flat"]
    return ids[flat & learner_cohort_mask(command.learner)[ids]]


def normal_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg):
    """Keep the native mixed reset, then normalize only the selected flat cohort."""
    from mixed_task import mixed_reset_root

    mixed_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg)
    ids = special_ids(env, env_ids)
    if len(ids):
        robot = env.scene["robot"]
        state = robot.data.root_state_w[ids].clone()
        state[:, :3] = env.scene.env_origins[ids] + robot.data.default_root_state[ids, :3]
        state[:, 7:13] = 0.0
        robot.write_root_state_to_sim(state, env_ids=ids)


def normal_reset_joints(env, env_ids, position_range, velocity_range, asset_cfg):
    """Keep pit and other native initialization; narrow only selected flat joints."""
    from isaaclab.envs.mdp.events import reset_joints_by_scale
    from mixed_task import mixed_reset_joints

    mixed_reset_joints(env, env_ids, position_range, velocity_range, asset_cfg)
    ids = special_ids(env, env_ids)
    if len(ids):
        reset_joints_by_scale(env, ids, (0.98, 1.02), (0.0, 0.0), asset_cfg)


def zero_command(commands: torch.Tensor) -> torch.Tensor:
    """A pure yaw request is motion: all three axes must be near zero."""
    return (commands.abs() <= ZERO_COMMAND_THRESHOLD).all(dim=-1)


def continuous_l1(q: torch.Tensor, q0: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """Actual-angle L1 with no dead zone and a stronger true-stop coefficient."""
    scale = torch.where(zero_command(commands), STANDING_SCALE, 1.0)
    return (q - q0).abs().sum(dim=-1) * scale


def _joint_indices(env) -> tuple[list[int], list[int]]:
    if not hasattr(env, "_early_restart_joint_ids"):
        names = env.scene["robot"].joint_names
        hip = [i for i, name in enumerate(names) if name.endswith("_hip_joint")]
        thigh_calf = [i for i, name in enumerate(names)
                      if name.endswith("_thigh_joint") or name.endswith("_calf_joint")]
        if len(hip) != 4 or len(thigh_calf) != 8:
            raise ValueError("Early restart requires four hip and eight thigh/calf joints")
        env._early_restart_joint_ids = (hip, thigh_calf)
    return env._early_restart_joint_ids


def phase_weight(env) -> tuple[torch.Tensor, dict]:
    """Return a geometry-only mask: flat, normal stairs/pit, or inactive."""
    from phase_transition import measure

    command = env.command_manager.get_term("base_velocity")
    phase = measure(env)
    flat = command.terrain_idxs == command.terrain_type2idx["flat"]
    weight = torch.zeros(env.num_envs, device=env.device)
    weight[flat] = 1.0
    relevant = phase["stairs"] | phase["pit"]
    weight = torch.where(relevant, phase["weight"], weight)
    return weight, phase


def raw_terms(env) -> dict[str, torch.Tensor]:
    command = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    data = robot.data
    hip, thigh_calf = _joint_indices(env)
    gate, phase = phase_weight(env)
    ground = env.scene.env_origins[:, 2]
    relevant = phase["stairs"] | phase["pit"]
    ground = torch.where(relevant, phase["ground"], ground)
    active = gate > 0
    # A missing terrain ray must not turn the height cost into zero through
    # phase_transition's geometric placeholder ground=root_z-target_height.
    valid_ground = torch.isfinite(phase["wheel_ground"]).all(-1)
    finite = torch.isfinite(torch.cat((
        data.joint_pos[:, hip + thigh_calf], data.default_joint_pos[:, hip + thigh_calf],
        command.command, data.root_pos_w[:, 2:3], data.root_lin_vel_w[:, 2:3],
        ground[:, None], gate[:, None]), dim=-1)).all(-1)
    if bool((~torch.isfinite(gate) | (active & (~finite | (relevant & ~valid_ground)))).any()):
        raise FloatingPointError("Non-finite active posture state or support terrain ray")
    costs = {
        "hip": continuous_l1(data.joint_pos[:, hip], data.default_joint_pos[:, hip], command.command),
        "thigh_calf": continuous_l1(
            data.joint_pos[:, thigh_calf], data.default_joint_pos[:, thigh_calf], command.command
        ),
        "height": (data.root_pos_w[:, 2] - ground - TARGET_HEIGHT_M).square(),
        "vertical": data.root_lin_vel_w[:, 2].square(),
    }
    return {name: torch.where(active, gate * value, torch.zeros_like(value))
            for name, value in costs.items()}


def reward(env, kind: str) -> torch.Tensor:
    terms = raw_terms(env)
    if kind not in terms:
        raise ValueError("Unknown early restart objective")
    return terms[kind]


def configure(cfg, profile: str) -> None:
    if profile not in ("off", PROFILE):
        raise ValueError("Unknown early restart profile")
    cfg.early_restart_profile = profile
    if profile == "off":
        return
    for name in ("startup_repair_profile", "pit_posture_profile",
                 "surface_posture_profile", "stance_strength_profile"):
        if getattr(cfg, name, "off") != "off":
            raise ValueError(f"Early restart requires {name}=off")

    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from mixed_task import mixed_curriculum
    from startup_sampling import command_class

    command = cfg.commands.base_velocity
    command.class_type = command_class()
    command.resampling_time = 5.0
    command.resampling_time_range = (5.0, 5.0)
    cfg.episode_length_s = 25.0
    if hasattr(cfg, "terminations") and hasattr(cfg.terminations, "startup_window"):
        cfg.terminations.startup_window = None
    cfg.curriculum.terrain_levels.func = mixed_curriculum
    cfg.events.reset_base.func = normal_reset_root
    cfg.events.reset_robot_joints.func = normal_reset_joints
    cfg.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot")
    cfg.rewards.early_actual_hip_l1 = RewardTermCfg(
        func=reward, weight=HIP_WEIGHT, params={"kind": "hip"}
    )
    cfg.rewards.early_actual_thigh_calf_l1 = RewardTermCfg(
        func=reward, weight=THIGH_CALF_WEIGHT, params={"kind": "thigh_calf"}
    )
    cfg.rewards.early_support_height = RewardTermCfg(
        func=reward, weight=HEIGHT_WEIGHT, params={"kind": "height"}
    )
    cfg.rewards.early_vertical_velocity = RewardTermCfg(
        func=reward, weight=VERTICAL_WEIGHT, params={"kind": "vertical"}
    )


def snapshot(env) -> dict | None:
    if getattr(env.cfg, "early_restart_profile", "off") == "off":
        return None
    command = env.command_manager.get_term("base_velocity")
    ids = special_ids(env, torch.arange(env.num_envs, device=env.device))
    terms = raw_terms(env)
    gate, phase = phase_weight(env)
    term_names = {
        "hip": "early_actual_hip_l1",
        "thigh_calf": "early_actual_thigh_calf_l1",
        "height": "early_support_height",
        "vertical": "early_vertical_velocity",
    }
    weights = {name: env.reward_manager.get_term_cfg(term).weight
               for name, term in term_names.items()}
    sampling = (command.startup_sampling_snapshot()
                if hasattr(command, "startup_sampling_snapshot") else None)
    terrain_counts = {
        name: int((command.terrain_idxs == index).sum())
        for name, index in command.terrain_type2idx.items()
    }
    return {
        "profile": PROFILE,
        "scope": "flat and geometry-normal stairs/high-platform phases; released while climbing",
        "actual_joint_l1_no_deadzone": True,
        "standing_all_axes_threshold": ZERO_COMMAND_THRESHOLD,
        "standing_multiplier": STANDING_SCALE,
        "target_height_m": TARGET_HEIGHT_M,
        "weights": weights,
        "raw_mean": {name: float(value.mean()) for name, value in terms.items()},
        "weighted_mean": {name: float((value * weights[name]).mean()) for name, value in terms.items()},
        "cohort": {
            "flat_count": len(ids),
            "fraction_per_learner": RESET_FRACTION,
            "teacher": int((command.learner[ids] == 0).sum()),
            "student": int((command.learner[ids] == 1).sum()),
        },
        "sampling": sampling,
        "phase_counts": {
            "flat_active": int(((command.terrain_idxs == command.terrain_type2idx["flat"]) & (gate > 0)).sum()),
            "stairs_normal": int((phase["stairs"] & (gate > 0)).sum()),
            "pit_normal": int((phase["pit"] & (gate > 0)).sum()),
            "climb_released": int(((phase["stairs"] | phase["pit"]) & (gate < 0.01)).sum()),
        },
        "terrain_counts": terrain_counts,
    }


def install_allocation(allocation, iteration: int) -> dict | None:
    """Install fixed V14 quotas through natural resets, preserving difficulty."""
    from terrain_allocation import (
        AllocationCourse,
        EARLY_BALANCED,
        EARLY_BALANCED_PROFILE,
        reassign,
    )

    if allocation.course.profile == EARLY_BALANCED_PROFILE:
        return None
    if allocation.pending:
        raise ValueError("Finish pending adaptive allocation moves before early restart allocation")
    terrain = allocation.env.scene.terrain
    levels, columns, receipt = reassign(
        terrain.terrain_levels.cpu().tolist(),
        terrain.terrain_types.cpu().tolist(),
        allocation.names,
        allocation.groups,
        EARLY_BALANCED,
        seed=iteration,
        profile=EARLY_BALANCED_PROFILE,
    )
    allocation.course = AllocationCourse(
        profile=EARLY_BALANCED_PROFILE,
        weights=dict(EARLY_BALANCED),
        last_check=iteration,
    )
    allocation.stats.zero_()
    allocation.pending = {i: (levels[i], columns[i]) for i, _ in receipt["moves"]}
    receipt["bounds_profile"] = EARLY_BALANCED_PROFILE
    receipt["application"] = "natural episode resets; retained difficulties preserved"
    receipt["optimality_claim"] = "none; fixed experimental coverage"
    return receipt
