# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary
# ruff: noqa: E402

"""Evaluate Thunder velocity tracking, actuator load, and contact force in one Isaac process."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import NamedTuple, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cli_args

from isaaclab.app import AppLauncher


class PoseSpec(NamedTuple):
    hip_magnitude: float
    base_z: float


POSE_SPECS = {
    0.20: PoseSpec(0.20, 0.4600),
    0.15: PoseSpec(0.15, 0.4526),
    0.10: PoseSpec(0.10, 0.4441),
    0.00: PoseSpec(0.00, 0.4239),
}


def first_sustained_recovery_step(
    values: Sequence[float], threshold: float, hold_steps: int
) -> int | None:
    """Return the first index ending a sustained in-threshold window."""
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if abs(value) <= threshold else 0
        if streak == hold_steps:
            return index
    return None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pose", type=float, choices=tuple(POSE_SPECS), default=0.20)
parser.add_argument("--initial-hip-scale", type=float, default=1.0)
parser.add_argument("--initial-roll-deg", type=float, default=0.0)
parser.add_argument("--static-friction", type=float)
parser.add_argument("--dynamic-friction", type=float)
parser.add_argument("--recovery-roll-deg", type=float, default=5.0)
parser.add_argument("--failure-roll-deg", type=float, default=30.0)
parser.add_argument("--recovery-hold-steps", type=int, default=25)
parser.add_argument("--recovery-hip-magnitude", type=float, default=0.30)
parser.add_argument("--recovery-time-limit-s", type=float, default=2.0)
parser.add_argument("--fall-tilt-deg", type=float, default=60.0)
parser.add_argument("--fall-height-m", type=float, default=0.25)
parser.add_argument("--require-recovery-pass", action="store_true")
parser.add_argument("--kp", type=float, default=120.0)
parser.add_argument("--kd", type=float, default=8.0)
parser.add_argument("--seed", type=int, default=20260901)
parser.add_argument("--num-envs-per-scenario", type=int, default=256)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--sample-steps", type=int, default=600)
parser.add_argument("--command-profile", choices=("fixed", "step", "ramp"), default="fixed")
parser.add_argument("--command-transition-steps", type=int, default=50)
parser.add_argument(
    "--scenarios",
    nargs="+",
    choices=("stand", "forward_0.6", "backward_-0.6", "left_0.6", "right_-0.6", "yaw_left_0.6", "yaw_right_-0.6"),
)
parser.add_argument("--gain-randomization", type=float, nargs=2, metavar=("LOW", "HIGH"))
parser.add_argument("--payload-mass", type=float, default=0.0, help="Deterministic payload mass in kg.")
parser.add_argument(
    "--payload-profile",
    choices=("center", "front_left", "front_right", "rear_left", "rear_right"),
    default="center",
)
parser.add_argument("--payload-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
parser.add_argument("--device", default="cuda:0")
cli_args.add_latency_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(headless=True, device=args.device).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat

import doso_train.tasks  # noqa: F401
from doso_train.tasks.terrain_inward.agents.rsl_rl_ppo_cfg import (
    DosoThunderTerrainInwardFlatPPORunnerCfg,
)
from doso_train.tasks.terrain_inward.env_cfg import (
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    DosoThunderTerrainInwardFlatEnvCfg,
)
from doso_train.tasks.terrain_inward.latency import ACTION_DIM, ThunderLatencyWrapper


TASK_ID = "DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0"
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_0.6": (0.6, 0.0, 0.0),
    "backward_-0.6": (-0.6, 0.0, 0.0),
    "left_0.6": (0.0, 0.6, 0.0),
    "right_-0.6": (0.0, -0.6, 0.0),
    "yaw_left_0.6": (0.0, 0.0, 0.6),
    "yaw_right_-0.6": (0.0, 0.0, -0.6),
}
CONTACT_THRESHOLD_N = 1.0
LEG_TORQUE_LIMIT_NM = 120.0
WHEEL_TORQUE_LIMIT_NM = 17.0
PAYLOAD_PROFILES = {
    "center": (0.0, 0.0, 0.125),
    "front_left": (0.15, 0.10, 0.20),
    "front_right": (0.15, -0.10, 0.20),
    "rear_left": (-0.15, 0.10, 0.20),
    "rear_right": (-0.15, -0.10, 0.20),
}

# argparse is constructed before simulator launch; validate scenario names here to keep the default schema unchanged.
if args.scenarios:
    unknown_scenarios = set(args.scenarios) - set(SCENARIOS)
    if unknown_scenarios:
        parser.error(f"unknown scenarios: {sorted(unknown_scenarios)}")
ACTIVE_SCENARIOS = {name: SCENARIOS[name] for name in (args.scenarios or SCENARIOS)}


def _zero_range(keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    return {key: (0.0, 0.0) for key in keys}


def _stats(values: torch.Tensor, torque_limit: float | None = None) -> dict[str, float]:
    absolute = values.abs().flatten()
    result = {
        "mean_abs": round(absolute.mean().item(), 9),
        "rms": round(torch.sqrt(torch.mean(values.flatten().square())).item(), 9),
        "p99_abs": round(torch.quantile(absolute, 0.99).item(), 9),
        "max_abs": round(absolute.max().item(), 9),
    }
    if torque_limit is not None:
        result["over_90_limit_fraction"] = round(
            (absolute > 0.90 * torque_limit).float().mean().item(), 12
        )
    return result


def _command_scale(profile: str, step: int, warmup_steps: int, transition_steps: int) -> float:
    """Return the target-command scale for the current control step."""
    if profile == "fixed":
        return 1.0
    sample_step = step - warmup_steps
    if sample_step < 0:
        return 0.0
    if profile == "step" or transition_steps == 1:
        return 1.0
    return min(sample_step / (transition_steps - 1), 1.0)


def _difference_stats(values: torch.Tensor) -> dict[str, dict[str, float]]:
    """Summarize first and second action differences over time."""
    first = torch.diff(values, dim=0)
    second = torch.diff(first, dim=0)

    def summarize(difference: torch.Tensor) -> dict[str, float]:
        if difference.numel() == 0:
            return {"rms": 0.0, "max_abs": 0.0}
        return {
            "rms": round(torch.sqrt(difference.square().mean()).item(), 9),
            "max_abs": round(difference.abs().max().item(), 9),
        }

    return {"first_difference": summarize(first), "second_difference": summarize(second)}


def _per_joint_stats(values: torch.Tensor, names: list[str]) -> dict[str, dict[str, float]]:
    return {name: _stats(values[..., index]) for index, name in enumerate(names)}


def _recovery_stats(
    roll: torch.Tensor,
    hip_angles: torch.Tensor,
    height: torch.Tensor,
    step_dt: float,
) -> dict[str, float | bool | None]:
    roll_abs = roll.abs()
    recovery_steps = [
        first_sustained_recovery_step(
            roll[:, env_index].tolist(),
            threshold=math.radians(args.recovery_roll_deg),
            hold_steps=args.recovery_hold_steps,
        )
        for env_index in range(roll.shape[1])
    ]
    recovery_times = torch.tensor(
        [(step + 1) * step_dt for step in recovery_steps if step is not None],
        device=roll.device,
    )
    late_steps = min(50, len(roll))
    late_roll = roll[-late_steps:]
    late_hips = hip_angles[-late_steps:]
    recovered_fraction = len(recovery_times) / roll.shape[1]
    failure_roll_fraction = (roll_abs.amax(dim=0) > math.radians(args.failure_roll_deg)).float().mean().item()
    fell = torch.logical_or(
        roll_abs.amax(dim=0) > math.radians(args.fall_tilt_deg),
        height.amin(dim=0) < args.fall_height_m,
    )
    hip_soft_limit_violated = hip_angles.abs().amax(dim=0).amax(dim=1) > 0.9 * 0.41
    late_hip_stuck_fraction = (
        late_hips.abs() > args.recovery_hip_magnitude
    ).float().mean().item()
    recovery_time_p95 = (
        round(torch.quantile(recovery_times, 0.95).item(), 9) if len(recovery_times) else None
    )
    result = {
        "recovered_fraction": round(recovered_fraction, 9),
        "recovery_time_median_s": (
            round(recovery_times.median().item(), 9) if len(recovery_times) else None
        ),
        "recovery_time_p95_s": recovery_time_p95,
        "roll_peak_abs_deg": round(math.degrees(roll_abs.max().item()), 9),
        "late_roll_rms_deg": round(
            math.degrees(torch.sqrt(late_roll.square().mean()).item()), 9
        ),
        "late_hip_abs_max_rad": round(late_hips.abs().max().item(), 9),
        "late_hip_stuck_fraction": round(late_hip_stuck_fraction, 9),
        "base_height_min_m": round(height.min().item(), 9),
        "failure_roll_fraction": round(failure_roll_fraction, 9),
        "fall_fraction": round(fell.float().mean().item(), 9),
        "hip_soft_limit_violation_fraction": round(
            hip_soft_limit_violated.float().mean().item(), 9
        ),
    }
    result["passed"] = bool(
        recovered_fraction >= 0.99
        and recovery_time_p95 is not None
        and recovery_time_p95 <= args.recovery_time_limit_s
        and failure_roll_fraction <= 0.01
        and result["late_roll_rms_deg"] <= args.recovery_roll_deg
        and late_hip_stuck_fraction <= 0.05
        and result["fall_fraction"] == 0.0
        and result["hip_soft_limit_violation_fraction"] == 0.0
    )
    return result


def _configure_payload(env_cfg: DosoThunderTerrainInwardFlatEnvCfg) -> tuple[float, float, float]:
    if args.payload_mass < 0.0:
        raise ValueError("--payload-mass must be non-negative")
    position = tuple(args.payload_position or PAYLOAD_PROFILES[args.payload_profile])
    payload_event = getattr(env_cfg.events, "randomize_base_payload", None)
    if payload_event is None:
        if args.payload_mass > 0.0:
            raise RuntimeError("This environment does not expose events.randomize_base_payload")
        return position
    payload_event.params.update(
        {
            "payload_mass_range": (args.payload_mass, args.payload_mass),
            "payload_zero_probability": 1.0 if args.payload_mass == 0.0 else 0.0,
            "payload_position_range": {
                axis: (value, value) for axis, value in zip(("x", "y", "z"), position, strict=True)
            },
            "base_mass_scale_range": (1.0, 1.0),
            "base_com_bias_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z")},
        }
    )
    return position


def _configure_environment() -> tuple[
    DosoThunderTerrainInwardFlatEnvCfg,
    DosoThunderTerrainInwardFlatPPORunnerCfg,
    tuple[float, float, float],
    float,
]:
    env_cfg = DosoThunderTerrainInwardFlatEnvCfg()
    agent_cfg = DosoThunderTerrainInwardFlatPPORunnerCfg()
    env_cfg.scene.num_envs = len(ACTIVE_SCENARIOS) * args.num_envs_per_scenario
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    payload_position = _configure_payload(env_cfg)

    pose = POSE_SPECS[args.pose]
    if not 0.0 < args.initial_hip_scale <= 2.0:
        raise ValueError("--initial-hip-scale must keep the default 0.20 rad hips within ±0.40 rad")
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, pose.base_z)
    for leg in ("fr", "rl"):
        env_cfg.scene.robot.init_state.joint_pos[f"{leg}_hip_joint"] = -pose.hip_magnitude
    for leg in ("fl", "rr"):
        env_cfg.scene.robot.init_state.joint_pos[f"{leg}_hip_joint"] = pose.hip_magnitude
    for actuator_name in ("hip", "thigh", "calf"):
        actuator = env_cfg.scene.robot.actuators[actuator_name]
        actuator.stiffness = args.kp
        actuator.damping = args.kd

    env_cfg.observations.policy.enable_corruption = False
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.debug_vis = False
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_pure_yaw_envs = 0.0
    command_cfg.rel_pure_lateral_envs = 0.0
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)

    disabled_events = [
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_push_robot",
    ]
    if args.static_friction is None and args.dynamic_friction is None:
        disabled_events.append("randomize_rigid_body_material")
    elif args.static_friction is None or args.dynamic_friction is None:
        raise ValueError("--static-friction and --dynamic-friction must be set together")
    else:
        if not 0.0 <= args.dynamic_friction <= args.static_friction:
            raise ValueError("friction must satisfy 0 <= dynamic <= static")
        material_params = env_cfg.events.randomize_rigid_body_material.params
        material_params["static_friction_range"] = (args.static_friction, args.static_friction)
        material_params["dynamic_friction_range"] = (args.dynamic_friction, args.dynamic_friction)
        material_params["restitution_range"] = (0.0, 0.0)
    for event_name in disabled_events:
        setattr(env_cfg.events, event_name, None)
    reset_pose_range = _zero_range(("x", "y", "z", "roll", "pitch", "yaw"))
    initial_roll = math.radians(args.initial_roll_deg)
    reset_pose_range["roll"] = (initial_roll, initial_roll)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": reset_pose_range,
        "velocity_range": _zero_range(("x", "y", "z", "roll", "pitch", "yaw")),
    }
    if args.gain_randomization is None:
        env_cfg.events.randomize_actuator_gains = None
    else:
        low, high = args.gain_randomization
        gain_params = env_cfg.events.randomize_actuator_gains.params
        gain_params["stiffness_distribution_params"] = (low, high)
        gain_params["damping_distribution_params"] = (low, high)
    return env_cfg, agent_cfg, payload_position, args.initial_hip_scale


def _write_csv(output: Path, result: dict) -> None:
    fieldnames = [
        "scenario",
        "target_vx",
        "target_vy",
        "target_yaw",
        "actual_vx",
        "actual_vy",
        "actual_yaw",
        "tracking_rmse",
        "leg_torque_mean_abs",
        "leg_torque_rms",
        "hip_torque_rms",
        "wheel_torque_rms",
        "leg_power_mean_abs",
        "wheel_power_mean_abs",
        "contact_history_peak_n",
        "contact_duty",
        "roll_rms_rad",
        "pitch_rms_rad",
        "base_height_std_m",
        "wheel_velocity_rms",
        "raw_action_delta_rms",
        "raw_action_delta_max_abs",
        "raw_action_second_delta_rms",
        "raw_action_second_delta_max_abs",
        "applied_action_delta_rms",
        "applied_action_delta_max_abs",
        "applied_action_second_delta_rms",
        "applied_action_second_delta_max_abs",
    ]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in result["scenarios"].items():
            target = metrics["target_velocity"]
            actual = metrics["actual_velocity_mean"]
            writer.writerow(
                {
                    "scenario": name,
                    "target_vx": target[0],
                    "target_vy": target[1],
                    "target_yaw": target[2],
                    "actual_vx": actual[0],
                    "actual_vy": actual[1],
                    "actual_yaw": actual[2],
                    "tracking_rmse": metrics["tracking_rmse"],
                    "leg_torque_mean_abs": metrics["leg_torque"]["mean_abs"],
                    "leg_torque_rms": metrics["leg_torque"]["rms"],
                    "hip_torque_rms": metrics["hip_torque"]["rms"],
                    "wheel_torque_rms": metrics["wheel_torque"]["rms"],
                    "leg_power_mean_abs": metrics["leg_power"]["mean_abs"],
                    "wheel_power_mean_abs": metrics["wheel_power"]["mean_abs"],
                    "contact_history_peak_n": metrics["contact"]["history_peak_n"],
                    "contact_duty": metrics["contact"]["duty"],
                    "roll_rms_rad": metrics["base_attitude"]["roll_rms_rad"],
                    "pitch_rms_rad": metrics["base_attitude"]["pitch_rms_rad"],
                    "base_height_std_m": metrics["base_height"]["std_m"],
                    "wheel_velocity_rms": metrics["wheel_velocity"]["rms"],
                    "raw_action_delta_rms": metrics["raw_action_smoothness"]["first_difference"]["rms"],
                    "raw_action_delta_max_abs": metrics["raw_action_smoothness"]["first_difference"]["max_abs"],
                    "raw_action_second_delta_rms": metrics["raw_action_smoothness"]["second_difference"]["rms"],
                    "raw_action_second_delta_max_abs": metrics["raw_action_smoothness"]["second_difference"]["max_abs"],
                    "applied_action_delta_rms": metrics["applied_action_smoothness"]["first_difference"]["rms"],
                    "applied_action_delta_max_abs": metrics["applied_action_smoothness"]["first_difference"]["max_abs"],
                    "applied_action_second_delta_rms": metrics["applied_action_smoothness"]["second_difference"]["rms"],
                    "applied_action_second_delta_max_abs": metrics["applied_action_smoothness"]["second_difference"]["max_abs"],
                }
            )


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.warmup_steps < 0 or args.sample_steps <= 0 or args.num_envs_per_scenario <= 0:
        raise ValueError("warmup, sample, and environment counts must be positive")
    if not 0 < args.recovery_hold_steps <= args.sample_steps:
        raise ValueError("recovery hold steps must be within the sampled interval")
    if not 0.0 < args.recovery_roll_deg < args.failure_roll_deg:
        raise ValueError("recovery roll must be positive and below failure roll")
    if not args.failure_roll_deg < args.fall_tilt_deg < 90.0:
        raise ValueError("fall tilt must be above failure roll and below 90 degrees")
    if not 0.0 < args.recovery_hip_magnitude <= 0.40:
        raise ValueError("recovery hip magnitude must be within (0.0, 0.40]")
    if args.recovery_time_limit_s <= 0.0 or args.fall_height_m <= 0.0:
        raise ValueError("recovery time and fall height must be positive")
    if args.command_transition_steps <= 0:
        raise ValueError("command transition steps must be positive")

    env_cfg, agent_cfg, payload_position, initial_hip_scale = _configure_environment()
    latency_cfg = cli_args.latency_kwargs(args, seed=args.seed)
    gym_env = gym.make(TASK_ID, cfg=env_cfg)
    if any(value for key, value in latency_cfg.items() if key != "seed"):
        gym_env = ThunderLatencyWrapper(gym_env, **latency_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy

        device = env.unwrapped.device
        scenario_count = len(ACTIVE_SCENARIOS)
        envs_per_scenario = args.num_envs_per_scenario
        targets = torch.tensor(list(ACTIVE_SCENARIOS.values()), device=device).repeat_interleave(
            envs_per_scenario, dim=0
        )
        command = env.unwrapped.command_manager.get_term("base_velocity")
        robot = env.unwrapped.scene["robot"]
        sensor = env.unwrapped.scene["contact_forces"]
        leg_ids, leg_names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
        hip_ids, hip_names = robot.find_joints(".*_hip_joint", preserve_order=True)
        foot_ids, foot_names = sensor.find_bodies(".*_foot_Link", preserve_order=True)
        initial_joint_pos = robot.data.default_joint_pos.clone()
        initial_joint_vel = torch.zeros_like(robot.data.joint_vel)
        initial_joint_pos[:, hip_ids] *= initial_hip_scale
        robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
        selected_joint_ids = leg_ids + wheel_ids
        selected_index = {joint_id: index for index, joint_id in enumerate(selected_joint_ids)}
        hip_positions = [selected_index[joint_id] for joint_id in hip_ids]

        shape = (args.sample_steps, scenario_count, envs_per_scenario)
        torque_samples = torch.empty((*shape, len(selected_joint_ids)), device=device)
        power_samples = torch.empty_like(torque_samples)
        joint_velocity_samples = torch.empty_like(torque_samples)
        velocity_samples = torch.empty((*shape, 3), device=device)
        attitude_samples = torch.empty((*shape, 2), device=device)
        hip_angle_samples = torch.empty((*shape, len(hip_ids)), device=device)
        height_samples = torch.empty(shape, device=device)
        history_force_samples = torch.empty((*shape, len(foot_ids)), device=device)
        contact_samples = torch.empty((*shape, len(foot_ids)), dtype=torch.bool, device=device)
        done_samples = torch.empty(shape, dtype=torch.bool, device=device)
        target_samples = torch.empty((*shape, 3), device=device)
        action_prefix_steps = min(args.warmup_steps, 2)
        action_shape = (args.sample_steps + action_prefix_steps, scenario_count, envs_per_scenario)
        raw_action_samples = torch.empty((*action_shape, ACTION_DIM), device=device)
        applied_action_samples = torch.empty_like(raw_action_samples)

        obs = env.get_observations()
        total_steps = args.warmup_steps + args.sample_steps
        sample_index = 0
        action_sample_index = 0
        with torch.inference_mode():
            for step in range(total_steps):
                command_targets = targets * _command_scale(
                    args.command_profile, step, args.warmup_steps, args.command_transition_steps
                )
                command.vel_command_b[:] = command_targets
                obs["policy"][:, 6:9] = command_targets
                obs["critic"][:, 9:12] = command_targets
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
                if step >= args.warmup_steps - action_prefix_steps:
                    raw_action_samples[action_sample_index] = actions.reshape(
                        scenario_count, envs_per_scenario, -1
                    )
                    applied_action_samples[action_sample_index] = (
                        env.unwrapped.action_manager.action.reshape(
                            scenario_count, envs_per_scenario, -1
                        )
                    )
                    action_sample_index += 1
                if step < args.warmup_steps:
                    continue

                torque = robot.data.applied_torque[:, selected_joint_ids]
                joint_velocity = robot.data.joint_vel[:, selected_joint_ids]
                velocity = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=1)
                roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                attitude = torch.stack(
                    (torch.atan2(torch.sin(roll), torch.cos(roll)), torch.atan2(torch.sin(pitch), torch.cos(pitch))),
                    dim=1,
                )
                current_force = torch.linalg.vector_norm(sensor.data.net_forces_w[:, foot_ids], dim=-1)
                history = sensor.data.net_forces_w_history
                if history is None:
                    history_force = current_force
                else:
                    history_force = torch.linalg.vector_norm(history[:, :, foot_ids], dim=-1).amax(dim=1)

                torque_samples[sample_index] = torque.reshape(scenario_count, envs_per_scenario, -1)
                power_samples[sample_index] = (torque * joint_velocity).abs().reshape(
                    scenario_count, envs_per_scenario, -1
                )
                joint_velocity_samples[sample_index] = joint_velocity.reshape(
                    scenario_count, envs_per_scenario, -1
                )
                velocity_samples[sample_index] = velocity.reshape(scenario_count, envs_per_scenario, 3)
                attitude_samples[sample_index] = attitude.reshape(scenario_count, envs_per_scenario, 2)
                hip_angle_samples[sample_index] = robot.data.joint_pos[:, hip_ids].reshape(
                    scenario_count, envs_per_scenario, -1
                )
                height_samples[sample_index] = robot.data.root_pos_w[:, 2].reshape(
                    scenario_count, envs_per_scenario
                )
                history_force_samples[sample_index] = history_force.reshape(
                    scenario_count, envs_per_scenario, -1
                )
                contact_samples[sample_index] = (current_force > CONTACT_THRESHOLD_N).reshape(
                    scenario_count, envs_per_scenario, -1
                )
                done_samples[sample_index] = dones.reshape(scenario_count, envs_per_scenario)
                target_samples[sample_index] = command_targets.reshape(
                    scenario_count, envs_per_scenario, 3
                )
                sample_index += 1

        scenarios = {}
        for scenario_index, (scenario_name, target_tuple) in enumerate(ACTIVE_SCENARIOS.items()):
            torque = torque_samples[:, scenario_index]
            power = power_samples[:, scenario_index]
            joint_velocity = joint_velocity_samples[:, scenario_index]
            velocity = velocity_samples[:, scenario_index]
            attitude = attitude_samples[:, scenario_index]
            hip_angles = hip_angle_samples[:, scenario_index]
            height = height_samples[:, scenario_index]
            sampled_targets = target_samples[:, scenario_index]
            velocity_error = velocity - sampled_targets
            raw_actions = raw_action_samples[:, scenario_index]
            applied_actions = applied_action_samples[:, scenario_index]
            history_force = history_force_samples[:, scenario_index]
            finite = all(
                torch.isfinite(values).all().item()
                for values in (
                    torque,
                    power,
                    joint_velocity,
                    velocity,
                    attitude,
                    hip_angles,
                    height,
                    history_force,
                )
            )
            scenarios[scenario_name] = {
                "target_velocity": list(target_tuple),
                "sampled_target_velocity_mean": [
                    round(value, 9) for value in sampled_targets.mean(dim=(0, 1)).tolist()
                ],
                "actual_velocity_mean": [round(value, 9) for value in velocity.mean(dim=(0, 1)).tolist()],
                "tracking_axis_rmse": [
                    round(value, 9) for value in torch.sqrt(velocity_error.square().mean(dim=(0, 1))).tolist()
                ],
                "tracking_rmse": round(
                    torch.sqrt(velocity_error.square().sum(dim=-1).mean()).item(), 9
                ),
                "leg_torque": _stats(torque[:, :, : len(leg_ids)], LEG_TORQUE_LIMIT_NM),
                "hip_torque": _stats(torque[:, :, hip_positions]),
                "wheel_torque": _stats(torque[:, :, len(leg_ids) :], WHEEL_TORQUE_LIMIT_NM),
                "leg_power": _stats(power[:, :, : len(leg_ids)]),
                "wheel_power": _stats(power[:, :, len(leg_ids) :]),
                "wheel_velocity": _stats(joint_velocity[:, :, len(leg_ids) :]),
                "raw_action_smoothness": _difference_stats(raw_actions),
                "applied_action_smoothness": _difference_stats(applied_actions),
                "per_joint_torque": _per_joint_stats(torque, leg_names + wheel_names),
                "per_joint_power": _per_joint_stats(power, leg_names + wheel_names),
                "base_attitude": {
                    "roll_rms_rad": round(torch.sqrt(attitude[..., 0].square().mean()).item(), 9),
                    "roll_peak_abs_rad": round(attitude[..., 0].abs().max().item(), 9),
                    "pitch_rms_rad": round(torch.sqrt(attitude[..., 1].square().mean()).item(), 9),
                    "pitch_peak_abs_rad": round(attitude[..., 1].abs().max().item(), 9),
                },
                "base_height": {
                    "mean_m": round(height.mean().item(), 9),
                    "std_m": round(height.std(unbiased=False).item(), 9),
                    "min_m": round(height.min().item(), 9),
                    "max_m": round(height.max().item(), 9),
                },
                "lateral_recovery": _recovery_stats(
                    attitude[..., 0], hip_angles, height, env.unwrapped.step_dt
                ),
                "contact": {
                    "history_p99_n": round(torch.quantile(history_force.flatten(), 0.99).item(), 9),
                    "history_peak_n": round(history_force.max().item(), 9),
                    "duty": round(contact_samples[:, scenario_index].float().mean().item(), 9),
                },
                "unexpected_done_fraction": round(done_samples[:, scenario_index].float().mean().item(), 12),
                "finite": finite,
            }

        result = {
            "meta": {
                "checkpoint": str(checkpoint),
                "evaluation_seed": args.seed,
                "pose": args.pose,
                "initial_hip_scale": initial_hip_scale,
                "initial_roll_deg": args.initial_roll_deg,
                "static_friction": args.static_friction,
                "dynamic_friction": args.dynamic_friction,
                "recovery_roll_deg": args.recovery_roll_deg,
                "failure_roll_deg": args.failure_roll_deg,
                "recovery_hold_steps": args.recovery_hold_steps,
                "recovery_hip_magnitude": args.recovery_hip_magnitude,
                "recovery_time_limit_s": args.recovery_time_limit_s,
                "fall_tilt_deg": args.fall_tilt_deg,
                "fall_height_m": args.fall_height_m,
                "kp": args.kp,
                "kd": args.kd,
                "gain_randomization": args.gain_randomization,
                "envs_per_scenario": args.num_envs_per_scenario,
                "warmup_steps": args.warmup_steps,
                "sample_steps": args.sample_steps,
                "command_profile": args.command_profile,
                "command_transition_steps": args.command_transition_steps,
                "joint_order": leg_names + wheel_names,
                "hip_order": hip_names,
                "foot_order": foot_names,
                "payload_mass_kg": args.payload_mass,
                "payload_profile": args.payload_profile,
                "payload_position_m": payload_position,
                "latency": latency_cfg,
            },
            "scenarios": scenarios,
            "recovery_passed": all(
                metrics["lateral_recovery"]["passed"] for metrics in scenarios.values()
            ),
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(output, result)
        print(f"DOSO_TORQUE_TRACKING_RESULT={output}", flush=True)
        print(f"DOSO_LATERAL_RECOVERY_PASS={result['recovery_passed']}", flush=True)
        if args.require_recovery_pass and not result["recovery_passed"]:
            raise RuntimeError("lateral recovery gate failed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
