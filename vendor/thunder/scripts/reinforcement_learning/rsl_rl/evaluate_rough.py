from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Deterministic DOSO Thunder rough-terrain evaluation"
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--terrain", default="random_rough")
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--suite", choices=("motions", "stairs"), default="motions")
parser.add_argument("--envs_per_scenario", type=int, default=16)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--policy_device", default="cpu")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

if args.envs_per_scenario < 1:
    parser.error("--envs_per_scenario must be positive")
if args.steps <= args.warmup_steps:
    parser.error("--steps must be greater than --warmup_steps")

simulation_app = AppLauncher(headless=True, device=args.device).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import doso_train.tasks  # noqa: F401, E402
from doso_train.tasks.terrain_inward.agents.rsl_rl_ppo_cfg import (
    DosoThunderTerrainInwardRoughPPORunnerCfg,
)
from doso_train.tasks.terrain_inward.rough_v0_env_cfg import (
    DosoThunderTerrainInwardRoughV0EnvCfg,
)


TASK_ID = "DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0"
NON_FOOT_CONTACT_THRESHOLD_N = 1.0
STUMBLE_FORCE_RATIO = 4.0
CONTACT_FORCE_HISTORY_LENGTH = 3
MOTION_SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.5, 0.0, 0.0),
    "lateral_left": (0.0, 0.5, 0.0),
    "lateral_right": (0.0, -0.5, 0.0),
    "yaw_left": (0.0, 0.0, 0.5),
    "yaw_right": (0.0, 0.0, -0.5),
    "mixed": (0.5, 0.3, 0.3),
}
STAIR_SCENARIOS = {"forward": (0.5, 0.0, 0.0)}


def zero_range(keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    return {key: (0.0, 0.0) for key in keys}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_stats(values: torch.Tensor) -> dict[str, int | float | None]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "p95": None, "p99": None, "peak": None}
    return {
        "count": values.numel(),
        "p95": torch.quantile(values, 0.95).item(),
        "p99": torch.quantile(values, 0.99).item(),
        "peak": values.max().item(),
    }


def configure_env(scenarios: dict[str, tuple[float, float, float]]):
    env_cfg = DosoThunderTerrainInwardRoughV0EnvCfg()
    agent_cfg = DosoThunderTerrainInwardRoughPPORunnerCfg()
    num_envs = len(scenarios) * args.envs_per_scenario

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.policy_device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.height_scanner = None
    env_cfg.observations.critic.height_scan = None

    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None or args.terrain not in generator.sub_terrains:
        available = [] if generator is None else list(generator.sub_terrains)
        raise ValueError(f"Unknown terrain {args.terrain!r}; available={available}")
    terrain = generator.sub_terrains[args.terrain]
    terrain.proportion = 1.0
    generator.sub_terrains = {args.terrain: terrain}
    generator.num_rows = 1
    generator.num_cols = num_envs
    generator.curriculum = False
    generator.difficulty_range = (args.difficulty, args.difficulty)
    generator.seed = agent_cfg.seed
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum.terrain_levels = None

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.debug_vis = False
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_pure_yaw_envs = 0.0
    command_cfg.rel_pure_lateral_envs = 0.0
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)

    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
        "randomize_foot_material_with_privileged",
    ):
        setattr(env_cfg.events, name, None)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": zero_range(("x", "y", "z", "roll", "pitch", "yaw")),
        "velocity_range": zero_range(("x", "y", "z", "roll", "pitch", "yaw")),
    }
    env_cfg.terminations.time_out = None
    env_cfg.terminations.terrain_out_of_bounds = None
    return env_cfg, agent_cfg, terrain


def main() -> None:
    started_at = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def progress(message: str) -> None:
        print(
            f"EVAL_PROGRESS elapsed={time.perf_counter() - started_at:.1f}s {message}",
            flush=True,
        )

    scenarios = STAIR_SCENARIOS if args.suite == "stairs" else MOTION_SCENARIOS
    env_cfg, agent_cfg, terrain_cfg = configure_env(scenarios)
    progress("config_ready")
    gym_env = gym.make(TASK_ID, cfg=env_cfg)
    progress("gym_ready")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    progress("wrapper_ready")
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    progress("runner_ready")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint is not a regular file: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    progress("checkpoint_file_loaded")
    progress(f"policy_device={next(runner.alg.policy.parameters()).device}")
    policy_parameters = dict(runner.alg.policy.named_parameters())
    actor_names = {
        name for name in policy_parameters if name == "std" or name.startswith("actor.")
    }
    checkpoint_actor_names = {
        name
        for name in checkpoint["model_state_dict"]
        if name == "std" or name.startswith("actor.")
    }
    if actor_names != checkpoint_actor_names:
        raise RuntimeError(
            f"Actor checkpoint mismatch: missing={actor_names - checkpoint_actor_names}, "
            f"unexpected={checkpoint_actor_names - actor_names}"
        )
    with torch.no_grad():
        for name in sorted(actor_names):
            source = checkpoint["model_state_dict"][name]
            target = policy_parameters[name]
            if target.shape != source.shape:
                raise RuntimeError(
                    f"Actor shape mismatch for {name}: {target.shape} != {source.shape}"
                )
            target.copy_(source)
    progress("checkpoint_loaded")
    policy = runner.get_inference_policy(device=agent_cfg.device)
    policy_nn = runner.alg.policy

    device = env.unwrapped.device
    targets = torch.tensor(list(scenarios.values()), device=device).repeat_interleave(
        args.envs_per_scenario, dim=0
    )
    command = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    foot_sensor_ids, foot_sensor_names = contact_sensor.find_bodies(
        ".*_foot_Link", preserve_order=True
    )
    non_foot_sensor_ids, _ = contact_sensor.find_bodies(
        "^(?!.*_foot_Link).*", preserve_order=True
    )
    foot_body_ids, foot_body_names = robot.find_bodies(
        ".*_foot_Link", preserve_order=True
    )
    if foot_sensor_names != foot_body_names:
        raise RuntimeError(
            "Foot body order differs between robot and contact sensor: "
            f"{foot_body_names} != {foot_sensor_names}"
        )
    if not foot_sensor_ids or not non_foot_sensor_ids:
        raise RuntimeError(
            "Contact sensor must expose both foot and non-foot robot bodies"
        )
    if contact_sensor.cfg.history_length != CONTACT_FORCE_HISTORY_LENGTH:
        raise RuntimeError(
            "Fixed rough evaluation requires contact force history_length="
            f"{CONTACT_FORCE_HISTORY_LENGTH}, got {contact_sensor.cfg.history_length}"
        )
    obs = env.get_observations()
    progress("observations_ready")
    initial_xy = robot.data.root_pos_w[:, :2].clone()
    previous_actions = None
    previous_foot_vel_z = robot.data.body_lin_vel_w[:, foot_body_ids, 2].clone()
    samples = 0
    num_envs = targets.shape[0]
    sample_steps = args.steps - args.warmup_steps
    foot_force_samples = torch.empty(
        sample_steps, num_envs, len(foot_sensor_ids), device=device
    )
    touchdown_speed_samples = torch.full_like(foot_force_samples, torch.nan)

    sums = {
        key: torch.zeros(num_envs, device=device)
        for key in (
            "vx",
            "vy",
            "wz",
            "xy_error_sq",
            "yaw_error_sq",
            "tilt_sq",
            "reward",
            "action_delta",
            "non_foot_contact",
            "stumble",
        )
    }
    min_upright = torch.ones(num_envs, device=device)
    max_tilt = torch.zeros(num_envs, device=device)
    max_progress = torch.zeros(num_envs, device=device)
    max_drift = torch.zeros(num_envs, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for step in range(args.steps):
        command.vel_command_b[:] = targets
        obs["policy"][:, 6:9] = targets
        obs["critic"][:, 9:12] = targets
        with torch.inference_mode():
            policy_obs = (
                obs
                if agent_cfg.device == str(device)
                else {name: value.to(agent_cfg.device) for name, value in obs.items()}
            )
            actions = policy(policy_obs).to(device)
            obs, rewards, dones, _ = env.step(actions)
            policy_nn.reset(dones.to(agent_cfg.device))

        current_foot_vel_z = robot.data.body_lin_vel_w[:, foot_body_ids, 2]
        current_foot_forces = contact_sensor.data.net_forces_w[:, foot_sensor_ids]
        first_contact = contact_sensor.compute_first_contact(env.unwrapped.step_dt)[
            :, foot_sensor_ids
        ]

        if step >= args.warmup_steps:
            lin_vel = robot.data.root_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_ang_vel_b[:, 2]
            upright = -robot.data.projected_gravity_b[:, 2]
            tilt = torch.acos(torch.clamp(upright, -1.0, 1.0))
            delta_xy = robot.data.root_pos_w[:, :2] - initial_xy
            command_norm = torch.linalg.vector_norm(targets[:, :2], dim=1)
            direction = targets[:, :2] / command_norm.clamp_min(1.0e-6).unsqueeze(1)
            forward_progress = torch.sum(delta_xy * direction, dim=1)
            force_history = contact_sensor.data.net_forces_w_history
            if force_history is None:
                raise RuntimeError("Contact force history is unavailable")
            foot_force = torch.linalg.vector_norm(
                force_history[:, :, foot_sensor_ids], dim=-1
            ).amax(dim=1)
            non_foot_contact = torch.any(
                torch.linalg.vector_norm(
                    force_history[:, :, non_foot_sensor_ids], dim=-1
                ).amax(dim=1)
                > NON_FOOT_CONTACT_THRESHOLD_N,
                dim=1,
            )
            stumble = torch.any(
                torch.linalg.vector_norm(current_foot_forces[..., :2], dim=-1)
                > STUMBLE_FORCE_RATIO * torch.abs(current_foot_forces[..., 2]),
                dim=1,
            )
            touchdown_speed = torch.where(
                first_contact,
                torch.clamp(-previous_foot_vel_z, min=0.0),
                torch.nan,
            )

            sums["vx"] += lin_vel[:, 0]
            sums["vy"] += lin_vel[:, 1]
            sums["wz"] += yaw_vel
            sums["xy_error_sq"] += torch.sum(
                torch.square(targets[:, :2] - lin_vel), dim=1
            )
            sums["yaw_error_sq"] += torch.square(targets[:, 2] - yaw_vel)
            sums["tilt_sq"] += torch.square(tilt)
            sums["reward"] += rewards
            sums["non_foot_contact"] += non_foot_contact.float()
            sums["stumble"] += stumble.float()
            foot_force_samples[samples].copy_(foot_force)
            touchdown_speed_samples[samples].copy_(touchdown_speed)
            if previous_actions is not None:
                sums["action_delta"] += torch.mean(
                    torch.abs(actions - previous_actions), dim=1
                )
            min_upright = torch.minimum(min_upright, upright)
            max_tilt = torch.maximum(max_tilt, tilt)
            max_progress = torch.maximum(max_progress, forward_progress)
            max_drift = torch.maximum(
                max_drift, torch.linalg.vector_norm(delta_xy, dim=1)
            )
            fell |= upright < 0.5
            samples += 1
        previous_actions = actions.clone()
        previous_foot_vel_z.copy_(current_foot_vel_z)
        if step == 0 or (step + 1) % 50 == 0:
            progress(f"step={step + 1}/{args.steps}")

    result: dict[str, object] = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_sha256,
            "terrain": args.terrain,
            "difficulty": args.difficulty,
            "step_height_m": (
                terrain_cfg.step_height_range[0]
                + args.difficulty
                * (terrain_cfg.step_height_range[1] - terrain_cfg.step_height_range[0])
                if hasattr(terrain_cfg, "step_height_range")
                else None
            ),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "envs_per_scenario": args.envs_per_scenario,
            "step_dt": env.unwrapped.step_dt,
            "policy_device": agent_cfg.device,
            "non_foot_contact_threshold_n": NON_FOOT_CONTACT_THRESHOLD_N,
            "stumble_horizontal_to_vertical_force_ratio": STUMBLE_FORCE_RATIO,
            "contact_rate_denominator": "environment_control_steps",
            "foot_force_sampling": "history_window_peak_normal_resultant",
            "foot_force_history_length": CONTACT_FORCE_HISTORY_LENGTH,
            "foot_order": foot_sensor_names,
        },
        "scenarios": {},
    }
    for index, (name, target_tuple) in enumerate(scenarios.items()):
        subset = slice(
            index * args.envs_per_scenario, (index + 1) * args.envs_per_scenario
        )
        force_stats = distribution_stats(foot_force_samples[:, subset].flatten())
        touchdown_stats = distribution_stats(
            touchdown_speed_samples[:, subset].flatten()
        )
        scenario = {
            "target": target_tuple,
            "mean_vx": sums["vx"][subset].mean().item() / samples,
            "mean_vy": sums["vy"][subset].mean().item() / samples,
            "mean_wz": sums["wz"][subset].mean().item() / samples,
            "xy_rmse": math.sqrt(sums["xy_error_sq"][subset].mean().item() / samples),
            "yaw_rmse": math.sqrt(sums["yaw_error_sq"][subset].mean().item() / samples),
            "tilt_rms_rad": math.sqrt(sums["tilt_sq"][subset].mean().item() / samples),
            "tilt_peak_rad": max_tilt[subset].max().item(),
            "min_upright": min_upright[subset].min().item(),
            "fall_rate": fell[subset].float().mean().item(),
            "mean_reward": sums["reward"][subset].mean().item() / samples,
            "mean_action_delta": sums["action_delta"][subset].mean().item() / samples,
            "mean_max_progress_m": max_progress[subset].mean().item(),
            "progress_p10_m": torch.quantile(max_progress[subset], 0.1).item(),
            "mean_max_drift_m": max_drift[subset].mean().item(),
            "contacts": {
                "non_foot_contact_rate": sums["non_foot_contact"][subset].mean().item()
                / samples,
                "foot_force_sample_count": force_stats["count"],
                "foot_force_p95_n": force_stats["p95"],
                "foot_force_p99_n": force_stats["p99"],
                "foot_force_peak_n": force_stats["peak"],
                "touchdown_count": touchdown_stats["count"],
                "touchdown_downward_speed_p95_mps": touchdown_stats["p95"],
                "touchdown_downward_speed_p99_mps": touchdown_stats["p99"],
                "touchdown_downward_speed_peak_mps": touchdown_stats["peak"],
                "stumble_rate": sums["stumble"][subset].mean().item() / samples,
            },
        }
        if args.suite == "stairs":
            success = (
                (max_progress[subset] >= 2.6)
                & ~fell[subset]
                & (min_upright[subset] >= 0.5)
            )
            scenario["stair_success_rate"] = success.float().mean().item()
            scenario["reached_2_9m_rate"] = (
                (max_progress[subset] >= 2.9).float().mean().item()
            )
        result["scenarios"][name] = scenario

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print("ROUGH_EVAL_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
