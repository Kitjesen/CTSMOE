# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def select_actor_warm_start_state(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Select compatible actor.* and std tensors from an RSL-RL policy checkpoint."""
    selected = {key: value for key, value in source.items() if key.startswith("actor.") or key == "std"}
    invalid = [key for key, value in selected.items() if key not in target or target[key].shape != value.shape]
    if invalid:
        raise ValueError(f"Actor warm-start keys missing or shape-mismatched in target: {invalid}")
    if not selected or not any(key.startswith("actor.") for key in selected) or "std" not in selected:
        raise ValueError("Actor warm-start requires actor.* keys and std")
    return selected


def add_latency_args(parser: argparse.ArgumentParser) -> None:
    """Add shape-preserving policy observation and action latency options."""
    group = parser.add_argument_group("policy latency")
    group.add_argument("--obs-delay-min-frames", type=int, default=0)
    group.add_argument("--obs-delay-max-frames", type=int, default=0)
    group.add_argument("--obs-drop-start-prob", type=float, default=0.0)
    group.add_argument("--obs-drop-max-frames", type=int, default=0)
    group.add_argument("--action-delay-min-frames", type=int, default=0)
    group.add_argument("--action-delay-max-frames", type=int, default=0)


def latency_kwargs(args: argparse.Namespace, seed: int) -> dict[str, int | float]:
    """Validate latency CLI values and return ``ThunderLatencyWrapper`` kwargs."""
    if not 0 <= args.obs_delay_min_frames <= args.obs_delay_max_frames:
        raise ValueError("observation delay must satisfy 0 <= min <= max")
    if not 0 <= args.action_delay_min_frames <= args.action_delay_max_frames:
        raise ValueError("action delay must satisfy 0 <= min <= max")
    if not 0.0 <= args.obs_drop_start_prob <= 1.0:
        raise ValueError("observation drop probability must be within [0, 1]")
    if args.obs_drop_max_frames < 0:
        raise ValueError("observation drop maximum must be non-negative")
    if args.obs_drop_start_prob > 0.0 and args.obs_drop_max_frames < 1:
        raise ValueError("positive observation drop probability requires at least one frame")
    return {
        "obs_delay_min_frames": args.obs_delay_min_frames,
        "obs_delay_max_frames": args.obs_delay_max_frames,
        "obs_drop_start_prob": args.obs_drop_start_prob,
        "obs_drop_max_frames": args.obs_drop_max_frames,
        "action_delay_min_frames": args.action_delay_min_frames,
        "action_delay_max_frames": args.action_delay_max_frames,
        "seed": seed,
    }


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    load_group = arg_group.add_mutually_exclusive_group()
    load_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    load_group.add_argument(
        "--actor-warm-start",
        type=str,
        default=None,
        help="Load only actor.* and std from a checkpoint; critic, optimizer, and iteration start fresh.",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    add_latency_args(parser)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
