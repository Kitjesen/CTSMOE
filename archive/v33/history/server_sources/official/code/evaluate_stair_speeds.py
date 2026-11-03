"""Matched teacher/student speed probes; never update weights or the training run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def evaluate(args):
    import torch
    from tensordict import TensorDict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from stair_speed import configure_stair_speed
    from training_profiles import configure_base_height_penalty
    from train_straight_speed_20260918 import apply_command_curriculum_start

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    columns = checkpoint["terrain_column_names"]
    cases = []
    for learner in ("teacher", "student"):
        for terrain_name in ("stairs_up", "stairs_down"):
            available = [i for i, name in enumerate(columns) if name == terrain_name]
            for level in (3, 5, 7):
                for bin_idx, speed in enumerate((0.3, 0.5, 0.7, 0.9)):
                    for repeat in range(4):
                        cases.append(dict(learner=learner, terrain=terrain_name, level=level,
                                          speed=speed, bin=bin_idx, column=available[repeat % len(available)]))

    class FirstEpisodeEnv(Go2Env):
        retain_terminal = False

        def _reset_idx(self, env_ids):
            if not self.retain_terminal:
                super()._reset_idx(env_ids)

    cfg = make_config(len(cases), args.root / "cache/usd")
    cfg.seed = 42
    cfg.log_dir = str(args.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, "intermediate")
    configure_base_height_penalty(cfg, "off")
    configure_stair_speed(cfg, "straight_speed_v1")
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    cfg.rewards.lin_vel_z_l2.weight = 0
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    for name in vars(cfg.events):
        if name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.func = reset_root_state_uniform
    cfg.events.reset_base.params["pose_range"] = {key: (0., 0.) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_base.params["velocity_range"] = dict(cfg.events.reset_base.params["pose_range"])
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    raw = FirstEpisodeEnv(cfg=cfg)
    try:
        raw.common_step_counter = checkpoint["common_step_counter"]
        terrain = raw.scene.terrain
        terrain.terrain_levels[:] = torch.tensor([case["level"] for case in cases], device=raw.device)
        terrain.terrain_types[:] = torch.tensor([case["column"] for case in cases], device=raw.device)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        raw.stair_axis = torch.zeros(len(cases), device=raw.device, dtype=torch.long)
        command = raw.command_manager.get_term("base_velocity")
        command._init_terrain_infos()
        command._update_env_command_ranges()
        original_resample = command._resample
        speeds = torch.tensor([case["speed"] for case in cases], device=raw.device)
        bins = torch.tensor([case["bin"] for case in cases], device=raw.device)

        def fixed_resample(ids):
            original_resample(ids)
            command.held_speed[ids] = speeds[ids]
            command.speed_bins[ids] = bins[ids]
            command.commands[ids, 0] = speeds[ids]

        command._resample = fixed_resample
        raw.reset()
        env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
        agent = MoECTSRunnerCfg().to_dict()
        agent["robogauge"] = {"enabled": False}
        runner = OnPolicyRunnerCTS(env, agent, log_dir=str(args.output / "loader"), device=raw.device)
        policy = runner.alg.policy
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        half = len(cases) // 2
        active = torch.ones(len(cases), dtype=torch.bool, device=raw.device)
        failed = torch.zeros_like(active)
        passed = torch.zeros_like(active)
        errors = torch.zeros(len(cases), device=raw.device)
        steps = torch.zeros(len(cases), device=raw.device)
        max_cross = torch.zeros_like(errors)
        max_forward = torch.zeros_like(errors)
        raw.retain_terminal = True
        obs = TensorDict(raw.obs_buf, batch_size=[len(cases)])
        with torch.inference_mode():
            for step in range(raw.max_episode_length):
                torch.testing.assert_close(obs["single_obs"][:, 6:9], command.command, rtol=0, atol=0)
                teacher_obs = obs[:half]
                latent = policy.teacher_encoder(policy.critic_obs_normalizer(policy.get_critic_obs(teacher_obs)))
                teacher_action = policy.actor(torch.cat((latent, policy.single_obs_normalizer(teacher_obs["single_obs"])), dim=-1))
                if policy.state_dependent_std:
                    teacher_action = teacher_action[..., 0, :]
                action = torch.cat((teacher_action, policy.act_inference(obs[half:])))
                assert torch.isfinite(action).all()
                if agent["clip_actions"] is not None:
                    action.clamp_(-agent["clip_actions"], agent["clip_actions"])
                action[~active] = 0
                next_obs, _, terminated, truncated, _ = raw.step(action)
                offset = raw.scene["robot"].data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]
                errors[active] += (raw.scene["robot"].data.root_lin_vel_b[:, 0] - speeds).abs()[active]
                steps[active] += 1
                max_cross[active] = torch.maximum(max_cross[active], offset[active, 1].abs())
                max_forward[active] = torch.maximum(max_forward[active], offset[active, 0])
                ended = active & (terminated | truncated)
                failed[ended] = terminated[ended]
                passed[ended] = (offset[ended, 0] > 4.) & (max_cross[ended] < 0.75) & ~terminated[ended]
                active &= ~ended
                obs = TensorDict(next_obs, batch_size=[len(cases)])
                if step % 250 == 0:
                    (args.output / "status.json").write_text(json.dumps(dict(state="evaluating", step=step, active=int(active.sum()))))
                if not active.any():
                    break
        metrics = torch.stack((passed, failed, errors / steps.clamp_min(1), steps * raw.step_dt,
                               max_forward, max_cross), dim=1).cpu().tolist()
        for case, values in zip(cases, metrics):
            case.update(zip(("reference_exit_pass", "physical_failure", "vx_mae_mps", "duration_s",
                             "max_forward_m", "max_cross_m"), values))
        result = {"checkpoint": str(args.checkpoint), "completed_updates": checkpoint["completed_updates"],
                  "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "cases": cases, "geometry_seed": 42,
                  "conditions": "nominal pose/joints/dynamics, observations without corruption; actuator delays retained",
                  "criterion": "forward x > 4m, cross track < 0.75m, no physical termination; reference exit, not all-wheel flight completion",
                  "scope": "192 diagnostic trials, 4 per learner/direction/level/speed; not a calibrated promotion gate"}
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
        (args.output / "status.json").write_text(json.dumps(dict(state="complete", trials=len(cases))))
        print("STAIR_SPEED_BASELINE_COMPLETE", len(cases), flush=True)
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "code"), str(args.root / "upstream/source/rsl_rl"),
                   str(args.root / "upstream/source/robot_lab"), str(args.root / "thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        evaluate(args)
    except BaseException as error:
        traceback.print_exc()
        (args.output / "status.json").write_text(json.dumps(dict(state="failed", error=str(error))))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
