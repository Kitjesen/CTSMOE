"""Compare stair command paths and physical execution without updating a policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def run(args):
    import numpy as np
    import torch
    from isaaclab.utils.io import dump_yaml
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from config import make_config

    torch.set_num_threads(2)
    updates = int(args.policy.stem.removeprefix("student_"))

    class DirectCommand(Go2RLGymCommand):
        def _resample(self, env_ids):
            self.time_left[env_ids] = 1e9
            self.commands[env_ids] = 0

        def _update_command(self):
            super()._update_command()
            self.commands[:] = 0
            self.commands[self._env.episode_length_buf * self._env.step_dt >= 2, 0] = 0.35

    class FirstEpisodeEnv(Go2Env):
        retain_terminal = False

        def _reset_idx(self, env_ids):
            if not self.retain_terminal:
                super()._reset_idx(env_ids)

    cfg = make_config(args.num_envs, args.root / "cache/usd")
    cfg.seed = args.seed
    cfg.log_dir = str(args.output)
    cfg.sim.render_interval = 10000
    cfg.commands.base_velocity.debug_vis = False
    if args.commands == "direct":
        cfg.commands.base_velocity.class_type = DirectCommand
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    # Match the mature training reward coefficients, although rewards do not
    # enter the exported deterministic student policy.
    cfg.rewards.lin_vel_z_l2.weight = 0.0
    cfg.rewards.base_height_l2.weight = -10.0
    if args.randomization == "nominal":
        cfg.observations.policy.enable_corruption = False
        cfg.observations.single_obs.enable_corruption = False
        for name in vars(cfg.events):
            if name not in ("reset_base", "reset_robot_joints"):
                setattr(cfg.events, name, None)
        cfg.events.reset_base.params["pose_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        cfg.events.reset_base.params["velocity_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    generator = cfg.scene.terrain.terrain_generator
    names = ["flat", "slope_up", "stairs_up", "stairs_down"]
    generator.sub_terrains = {name: generator.sub_terrains[name] for name in names}
    for sub in generator.sub_terrains.values():
        sub.proportion = 0.25
    generator.num_cols, generator.num_rows, generator.seed = 4, 10, 42
    cfg.scene.terrain.max_init_terrain_level = 0
    env = FirstEpisodeEnv(cfg=cfg)
    try:
        env.common_step_counter = updates * 24
        terrain = env.scene.terrain
        terrain.terrain_levels.fill_(args.level)
        terrain.terrain_types.fill_(2)
        terrain.env_origins[:] = terrain.terrain_origins[args.level, 2]
        env.scene.env_origins[:] = terrain.env_origins
        command = env.command_manager.get_term("base_velocity")
        command._init_terrain_infos()
        command._update_env_command_ranges()
        robot = env.scene["robot"]
        contact = env.scene["contact_forces"]
        policy = torch.jit.load(str(args.policy), map_location=env.device).eval()
        clip_actions = MoECTSRunnerCfg().clip_actions
        dump_yaml(str(args.output / "env.yaml"), cfg)
        with torch.inference_mode():
            obs, _ = env.reset()
            env.retain_terminal = True
            assert obs["policy"].shape == (args.num_envs, 530)
            assert obs["single_obs"].shape == (args.num_envs, 53)
            origin = terrain.env_origins[:, :2].cpu().numpy().copy()
            initial_position = robot.data.root_pos_w.cpu().numpy().copy()
            initial_quaternion = robot.data.root_quat_w.cpu().numpy().copy()
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            end_step = torch.full((args.num_envs,), 1250, dtype=torch.long, device=env.device)
            terminated = torch.zeros_like(active)
            timeout = torch.zeros_like(active)
            histories = []
            for step in range(1250):
                used_command = command.command.clone()
                before_quat = robot.data.root_quat_w.clone()
                if step in (0, 1249):
                    torch.testing.assert_close(obs["single_obs"][:, 6:9], used_command, atol=0, rtol=0)
                action = policy(obs["policy"], obs["single_obs"])
                assert torch.isfinite(action).all()
                if clip_actions is not None:
                    action = action.clamp(-clip_actions, clip_actions)
                action[~active] = 0
                obs, _, term, trunc, _ = env.step(action)
                # Save all command schedules. State metrics are cut at each
                # environment's FIRST terminal frame using end_step below.
                histories.append(torch.cat((
                    used_command, robot.data.root_pos_w, before_quat,
                    robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3],
                    -robot.data.projected_gravity_b[:, 2:3],
                    robot.data.computed_torque, robot.data.applied_torque,
                    torch.linalg.vector_norm(contact.data.net_forces_w, dim=-1),
                    command.commands_xy_accumulation, command.max_move_distance[:, None],
                ), dim=1).cpu().numpy())
                ended = active & (term | trunc)
                end_step[ended] = step + 1
                terminated[ended] = term[ended]
                timeout[ended] = trunc[ended]
                active &= ~ended
                if step % 100 == 0:
                    (args.output / "status.json").write_text(json.dumps(dict(state="running", step=step, active=int(active.sum()))))
            end_step = end_step.cpu().numpy()
            terminated = terminated.cpu().numpy()
            timeout = timeout.cpu().numpy()
        history = np.stack(histories)
        assert np.isfinite(history).all()
        np.savez_compressed(args.output / "traces.npz", history=history, origin_xy=origin,
                            initial_position=initial_position, initial_quaternion=initial_quaternion,
                            end_step=end_step, terminated=terminated, timeout=timeout)
        # Exact field offsets: cmd[0:3], position[3:6], pre-step quat[6:10],
        # body velocity vx/vy/wz[10:13], upright[13], tau requested[14:30],
        # tau applied[30:46], contact norms[46:46+B], accum[46+B:48+B], maxdist[48+B].
        bodies = contact.body_names
        meta = dict(updates=updates, policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    num_envs=args.num_envs, level=args.level, commands=args.commands,
                    randomization=args.randomization, seed=args.seed, dt=env.step_dt,
                    body_names=bodies, joint_names=robot.joint_names,
                    max_radius_threshold_m=command.terrain_length / 2,
                    command_zero_probability=float(command.zero_command_prob),
                    full_command_schedule_retained_after_first_termination=True,
                    student_actions="deterministic mean; no policy updates",
                    caveat="Analysis must cut each robot state at end_step. Full randomization includes reset pose, dynamics, observation noise and interval disturbances. Torque/contact fields sample the last physics substep at 50 Hz.")
        (args.output / "manifest.json").write_text(json.dumps(meta, indent=2))
        (args.output / "status.json").write_text(json.dumps(dict(state="complete", num_envs=args.num_envs)))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commands", choices=("direct", "original"), required=True)
    parser.add_argument("--randomization", choices=("nominal", "full"), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "upstream/source/rsl_rl"), str(args.root / "upstream/source/robot_lab"),
                   str(args.root / "thunder/source/doso_train"), str(args.root / "code")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        run(args)
    except BaseException as error:
        (args.output / "status.json").write_text(json.dumps(dict(state="failed", error=str(error))))
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
