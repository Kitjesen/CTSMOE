"""Record a fixed student checkpoint in real Isaac simulation, without learning."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from isaaclab.sim import PreviewSurfaceCfg
    from isaaclab.utils.io import dump_yaml
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand
    from robot_lab.tasks.go2.mdp.terrains import PerSubTerrainSlopeThresholdGenerator
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from config import make_config

    policy_prefix, separator, update_text = args.policy.stem.rpartition("_")
    assert policy_prefix == "student" and separator and update_text.isdigit(), "Expected student_<completed_updates>.pt"
    completed_updates = int(update_text)
    torch.set_num_threads(2)
    zero = (0.0, 0.0, 0.0)
    flat = [(2, zero), (8, (0.4, 0.0, 0.0)), (10, zero),
            (13, (0.0, 0.0, 0.5)), (16, (0.0, 0.0, -0.5)),
            (18, (0.0, 0.3, 0.0)), (20, (0.0, -0.3, 0.0)), (22, zero)]
    traverse = [(2, zero), (12, (0.35, 0.0, 0.0)), (16, zero)]
    cases = [("flat_commands", "flat", 0, flat), ("slope_up_L4", "slope_up", 4, traverse),
             ("stairs_up_L4", "stairs_up", 4, traverse), ("stairs_down_L4", "stairs_down", 4, traverse)]
    if args.stairs_probe:
        cases = [(f"stairs_up_L{level}", "stairs_up", level, [(2, zero), (25, (0.35, 0.0, 0.0))])
                 for level in (3, 4, 5, 6)]
    if args.smoke:
        cases = [("smoke", "flat", 0, [(2, zero)])]
    terrain_exports = {(terrain_name, level) for _, terrain_name, level, _ in cases}
    schedule = {"segments": cases[0][3]}
    generated = []

    class RecordedTerrainGenerator(PerSubTerrainSlopeThresholdGenerator):
        def _get_terrain_mesh(self, difficulty, cfg):
            generated.append(dict(name=cfg.recording_name, difficulty=float(difficulty)))
            return super()._get_terrain_mesh(difficulty, cfg)

        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
            if (sub_terrain_cfg.recording_name, row) in terrain_exports:
                saved = mesh.copy()
                saved.apply_translation([-self.cfg.size[0] * self.cfg.num_rows * 0.5,
                                         -self.cfg.size[1] * self.cfg.num_cols * 0.5, 0.0])
                saved.export(args.output / f"terrain_{sub_terrain_cfg.recording_name}_L{row}.obj")

    class ScriptedCommand(Go2RLGymCommand):
        def _resample(self, env_ids):
            self.time_left[env_ids] = 1.0e9
            self.commands[env_ids] = 0.0

        def _update_command(self):
            super()._update_command()
            now = float(self._env.episode_length_buf[0]) * self._env.step_dt
            values = zero
            for end, candidate in schedule["segments"]:
                if now < end - 1.0e-7:
                    values = candidate
                    break
            self.commands[:] = torch.tensor(values, device=self.device)

    class RecordingEnv(Go2Env):
        keep_terminal_frame = False

        def _reset_idx(self, env_ids):
            # Stop at first termination and retain its actual pose for the video.
            # Explicit resets between clips still use the complete original path.
            if not self.keep_terminal_frame:
                super()._reset_idx(env_ids)

    cfg = make_config(1, args.root / "cache/usd")
    cfg.seed = 42
    cfg.log_dir = str(args.output)
    cfg.viewer.resolution = (960, 640)
    cfg.sim.render_interval = 8  # 200 Hz physics; 25 Hz video.
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    cfg.commands.base_velocity.class_type = ScriptedCommand
    cfg.commands.base_velocity.debug_vis = False
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    for name in vars(cfg.events):
        if name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_robot_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    terrain_names = ["flat", "slope_up", "stairs_up", "stairs_down"]
    generator = cfg.scene.terrain.terrain_generator
    generator.sub_terrains = {name: generator.sub_terrains[name] for name in terrain_names}
    for name, sub in generator.sub_terrains.items():
        sub.proportion = 0.25
        sub.recording_name = name
    generator.class_type = RecordedTerrainGenerator
    generator.num_cols = 4
    generator.num_rows = 10
    generator.curriculum = True
    generator.seed = 42
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.visual_material = PreviewSurfaceCfg(diffuse_color=(0.43, 0.47, 0.52), roughness=0.9)
    raw = RecordingEnv(cfg=cfg, render_mode=None if args.trace_only else "rgb_array")
    try:
        policy = torch.jit.load(str(args.policy), map_location=raw.device).eval()
        assert raw.observation_manager.group_obs_dim["policy"] == (530,)
        assert raw.observation_manager.group_obs_dim["single_obs"] == (53,)
        assert raw.action_manager.total_action_dim == 16
        assert raw.step_dt == 0.02
        robot = raw.scene["robot"]
        command = raw.command_manager.get_term("base_velocity")
        clip_actions = MoECTSRunnerCfg().clip_actions
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
        manifest = dict(policy=str(args.policy), policy_sha256=digest(args.policy),
                        completed_updates=completed_updates, policy_kind="student-only deterministic mean action",
                        code_sha256=digest(__file__), fps=25, playback_speed=1,
                        num_envs=1, physics_dt=raw.physics_dt, control_dt=raw.step_dt,
                        observation_noise=False, push_and_parameter_randomization=False,
                        actuator_delay="original independent 0-4 physics steps, resampled at reset",
                        terrain_source="pinned official generator; four equal columns, ten original difficulty rows",
                        terrain_cells=generated, joint_names=robot.joint_names, clips=[],
                        body_names=robot.body_names, urdf_path=cfg.scene.robot.spawn.asset_path,
                        urdf_sha256=digest(cfg.scene.robot.spawn.asset_path), trace_only=args.trace_only,
                        limitations="One nominal rollout per case. Not a terrain success-rate or robustness test. "
                                    "All first terminations retained; no teacher assistance or policy updates.")
        if args.stairs_probe:
            manifest["probe"] = dict(command_vx=0.35, settle_s=2, maximum_duration_s=25,
                                     distance_threshold_m=command.terrain_length / 2,
                                     stop_at_distance_threshold=True,
                                     torque_sampling="last physics substep at each 50 Hz control step",
                                     contact_body_names=raw.scene["contact_forces"].body_names)
        dump_yaml(str(args.output / "eval_env.yaml"), cfg)
        for terrain_name in terrain_names:
            values = [cell["difficulty"] for cell in generated if cell["name"] == terrain_name]
            assert len(values) == 10 and values == sorted(values), "Unexpected terrain row metadata"
        terrain = raw.scene.terrain
        for name, terrain_name, level, segments in cases:
            col = terrain_names.index(terrain_name)
            schedule["segments"] = segments
            with torch.inference_mode():
                terrain.terrain_levels.fill_(level)
                terrain.terrain_types.fill_(col)
                terrain.env_origins[:] = terrain.terrain_origins[level, col]
                raw.scene.env_origins[:] = terrain.env_origins
                command._init_terrain_infos()
                command._update_env_command_ranges()
                raw.keep_terminal_frame = False
                obs, _ = raw.reset()
                raw.keep_terminal_frame = True
            records = []
            reached_distance = False
            # Both observations are read directly from the environment. Never
            # truncate 530 field-major history to reconstruct the current frame.
            video = args.output / f"{name}.mp4"
            duration = segments[-1][0]
            difficulty = [cell["difficulty"] for cell in generated if cell["name"] == terrain_name][level]
            terrain_details = dict(name=terrain_name, level=level, difficulty=difficulty)
            if "stairs" in terrain_name:
                height = (0.05 + 0.207 * difficulty)
                terrain_details.update(nominal_step_height_m=height, height_quantization_m=0.005, tread_m=0.31)
            if "slope" in terrain_name:
                slope = 0.1 + 0.468 * difficulty
                terrain_details.update(slope=slope, angle_deg=float(np.degrees(np.arctan(slope))))
            video_context = nullcontext(None) if args.trace_only else imageio.get_writer(
                str(video), fps=25, codec="libx264", quality=7, macro_block_size=16,
                output_params=["-pix_fmt", "yuv420p"])
            with video_context as writer:
                with torch.inference_mode():
                    for step in range(round(duration / raw.step_dt)):
                        used_command = command.command[0].clone()
                        torch.testing.assert_close(obs["single_obs"][0, 6:9], used_command, atol=0, rtol=0)
                        torch.testing.assert_close(obs["policy"][0, 87:90], used_command, atol=0, rtol=0)
                        action = policy(obs["policy"], obs["single_obs"])
                        assert torch.isfinite(action).all()
                        if clip_actions is not None:
                            action = action.clamp(-clip_actions, clip_actions)
                        obs, reward, terminated, truncated, _ = raw.step(action)
                        done = bool((terminated | truncated).any())
                        row = dict(time=(step + 1) * raw.step_dt, command=used_command.cpu().tolist(),
                                   velocity=torch.cat((robot.data.root_lin_vel_b[0, :2], robot.data.root_ang_vel_b[0, 2:3])).cpu().tolist(),
                                   root_pos=robot.data.root_pos_w[0].cpu().tolist(),
                                   root_quat=robot.data.root_quat_w[0].cpu().tolist(),
                                   joint_pos=robot.data.joint_pos[0].cpu().tolist(),
                                   joint_vel=robot.data.joint_vel[0].cpu().tolist(),
                                   body_pos=robot.data.body_link_pos_w[0].cpu().tolist(),
                                   body_quat=robot.data.body_link_quat_w[0].cpu().tolist(),
                                   upright=float(-robot.data.projected_gravity_b[0, 2]),
                                   terminated=bool(terminated.any()), truncated=bool(truncated.any()))
                        if args.stairs_probe:
                            distance = torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - terrain.env_origins[0, :2])
                            row.update(horizontal_distance_from_origin=float(distance),
                                       computed_torque=robot.data.computed_torque[0].cpu().tolist(),
                                       applied_torque=robot.data.applied_torque[0].cpu().tolist(),
                                       joint_pos_target=robot.data.joint_pos_target[0].cpu().tolist(),
                                       joint_vel_target=robot.data.joint_vel_target[0].cpu().tolist(),
                                       contact_forces_w=raw.scene["contact_forces"].data.net_forces_w[0].cpu().tolist())
                            reached_distance = float(distance) > command.terrain_length / 2
                        records.append(row)
                        assert torch.isfinite(reward).all()
                        if writer is not None and (step % 2 == 1 or done):
                            target = np.asarray(row["root_pos"])
                            raw.sim.set_camera_view(tuple(target + np.array([1.6, -2.4, 1.25])),
                                                    tuple(target + np.array([0.0, 0.0, -0.08])))
                            raw.sim.render()
                            frame = raw.render()
                            assert frame is not None and frame.std() > 1.0, "Missing/blank rendered frame"
                            picture = Image.fromarray(frame[..., :3])
                            draw = ImageDraw.Draw(picture)
                            draw.rectangle((0, 0, 960, 106), fill=(17, 23, 32))
                            draw.text((12, 6), f"Thunder CTS-MoE STUDENT | update {completed_updates} | {name} | 1x", font=font, fill="white")
                            draw.text((12, 31), f"t={row['time']:.2f}s  cmd vx/vy/wz={row['command']}  nominal / no push", font=font, fill="white")
                            actual = row["velocity"]
                            draw.text((12, 56), f"measured {actual[0]:+.2f}, {actual[1]:+.2f} m/s, {actual[2]:+.2f} rad/s", font=font, fill=(120, 220, 255))
                            detail = "flat"
                            if "stairs" in terrain_name:
                                detail = f"L{level} | riser about {terrain_details['nominal_step_height_m'] * 100:.1f} cm | tread 31 cm"
                            elif "slope" in terrain_name:
                                detail = f"L{level} | slope about {terrain_details['angle_deg']:.1f} deg"
                            draw.text((12, 81), detail + (" | TERMINATED - CLIP ENDS" if done else ""),
                                      font=font, fill=(255, 110, 100) if done else "white")
                            writer.append_data(np.asarray(picture))
                            if step in (1, 199, 399, 599) or done:
                                picture.save(args.output / f"{name}_{step:04d}.jpg")
                        if done or reached_distance:
                            break
                        if step % 100 == 0:
                            (args.output / "status.json").write_text(json.dumps(dict(state="recording", clip=name, step=step)))
                if writer is not None and (records[-1]["terminated"] or records[-1]["truncated"]):
                    for _ in range(25):
                        writer.append_data(np.asarray(picture))  # explicit terminal freeze, never a hidden retry
            (args.output / f"{name}.json").write_text(json.dumps(records, allow_nan=False))
            measured = np.asarray([row["velocity"] for row in records])
            commanded = np.asarray([row["command"] for row in records])
            clip = dict(name=name, file=video.name, terrain=terrain_details, duration_s=len(records) * raw.step_dt,
                        terminated=records[-1]["terminated"], truncated=records[-1]["truncated"],
                        min_upright=min(row["upright"] for row in records),
                        delta_world_xyz=(np.asarray(records[-1]["root_pos"]) - np.asarray(records[0]["root_pos"])).tolist(),
                        rmse_vx_vy_wz=np.sqrt(np.mean((measured - commanded) ** 2, axis=0)).tolist())
            if args.stairs_probe:
                clip.update(reached_upgrade_distance=reached_distance,
                            max_horizontal_distance_m=max(row["horizontal_distance_from_origin"] for row in records),
                            stop_reason="environment_termination" if records[-1]["terminated"] else
                                        "distance_threshold" if reached_distance else
                                        "time_limit" if records[-1]["truncated"] else "duration_completed")
            manifest["clips"].append(clip)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        (args.output / "status.json").write_text(json.dumps(dict(state="complete", clips=len(cases))))
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--stairs-probe", action="store_true", help="Probe L3-L6 upstairs until 4 m displacement or 25 s")
    args = parser.parse_args()
    if args.stairs_probe and (not args.trace_only or args.smoke):
        parser.error("--stairs-probe requires --trace-only and cannot use --smoke")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.root / "upstream/source/rsl_rl"), str(args.root / "upstream/source/robot_lab"),
                   str(args.root / "thunder/source/doso_train"), str(args.root / "code")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=not args.trace_only, device="cuda:0").app
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
