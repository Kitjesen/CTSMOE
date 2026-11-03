"""Isolated slope-up approach curriculum; original rewards and actuator guard retained."""
import math

import torch

PROFILE = "uphill_approach_v16"
SPEEDS = (.6, 1., 1.2, 1.5)
GAPS = (.1, 1.5)


def selected(command, ids):
    return (command.terrain_idxs[ids] == command.terrain_type2idx["slope_up"]) & (ids % 4 != 0)


def place(env, ids, gaps, axes, lateral=None):
    """Place full collision geometry on the central platform, with an explicit front gap.

    Gap means the foremost collision surface to the nominal 1.5m slope foot,
    not center-to-foot distance. All collision points must stay within the flat
    platform with 10cm margin. No fallback to an unsafe pose is allowed.
    """
    if not len(ids):
        return
    robot = env.scene["robot"]
    geometry = env.command_manager.get_term("base_velocity").pit.geometry
    origins = env.scene.env_origins[ids]
    yaw = axes * (math.pi / 2)
    forward = torch.stack((yaw.cos(), yaw.sin()), dim=1)
    sideways = torch.stack((-yaw.sin(), yaw.cos()), dim=1)
    pose = robot.data.default_root_state[ids, :7].clone()
    pose[:, :3] = origins
    pose[:, 3:] = 0.
    pose[:, 3], pose[:, 6] = (yaw / 2).cos(), (yaw / 2).sin()
    points = geometry.world_points(robot.data.joint_pos[ids], pose)
    front = ((points[..., :2] - origins[:, None, :2]) * forward[:, None]).sum(-1).amax(-1)
    shift = 1.5 - gaps - front
    pose[:, :2] += forward * shift[:, None]
    if lateral is not None:
        pose[:, :2] += sideways * lateral[:, None]
    points = geometry.world_points(robot.data.joint_pos[ids], pose)
    margin = 1.5 - (points[..., :2] - origins[:, None, :2]).abs().amax(dim=(1, 2))
    if not bool(torch.isfinite(points).all() & (margin >= .0998).all()):
        raise RuntimeError("Uphill spawn does not fit inside the validated flat platform")
    pose[:, 2] += origins[:, 2] + .03 - points[..., 2].amin(-1)
    final_points = geometry.world_points(robot.data.joint_pos[ids], pose)
    actual_gap = 1.5 - ((final_points[..., :2] - origins[:, None, :2]) * forward[:, None]).sum(-1).amax(-1)
    clearance = (final_points[..., 2] - origins[:, None, 2]).amin(-1)
    torch.testing.assert_close(actual_gap, gaps, atol=2e-4, rtol=0.)
    if not bool((clearance >= .0298).all()):
        raise RuntimeError("Uphill spawn clearance verification failed")
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    receipt = dict(front_gap_min_m=float(actual_gap.min()), front_gap_max_m=float(actual_gap.max()),
                   collision_platform_margin_min_m=float(margin.min()),
                   actual_collision_clearance_min_m=float(clearance.min()),
                   nominal_collision_clearance_m=.03, count=len(ids),
                   definition="all collision surface to nominal central-platform edge")
    env.slope_runup_placement_receipt = receipt
    return receipt


def reset_approach(env, env_ids):
    command = env.command_manager.get_term("base_velocity")
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[selected(command, ids)]
    if not len(ids):
        return
    def draw(n):
        return torch.randint(n, (len(ids),), device=env.device, generator=command.approach_rng)
    command.approach_axis[ids] = draw(4)
    command.approach_gap[ids] = torch.tensor(GAPS, device=env.device)[draw(2)]
    command.approach_speed[ids] = torch.tensor(SPEEDS, device=env.device)[draw(4)]
    place(env, ids, command.approach_gap[ids], command.approach_axis[ids])
    command.approach_reset_count += len(ids)


def route_exit(env):
    # The 25s cap remains. A successful outward traverse is truncated at x=4m
    # to avoid training on a neighboring atlas tile under a false slope label.
    # This deliberate slope-only boundary is not whole-body crest completion.
    command = env.command_manager.get_term("base_velocity")
    ids = torch.arange(env.num_envs, device=env.device)
    angle = command.approach_axis * (math.pi / 2)
    axis = torch.stack((angle.cos(), angle.sin()), dim=1)
    relative = command.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return selected(command, ids) & ((relative * axis).sum(-1) >= 4.)


def snapshot(env):
    command = env.command_manager.get_term("base_velocity")
    ids = torch.arange(env.num_envs, device=env.device)
    mask = selected(command, ids)
    expected = torch.zeros_like(command.commands[mask])
    expected[:, 0] = command.approach_speed[mask]
    torch.testing.assert_close(command.commands[mask], expected, atol=1e-6, rtol=0.)
    return dict(profile=PROFILE, scope="slope_up only; 3/4 held-forward and 1/4 original mixed",
                held_forward_count=int(mask.sum()), speeds=list(SPEEDS), front_gaps_m=list(GAPS),
                placements=command.approach_reset_count,
                placement=getattr(env, "slope_runup_placement_receipt", None),
                heading_feedback=False, reward_changes=False, actuator_limit_changes=False,
                crest_deceleration=False, route_exit="base projection reaches 4m from tile center")


def configure(cfg):
    from isaaclab.managers import EventTermCfg, TerminationTermCfg
    parent = cfg.commands.base_velocity.class_type

    class UphillApproachCommand(parent):
        def __init__(self, command_cfg, env):
            super().__init__(command_cfg, env)
            self.approach_axis = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.approach_gap = torch.full((self.num_envs,), .1, device=self.device)
            self.approach_speed = torch.full((self.num_envs,), .6, device=self.device)
            self.approach_rng = torch.Generator(device=self.device).manual_seed(160922)
            self.approach_reset_count = 0

        def _resample(self, ids):
            super()._resample(ids)
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            ids = ids[selected(self, ids)]
            self.commands[ids] = 0.
            self.commands[ids, 0] = self.approach_speed[ids]
            self.time_left[ids] = 1e9

        def _update_command(self):
            super()._update_command()
            ids = torch.arange(self.num_envs, device=self.device)
            ids = ids[selected(self, ids)]
            self.commands[ids] = 0.
            self.commands[ids, 0] = self.approach_speed[ids]
            elapsed = self._env.episode_length_buf[ids] * self._env.step_dt
            self.commands_xy_accumulation[ids] = 0.
            self.commands_xy_accumulation[ids, 0] = self.approach_speed[ids] * elapsed / self.cfg.resampling_time

    cfg.commands.base_velocity.class_type = UphillApproachCommand
    cfg.events.uphill_approach_reset = EventTermCfg(func=reset_approach, mode="reset")
    cfg.terminations.uphill_approach_exit = TerminationTermCfg(func=route_exit, time_out=True)
    cfg.uphill_approach_profile = PROFILE
