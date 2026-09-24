"""Explicit body-frame side stepping and mixed commands on V30 measured routes.

Reference columns retain their historical evaluation contract. New route groups
use full-platform support and command tracking, never radial-distance promotion.
"""
import math
import torch

PROFILE = 'omni_routes_v31'
NAMES = ('straight', 'side', 'forward_yaw', 'forward_side', 'combined')
WEIGHTS = (.6, 0., .15, .15, .1)
YAW_PROFILE = 'initial_yaw45_v32'

def sample_yaw_offsets(n, device):
    angles = (torch.rand(n, device=device) * 2 - 1) * (math.pi / 4)
    return torch.where(torch.rand(n, device=device) < .25, 0., angles)

def rotate_world_yaw(quat, angle):
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    w, x, y, z = quat.unbind(-1)
    return torch.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w), -1)



def sample_groups(n, device):
    # Reset batches can contain only non-route terrain. No command cohort
    # exists for that batch, and torch.multinomial rejects zero draws.
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.multinomial(torch.tensor(WEIGHTS, device=device), n, replacement=True)


def body_command(group, sign, forward_sign, speed, elapsed, triggered):
    """Three-second, net-heading-balanced manoeuvre after approaching the steps."""
    u = torch.zeros((len(group), 3), device=group.device)
    side = group == 1
    u[:, 0] = torch.where(side, 0., speed * forward_sign)
    u[:, 1] = torch.where(side, sign * speed, 0.)
    active = triggered & (elapsed >= 0) & (elapsed < 3.)
    turn = torch.where(elapsed < 1.5, 1., -1.)
    u[:, 1] += torch.where(active & ((group == 3) | (group == 4)), sign * .15 * turn, 0.)
    u[:, 2] = torch.where(active & ((group == 2) | (group == 4)), sign * .25 * turn, 0.)
    return u


def world_xy(body, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * body[:, 0] - s * body[:, 1], s * body[:, 0] + c * body[:, 1]), -1)


def mask(env):
    from terrain_variety import VARIANTS
    cols = env.scene.terrain.terrain_types
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for col in VARIANTS:
        result |= cols == col
    return result


def reset_root(env, env_ids, pose_range, velocity_range, asset_cfg, original):
    from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform
    # Retain the mixed-terrain spawn bookkeeping and wave-height repair owned
    # by the existing reset event before changing only route body orientation.
    original(env, env_ids, pose_range, velocity_range, asset_cfg)
    if not hasattr(env, 'omni_route_group'):
        env.omni_route_group = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.omni_route_sign = torch.ones(env.num_envs, device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[mask(env)[ids]]
    groups = sample_groups(len(ids), env.device)
    signs = torch.where(torch.rand(len(ids), device=env.device) < .5, -1., 1.)
    env.omni_route_group[ids], env.omni_route_sign[ids] = groups, signs
    # Mixed forward routes face outward; backwards remains a straight cohort.
    mixed_ids = ids[groups >= 2]
    env.stair_motion_sign[mixed_ids] = 1.
    for axis in range(4):
        selected = mixed_ids[env.stair_axis[mixed_ids] == axis]
        yaw = axis * math.pi / 2
        reset_root_state_uniform(env, selected, dict(pose_range, yaw=(yaw, yaw)), velocity_range, asset_cfg)


    # Final world-yaw rotation after all inherited reset alignment. Keep sampled
    # position, roll/pitch and velocities; do not rerandomize the whole root.
    all_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    cmd = env.command_manager.get_term('base_velocity')
    stairs = all_ids[cmd.stair_mask()[all_ids]]
    if not hasattr(env, 'stair_initial_yaw_offset'):
        env.stair_initial_yaw_offset = torch.zeros(env.num_envs, device=env.device)
        env.stair_yaw_reset_bins = torch.zeros(7, dtype=torch.long, device=env.device)
    offsets = sample_yaw_offsets(len(stairs), env.device)
    env.stair_initial_yaw_offset[stairs] = offsets
    if len(stairs):
        robot = env.scene[asset_cfg.name]
        pose = robot.data.root_state_w[stairs, :7].clone()
        pose[:, 3:7] = rotate_world_yaw(pose[:, 3:7], offsets)
        robot.write_root_pose_to_sim(pose, env_ids=stairs)
        bins = torch.bucketize(offsets, torch.tensor([-math.pi/6, -math.pi/12, -1e-7, 1e-7, math.pi/12, math.pi/6], device=env.device))
        env.stair_yaw_reset_bins += torch.bincount(bins, minlength=7)


class CommandMixin:
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = self.num_envs, self.device
        self.route_elapsed = torch.zeros(n, device=device)
        self.route_triggered = torch.zeros(n, dtype=torch.bool, device=device)
        self.route_tracking = torch.zeros((n, 3), device=device)
        self.route_samples = torch.zeros(n, device=device)
        self.route_hold = torch.zeros(n, device=device)
        self.route_complete = torch.zeros(n, dtype=torch.bool, device=device)
        self.route_totals = torch.zeros((5, 2, 5, 4), dtype=torch.long, device=device)
        self.route_commands_seen = torch.zeros(5, dtype=torch.long, device=device)
        self.route_start = torch.zeros(n, device=device)
        self.route_z = torch.zeros(n, device=device)
        self.route_desired = torch.zeros((n, 3), device=device)
        self.route_initial_x = torch.zeros(n, device=device)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[mask(self._env)[ids]]
        if not len(ids) or not hasattr(self, 'route_elapsed'):
            return
        from terrain_variety import VARIANTS, segments
        from mixed_stair_logic import actual_pose_in_stair_frame
        group = self._env.omni_route_group[ids]
        speed = .3 + .5 * torch.rand(len(ids), device=self.device)
        speed = torch.where(group == 1, .2 + .15 * torch.rand(len(ids), device=self.device), speed)
        self.held_speed[ids] = speed
        self.route_elapsed[ids] = 0.
        self.route_triggered[ids] = False
        self.route_tracking[ids] = 0.
        self.route_samples[ids] = 0.
        self.route_hold[ids] = 0.
        self.route_complete[ids] = False
        terrain = self._env.scene.terrain
        for col, kind in VARIANTS.items():
            for level in range(10):
                selected = ids[(terrain.terrain_types[ids] == col) & (terrain.terrain_levels[ids] == level)]
                last = segments(kind, level)[-1]
                self.route_start[selected], self.route_z[selected] = last[0], last[3]
        self.route_desired[ids] = actual_pose_in_stair_frame(self.robot.data.root_pos_w[ids, :2],
            self.robot.data.root_quat_w[ids], self._env.scene.env_origins[ids, :2], self._env.stair_axis[ids])
        self.route_initial_x[ids] = self.route_desired[ids, 0]
        self._set_route_commands(ids)

    def _set_route_commands(self, ids):
        e = self._env
        self.commands[ids] = body_command(e.omni_route_group[ids], e.omni_route_sign[ids],
            e.stair_motion_sign[ids], self.held_speed[ids], self.route_elapsed[ids], self.route_triggered[ids])

    def _update_command(self):
        super()._update_command()
        if hasattr(self, 'route_elapsed'):
            self._set_route_commands(mask(self._env).nonzero().flatten())

    def _update_metrics(self):
        super()._update_metrics()
        if not hasattr(self, 'route_elapsed'):
            return
        from mixed_stair_logic import actual_pose_in_stair_frame, integrate_body_twist
        e = self._env
        ids = (mask(e) & (e.episode_length_buf > 0)).nonzero().flatten()
        if not len(ids):
            return
        actual = torch.cat((self.robot.data.root_lin_vel_b[ids, :2], self.robot.data.root_ang_vel_b[ids, 2:3]), -1)
        self.route_tracking[ids] += (actual - self.commands[ids]).abs()
        self.route_samples[ids] += 1
        self.route_commands_seen.index_add_(0, e.omni_route_group[ids], torch.ones_like(ids))
        dt = torch.full((len(ids),), e.step_dt, device=self.device)
        self.route_desired[ids] = integrate_body_twist(self.route_desired[ids], self.commands[ids], dt)
        pose = actual_pose_in_stair_frame(self.robot.data.root_pos_w[ids, :2], self.robot.data.root_quat_w[ids],
            e.scene.env_origins[ids, :2], e.stair_axis[ids])
        self.route_triggered[ids] |= pose[:, 0] >= 1.2
        self.route_elapsed[ids] += self.route_triggered[ids] * e.step_dt
        # Only evaluate expensive hulls when the body is approaching the final platform.
        candidates = ids[pose[:, 0] > self.route_start[ids] - .2]
        supported = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if len(candidates):
            supported[torch.isin(ids, candidates)] = platform_support(e, self, candidates)
        self.route_hold[ids] = torch.where(supported, self.route_hold[ids] + e.step_dt, 0.)
        self.route_complete[ids] |= self.route_hold[ids] >= .2

    def snapshot(self):
        result = super().snapshot()
        result['initial_yaw_v32'] = dict(profile=YAW_PROFILE, range_deg=[-45, 45], aligned_fraction=.25, reset_bins=self._env.stair_yaw_reset_bins.cpu().tolist())
        result['omni_routes_v31'] = dict(groups=NAMES, group_weights=WEIGHTS,
            command_control_samples=self.route_commands_seen.cpu().tolist(),
            episodes_complete_physical_promoted=self.route_totals.cpu().tolist(),
            axes=['variant', 'teacher_student', 'command_group', 'episodes_complete_physical_promoted'])
        return result


def platform_support(env, command, ids):
    """Conservative collider-envelope/hull support witness, not contact point measurement."""
    from stair_platform_witness import validated_wheel_envelope
    _, envelope = validated_wheel_envelope(env.cfg.scene.robot.spawn.asset_path)
    robot = command.robot
    wheel = robot.data.body_pos_w[ids][:, command.pit.wheel_ids]
    hull = command.pit.geometry.world_points(robot.data.joint_pos[ids], robot.data.root_state_w[ids, :7])
    hull = hull[:, ~command.pit.geometry.wheel_points, :2]
    origin = env.scene.env_origins[ids]
    yaw = env.stair_axis[ids] * (math.pi / 2)
    axis = torch.stack((yaw.cos(), yaw.sin()), -1)
    wheel_delta = wheel[..., :2] - origin[:, None, :2]
    hull_delta = hull - origin[:, None, :2]
    progress = (wheel_delta * axis[:, None]).sum(-1)
    hull_progress = (hull_delta * axis[:, None]).sum(-1)
    forces = torch.stack([env.scene[f'pit_contact_{s}'].data.force_matrix_w[ids, 0, 0, 2]
                          for s in ('fr', 'fl', 'rr', 'rl')], -1)
    above = wheel[..., 2] > origin[:, None, 2] + command.route_z[ids, None] + .05
    return ((progress - envelope > command.route_start[ids, None]).all(-1)
            & (progress + envelope < 4.45).all(-1)
            & (wheel_delta.abs() + envelope < 4.45).all(dim=(1, 2))
            & (hull_progress > command.route_start[ids, None]).all(-1)
            & (hull_delta.abs() < 4.45).all(dim=(1, 2))
            & (forces > 5.).all(-1) & above.all(-1))


def route_exit(env):
    from mixed_stair_runtime import mixed_stair_route_exit
    result = mixed_stair_route_exit(env)
    command = env.command_manager.get_term('base_velocity')
    selected = mask(env)
    offset = command.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    result[selected] = command.route_complete[selected] | (offset[selected].abs().amax(-1) > 4.)
    return result


def curriculum(env, env_ids):
    from mixed_stair_runtime import mixed_stair_curriculum
    from mixed_stair_logic import actual_pose_in_stair_frame, signed_path_error
    from terrain_variety import VARIANTS
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    selected = mask(env)[ids]
    others = ids[~selected]
    if len(others):
        mixed_stair_curriculum(env, others)
    command = env.command_manager.get_term('base_velocity')
    ids = ids[selected & (env.episode_length_buf[ids] > 0)]
    if len(ids):
        physical = env.termination_manager.terminated[ids]
        tracking = command.route_tracking[ids] / command.route_samples[ids, None].clamp_min(1)
        pose = actual_pose_in_stair_frame(command.robot.data.root_pos_w[ids, :2], command.robot.data.root_quat_w[ids],
            env.scene.env_origins[ids, :2], env.stair_axis[ids])
        error = signed_path_error(pose, command.route_desired[ids])
        good = (tracking <= torch.tensor([.25, .18, .2], device=env.device)).all(-1)
        good &= (error[:, 1].abs() <= .4) & (error[:, 2].abs() <= .3)
        complete = command.route_complete[ids]
        promote = complete & good & ~physical
        # Preserve difficulty on nonphysical incomplete episodes; downgrade on
        # physical failure or clear failure to make half the commanded progress.
        # Compare progress along the stair axis with the commanded projected
        # displacement, not body speed*time (which overstates diagonal progress).
        expected_x = (command.route_desired[ids, 0] - command.route_initial_x[ids]).clamp_min(0.)
        actual_x = pose[:, 0] - command.route_initial_x[ids]
        down = physical | (~complete & (actual_x < .5 * expected_x))
        for index, col in enumerate(VARIANTS):
            choose = env.scene.terrain.terrain_types[ids] == col
            sample = ids[choose]
            key = command.learner[sample] * 5 + env.omni_route_group[sample]
            values = torch.stack((torch.ones_like(key), complete[choose].long(), physical[choose].long(), promote[choose].long()), -1)
            command.route_totals[index].view(10, 4).index_add_(0, key, values)
        env.scene.terrain.update_env_origins(ids, promote, down)
    return env.scene.terrain.terrain_levels.float().mean()


def configure(cfg):
    parent = cfg.commands.base_velocity.class_type
    cls = type('OmniRoutesV31Command', (CommandMixin, parent), {'__module__': __name__})
    globals()[cls.__name__] = cls
    cfg.commands.base_velocity.class_type = cls
    cfg.events.reset_base.params['original'] = cfg.events.reset_base.func
    cfg.events.reset_base.func = reset_root
    cfg.curriculum.terrain_levels.params = {'original': curriculum}
    cfg.terminations.stair_route_exit.func = route_exit
    cfg.omni_routes_profile = PROFILE
