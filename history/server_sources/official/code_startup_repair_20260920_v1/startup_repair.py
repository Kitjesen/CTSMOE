"""Bounded all-direction startup repair, without clamping policy or robot state."""
import torch

PROFILE = 'omni_start_v1'


def target_excursion(target, nominal, tolerance):
    return ((target-nominal).abs()/tolerance-1.).clamp_min(0.).square().mean(-1)


def transition_command(age, hold, period, goal, ramp):
    elapsed = (age-hold).clamp_min(0.)
    phase = torch.floor(elapsed/period).long()
    local = elapsed-phase*period
    moving = (age >= hold) & (phase.remainder(2) == 0)
    factor = torch.where(ramp, (local/.6).clamp(0., 1.), torch.ones_like(local))
    return goal*(factor*moving)[:, None], phase


def sample_goals(count, device):
    kind = torch.randint(0, 10, (count,), device=device)
    vx = .2+torch.rand(count, device=device)
    vy = .2+.4*torch.rand(count, device=device)
    yaw = .3+.6*torch.rand(count, device=device)
    goals = torch.zeros(count, 3, device=device)
    goals[:, 0] = torch.where((kind == 0) | (kind == 6) | (kind == 8), vx,
                              torch.where((kind == 1) | (kind == 7), -vx, 0.))
    goals[:, 1] = torch.where((kind == 2) | (kind == 6) | (kind == 9), vy,
                              torch.where((kind == 3) | (kind == 7), -vy, 0.))
    goals[:, 2] = torch.where((kind == 4) | (kind == 8), yaw,
                              torch.where((kind == 5) | (kind == 9), -yaw, 0.))
    # Diagonal/reversing arcs include both signs, rather than one preferred hand.
    combined = kind >= 6
    signs = torch.where(torch.rand(count, 3, device=device) < .5, -1., 1.)
    goals[combined] *= signs[combined]
    return goals


def special_ids(env, ids):
    command = env.command_manager.get_term('base_velocity')
    ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    flat = command.terrain_idxs[ids] == command.terrain_type2idx['flat']
    # A fixed 3/4 cohort makes reset and command membership agree without
    # depending on event ordering. The remaining flat cohort retains free gait.
    return ids[flat & (ids.remainder(4) != 0)]


def normal_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg):
    from mixed_task import mixed_reset_root
    mixed_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg)
    ids = special_ids(env, env_ids)
    if len(ids):
        robot = env.scene['robot']
        state = robot.data.root_state_w[ids].clone()
        state[:, :3] = env.scene.env_origins[ids]+robot.data.default_root_state[ids, :3]
        # Keep the sampled yaw; remove the airborne/velocity recovery start from
        # this cohort so ordinary standing-to-motion is explicitly represented.
        state[:, 7:13] = 0.
        robot.write_root_state_to_sim(state, env_ids=ids)


def normal_reset_joints(env, env_ids, position_range, velocity_range, asset_cfg):
    from mixed_task import mixed_reset_joints
    from isaaclab.envs.mdp.events import reset_joints_by_scale
    mixed_reset_joints(env, env_ids, position_range, velocity_range, asset_cfg)
    ids = special_ids(env, env_ids)
    if len(ids):
        reset_joints_by_scale(env, ids, (.98, 1.02), (0., 0.), asset_cfg)


def command_class():
    from mixed_task import MixedCommand

    class StartupCommand(MixedCommand):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.start_goal = torch.zeros(self.num_envs, 3, device=self.device)
            self.start_hold = torch.zeros(self.num_envs, device=self.device)
            self.start_period = torch.ones(self.num_envs, device=self.device)*2.
            self.start_phase = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
            self.start_ramp = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        def _resample(self, env_ids):
            super()._resample(env_ids)
            ids = special_ids(self._env, env_ids)
            if len(ids):
                self.start_goal[ids] = sample_goals(len(ids), self.device)
                self.start_hold[ids] = torch.where(torch.rand(len(ids), device=self.device) < .5,
                    0., .5+torch.rand(len(ids), device=self.device))
                self.start_period[ids] = 1.5+torch.rand(len(ids), device=self.device)
                self.start_ramp[ids] = torch.rand(len(ids), device=self.device) < .25
                self.start_phase[ids] = -1
                self.time_left[ids] = 1e9
                self._update_startup(ids)

        def _update_startup(self, ids):
            age = self._env.episode_length_buf[ids]*self._env.step_dt
            _, phase = transition_command(age, self.start_hold[ids], self.start_period[ids],
                                           self.start_goal[ids], self.start_ramp[ids])
            change = (phase != self.start_phase[ids]) & (phase.remainder(2) == 0)
            changed = ids[change]
            if len(changed):
                self.start_goal[changed] = sample_goals(len(changed), self.device)
            self.start_phase[ids] = phase
            values, _ = transition_command(age, self.start_hold[ids], self.start_period[ids],
                                            self.start_goal[ids], self.start_ramp[ids])
            self.commands[ids] = values

        def _update_command(self):
            super()._update_command()
            self._update_startup(special_ids(self._env, torch.arange(self.num_envs, device=self.device)))

    return StartupCommand


def reward(env, kind):
    from stance_strength import measure
    gate, _, _ = measure(env)
    robot = env.scene['robot']
    ids = env._surface_posture_ids
    if kind == 'target':
        value = target_excursion(robot.data.joint_pos_target[:, ids],
                                 robot.data.default_joint_pos[:, ids], env._surface_posture_tolerance)
    elif kind == 'vertical':
        value = robot.data.root_lin_vel_w[:, 2].square()
    else:
        raise ValueError('Unknown startup objective')
    return gate*value


def configure(cfg, profile):
    if profile not in ('off', PROFILE):
        raise ValueError('Unknown startup repair profile')
    cfg.startup_repair_profile = profile
    if profile == 'off':
        return
    if getattr(cfg, 'stance_strength_profile', 'off') != 'strong_stand_v1':
        raise ValueError('Startup repair requires existing phase-aware posture objectives')
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    cfg.commands.base_velocity.class_type = command_class()
    cfg.events.reset_base.func = normal_reset_root
    cfg.events.reset_robot_joints.func = normal_reset_joints
    cfg.events.reset_robot_joints.params['asset_cfg'] = SceneEntityCfg('robot')
    cfg.rewards.surface_joint_posture.weight = -2.
    cfg.rewards.normal_target_posture = RewardTermCfg(func=reward, weight=-2., params={'kind':'target'})
    cfg.rewards.normal_vertical_velocity = RewardTermCfg(func=reward, weight=-2., params={'kind':'vertical'})


def snapshot(env):
    if getattr(env.cfg, 'startup_repair_profile', 'off') == 'off':
        return None
    from stance_strength import measure
    gate, _, _ = measure(env)
    ids = special_ids(env, torch.arange(env.num_envs, device=env.device))
    c = env.command_manager.get_term('base_velocity')
    return dict(profile=PROFILE, special_flat_count=len(ids),
                standing_count=int((c.commands[ids].norm(dim=-1) < .05).sum()),
                active_posture_count=int((gate > .5).sum()),
                target_weight=env.reward_manager.get_term_cfg('normal_target_posture').weight,
                vertical_weight=env.reward_manager.get_term_cfg('normal_vertical_velocity').weight,
                command_abs_max=c.commands[ids].abs().amax(0).cpu().tolist() if len(ids) else None)


def install_allocation(allocation, iteration):
    from terrain_allocation import AllocationCourse, STARTUP_PROFILE, BASE, reassign
    if allocation.course.profile == STARTUP_PROFILE:
        return None
    if allocation.pending:
        raise ValueError('Finish pending allocation moves before switching the repair cohort')
    terrain = allocation.env.scene.terrain
    weights = {**BASE, 'pit': .35, 'flat': .08}
    levels, columns, receipt = reassign(terrain.terrain_levels.cpu().tolist(),
        terrain.terrain_types.cpu().tolist(), allocation.names, allocation.groups,
        weights, seed=iteration, profile=STARTUP_PROFILE)
    allocation.course = AllocationCourse(profile=STARTUP_PROFILE, weights=weights, last_check=iteration)
    allocation.stats.zero_()
    allocation.pending = {i: (levels[i], columns[i]) for i, _ in receipt['moves']}
    receipt['bounds_profile'] = STARTUP_PROFILE
    receipt['application'] = 'natural episode resets; preserved retained terrain difficulties'
    return receipt
