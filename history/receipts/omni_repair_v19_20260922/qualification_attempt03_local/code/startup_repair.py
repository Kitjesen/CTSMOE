"""Bounded all-direction startup repair, without clamping policy or robot state."""
import torch

PROFILE = 'omni_start_v1'
REPEAT_PROFILE = 'omni_start_v2'
REQUEST_PROFILE = 'omni_start_v3'
PROTECTED_PROFILE = 'omni_start_v4'
BALANCED_PROFILE = 'omni_start_v5'
FLAT_FOCUS_PROFILE = 'omni_start_v6'
CONTINUOUS_PROFILE = 'omni_start_v7'
PHASE_PROFILE = 'omni_start_v8'
FLAT_CONTINUOUS_PROFILE = 'omni_start_v9'
FLAT_OUTLIER_PROFILE = 'omni_start_v10'
FULL_EPISODE_PROFILE = 'omni_full_episode_v1'
EXPLICIT_EPISODE_PROFILE = 'omni_full_episode_v13'
FULL_EPISODE_PROFILES = (FULL_EPISODE_PROFILE, EXPLICIT_EPISODE_PROFILE)
PHASE_PROFILES = (PHASE_PROFILE, FLAT_CONTINUOUS_PROFILE, FLAT_OUTLIER_PROFILE,
                  *FULL_EPISODE_PROFILES)
FOCUSED_PROFILES = (FLAT_FOCUS_PROFILE, CONTINUOUS_PROFILE, *PHASE_PROFILES)
REPEAT_PROFILES = (REPEAT_PROFILE, REQUEST_PROFILE, PROTECTED_PROFILE, BALANCED_PROFILE, *FOCUSED_PROFILES)
REQUEST_PROFILES = (REQUEST_PROFILE, PROTECTED_PROFILE, BALANCED_PROFILE, *FOCUSED_PROFILES)
SHORT_EPISODE_PROFILES = tuple(profile for profile in REPEAT_PROFILES
                               if profile not in FULL_EPISODE_PROFILES)


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


def learner_cohort_mask(learner):
    """Select three of four members within each actual CTS learner group.

    Global id modulo four is the CTS teacher/student split itself. Reusing it
    as a repair mask excluded every student. Group ranks avoid that collision,
    remain stable when terrain membership changes, and support other ratios.
    """
    result = torch.zeros_like(learner, dtype=torch.bool)
    for group in (0, 1):
        member = learner == group
        rank = member.long().cumsum(0)-1
        result |= member & (rank.remainder(4) != 0)
    return result


def special_ids(env, ids):
    command = env.command_manager.get_term('base_velocity')
    ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    flat = command.terrain_idxs[ids] == command.terrain_type2idx['flat']
    if env.cfg.startup_repair_profile in (BALANCED_PROFILE, *FOCUSED_PROFILES):
        return ids[flat & learner_cohort_mask(command.learner)[ids]]
    # Keep v1-v4 behavior for reproducible checkpoint experiments. These older
    # profiles have the audited student-exclusion bug; v5 is the corrected one.
    return ids[flat & (ids.remainder(4) != 0)]


def short_episode(env):
    """Truncate the repair cohort, without treating a planned reset as failure."""
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ids = special_ids(env, torch.arange(env.num_envs, device=env.device))
    distance = (env.scene['robot'].data.root_pos_w[ids, :2]-env.scene.env_origins[ids, :2]).norm(dim=-1)
    result[ids] = (env.episode_length_buf[ids]*env.step_dt >= 4.) | (distance >= 3.)
    return result


def short_curriculum(env, env_ids):
    # Four-second diagnostic episodes cannot satisfy the original 4 m terrain
    # promotion rule. Preserve their flat level and exclude planned truncations
    # from terrain-allocation evidence; other terrains keep the existing rule.
    from mixed_task import mixed_curriculum
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    repair = special_ids(env, ids)
    result = mixed_curriculum(env, ids[~torch.isin(ids, repair)])
    allocation = getattr(env, 'terrain_allocation', None)
    if allocation is not None:
        allocation.after_curriculum(repair)
    return result


def normal_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg):
    from mixed_task import mixed_reset_root
    if env.cfg.startup_repair_profile in PHASE_PROFILES:
        command = env.command_manager.get_term('base_velocity')
        candidates = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        pit_ids = candidates[command.pit.active[candidates]]
        robot = env.scene['robot']
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[pit_ids],
                                      torch.zeros_like(robot.data.default_joint_pos[pit_ids]), env_ids=pit_ids)
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
    if env.cfg.startup_repair_profile in PHASE_PROFILES:
        candidates = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        pit_ids = candidates[env.command_manager.get_term('base_velocity').pit.active[candidates]]
        ids = torch.unique(torch.cat((ids, pit_ids)))
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
                if self._env.cfg.startup_repair_profile in REPEAT_PROFILES:
                    self.start_hold[ids] = torch.where(torch.rand(len(ids), device=self.device) < .8,
                        0., .5+torch.rand(len(ids), device=self.device))
                    self.start_ramp[ids] = torch.rand(len(ids), device=self.device) < .1
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
                if self._env.cfg.startup_repair_profile in REPEAT_PROFILES:
                    # Half of the samples target the measured forward-start
                    # failure; the other half still spans all directions.
                    focused = changed[torch.rand(len(changed), device=self.device) < .5]
                    self.start_goal[focused] = 0.
                    self.start_goal[focused, 0] = .2+torch.rand(len(focused), device=self.device)
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
    elif kind == 'request':
        if env.cfg.startup_repair_profile in (PROTECTED_PROFILE, BALANCED_PROFILE, *FOCUSED_PROFILES):
            # Controlled ablation of v3: retain all v2 pit objectives, but
            # exclude pit approaches from this new pre-limit request cost.
            # The same actor still trains on pit episodes; no policy switching.
            gate = gate*(~env.command_manager.get_term('base_velocity').pit.active)
        term = env.action_manager.get_term('joint_pos')
        if not hasattr(env, '_startup_request_tolerance'):
            from pit_posture import JOINT_TOLERANCE
            env._startup_request_tolerance = torch.tensor([
                next(v for name,v in JOINT_TOLERANCE.items()
                     if robot.joint_names[j].endswith('_'+name+'_joint'))
                for j in term.joint_indices], device=env.device)
        value = target_excursion(term.requested_actions,
            robot.data.default_joint_pos[:, term.joint_indices], env._startup_request_tolerance)
    else:
        raise ValueError('Unknown startup objective')
    if env.cfg.startup_repair_profile == CONTINUOUS_PROFILE and kind in ('target', 'request'):
        from flat_posture import flat_mask
        # On flat, judge actual pose with upstream continuous L1. Existing
        # action-rate/smoothness and actuator boundaries remain in force.
        gate = gate * ~flat_mask(env)
    return gate*value


def configure(cfg, profile):
    if profile not in ('off', PROFILE, *REPEAT_PROFILES):
        raise ValueError('Unknown startup repair profile')
    cfg.startup_repair_profile = profile
    if profile == 'off':
        return
    if getattr(cfg, 'stance_strength_profile', 'off') != 'strong_stand_v1':
        raise ValueError('Startup repair requires existing phase-aware posture objectives')
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    if profile in FULL_EPISODE_PROFILES:
        from mixed_task import MixedCommand, mixed_curriculum
        command = cfg.commands.base_velocity
        command.class_type = MixedCommand
        if profile == EXPLICIT_EPISODE_PROFILE:
            from startup_sampling import command_class as explicit_command_class
            command.class_type = explicit_command_class()
        command.resampling_time = 5.
        command.resampling_time_range = (5., 5.)
        cfg.curriculum.terrain_levels.func = mixed_curriculum
        cfg.terminations.startup_window = None
        cfg.episode_length_s = 25.
    else:
        cfg.commands.base_velocity.class_type = command_class()
    cfg.events.reset_base.func = normal_reset_root
    cfg.events.reset_robot_joints.func = normal_reset_joints
    cfg.events.reset_robot_joints.params['asset_cfg'] = SceneEntityCfg('robot')
    cfg.rewards.surface_joint_posture.weight = -2.
    cfg.rewards.normal_target_posture = RewardTermCfg(func=reward, weight=-2., params={'kind':'target'})
    cfg.rewards.normal_vertical_velocity = RewardTermCfg(func=reward, weight=-2., params={'kind':'vertical'})
    if profile in SHORT_EPISODE_PROFILES:
        cfg.terminations.startup_window = TerminationTermCfg(func=short_episode, time_out=True)
        cfg.curriculum.terrain_levels.func = short_curriculum
    if profile in REQUEST_PROFILES:
        # Limited targets are constant inside saturation regions. This term
        # supplies a graded signal for excess policy requests without changing
        # the actuator boundary or the inference action itself.
        cfg.rewards.normal_requested_posture = RewardTermCfg(func=reward, weight=-1., params={'kind':'request'})
    if profile == CONTINUOUS_PROFILE:
        from flat_posture import configure as configure_flat_posture
        configure_flat_posture(cfg)
    if profile in PHASE_PROFILES:
        from phase_transition import configure as configure_phase
        configure_phase(cfg)
    if profile in (FLAT_CONTINUOUS_PROFILE, *FULL_EPISODE_PROFILES):
        from flat_continuous import configure as configure_flat_continuous
        configure_flat_continuous(cfg)
    if profile == FLAT_OUTLIER_PROFILE:
        from flat_outlier import configure as configure_flat_outlier
        configure_flat_outlier(cfg)


def snapshot(env):
    if getattr(env.cfg, 'startup_repair_profile', 'off') == 'off':
        return None
    from stance_strength import measure
    gate, _, _ = measure(env)
    ids = special_ids(env, torch.arange(env.num_envs, device=env.device))
    c = env.command_manager.get_term('base_velocity')
    result = dict(profile=env.cfg.startup_repair_profile, special_flat_count=len(ids),
                special_flat_by_learner={name:int((c.learner[ids] == group).sum())
                    for group,name in ((0,'teacher'),(1,'student'))},
                standing_count=int((c.commands[ids].norm(dim=-1) < .05).sum()),
                active_posture_count=int((gate > .5).sum()),
                target_weight=env.reward_manager.get_term_cfg('normal_target_posture').weight,
                vertical_weight=env.reward_manager.get_term_cfg('normal_vertical_velocity').weight,
                request_weight=(env.reward_manager.get_term_cfg('normal_requested_posture').weight
                    if env.cfg.startup_repair_profile in REQUEST_PROFILES else 0.),
                request_pit_enabled=env.cfg.startup_repair_profile == REQUEST_PROFILE,
                command_abs_max=c.commands[ids].abs().amax(0).cpu().tolist() if len(ids) else None)
    if env.cfg.startup_repair_profile in FULL_EPISODE_PROFILES:
        startup_window = getattr(env.cfg.terminations, 'startup_window', None)
        result.update(command_schedule_s=float(c.cfg.resampling_time),
                      full_episode_s=float(env.cfg.episode_length_s),
                      extra_truncation=startup_window is not None)
    if env.cfg.startup_repair_profile == EXPLICIT_EPISODE_PROFILE:
        result['sampling'] = c.startup_sampling_snapshot()
    return result


def install_allocation(allocation, iteration):
    from terrain_allocation import (AllocationCourse, STARTUP_PROFILE, BASE, reassign,
                                    FLAT_FOCUS_PROFILE as FOCUS_BOUNDS, FLAT_FOCUS)
    focused = allocation.env.cfg.startup_repair_profile in FOCUSED_PROFILES
    profile = FOCUS_BOUNDS if focused else STARTUP_PROFILE
    if allocation.course.profile == profile:
        return None
    if allocation.pending:
        raise ValueError('Finish pending allocation moves before switching the repair cohort')
    terrain = allocation.env.scene.terrain
    weights = dict(FLAT_FOCUS) if focused else {**BASE, 'pit': .35, 'flat': .08}
    levels, columns, receipt = reassign(terrain.terrain_levels.cpu().tolist(),
        terrain.terrain_types.cpu().tolist(), allocation.names, allocation.groups,
        weights, seed=iteration, profile=profile)
    allocation.course = AllocationCourse(profile=profile, weights=weights, last_check=iteration)
    allocation.stats.zero_()
    allocation.pending = {i: (levels[i], columns[i]) for i, _ in receipt['moves']}
    receipt['bounds_profile'] = profile
    receipt['application'] = 'natural episode resets; preserved retained terrain difficulties'
    return receipt
