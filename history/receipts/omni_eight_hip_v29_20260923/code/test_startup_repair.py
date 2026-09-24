import pytest
import torch
from types import SimpleNamespace

from startup_repair import sample_goals, target_excursion, transition_command
from terrain_allocation import BASE, STARTUP_PROFILE, AllocationCourse, validate_weights


def test_all_direction_sampler_includes_sides_turns_and_signed_combinations():
    torch.manual_seed(31)
    goals = sample_goals(4096, 'cpu')
    assert (goals[:, 0] > 0).any() and (goals[:, 0] < 0).any()
    assert (goals[:, 1] > 0).any() and (goals[:, 1] < 0).any()
    pure_turn = goals[:, :2].norm(dim=-1) == 0
    assert (goals[pure_turn, 2] > 0).any() and (goals[pure_turn, 2] < 0).any()
    assert ((goals[:, :2].norm(dim=-1) > 0) & (goals[:, 2] != 0)).any()
    assert torch.all(goals.abs().amax(0) <= torch.tensor([1.2, .6, .9])+1e-6)


def test_stop_clears_all_axes_and_immediate_starts_are_not_hidden_by_a_ramp():
    age = torch.tensor([0., 1., 2.1, 4.1])
    goal = torch.tensor([[.6, .3, .5]]).expand(4, -1)
    cmd, _ = transition_command(age, torch.zeros(4), torch.full((4,), 2.), goal,
                                 torch.zeros(4, dtype=torch.bool))
    torch.testing.assert_close(cmd[0], goal[0])
    torch.testing.assert_close(cmd[2], torch.zeros(3))
    torch.testing.assert_close(cmd[3], goal[0])


def test_settle_and_ramp_are_distinct_from_immediate_start():
    goal = torch.ones(3, 3)
    cmd, _ = transition_command(torch.tensor([.5, 1.3, 1.6]), torch.ones(3),
                                 torch.full((3,), 2.), goal, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(cmd[0], torch.zeros(3))
    torch.testing.assert_close(cmd[1], torch.full((3,), .5))
    torch.testing.assert_close(cmd[2], torch.ones(3))


def test_target_cost_preserves_targets_and_keeps_gait_deadband():
    target = torch.tensor([[.1, .2, .3], [.1, 1.6, .3]])
    before = target.clone()
    cost = target_excursion(target, torch.zeros_like(target), torch.tensor([.25, .45, .55]))
    assert cost[0] == 0 and cost[1] > 2
    torch.testing.assert_close(before, target)


def test_requested_cost_distinguishes_outputs_hidden_by_saturation():
    request = torch.tensor([[1.], [4.]])
    tolerance = torch.tensor([.25])
    limited = request.clamp(-.5, .5)
    costs = target_excursion(request, torch.zeros_like(request), tolerance)
    clipped_costs = target_excursion(limited, torch.zeros_like(limited), tolerance)
    assert clipped_costs[0] == clipped_costs[1]
    assert costs[1] > costs[0]


def test_repair_allocation_has_explicit_bounds_and_original_bounds_stay_unchanged():
    weights = {**BASE, 'pit': .35, 'flat': .08}
    validate_weights(weights, STARTUP_PROFILE)
    with pytest.raises(ValueError):
        validate_weights(weights)
    course = AllocationCourse(profile=STARTUP_PROFILE, weights=weights)
    restored = AllocationCourse(**course.state_dict())
    restored.evaluate({}, 500)
    assert restored.weights == weights


def test_short_window_only_truncates_special_flat_cohort():
    from startup_repair import short_episode
    command = SimpleNamespace(terrain_idxs=torch.tensor([0, 0, 1, 0, 0, 0]), terrain_type2idx={'flat':0})
    position = torch.tensor([[4., 0., 0.], [0., 0., 0.], [4., 0., 0.],
                             [3.1, 0., 0.], [4., 0., 0.], [0., 0., 0.]])
    scene = {'robot':SimpleNamespace(data=SimpleNamespace(root_pos_w=position))}
    class Scene(dict):
        env_origins = torch.zeros(6, 3)
    env = SimpleNamespace(num_envs=6, device='cpu', step_dt=.02,
        cfg=SimpleNamespace(startup_repair_profile='omni_start_v2'),
        episode_length_buf=torch.tensor([200, 200, 200, 10, 200, 10]), scene=Scene(scene),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    assert short_episode(env).tolist() == [False, True, False, True, False, False]


def test_protected_request_cost_excludes_pit_without_changing_requested_targets(monkeypatch):
    import stance_strength
    from startup_repair import reward, PROTECTED_PROFILE, REQUEST_PROFILE
    monkeypatch.setattr(stance_strength, 'measure', lambda env: (torch.ones(2), None, None))
    request = torch.tensor([[2.], [2.]])
    term = SimpleNamespace(requested_actions=request, joint_indices=[0])
    robot = SimpleNamespace(joint_names=['fr_hip_joint'],
        data=SimpleNamespace(default_joint_pos=torch.zeros(2, 1)))
    command = SimpleNamespace(pit=SimpleNamespace(active=torch.tensor([False, True])))
    env = SimpleNamespace(cfg=SimpleNamespace(startup_repair_profile=PROTECTED_PROFILE),
        scene={'robot':robot}, device='cpu', _surface_posture_ids=[0],
        action_manager=SimpleNamespace(get_term=lambda name: term),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    protected = reward(env, 'request')
    assert protected[0]>0 and protected[1]==0
    env.cfg.startup_repair_profile=REQUEST_PROFILE
    broad = reward(env, 'request')
    torch.testing.assert_close(broad[0],protected[0])
    assert broad[1]>0
    torch.testing.assert_close(request,torch.full((2,1),2.))


@pytest.mark.parametrize('student_stride',[2,4])
def test_repair_cohort_includes_both_cts_groups_and_retains_free_gait(student_stride):
    from startup_repair import learner_cohort_mask
    ids = torch.arange(64)
    learner = (ids.remainder(student_stride) == 0).long()
    mask = learner_cohort_mask(learner)
    for group in (0,1):
        membership = learner == group
        assert int((mask & membership).sum()) == int(membership.sum())*3//4
        assert (membership & ~mask).any()


def test_balanced_cohort_is_consistent_for_reset_subsets_and_does_not_select_other_terrain():
    from startup_repair import special_ids, BALANCED_PROFILE, PROTECTED_PROFILE
    ids = torch.arange(32)
    command = SimpleNamespace(terrain_idxs=torch.zeros(32,dtype=torch.long),
        terrain_type2idx={'flat':0}, learner=(ids.remainder(4)==0).long())
    command.terrain_idxs[12:16] = 1
    env = SimpleNamespace(cfg=SimpleNamespace(startup_repair_profile=BALANCED_PROFILE),device='cpu',
        command_manager=SimpleNamespace(get_term=lambda name:command))
    selected = special_ids(env,ids)
    subset = torch.tensor([0,4,7,8,12,19,20,28,31])
    torch.testing.assert_close(special_ids(env,subset),subset[torch.isin(subset,selected)])
    assert (command.learner[selected]==1).any()
    assert (command.learner[selected]==0).any()
    assert (command.terrain_idxs[selected]==0).all()
    env.cfg.startup_repair_profile = PROTECTED_PROFILE
    legacy = special_ids(env,ids)
    assert not (command.learner[legacy]==1).any()


@pytest.mark.parametrize('explicit', [False, True])
def test_full_episode_profile_uses_v9_objectives_with_normal_command_and_curriculum(monkeypatch, explicit):
    import sys
    from startup_repair import FULL_EPISODE_PROFILE, EXPLICIT_EPISODE_PROFILE, configure

    class MixedCommand:
        pass

    def mixed_curriculum(*args):
        return args

    configured = []
    monkeypatch.setitem(sys.modules, 'mixed_task', SimpleNamespace(
        MixedCommand=MixedCommand, mixed_curriculum=mixed_curriculum))
    monkeypatch.setitem(sys.modules, 'phase_transition', SimpleNamespace(
        configure=lambda cfg: configured.append('phase')))
    monkeypatch.setitem(sys.modules, 'flat_continuous', SimpleNamespace(
        configure=lambda cfg: configured.append('flat_continuous')))

    class TermCfg:
        def __init__(self, func, weight=None, params=None, time_out=None):
            self.func, self.weight, self.params, self.time_out = func, weight, params, time_out

    class EntityCfg:
        def __init__(self, name):
            self.name = name

    managers = SimpleNamespace(RewardTermCfg=TermCfg, SceneEntityCfg=EntityCfg,
                               TerminationTermCfg=TermCfg)
    monkeypatch.setitem(sys.modules, 'isaaclab', SimpleNamespace(managers=managers))
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', managers)

    def original_curriculum(*args):
        return args
    cfg = SimpleNamespace(
        stance_strength_profile='strong_stand_v1', episode_length_s=4.,
        commands=SimpleNamespace(base_velocity=SimpleNamespace(
            class_type=object, resampling_time=1e9, resampling_time_range=(1e9, 1e9))),
        events=SimpleNamespace(
            reset_base=SimpleNamespace(func=None),
            reset_robot_joints=SimpleNamespace(func=None, params={})),
        rewards=SimpleNamespace(surface_joint_posture=SimpleNamespace(weight=0.)),
        terminations=SimpleNamespace(startup_window=SimpleNamespace(time_out=True)),
        curriculum=SimpleNamespace(terrain_levels=SimpleNamespace(func=original_curriculum)))

    configure(cfg, EXPLICIT_EPISODE_PROFILE if explicit else FULL_EPISODE_PROFILE)

    if explicit:
        assert issubclass(cfg.commands.base_velocity.class_type, MixedCommand)
    else:
        assert cfg.commands.base_velocity.class_type is MixedCommand
    assert cfg.curriculum.terrain_levels.func is mixed_curriculum
    assert cfg.episode_length_s == 25.
    assert cfg.commands.base_velocity.resampling_time == 5.
    assert cfg.commands.base_velocity.resampling_time_range == (5., 5.)
    assert cfg.terminations.startup_window is None
    assert cfg.rewards.surface_joint_posture.weight == -2.
    assert cfg.rewards.normal_target_posture.weight == -2.
    assert cfg.rewards.normal_vertical_velocity.weight == -2.
    assert cfg.rewards.normal_requested_posture.weight == -1.
    assert configured == ['phase', 'flat_continuous']


def test_full_episode_profile_keeps_three_of_four_per_learner_on_flat():
    from startup_repair import (FOCUSED_PROFILES, FULL_EPISODE_PROFILE, PHASE_PROFILES,
                                special_ids)

    ids = torch.arange(32)
    learner = (ids.remainder(4) == 0).long()
    command = SimpleNamespace(terrain_idxs=torch.zeros(32, dtype=torch.long),
                              terrain_type2idx={'flat': 0}, learner=learner)
    command.terrain_idxs[24:] = 1
    env = SimpleNamespace(cfg=SimpleNamespace(startup_repair_profile=FULL_EPISODE_PROFILE),
                          device='cpu', command_manager=SimpleNamespace(
                              get_term=lambda name: command))

    selected = special_ids(env, ids)

    assert (command.terrain_idxs[selected] == 0).all()
    assert FULL_EPISODE_PROFILE in PHASE_PROFILES
    assert FULL_EPISODE_PROFILE in FOCUSED_PROFILES
    for group in (0, 1):
        flat_group = (command.terrain_idxs == 0) & (learner == group)
        assert int((learner[selected] == group).sum()) == int(flat_group.sum()) * 3 // 4


def test_full_episode_snapshot_reads_runtime_schedule_without_startup_command_fields(monkeypatch):
    import sys
    from startup_repair import FULL_EPISODE_PROFILE, snapshot

    monkeypatch.setitem(sys.modules, 'stance_strength', SimpleNamespace(
        measure=lambda env: (torch.ones(env.num_envs), None, None)))
    ids = torch.arange(8)
    command = SimpleNamespace(
        terrain_idxs=torch.zeros(8, dtype=torch.long), terrain_type2idx={'flat': 0},
        learner=(ids.remainder(2) == 0).long(), commands=torch.zeros(8, 3),
        cfg=SimpleNamespace(resampling_time=4.75))
    rewards = {
        'normal_target_posture': SimpleNamespace(weight=-2.),
        'normal_vertical_velocity': SimpleNamespace(weight=-2.),
        'normal_requested_posture': SimpleNamespace(weight=-1.),
    }
    env = SimpleNamespace(
        num_envs=8, device='cpu',
        cfg=SimpleNamespace(startup_repair_profile=FULL_EPISODE_PROFILE,
                            episode_length_s=23., terminations=SimpleNamespace()),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        reward_manager=SimpleNamespace(get_term_cfg=lambda name: rewards[name]))

    result = snapshot(env)

    assert result['command_schedule_s'] == 4.75
    assert result['full_episode_s'] == 23.
    assert result['extra_truncation'] is False
