from types import SimpleNamespace as NS

import pytest
import torch

import early_restart


def test_continuous_actual_angle_l1_has_no_deadzone_and_true_stop_multiplier():
    q0 = torch.zeros(3, 2)
    q = torch.tensor([[0.001, -0.002], [0.001, -0.002], [0.001, -0.002]])
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.0, 0.06]])
    value = early_restart.continuous_l1(q, q0, commands)
    assert value[0] > 0
    assert value[0] == pytest.approx(3 * value[1])
    assert value[2] == pytest.approx(value[1])


def test_phase_mask_is_flat_constant_geometry_gated_and_unrelated_zero(monkeypatch):
    import phase_transition

    phase = {
        "weight": torch.tensor([0.0, 0.4, 0.0, 1.0]),
        "stairs": torch.tensor([False, True, False, False]),
        "pit": torch.tensor([False, False, True, False]),
        "ground": torch.zeros(4),
    }
    monkeypatch.setattr(phase_transition, "measure", lambda env: phase)
    command = NS(terrain_idxs=torch.tensor([0, 1, 2, 3]), terrain_type2idx={"flat": 0})
    env = NS(num_envs=4, device="cpu", command_manager=NS(get_term=lambda _: command))
    weight, _ = early_restart.phase_weight(env)
    torch.testing.assert_close(weight, torch.tensor([1.0, 0.4, 0.0, 0.0]))


@pytest.mark.parametrize("invalid", [None, "inactive", "active_q", "active_vz", "active_root", "active_ray", "active_gate"])
def test_raw_rewards_use_valid_ground_and_reject_active_invalid_state(monkeypatch, invalid):
    import phase_transition

    phase = {
        "weight": torch.ones(3),
        "stairs": torch.tensor([False, True, False]),
        "pit": torch.tensor([False, False, False]),
        "ground": torch.tensor([99.0, 1.0, 99.0]),
        "wheel_ground": torch.zeros(3, 4),
    }
    monkeypatch.setattr(phase_transition, "measure", lambda env: phase)
    names = [f"{leg}_{joint}_joint" for leg in ("fl", "fr", "rl", "rr")
             for joint in ("hip", "thigh", "calf")]
    data = NS(
        joint_pos=torch.ones(3, 12) * 0.01,
        default_joint_pos=torch.zeros(3, 12),
        root_pos_w=torch.tensor([[0.0, 0.0, 0.46], [0.0, 0.0, 1.46], [0.0, 0.0, 5.0]]),
        root_lin_vel_w=torch.tensor([[0.0, 0.0, 0.2]]).expand(3, -1).clone(),
    )
    command = NS(command=torch.zeros(3, 3), terrain_idxs=torch.tensor([0, 1, 2]),
                 terrain_type2idx={"flat": 0})
    scene = {"robot": NS(joint_names=names, data=data)}
    scene_obj = NS(env_origins=torch.zeros(3, 3))
    scene_obj.__getitem__ = scene.__getitem__
    class Scene(dict):
        env_origins = torch.zeros(3, 3)
    env = NS(num_envs=3, device="cpu", scene=Scene(scene),
             command_manager=NS(get_term=lambda _: command))
    if invalid == "inactive":
        data.joint_pos[2] = float("nan")
        data.root_pos_w[2, 2] = float("inf")
        data.root_lin_vel_w[2, 2] = float("inf")
    elif invalid == "active_q":
        data.joint_pos[0, 0] = float("nan")
    elif invalid == "active_vz":
        data.root_lin_vel_w[0, 2] = float("inf")
    elif invalid == "active_root":
        data.root_pos_w[0, 2] = float("nan")
    elif invalid == "active_ray":
        phase["wheel_ground"][1, 0] = float("nan")
    elif invalid == "active_gate":
        phase["weight"][1] = float("nan")
    if invalid is not None and invalid.startswith("active_"):
        with pytest.raises(FloatingPointError, match="support terrain ray"):
            early_restart.raw_terms(env)
        return
    terms = early_restart.raw_terms(env)
    assert all(torch.isfinite(term).all() and term[2] == 0 for term in terms.values())
    assert terms["height"][0] == pytest.approx(0.0)
    assert terms["height"][1] == pytest.approx(0.0)
    assert terms["height"][2] == pytest.approx(0.0)
    assert terms["hip"][0] == pytest.approx(0.12)
    assert terms["thigh_calf"][0] == pytest.approx(0.24)


def test_flat_reset_cohort_has_three_of_four_for_both_learners():
    ids = torch.arange(40)
    learner = (ids.remainder(4) == 0).long()
    command = NS(terrain_idxs=torch.zeros(40, dtype=torch.long),
                 terrain_type2idx={"flat": 0}, learner=learner)
    command.terrain_idxs[32:] = 1
    env = NS(device="cpu", command_manager=NS(get_term=lambda _: command))
    selected = early_restart.special_ids(env, ids)
    assert (command.terrain_idxs[selected] == 0).all()
    for group in (0, 1):
        eligible = (command.terrain_idxs == 0) & (learner == group)
        assert int((learner[selected] == group).sum()) == int(eligible.sum()) * 3 // 4


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown early restart"):
        early_restart.configure(NS(), "latest")


def test_configure_keeps_official_l1_and_installs_independent_profile(monkeypatch):
    import sys

    class Term:
        def __init__(self, func=None, weight=None, params=None):
            self.func, self.weight, self.params = func, weight, params

    class Entity:
        def __init__(self, name):
            self.name = name

    managers = NS(RewardTermCfg=Term, SceneEntityCfg=Entity)
    monkeypatch.setitem(sys.modules, "isaaclab", NS(managers=managers))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    marker = type("EarlyCommand", (), {})
    monkeypatch.setattr("startup_sampling.command_class", lambda: marker)
    native_curriculum = object()
    monkeypatch.setitem(sys.modules, "mixed_task", NS(mixed_curriculum=native_curriculum))
    official_hip = Term(weight=-0.5)
    official_joint = Term(weight=-0.1)
    cfg = NS(
        startup_repair_profile="off", pit_posture_profile="off",
        surface_posture_profile="off", stance_strength_profile="off",
        commands=NS(base_velocity=NS(class_type=object, resampling_time=99.0,
                                     resampling_time_range=(99.0, 99.0))),
        curriculum=NS(terrain_levels=NS(func=object())),
        events=NS(reset_base=NS(func=None), reset_robot_joints=NS(func=None, params={})),
        terminations=NS(startup_window=object()),
        rewards=NS(hip_pos_penalty_l1=official_hip, joint_pos_penalty_l1=official_joint),
    )

    early_restart.configure(cfg, early_restart.PROFILE)

    assert cfg.early_restart_profile == early_restart.PROFILE
    assert cfg.episode_length_s == 25.0
    assert cfg.commands.base_velocity.class_type is marker
    assert cfg.commands.base_velocity.resampling_time == 5.0
    assert cfg.commands.base_velocity.resampling_time_range == (5.0, 5.0)
    assert cfg.curriculum.terrain_levels.func is native_curriculum
    assert cfg.terminations.startup_window is None
    assert cfg.rewards.hip_pos_penalty_l1 is official_hip
    assert cfg.rewards.joint_pos_penalty_l1 is official_joint
    assert cfg.rewards.early_actual_hip_l1.weight == early_restart.HIP_WEIGHT
    assert cfg.rewards.early_actual_thigh_calf_l1.weight == early_restart.THIGH_CALF_WEIGHT
    assert cfg.rewards.early_support_height.weight == early_restart.HEIGHT_WEIGHT
    assert cfg.rewards.early_vertical_velocity.weight == early_restart.VERTICAL_WEIGHT


def test_early_allocation_is_exact_and_bounded():
    from terrain_allocation import EARLY_BALANCED, EARLY_BALANCED_PROFILE, bounds, validate_weights

    floor, ceiling = bounds(EARLY_BALANCED_PROFILE)
    assert sum(EARLY_BALANCED.values()) == pytest.approx(1.0)
    assert floor == ceiling == EARLY_BALANCED
    validate_weights(EARLY_BALANCED, EARLY_BALANCED_PROFILE)
    assert EARLY_BALANCED == {
        "flat": 0.25, "pit": 0.25, "stairs_up": 0.15, "stairs_down": 0.10,
        "wave": 0.05, "slope_up": 0.05, "slope_down": 0.05,
        "rough_slope": 0.05, "obstacles": 0.05,
    }


def test_install_allocation_preserves_existing_pending():
    allocation = NS(course=NS(profile="balanced_adaptive_v1"), pending={3: (2, 1)})
    with pytest.raises(ValueError, match="pending adaptive"):
        early_restart.install_allocation(allocation, 10)
