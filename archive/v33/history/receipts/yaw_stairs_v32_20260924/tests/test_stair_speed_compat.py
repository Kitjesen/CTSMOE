# The runtime imports the local stop_commands fork. Mock that actual dependency
# for this CPU-only inherited curriculum test; no simulator behavior is bypassed.
import torch

from stair_speed_logic import SPEED_EDGES, sample_speeds, terrain_decisions


def test_all_speed_bands_are_sampled_and_respect_boundaries():
    torch.manual_seed(42)
    bins, speeds = sample_speeds(4096, "cpu")
    edges = torch.tensor(SPEED_EDGES)
    assert torch.all(speeds >= edges[bins])
    assert torch.all(speeds < edges[bins + 1])
    assert torch.all(torch.bincount(bins, minlength=4) > 800)


def test_slow_budget_does_not_inflate_or_reduce_difficulty_but_falls_still_count():
    speed = torch.tensor([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])
    distance = torch.tensor([1.0, 4.5, 1.0, 4.5, 1.0, 4.5])
    failed = torch.tensor([False, False, True, False, False, True])
    up, down, eligible = terrain_decisions(speed, torch.full_like(speed, 25), distance, failed, 25, 4)
    assert eligible.tolist() == [False, False, False, True, True, True]
    assert up.tolist() == [False, False, False, True, False, False]
    assert down.tolist() == [False, False, True, False, True, True]


def test_downgrade_uses_elapsed_command_not_full_future_episode():
    speed = torch.tensor([0.8, 0.8])
    up, down, _ = terrain_decisions(speed, torch.tensor([2., 25.]), torch.tensor([1., 1.]),
                                    torch.tensor([False, False]), 25, 4)
    assert not up.any()
    assert down.tolist() == [False, True]


def test_full_episode_profile_keeps_phase_stability_gate_for_stair_promotion(monkeypatch):
    import importlib
    import sys
    from types import SimpleNamespace
    from startup_repair import FULL_EPISODE_PROFILE

    monkeypatch.setitem(sys.modules, 'stop_commands', SimpleNamespace(
        Go2RLGymCommand=object))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.curriculums', SimpleNamespace(
        terrain_levels_vel_gym=lambda env, ids: None))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.events', SimpleNamespace(
        reset_root_state_uniform=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.utils', SimpleNamespace(
        is_env_assigned_to_terrain=lambda *args, **kwargs: torch.tensor([True])))
    monkeypatch.setitem(sys.modules, 'phase_transition', SimpleNamespace(
        measure=lambda env: {'stable_s': torch.tensor([0.]), 'weight': torch.tensor([1.])}))
    sys.modules.pop('stair_speed', None)
    stair_speed = importlib.import_module('stair_speed')

    recorded = {}
    command = SimpleNamespace(
        stair_mask=lambda: torch.tensor([True]), speed_bins=torch.tensor([0]),
        held_speed=torch.tensor([.6]), max_move_distance=torch.tensor([5.]),
        terrain_length=8., robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[5., 0., 0.]]))),
        record_episodes=lambda ids, up, terminated, eligible: recorded.update(up=up.clone()))
    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([1]),
        update_env_origins=lambda ids, up, down: recorded.update(updated=up.clone()))
    env = SimpleNamespace(
        device='cpu', num_envs=1, episode_length_buf=torch.tensor([100]), step_dt=.02,
        max_episode_length_s=25., stair_axis=torch.tensor([0]),
        cfg=SimpleNamespace(startup_repair_profile=FULL_EPISODE_PROFILE),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])),
        scene=SimpleNamespace(terrain=terrain, env_origins=torch.zeros(1, 3)))

    stair_speed.straight_stair_terrain_levels(env, torch.tensor([0]))

    assert recorded['up'].tolist() == [False]
    assert recorded['updated'].tolist() == [False]
