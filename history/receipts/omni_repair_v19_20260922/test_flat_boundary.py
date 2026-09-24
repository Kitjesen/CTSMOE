import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flat_boundary import (
    boundary_safe_curriculum,
    cardinal_recovery,
    clear_guard,
    contain_goals,
    flat_boundary_timeout,
    paths,
    reflect_goals,
)
from omni_commands import sample_goals


def test_body_twist_not_world_velocity_and_curved_path():
    position = torch.zeros(1, 2)
    result = paths(position, torch.tensor([math.pi / 2]), torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(result[0, -1], torch.tensor([0.0, 5.0]), atol=1e-6, rtol=0)
    curve = paths(position, torch.zeros(1), torch.tensor([[1.0, 0.0, 1.0]]), duration=math.pi / 2)
    torch.testing.assert_close(curve[0, -1], torch.tensor([1.0, 1.0]), atol=1e-6, rtol=0)


def test_exact_failed_runtime_arc_is_resolved_by_yaw_and_translation_reflection():
    receipt = json.loads((Path(__file__).parent / "BOUNDARY_FAILURE_DIAGNOSTIC.json").read_text())
    position = torch.tensor(receipt["position"])
    yaw = torch.tensor(receipt["yaw"])
    goals = torch.tensor(receipt["goals"])
    lower, upper = torch.tensor(receipt["lower"]), torch.tensor(receipt["upper"])
    result, reflected, resolved = reflect_goals(position, yaw, goals, lower, upper)
    assert resolved.tolist() == [True]
    assert reflected.tolist() == [True]
    assert result[0, 0] < 0 and result[0, 2] < 0
    trajectory = paths(position, yaw, result)
    assert bool(((trajectory >= lower - 1e-6) & (trajectory <= upper + 1e-6)).all())


def test_outer_row_and_column_reflect_without_shortening_or_zeroing():
    positions = torch.tensor([[1.5, 5.0], [5.0, 8.0]])
    yaw = torch.tensor([0.0, math.pi / 2])
    goals = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lower, upper = torch.tensor([1.0, 1.0]), torch.tensor([9.0, 9.0])
    result, reflected = contain_goals(positions, yaw, goals, lower, upper)
    assert reflected.tolist() == [True, True]
    torch.testing.assert_close(result.abs(), goals.abs())
    trajectory = paths(positions, yaw, result)
    assert bool(((trajectory >= lower - 1e-6) & (trajectory <= upper + 1e-6)).all())


def test_in_margin_can_recover_or_stop_but_not_move_further_out():
    positions = torch.tensor([[0.8, 5.0], [0.8, 5.0]])
    goals = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result, _ = contain_goals(
        positions, torch.zeros(2), goals, torch.tensor([1.0, 1.0]), torch.tensor([9.0, 9.0])
    )
    assert result[0, 0] == 1.0
    assert bool((result[1] == 0).all())


def test_500_random_edge_goals_resolve_by_reflection_or_cardinal_recovery():
    generator = torch.Generator().manual_seed(190922)
    lower, upper = torch.tensor([-9.0, -4.0]), torch.tensor([9.0, 4.0])
    positions = lower + (upper - lower) * torch.rand(500, 2, generator=generator)
    sides = torch.randint(0, 4, (500,), generator=generator)
    positions[sides == 0, 0] = lower[0]
    positions[sides == 1, 0] = upper[0]
    positions[sides == 2, 1] = lower[1]
    positions[sides == 3, 1] = upper[1]
    yaw = (torch.rand(500, generator=generator) * 2 - 1) * math.pi
    goals, _ = sample_goals(500, generator=generator, moving_only=True)
    result, _, resolved = reflect_goals(positions, yaw, goals, lower, upper)
    missing = (~resolved).nonzero().flatten()
    if len(missing):
        recovery, _ = cardinal_recovery(positions[missing], yaw[missing], lower, upper)
        result[missing] = recovery
    trajectory = paths(positions, yaw, result)
    assert bool(((trajectory >= lower - 1e-6) & (trajectory <= upper + 1e-6)).all())
    assert bool((result.abs().sum(-1) > 0).all())


def _command(positions):
    count = len(positions)
    quaternion = torch.zeros(count, 4)
    quaternion[:, 0] = 1.0
    return SimpleNamespace(
        num_envs=count,
        device="cpu",
        flat_band_bounds=(torch.tensor([-10.0, -5.0]), torch.tensor([10.0, 5.0])),
        terrain_idxs=torch.zeros(count, dtype=torch.long),
        terrain_type2idx={"flat": 0},
        robot=SimpleNamespace(
            data=SimpleNamespace(root_pos_w=positions.clone(), root_quat_w=quaternion)
        ),
    )


def test_guard_timeout_counts_once_per_episode_and_true_crossing_is_hard_failure():
    command = _command(torch.tensor([[0.0, 4.2, 0.0], [0.0, 0.0, 0.0]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    assert flat_boundary_timeout(env).tolist() == [True, False]
    assert flat_boundary_timeout(env).tolist() == [True, False]
    assert command.flat_boundary_guard_timeouts == 1
    assert command.flat_boundary_crossings == 0
    clear_guard(command, torch.tensor([0]))
    assert flat_boundary_timeout(env).tolist() == [True, False]
    assert command.flat_boundary_guard_timeouts == 2
    command.robot.data.root_pos_w[0, 1] = 5.01
    with pytest.raises(RuntimeError, match="true physical"):
        flat_boundary_timeout(env)
    assert command.flat_boundary_crossings == 1


def test_guard_curriculum_restores_level_only_when_terrain_type_is_unchanged():
    command = _command(torch.zeros(3, 3))
    command.pit = SimpleNamespace(active=torch.zeros(3, dtype=torch.bool))
    command.flat_boundary_timeout_mask = torch.ones(3, dtype=torch.bool)
    origins = torch.arange(10 * 4 * 3, dtype=torch.float).reshape(10, 4, 3)
    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([2, 3, 4]),
        terrain_types=torch.tensor([1, 1, 1]),
        terrain_origins=origins,
        env_origins=torch.zeros(3, 3),
    )
    env = SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, True])),
        scene=SimpleNamespace(terrain=terrain, env_origins=torch.zeros(3, 3)),
    )

    def original(_, ids):
        terrain.terrain_levels[ids] = torch.tensor([7, 8, 9])
        terrain.terrain_types[ids] = torch.tensor([1, 2, 1])
        return torch.tensor(0.0)

    result = boundary_safe_curriculum(env, torch.tensor([0, 1, 2]), original)
    assert terrain.terrain_levels.tolist() == [2, 8, 9]
    assert terrain.terrain_types.tolist() == [1, 2, 1]
    torch.testing.assert_close(terrain.env_origins[0], origins[2, 1])
    torch.testing.assert_close(env.scene.env_origins[0], origins[2, 1])
    torch.testing.assert_close(result, torch.tensor(19 / 3))


def test_first_reset_curriculum_precedes_command_reset_and_termination():
    command = _command(torch.zeros(3, 3))
    assert not hasattr(command, 'flat_boundary_timeout_mask')
    terrain = SimpleNamespace(terrain_levels=torch.tensor([2, 3, 4]),
                              terrain_types=torch.tensor([1, 1, 1]))
    env = SimpleNamespace(device='cpu',
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool)),
        scene=SimpleNamespace(terrain=terrain))
    result = boundary_safe_curriculum(env, torch.arange(3), lambda env, ids: 3.)
    assert result == 3.
    assert not bool(command.flat_boundary_timeout_mask.any())
