from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


OFFICIAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OFFICIAL_DIR))

from audit_stats import (  # noqa: E402
    AGE_EDGES,
    AGE_LABELS,
    Cell,
    CommandClock,
    Reservoir,
    Stats,
    classify_commands,
    stable_priority,
)
from startup_repair import learner_cohort_mask  # noqa: E402


def test_stable_priority_is_deterministic_unsigned_64_bit_and_input_sensitive():
    inputs = [(0, 0, 0), (7, 3, 19), (4095, 17, 479)]
    first = [stable_priority(*item) for item in inputs]
    second = [stable_priority(*item) for item in inputs]

    assert first == second
    assert len(set(first)) == len(inputs)
    assert all(isinstance(value, int) and 0 <= value < 2**64 for value in first)


def test_reservoir_keeps_smallest_priorities_and_rejects_equal_cutoff():
    reservoir = Reservoir(3)
    for priority in (8, 2, 5, 1, 3):
        reservoir.add(priority, f"record-{priority}")

    assert reservoir.records() == ["record-1", "record-2", "record-3"]
    assert not reservoir.accepts(3)
    reservoir.add(0, "record-0")
    assert reservoir.records() == ["record-0", "record-1", "record-2"]


@pytest.mark.parametrize("capacity", [0, -1])
def test_reservoir_rejects_nonpositive_capacity(capacity):
    with pytest.raises(ValueError, match="capacity must be positive"):
        Reservoir(capacity)


def test_classify_commands_uses_documented_deadband_and_exclusive_categories():
    command = np.array(
        [
            [0.0, 0.05, -0.05],
            [0.051, 0.05, 0.0],
            [-0.051, 0.0, 0.0],
            [0.05, -0.051, 0.05],
            [0.0, 0.05, -0.051],
            [0.051, 0.051, 0.0],
            [0.051, 0.0, 0.051],
        ]
    )

    np.testing.assert_array_equal(
        classify_commands(command),
        np.array([0, 1, 2, 3, 4, 5, 5]),
    )


def test_command_clock_first_prestep_and_ordinary_segment_changes():
    clock = CommandClock(2, 0.1)
    special = np.array([False, False])
    phase = np.array([0, 0])
    goal = np.zeros((2, 3))
    episode = np.array([0, 0])
    command = np.array([[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]])

    nonzero_age, segment_age = clock.update(command, special, phase, goal, episode)
    np.testing.assert_allclose(nonzero_age, [0.0, 0.0])
    np.testing.assert_allclose(segment_age, [0.0, 0.0])

    nonzero_age, segment_age = clock.update(command, special, phase, goal, episode)
    np.testing.assert_allclose(nonzero_age, [0.1, 0.0])
    np.testing.assert_allclose(segment_age, [0.1, 0.0])

    command[0, 0] = 0.8
    nonzero_age, segment_age = clock.update(command, special, phase, goal, episode)
    np.testing.assert_allclose(nonzero_age, [0.2, 0.0])
    np.testing.assert_allclose(segment_age, [0.0, 0.0])

    command[0] = [0.0, 0.8, 0.0]
    nonzero_age, segment_age = clock.update(command, special, phase, goal, episode)
    np.testing.assert_allclose(nonzero_age, [0.3, 0.0])
    np.testing.assert_allclose(segment_age, [0.0, 0.0])

    command[0] = 0.0
    nonzero_age, segment_age = clock.update(command, special, phase, goal, episode)
    np.testing.assert_allclose(nonzero_age, [0.0, 0.0])
    np.testing.assert_allclose(segment_age, [0.0, 0.0])


def test_special_ramp_does_not_reset_segment_until_goal_phase_or_episode_changes():
    clock = CommandClock(1, 0.1)
    special = np.array([True])
    phase = np.array([2])
    goal = np.array([[0.6, 0.0, 0.0]])
    episode = np.array([0])

    for step, magnitude in enumerate((0.1, 0.2, 0.3)):
        nonzero_age, segment_age = clock.update(
            np.array([[magnitude, 0.0, 0.0]]),
            special,
            phase,
            goal,
            episode,
        )
        np.testing.assert_allclose(nonzero_age, [step * 0.1])
        np.testing.assert_allclose(segment_age, [step * 0.1])

    goal = np.array([[0.7, 0.0, 0.0]])
    nonzero_age, segment_age = clock.update(
        np.array([[0.4, 0.0, 0.0]]), special, phase, goal, episode
    )
    np.testing.assert_allclose(nonzero_age, [0.3])
    np.testing.assert_allclose(segment_age, [0.0])

    phase = np.array([3])
    nonzero_age, segment_age = clock.update(
        np.array([[0.4, 0.0, 0.0]]), special, phase, goal, episode
    )
    np.testing.assert_allclose(nonzero_age, [0.4])
    np.testing.assert_allclose(segment_age, [0.0])

    episode = np.array([1])
    nonzero_age, segment_age = clock.update(
        np.array([[0.4, 0.0, 0.0]]), special, phase, goal, episode
    )
    np.testing.assert_allclose(nonzero_age, [0.0])
    np.testing.assert_allclose(segment_age, [0.0])


def test_episode_generations_count_local_episode_ordinals_not_env_episode_pairs():
    cell = Cell()
    values = np.ones(15)
    cell.add(env=10, episode=0, values=values)
    cell.add(env=11, episode=0, values=values)

    summary = cell.summary()
    assert summary["distinct_episodes"] == 2
    assert summary["episode_generations"] == {"0": 2}
    assert len(summary["episode_generations"]) == 1

    cell.add(env=10, episode=1, values=values)
    assert cell.summary()["episode_generations"] == {"0": 2, "1": 1}


def test_age_bins_match_finite_experiment_definition():
    np.testing.assert_allclose(AGE_EDGES, [0.5, 1.0, 2.0, 2.5, 3.0, 3.3, 4.0])
    assert AGE_LABELS == [
        "0-.5",
        ".5-1",
        "1-2",
        "2-2.5",
        "2.5-3",
        "3-3.3",
        "3.3-4",
        "4+",
    ]
    assert np.searchsorted(AGE_EDGES, 2.5, side="right") == 4
    assert np.searchsorted(AGE_EDGES, 3.0, side="right") == 5


def test_state_and_episode_reservoirs_have_independent_caps(tmp_path):
    count = 4
    learner = torch.tensor([0, 0, 1, 1])
    robot = SimpleNamespace(
        joint_names=["fl_hip_joint", "fl_foot_joint"],
        data=SimpleNamespace(default_joint_pos=torch.zeros(count, 2)),
    )
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        scene={"robot": robot},
        termination_manager=SimpleNamespace(active_terms=[], compute=lambda *args: None),
        action_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(joint_indices=[0])
        ),
    )
    algorithm = SimpleNamespace(
        student_env_idxs=torch.tensor([2, 3]),
        anchor_cohort=learner_cohort_mask(learner),
    )
    stats = Stats(env, algorithm, reference_policy=None, output=tmp_path)

    for priority in range(5000):
        stats.candidate_samples.add(priority, {"kind": "state", "priority": priority})
    for episode in range(100):
        stats.episode_best[(episode % count, episode)] = (
            priority,
            {"kind": "episode", "episode": episode},
        )

    stats.finish()
    samples = torch.load(tmp_path / "samples.pt", weights_only=False)

    assert len(samples["candidate_state_weighted"]) == 4096
    assert len(samples["candidate_episode_balanced"]) == 64
    assert {record["kind"] for record in samples["candidate_state_weighted"]} == {"state"}
    assert {record["kind"] for record in samples["candidate_episode_balanced"]} == {
        "episode"
    }
