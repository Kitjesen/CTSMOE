import pytest
import torch

from omni_lift import (
    COMMAND_CATEGORIES,
    LANDING_EVENT_WEIGHT,
    LIFT_EVENT_WEIGHT,
    LiftState,
    command_categories,
    command_tracking,
)


def batch(command=(0.0, 0.6, 0.0), actual=(0.0, 0.5, 0.0)):
    return dict(
        contacts=torch.ones((1, 4), dtype=torch.bool),
        clearance=torch.zeros((1, 4)),
        flat=torch.tensor([True]),
        commands=torch.tensor([command]),
        actual_velocity=torch.tensor([actual]),
        reset=torch.tensor([False]),
        dt=0.04,
    )


def complete_flight(state, values, wheel=0, height=0.045, steps=3):
    values["contacts"][0, wheel] = False
    values["clearance"][0, wheel] = height
    result = None
    for _ in range(steps):
        result = state.update(**values)
    values["contacts"][0, wheel] = True
    values["clearance"][0, wheel] = 0.0
    return state.update(**values), result


def test_actual_tracking_is_required_and_accepts_lateral_or_pure_yaw():
    commands = torch.tensor([[0.0, 0.6, 0.0], [0.0, 0.0, -0.8], [0.0, 0.6, 0.0]])
    actual = torch.tensor([[0.0, 0.4, 0.0], [0.0, 0.0, -0.4], [0.0, -0.4, 0.0]])
    result = command_tracking(commands, actual)
    assert result["relevant"].tolist() == [True, True, True]
    assert result["tracking"].tolist() == [True, True, False]


def test_command_categories_are_exclusive_and_lateral_has_priority():
    commands = torch.tensor(
        [
            [0.0, 0.6, 0.0],
            [0.0, -0.6, 0.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, -0.8],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, -0.8],
        ]
    )
    names = [COMMAND_CATEGORIES[index] for index in command_categories(commands).tolist()]
    assert names == ["left", "right", "yawL", "yawR", "other", "left"]


def test_zero_and_pure_forward_commands_never_score():
    for command, actual in [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.8, 0.0, 0.0), (0.8, 0.0, 0.0))]:
        state = LiftState(1)
        result, _ = complete_flight(state, batch(command, actual))
        assert result["landing_reward"].item() == 0.0
        assert result["lift_reward"].item() == 0.0


def test_small_scrape_is_not_a_lift_event():
    state = LiftState(1)
    result, _ = complete_flight(state, batch(), height=0.02)
    assert result["landing_reward"].item() == 0.0
    assert state.landing_count.sum().item() == 0


def test_landing_event_counts_once_and_rewards_ideal_lift():
    state = LiftState(1)
    values = batch()
    result, _ = complete_flight(state, values)
    assert result["landing_reward"].item() == 1.0
    assert result["lift_reward"].item() == pytest.approx(1.0)
    repeated = state.update(**values)
    assert repeated["landing_reward"].item() == 0.0
    assert state.landing_count.sum().item() == 1


def test_cumulative_command_metrics_record_the_physical_event_chain():
    state = LiftState(1)
    values = batch()
    complete_flight(state, values)
    left = COMMAND_CATEGORIES.index("left")
    assert state.metric_tracking_steps[left].item() == 4
    assert state.metric_takeoffs[left].item() == 1
    assert state.metric_valid_flight_steps[left].item() == 3
    assert state.metric_landings[left].item() == 1
    expected = (LIFT_EVENT_WEIGHT + LANDING_EVENT_WEIGHT) * values["dt"]
    assert state.metric_reward_integral[left].item() == pytest.approx(expected)
    assert state.metric_landings.sum().item() == 1


def test_simultaneous_four_wheel_jump_is_invalid():
    state = LiftState(1)
    values = batch()
    values["contacts"][:] = False
    values["clearance"][:] = 0.045
    for _ in range(3):
        state.update(**values)
    values["contacts"][:] = True
    values["clearance"][:] = 0.0
    result = state.update(**values)
    assert result["landing_reward"].item() == 0.0
    assert state.landing_count.sum().item() == 0


def test_tracking_loss_during_flight_invalidates_landing():
    state = LiftState(1)
    values = batch()
    values["contacts"][0, 0] = False
    values["clearance"][0, 0] = 0.045
    state.update(**values)
    values["actual_velocity"][:] = 0.0
    state.update(**values)
    values["actual_velocity"][0, 1] = 0.5
    values["contacts"][0, 0] = True
    result = state.update(**values)
    assert result["landing_reward"].item() == 0.0


def test_airborne_hover_beyond_max_duration_never_scores():
    state = LiftState(1)
    values = batch()
    values["contacts"][0, 0] = False
    values["clearance"][0, 0] = 0.045
    for _ in range(13):  # 0.52 s > 0.45 s maximum
        state.update(**values)
    values["contacts"][0, 0] = True
    result = state.update(**values)
    assert result["landing_reward"].item() == 0.0
    assert result["lift_reward"].item() == 0.0


def test_reset_clears_partial_flight_and_counters_without_fake_landing():
    state = LiftState(1)
    values = batch()
    complete_flight(state, values)
    values["contacts"][0, 1] = False
    values["clearance"][0, 1] = 0.045
    state.update(**values)
    values["reset"][:] = True
    values["contacts"][:] = True
    result = state.update(**values)
    assert not state.airborne.any()
    assert state.landing_count.sum().item() == 0
    assert state.total_events.item() == 1
    assert result["landing_reward"].item() == 0.0


def test_nonflat_never_scores():
    state = LiftState(1)
    values = batch()
    values["flat"][:] = False
    result, _ = complete_flight(state, values)
    assert result["landing_reward"].item() == 0.0
