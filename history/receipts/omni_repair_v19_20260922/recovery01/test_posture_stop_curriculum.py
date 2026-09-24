from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'code'))

from posture_stop import (  # noqa: E402
    APPLIED_STEP_ATTR,
    height_curriculum,
    height_weight,
    validate_height_curriculum_state,
)


class RewardManager:
    def __init__(self, weight=-1.):
        self.term = SimpleNamespace(weight=weight)

    def get_term_cfg(self, name):
        assert name == 'base_height_l2'
        return self.term

    def set_term_cfg(self, name, term):
        assert name == 'base_height_l2'
        self.term = term


def test_two_update_reset_lag_is_valid_and_matches_applied_step():
    applied_step = 32837 * 24
    current_step = 32839 * 24
    actual = height_weight(32837)

    state = validate_height_curriculum_state(actual, applied_step, current_step, 1250)

    assert state['lag_steps'] == 48
    assert state['lag_updates'] == 2.
    assert state['expected_at_apply'] == actual
    assert actual != height_weight(32839)


def test_weight_overwrite_is_detected_at_the_recorded_application_step():
    applied_step = 32837 * 24
    with pytest.raises(RuntimeError, match='weight mismatch at applied step'):
        validate_height_curriculum_state(-9., applied_step, applied_step + 1, 1250)


def test_missing_future_and_episode_length_stale_application_are_rejected():
    with pytest.raises(RuntimeError, match='no recorded application step'):
        validate_height_curriculum_state(-1., None, 0, 1250)
    with pytest.raises(RuntimeError, match='in the future'):
        validate_height_curriculum_state(height_weight(31832), 24, 23, 1250)

    applied_step = 32837 * 24
    boundary = validate_height_curriculum_state(
        height_weight(32837), applied_step, applied_step + 1250, 1250
    )
    assert boundary['lag_steps'] == 1250
    with pytest.raises(RuntimeError, match='is stale'):
        validate_height_curriculum_state(
            height_weight(32837), applied_step, applied_step + 1251, 1250
        )


def test_restore_style_call_records_restored_common_step_and_exact_weight():
    restored_step = 32831 * 24
    env = SimpleNamespace(common_step_counter=restored_step, reward_manager=RewardManager())

    applied = height_curriculum(env, None)

    assert getattr(env, APPLIED_STEP_ATTR) == restored_step
    assert applied == height_weight(32831)
    assert env.reward_manager.term.weight == applied
