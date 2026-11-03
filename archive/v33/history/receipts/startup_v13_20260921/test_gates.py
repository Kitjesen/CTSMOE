"""Exercise the delayed-regression and completion criteria independently."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('v12_gates', Path(__file__).with_name('gates.py'))
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


def cases():
    rows = []
    for label in ('old', 'new'):
        for mode in ('direct', 'settle'):
            for direction in ('forward', 'reverse', 'left', 'right', 'yaw_left', 'yaw_right',
                              'diagonal', 'diagonal_back', 'forward_turn', 'side_turn'):
                rows.append(dict(label=label, terrain='flat', mode=mode, direction=direction,
                    level=0, velocity=[.6, 0, 0], first_peak_rms=.4 if label == 'old' else .2,
                    first_peak_joint_rad=.4, max_rms=.5 if label == 'old' else .22,
                    last_xy_error=.1, last_yaw_error=.1, completed_25s=True, physical=False,
                    **{'pass': label == 'new'}))
    return {'cases': rows}


def test_late_excursion_cannot_hide_behind_better_start():
    analysis = cases()
    assert gates.flat25_gate(analysis, 'old', 'new')['accepted']
    analysis['cases'][20]['max_rms'] = .5
    assert not gates.flat25_gate(analysis, 'old', 'new')['accepted']


def test_physical_failure_and_incomplete_rollout_are_rejected():
    for field, value in [('physical', True), ('completed_25s', False), ('last_xy_error', float('nan'))]:
        analysis = cases()
        analysis['cases'][21][field] = value
        assert not gates.flat25_gate(analysis, 'old', 'new')['accepted']


def test_final_requires_all_directions_and_modes():
    analysis = cases()
    assert gates.flat25_gate(analysis, 'old', 'new', final=True)['accepted']
    analysis['cases'][21]['pass'] = False
    assert gates.flat25_gate(analysis, 'old', 'new')['accepted']
    assert not gates.flat25_gate(analysis, 'old', 'new', final=True)['accepted']


def test_matched_case_identity_cannot_change():
    analysis = copy.deepcopy(cases())
    analysis['cases'][21]['velocity'] = [1., 0., 0.]
    with pytest.raises(ValueError, match='ordering or commands'):
        gates.flat25_gate(analysis, 'old', 'new')


def test_duplicate_directions_and_nonfinite_baseline_cannot_pass():
    analysis = cases()
    for index in (1, 21):
        analysis['cases'][index]['direction'] = 'forward'
    with pytest.raises(ValueError, match='duplicate or missing'):
        gates.flat25_gate(analysis, 'old', 'new', final=True)
    analysis = cases()
    analysis['cases'][0]['first_peak_rms'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite baseline'):
        gates.flat25_gate(analysis, 'old', 'new')
