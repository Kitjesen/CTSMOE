import json

import numpy as np
import pytest

from focus_metrics import analyze_braking, write_reports


def fixture():
    steps, dt = 250, .1
    names = [f'{side}_{part}_joint' for part in ('hip', 'thigh', 'calf', 'foot')
             for side in ('fl', 'fr', 'rl', 'rr')]
    ids = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    case = dict(id=0, policy=0, terrain='flat', direction='forward_high', mode='direct',
                velocity=[2., 0., 0.], end_step=steps-1, physical=False, terms=['time_out'])
    manifest = dict(dt=dt, joint_names=names, body_names=['base_link'], default_q=[0.]*16,
                    request_joint_ids=ids, protocol_id='brake25', cases=[case],
                    history=[dict(label='source33174', path='source.pt', teacher=False)])
    command = np.zeros((steps, 1, 3))
    command[:50, 0, 0] = 2.
    command[100:150, 0, 0] = -2.
    command[200:, 0, 0] = 2.
    velocity = command.copy()
    velocity[50:54, 0, 0] = [1.5, .8, .3, .05]
    velocity[150:154, 0, 0] = [-1.5, -.8, -.3, -.05]
    body_pos = np.zeros((steps, 1, 1, 3))
    body_pos[:, 0, 0, :2] = np.cumsum(velocity[:, 0, :2], axis=0) * dt
    body_pos[:, 0, 0, 2] = .5
    gravity = np.tile([np.sin(.1), 0., -np.cos(.1)], (steps, 1, 1))
    data = dict(command=command, velocity=velocity, angular_velocity=np.zeros_like(command),
                body_pos=body_pos, gravity=gravity, joint_pos=np.zeros((steps, 1, 16)),
                target=np.zeros((steps, 1, 16)), requested=np.zeros((steps, 1, 12)),
                torque=np.zeros((steps, 1, 16)), speed_peak=np.zeros((steps, 1, 16)),
                root_clearance=np.full((steps, 1), .5), base_contact=np.zeros((steps, 1)),
                active=np.ones((steps, 1), dtype=bool))
    return manifest, data


def events(manifest, data):
    return analyze_braking(manifest, data)['source33174']['cases'][0]['stop_events']


def test_two_actual_stop_events_and_sample_timing():
    manifest, data = fixture()
    result = events(manifest, data)
    assert [row['stop_start_step'] for row in result] == [50, 150]
    for event in result:
        assert event['entry_speed_ratio'] == 1.
        assert event['reached_command_before_stop']
        assert event['qualified_stop']
        assert event['stable_stop_time_s'] == pytest.approx(.4)
        assert event['stable_stop_confirmation_s'] == pytest.approx(.9)
        assert event['path_distance_xy_m'] == pytest.approx(.265)
        assert event['path_until_stable_stop_m'] == pytest.approx(.265)
        assert event['zero_window_observed_s'] == 5.
        assert not event['zero_window_censored']


def test_never_stopping_is_none_not_a_success():
    manifest, data = fixture()
    data['velocity'][50:100, 0, 0] = .2
    event = events(manifest, data)[0]
    assert event['stable_stop_time_s'] is None
    assert event['stable_stop_confirmation_s'] is None
    assert not event['qualified_stop']


def test_early_failure_truncates_before_later_fake_stop_and_poisoned_metrics():
    manifest, data = fixture()
    manifest['cases'][0].update(end_step=53, physical=True, terms=['bad_orientation'])
    data['joint_pos'][54:] = 99.
    result = events(manifest, data)
    assert len(result) == 1
    event = result[0]
    assert event['zero_command_end_step'] == 53
    assert event['physical_failure_during_stop']
    assert event['stable_stop_time_s'] is None
    assert not event['qualified_stop']
    assert event['joint_actual_offset_peak_abs_rad_by_joint'] == [0.]*12


def test_active_prefix_is_additional_cutoff_not_post_failure_coasting():
    manifest, data = fixture()
    data['active'][54:] = False
    report = analyze_braking(manifest, data)['source33174']['cases'][0]
    assert report['inactive_truncated']
    assert report['valid_steps'] == 54
    assert len(report['stop_events']) == 1
    assert report['stop_events'][0]['stable_stop_time_s'] is None


def test_initial_zero_is_not_braking_and_clock_does_not_fabricate_events():
    manifest, data = fixture()
    data['command'][:20] = 0.
    assert [event['stop_start_step'] for event in events(manifest, data)] == [50, 150]
    data['command'][:] = 0.
    assert events(manifest, data) == []


def test_low_actual_entry_speed_never_qualifies_as_high_speed_braking():
    manifest, data = fixture()
    data['velocity'][:100, 0, 0] = .02
    event = events(manifest, data)[0]
    assert event['entry_speed_ratio'] == pytest.approx(.01)
    assert event['stable_stop_time_s'] == 0.
    assert not event['reached_command_before_stop']
    assert not event['qualified_stop']


def test_wrong_direction_and_brief_spike_do_not_pass_last_second_tracking():
    manifest, data = fixture()
    data['velocity'][40:49, 0, 0] = -2.
    assert not events(manifest, data)[0]['reached_command_before_stop']
    data['velocity'][49, 0, 0] = -2.
    assert events(manifest, data)[0]['entry_speed_ratio'] == -1.


def test_true_joint_id_mapping_body_frame_labels_and_peaks():
    manifest, data = fixture()
    data['joint_pos'][50:100, 0, 1] = .4
    data['target'][50:100, 0, 1] = .6
    data['requested'][50:100, 0, 0] = .5
    data['torque_peak'] = np.zeros_like(data['torque'])
    data['torque_peak'][50, 0, 1] = 8.
    data['torque_peak'][50, 0, 12] = 7.
    data['speed_peak'][50, 0, 1] = 3.
    data['speed_peak'][50, 0, 12] = 4.
    data['velocity'][70, 0, 0] = -.3
    data['velocity'][70, 0, 2] = .2
    report = analyze_braking(manifest, data)['source33174']
    event = report['cases'][0]['stop_events'][0]
    assert report['requested_joint_names'][0] == 'fr_hip_joint'
    assert event['joint_actual_offset_peak_abs_rad_by_joint'][0] == .4
    assert event['joint_target_offset_peak_abs_rad_by_joint'][0] == .6
    assert event['joint_requested_offset_peak_abs_rad_by_joint'][0] == .5
    assert event['posture_tilt_rms_rad'] == pytest.approx(.1)
    assert event['posture_tilt_peak_rad'] == pytest.approx(.1)
    assert event['reverse_rebound_body_projection_peak_m_s'] == .3
    assert event['vertical_velocity_body_z_peak_abs_m_s'] == .2
    assert (event['leg_torque_abs_peak_nm'], event['wheel_torque_abs_peak_nm']) == (8., 7.)
    assert (event['leg_speed_abs_peak_rad_s'], event['wheel_speed_abs_peak_rad_s']) == (3., 4.)
    assert 'not world' in report['coordinate_notes']['angular_integral']


def test_pure_yaw_still_requires_yaw_stop_and_tracks_signed_ratio():
    manifest, data = fixture()
    data['command'][:, 0, 2] = data['command'][:, 0, 0]
    data['command'][:, 0, 0] = 0.
    data['angular_velocity'][:, 0, 2] = data['velocity'][:, 0, 0]
    data['velocity'][:] = 0.
    first, second = events(manifest, data)
    assert first['entry_speed_ratio_components'] == {'yaw_signed': 1.}
    assert second['entry_speed_ratio_components'] == {'yaw_signed': 1.}
    assert first['reverse_rebound_body_projection_peak_m_s'] is None
    assert first['body_z_angular_velocity_integral_rad'] > 0.
    assert second['body_z_angular_velocity_integral_rad'] < 0.


def test_control_peaks_do_not_include_previous_nonzero_command_interval():
    manifest, data = fixture()
    for key in ('torque', 'speed_peak', 'requested', 'target'):
        data[key][49] = 99.
    event = events(manifest, data)[0]
    assert event['leg_torque_abs_peak_nm'] == 0.
    assert event['leg_speed_abs_peak_rad_s'] == 0.
    assert event['joint_requested_offset_peak_abs_rad_by_joint'] == [0.]*12
    assert event['joint_target_offset_peak_abs_rad_by_joint'] == [0.]*12


def test_non_physical_termination_cannot_be_declared_braking_success():
    manifest, data = fixture()
    manifest['cases'][0].update(end_step=90, physical=False, terms=['bad_orientation'])
    event = events(manifest, data)[0]
    assert event['stable_stop_time_s'] is not None
    assert event['terminal_failure_during_stop']
    assert not event['qualified_stop']


def test_exact_half_second_span_and_equality_thresholds():
    manifest, data = fixture()
    manifest['cases'][0]['end_step'] = 54
    data['velocity'][50:55, 0, 0] = 0.
    assert events(manifest, data)[0]['stable_stop_time_s'] is None
    manifest['cases'][0]['end_step'] = 55
    assert events(manifest, data)[0]['stable_stop_time_s'] == pytest.approx(.1)
    data['velocity'][50:56, 0, 0] = .1
    assert events(manifest, data)[0]['stable_stop_time_s'] is None


def test_real_manifest_npz_cli_path_writes_each_policy_separately(tmp_path):
    manifest, data = fixture()
    manifest['history'].append(dict(label='candidate', path='candidate.pt', teacher=False))
    manifest['cases'].append(dict(manifest['cases'][0], id=1, policy=1))
    data = {key: np.repeat(value, 2, axis=1) for key, value in data.items()}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    np.savez_compressed(tmp_path / 'traces.npz', **data)
    outputs = write_reports(tmp_path)
    assert {path.name for path in outputs} == {'BRAKE_METRICS_source33174.json', 'BRAKE_METRICS_candidate.json'}
    assert [json.loads(path.read_text())['cases'][0]['id'] for path in outputs] == [0, 1]


def test_nonfinite_and_bad_end_step_fail_explicitly():
    manifest, data = fixture()
    manifest['cases'][0]['end_step'] = 250
    with pytest.raises(ValueError, match='end_step'):
        events(manifest, data)
    manifest['cases'][0]['end_step'] = 249
    data['velocity'][50, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite velocity'):
        events(manifest, data)


def test_optional_world_vertical_velocity_is_distinct_or_explicitly_missing():
    manifest, data = fixture()
    data['velocity'][50, 0, 2] = .2
    assert events(manifest, data)[0]['vertical_velocity_world_z_peak_abs_m_s'] is None
    data['world_velocity'] = np.zeros_like(data['velocity'])
    data['world_velocity'][50, 0, 2] = -.4
    event = events(manifest, data)[0]
    assert event['vertical_velocity_body_z_peak_abs_m_s'] == .2
    assert event['vertical_velocity_world_z_peak_abs_m_s'] == .4
