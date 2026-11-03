import numpy as np
import pytest

from joint_assessment import window_metrics, write_joint_assessment, plot_joint_terrain_peaks


def test_large_joint_is_not_diluted_and_first_target_step_is_counted():
    result = window_metrics([.1, -.9, .2], [.5, -.4, .2], [.7, -.8, .4], 0., .02)
    assert result['actual_peak_abs_rad'] == .9
    assert result['actual_at_peak_rad'] == -.9
    assert result['target_at_actual_peak_rad'] == -.4
    assert result['request_at_actual_peak_rad'] == -.8
    assert result['target_step_peak_rad'] == .9
    assert result['actual_rms_rad'] == pytest.approx(np.sqrt(.86/3))
    first = window_metrics([.1, .2], [.8, .8], [1., 1.], 0., .02)
    assert first['target_step_peak_rad'] == .8
    assert first['target_rate_peak_rad_s'] == 40.


def test_settled_window_uses_previous_target_not_default():
    result = window_metrics([.1, .1], [.5, .5], [.5, .5], .48, .02)
    assert result['target_step_peak_rad'] == pytest.approx(.02)
    assert result['tracking_mae_rad'] == pytest.approx(.4)


def test_invalid_or_empty_data_is_not_a_pass():
    with pytest.raises(ValueError):
        window_metrics([], [], [], 0., .02)
    with pytest.raises(ValueError):
        window_metrics([np.nan], [0.], [0.], 0., .02)


def test_report_covers_all_joints_and_excludes_post_failure_frames(tmp_path):
    names = [f'leg_{i}_joint' for i in range(12)]+[f'wheel_{i}_joint' for i in range(4)]
    manifest = dict(dt=.02, joint_names=names, request_joint_ids=list(range(12)), default_q=[0.]*16,
                    history=[{'label':'synthetic_fixture'}], cases=[dict(id=0,end_step=1,policy=0,
                    teacher=False,terrain='flat',level=0,direction='forward',mode='direct')])
    data = {key:np.zeros((4,1,16)) for key in
            ('joint_pos','target','torque_peak','speed_peak','hard_excess','joint_vel')}
    data['requested'] = np.zeros((4,1,12))
    data['initial_q'] = np.zeros((1,16))
    for values in data.values():
        if values.ndim == 3:
            values[2:] = 1000.  # Invalid frames following episode end must disappear.
    data['joint_pos'][:2,0,7] = -.8
    data['target'][:2,0,7] = -.6
    rows = write_joint_assessment(tmp_path, manifest, data)
    startup = [r for r in rows if r['window']=='startup_0_1']
    assert len(startup) == 12
    assert max(r['actual_peak_abs_rad'] for r in startup) == .8
    assert all(not r['window_complete'] for r in startup)
    assert all(r['torque_peak_nm']==0 for r in rows)
    import json
    assert len(json.loads((tmp_path/'wheel_assessment.json').read_text())) == 4
    plot_joint_terrain_peaks(tmp_path, rows)
    assert (tmp_path/'synthetic_fixture_joint_terrain_peaks.png').is_file()
