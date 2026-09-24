import numpy as np
from focus_protocol import stair_window_metrics


def test_orientation_failure_is_not_reported_as_platform_success():
    names = [f'{side}_{joint}_joint' for side in ('fl', 'fr', 'rl', 'rr')
             for joint in ('hip', 'thigh', 'calf')]
    case = dict(id=0, policy=0, terrain='stairs_up', level=9, landing_mode='stop', end_step=9,
                physical=True, terms=['bad_orientation'], end_reason='physical_termination',
                platform_hold_s=.04, supported_platform_reached=True)
    manifest = dict(dt=.02, default_q=[0]*12, joint_names=names, cases=[case])
    scalar = np.zeros((10, 1))
    arrays = {name: scalar.copy() for name in ('stair_platform_hull', 'stair_edge_window',
              'phase_weight', 'phase_stable_s', 'stair_platform_supported')}
    arrays['joint_pos'] = np.zeros((10, 1, 12))
    arrays['world_velocity'] = np.zeros((10, 1, 3))
    arrays['stair_platform_hull'][4:, 0] = 1
    arrays['stair_platform_supported'][4:6, 0] = 1
    result = stair_window_metrics(manifest, arrays)['cases'][0]
    assert result['actual_hull_reached_platform']
    assert result['physical']
    assert result['end_reason'] == 'physical_termination'
    assert result['terms'] == ['bad_orientation']
    assert result['platform_hold_s'] == .04
    assert result['supported_platform_steps'] == 2
