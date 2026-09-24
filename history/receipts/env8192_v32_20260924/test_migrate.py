import torch
import pytest
from migrate import ENV_FIELDS, migrate, exact


def source():
    return dict(terrain_levels=torch.arange(4096) % 10, terrain_types=torch.arange(4096) % 20,
                mixed_stair_course=dict(num_envs=4096, episode_totals=torch.ones(2, 3),
                                        **{k: torch.zeros(4096) for k in ENV_FIELDS}),
                terrain_allocation={'pending': {}, 'stats': torch.ones(9, 22)},
                model_state_dict={'weight': torch.ones(4, 4)}, optimizer_state_dict={'step': 55},
                optimizer_stu_enc_state_dict={'step': 55}, completed_updates=43999)


def test_double_without_reset_or_counter_inflation():
    s = source()
    out = migrate(s, .75)
    assert out['terrain_levels'].shape == (8192,)
    assert out['mixed_stair_course']['num_envs'] == 8192
    exact(out['terrain_allocation'], s['terrain_allocation'])
    exact(out['model_state_dict'], s['model_state_dict'])
    assert s['terrain_levels'].shape == (4096,)


def test_unknown_environment_state_rejected():
    s = source()
    s['unknown'] = torch.zeros(4096, 2)
    with pytest.raises(ValueError, match='Unhandled'):
        migrate(s, .75)


def test_pending_rejected():
    s = source()
    s['terrain_allocation']['pending'] = {1: 2}
    with pytest.raises(ValueError, match='pending'):
        migrate(s, .75)
