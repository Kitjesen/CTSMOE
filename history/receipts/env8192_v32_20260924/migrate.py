"""Explicit 4096 -> 8192 curriculum migration; model and optimizers unchanged."""
import copy
import torch

ENV_FIELDS = ('group', 'sign', 'speed_bin', 'speed', 'elapsed_s', 'desired_pose',
              'active', 'support_streak', 'platform_complete', 'tracking_error_sum',
              'tracking_count', 'pulse_count', 'pulse_response_sum', 'pulse_abs_error_sum')


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def migrate(source, teacher_ratio):
    if source['terrain_levels'].shape != (4096,) or source['mixed_stair_course']['num_envs'] != 4096:
        raise ValueError('Only explicit 4096 to 8192 migration is supported')
    period = int(1 / (1 - teacher_ratio))
    if 4096 % period or source['terrain_allocation']['pending']:
        raise ValueError('Learner mapping or pending allocation is not compatible')
    allowed = {'terrain_levels', 'terrain_types'} | {'mixed_stair_course.' + k for k in ENV_FIELDS}

    def check(value, path=''):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 4096:
            if path not in allowed:
                raise ValueError('Unhandled environment-shaped state: ' + path)
        elif isinstance(value, dict):
            for key, v in value.items():
                check(v, f'{path}.{key}' if path else str(key))
    check(source)
    result = copy.deepcopy(source)
    for name in ('terrain_levels', 'terrain_types'):
        result[name] = source[name].repeat(2)
    course = result['mixed_stair_course']
    course['num_envs'] = 8192
    for name in ENV_FIELDS:
        original = source['mixed_stair_course'][name]
        assert original.shape[0] == 4096
        course[name] = original.repeat((2,) + (1,) * (original.ndim - 1))
    # The normal resume path resets all physical episodes and in-flight commands.
    # Aggregate episode totals must remain unchanged rather than being duplicated.
    for key in source:
        if key not in ('terrain_levels', 'terrain_types', 'mixed_stair_course'):
            exact(result[key], source[key])
    torch.testing.assert_close(course['episode_totals'], source['mixed_stair_course']['episode_totals'], rtol=0, atol=0)
    for name in ('terrain_levels', 'terrain_types'):
        assert torch.equal(result[name][:4096], source[name])
        assert torch.equal(result[name][4096:], source[name])
    return result
