"""Keep deterministic geometry corrections when disabling random evaluation events."""


def retain_deterministic_reset_events(cfg):
    keep = {'reset_base', 'reset_robot_joints'}
    if cfg.wave_reset_profile == 'wave_surface_v1':
        event = getattr(cfg.events, 'wave_surface', None)
        if event is None or event.mode != 'reset':
            raise ValueError('Configured wave surface reset must remain active in evaluation')
        keep.add('wave_surface')
    elif cfg.wave_reset_profile != 'original':
        raise ValueError(f'Unsupported evaluation wave reset: {cfg.wave_reset_profile}')
    for name in vars(cfg.events):
        if name not in keep:
            setattr(cfg.events, name, None)
    return sorted(keep)
