from types import SimpleNamespace

import pytest

from evaluation_reset_events import retain_deterministic_reset_events


def config(profile='wave_surface_v1', event=True):
    return SimpleNamespace(wave_reset_profile=profile, events=SimpleNamespace(
        reset_base=object(), reset_robot_joints=object(), push=object(),
        wave_surface=SimpleNamespace(mode='reset') if event else None))


def test_preserves_wave_geometry_but_disables_push():
    cfg = config()
    wave = cfg.events.wave_surface
    base = cfg.events.reset_base
    retain_deterministic_reset_events(cfg)
    assert cfg.events.wave_surface is wave
    assert cfg.events.reset_base is base
    assert cfg.events.push is None


def test_original_profile_does_not_inject_correction():
    cfg = config('original', False)
    retain_deterministic_reset_events(cfg)
    assert cfg.events.wave_surface is None
    assert cfg.events.reset_robot_joints is not None


def test_missing_configured_correction_fails_closed():
    with pytest.raises(ValueError, match='must remain active'):
        retain_deterministic_reset_events(config(event=False))


def test_wrong_event_mode_rejected():
    cfg = config()
    cfg.events.wave_surface.mode = 'interval'
    with pytest.raises(ValueError, match='must remain active'):
        retain_deterministic_reset_events(cfg)
