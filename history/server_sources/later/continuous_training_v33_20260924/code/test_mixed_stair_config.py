"""The V24 wrapper must preserve V23's flat-boundary curriculum guard."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from mixed_stair_runtime import configure, mixed_stair_curriculum


def test_configure_keeps_boundary_wrapper_and_replaces_only_its_inner_course(monkeypatch):
    mixed = ModuleType("mixed_task")
    boundary = ModuleType("flat_boundary")
    stair = ModuleType("stair_speed")

    class MixedCommand:
        pass

    class WrappedCommand(MixedCommand):
        pass

    def mixed_curriculum():
        pass

    def boundary_safe_curriculum():
        pass

    def stair_route_exit():
        pass

    mixed.MixedCommand = MixedCommand
    mixed.mixed_curriculum = mixed_curriculum
    boundary.boundary_safe_curriculum = boundary_safe_curriculum
    stair.stair_route_exit = stair_route_exit
    for name, module in (("mixed_task", mixed), ("flat_boundary", boundary), ("stair_speed", stair)):
        monkeypatch.setitem(sys.modules, name, module)
    curriculum = SimpleNamespace(func=boundary_safe_curriculum, params={"original": mixed_curriculum})
    cfg = SimpleNamespace(
        commands=SimpleNamespace(base_velocity=SimpleNamespace(class_type=WrappedCommand)),
        curriculum=SimpleNamespace(terrain_levels=curriculum),
        terminations=SimpleNamespace(stair_route_exit=SimpleNamespace(func=stair_route_exit)),
    )
    def provider(env, ids):
        return None
    configure(cfg, provider)
    assert curriculum.func is boundary_safe_curriculum
    assert curriculum.params == {"original": mixed_stair_curriculum}
    assert issubclass(cfg.commands.base_velocity.class_type, WrappedCommand)
    assert cfg.terminations.stair_route_exit.func is not stair_route_exit

    unguarded = SimpleNamespace(
        commands=SimpleNamespace(base_velocity=SimpleNamespace(class_type=WrappedCommand)),
        curriculum=SimpleNamespace(terrain_levels=SimpleNamespace(func=mixed_curriculum, params={})),
        terminations=SimpleNamespace(stair_route_exit=SimpleNamespace(func=stair_route_exit)),
    )
    with pytest.raises(ValueError, match="flat-boundary"):
        configure(unguarded, provider)
