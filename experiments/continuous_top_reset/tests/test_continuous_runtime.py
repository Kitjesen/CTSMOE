from __future__ import annotations

import sys
import types
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


CODE = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE))

runtime = importlib.import_module("continuous_runtime")


def install_geometry(monkeypatch, tmp_path, metadata=None):
    source = tmp_path / "route_geometry.py"
    source.write_text("# stable geometry source\n")
    module = types.ModuleType("route_geometry")
    module.__file__ = str(source)
    base = {"spawn": (-3.5, 0., 0.), "final_obstacle_x": 2.8,
            "finish_x": 3.2, "exit_bound_x": 4.2}
    if metadata:
        base.update(metadata)
    module.mesh_for = lambda family, level: (f"{family}:{level}", dict(base))
    monkeypatch.setitem(sys.modules, "route_geometry", module)
    runtime.metadata_for.cache_clear()
    return module


def install_isaac_stubs(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    terrains = types.ModuleType("isaaclab.terrains")

    class TerminationTermCfg:
        def __init__(self, func, time_out):
            self.func, self.time_out = func, time_out

    class SubTerrainBaseCfg:
        def __init__(self, function, proportion):
            self.function, self.proportion = function, proportion

    managers.TerminationTermCfg = TerminationTermCfg
    terrains.SubTerrainBaseCfg = SubTerrainBaseCfg
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrains)


def test_configure_inserts_exact_six_columns_and_preserves_legacy(monkeypatch, tmp_path):
    install_geometry(monkeypatch, tmp_path)
    install_isaac_stubs(monkeypatch)

    class Generator:
        pass

    class Command:
        pass

    legacy = {f"old_{i}": NS(function=object(), proportion=.5 / 8) for i in range(8)}
    pit = NS(function=object(), proportion=.5)
    legacy["pit"] = pit
    old_objects = list(legacy.values())[:-1]

    def old_reset(*args, **kwargs):
        pass

    def inherited_reset(*args, **kwargs):
        pass

    def old_course(*args, **kwargs):
        pass

    def boundary_course(*args, **kwargs):
        pass

    terrain = NS(num_cols=40, num_rows=22, size=(9., 9.), sub_terrains=legacy,
                 class_type=Generator)
    cfg = NS(
        scene=NS(terrain=NS(terrain_generator=terrain)),
        commands=NS(base_velocity=NS(class_type=Command, terrain_max_command_ranges={})),
        events=NS(reset_base=NS(func=inherited_reset, params={"original": old_reset})),
        curriculum=NS(terrain_levels=NS(func=boundary_course, params={"original": old_course})),
        terminations=NS(continuous_route_done=None),
    )
    receipt = runtime.configure(cfg)

    assert list(terrain.sub_terrains) == [*(f"old_{i}" for i in range(8)),
                                          *runtime.FAMILIES, "pit"]
    assert list(terrain.sub_terrains.values())[:8] == old_objects
    assert [terrain.sub_terrains[name].proportion for name in runtime.FAMILIES] == [.025] * 6
    assert [terrain.sub_terrains[name].function for name in runtime.FAMILIES] == [
        runtime.TERRAIN_FUNCTIONS[name] for name in runtime.FAMILIES]
    assert terrain.sub_terrains["pit"] is pit and pit.proportion == .35
    assert terrain.continuous_routes_profile == runtime.PROFILE
    assert issubclass(terrain.class_type, Generator)
    assert issubclass(cfg.commands.base_velocity.class_type, Command)
    assert cfg.events.reset_base.params == {
        "original": old_reset, "continuous_original": inherited_reset}
    assert cfg.curriculum.terrain_levels.params == {
        "original": old_course, "continuous_original": boundary_course}
    assert cfg.terminations.continuous_route_done.time_out is True
    assert receipt["columns"] == runtime.COLUMNS
    for name in runtime.FAMILIES:
        assert cfg.commands.base_velocity.terrain_max_command_ranges[name] == {
            "lin_vel_x": (.4, .9), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}


def test_manager_function_signatures_exactly_match_configured_named_params():
    reset = inspect.signature(runtime.reset_root).parameters
    assert list(reset) == ["env", "env_ids", "pose_range", "velocity_range", "asset_cfg",
                           "original", "continuous_original"]
    assert all(item.kind is not inspect.Parameter.VAR_KEYWORD for item in reset.values())
    course = inspect.signature(runtime.curriculum).parameters
    assert list(course) == ["env", "env_ids", "original", "continuous_original"]
    assert all(item.kind is not inspect.Parameter.VAR_KEYWORD for item in course.values())


@pytest.mark.parametrize("family", runtime.FAMILIES)
def test_family_specific_terrain_function_survives_config_copy(monkeypatch, tmp_path, family):
    module = install_geometry(monkeypatch, tmp_path)
    calls = []
    module.mesh_for = lambda received, level: (
        calls.append((received, level)) or object(),
        {"spawn": (-3.5, 0., 0.), "final_obstacle_x": 2.8,
         "finish_x": 3.2, "exit_bound_x": 4.2},
    )
    # Isaac copies only declared configclass fields.  The function itself must
    # therefore carry family identity and work with a field-only copied cfg.
    copied_cfg = NS(size=(9., 9.), function=runtime.TERRAIN_FUNCTIONS[family])
    meshes, origin = copied_cfg.function(.95, copied_cfg)
    assert len(meshes) == 1
    assert calls == [(family, 9)]
    np.testing.assert_allclose(origin, (-3.5, 0., 0.))


def test_geometry_contract_and_generator_only_replace_new_columns(monkeypatch, tmp_path):
    install_geometry(monkeypatch, tmp_path)
    calls = []

    class Parent:
        def _add_sub_terrain(self, mesh, origin, row, col, cfg):
            calls.append((mesh, np.asarray(origin), row, col, cfg))
            return "added"

    class Generator(runtime.ContinuousTerrainMixin, Parent):
        def __init__(self):
            self.cfg = NS()

    generator = Generator()
    sentinel = object()
    assert generator._add_sub_terrain(sentinel, (1., 2., 3.), 17, 3, "old") == "added"
    assert calls[-1][0] is sentinel
    generator._add_sub_terrain(sentinel, (1., 2., 3.), 17, 20, "new")
    assert calls[-1][0] == "continuous_curbs:9"
    np.testing.assert_allclose(calls[-1][1], (-3.5, 0., 0.))
    assert generator.cfg.continuous_route_receipts[-1]["level"] == 9

    install_geometry(monkeypatch, tmp_path, {"spawn": (0., 0., 0.)})
    with pytest.raises(ValueError, match="spawn"):
        runtime._validated_geometry("continuous_curbs", 0)


def test_direct_route_logic_requires_whole_body_finish_tracking_and_separates_boundary():
    success, boundary = runtime.evaluate_route_logic(
        torch.tensor([3.0, 2.7, 3.0, 3.0]),
        torch.tensor([3.8, 3.8, 3.8, 4.46]),
        torch.tensor([3.3, 3.3, 3.3, 4.0]),
        torch.tensor([0.2, 0.2, 0.2, 0.2]),
        torch.full((4,), 2.8), torch.full((4,), 3.2), torch.full((4,), 4.2),
        torch.tensor([True, True, False, True]),
    )
    assert success.tolist() == [True, False, False, False]
    assert boundary.tolist() == [False, False, False, True]


def test_command_resample_holds_forward_only_speed_for_five_seconds(monkeypatch, tmp_path):
    install_geometry(monkeypatch, tmp_path)

    class Parent:
        def __init__(self, cfg, env):
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self.commands = torch.full((env.num_envs, 3), -7.)
            self.time_left = torch.zeros(env.num_envs)

        def _resample(self, env_ids):
            self.commands[env_ids] = -1.

        def reset(self, env_ids=None):
            ids = torch.arange(self.num_envs) if env_ids is None else env_ids
            self._resample(ids)
            return {"parent_reset": len(ids)}

        def _update_command(self):
            pass

        def _update_metrics(self):
            pass

        def snapshot(self):
            return {}

    class Command(runtime.ContinuousCommandMixin, Parent):
        pass

    terrain = NS(terrain_types=torch.tensor([20, 25, 0]),
                 terrain_levels=torch.tensor([0, 9, 3]))
    env = NS(num_envs=3, device="cpu", scene=NS(terrain=terrain),
             episode_length_buf=torch.zeros(3, dtype=torch.long))
    command = Command(NS(), env)
    command._resample(torch.arange(3))
    assert bool(((command.commands[:2, 0] >= .4) & (command.commands[:2, 0] <= .9)).all())
    assert command.commands[:2, 1:].eq(0).all()
    assert command.commands[2].tolist() == [-1., -1., -1.]
    assert command.time_left[:2].tolist() == [5., 5.]
    assert command.continuous_final_obstacle_x[:2].tolist() == pytest.approx([2.8, 2.8])
    env.episode_length_buf[:] = torch.tensor([10, 10, 0])
    command.continuous_tracking_sum[:2] = 3.
    command.continuous_success[0] = True
    command.continuous_boundary[1] = True
    command._resample(torch.arange(3))
    assert command.continuous_tracking_sum[:2].eq(3.).all()
    assert command.continuous_success[0] and command.continuous_boundary[1]

    # Isaac resets command terms while episode_length_buf still holds the
    # terminal non-zero length.  Explicit reset, not _resample, clears latches.
    result = command.reset(slice(0, 2))
    assert result == {"parent_reset": 2}
    assert command.continuous_tracking_sum[:2].eq(0.).all()
    assert not command.continuous_success[:2].any()
    assert not command.continuous_boundary[:2].any()
    command.continuous_success[:] = True
    command.reset(None)
    assert not command.continuous_success.any()


def test_reset_wrapper_preserves_manager_params_and_fixes_safe_spawn_xy(monkeypatch):
    events = types.ModuleType("robot_lab.tasks.go2.mdp.events")
    reset_calls = []
    events.reset_root_state_uniform = lambda env, ids, pose, velocity, asset: reset_calls.append(
        (ids.clone(), pose, velocity, asset))
    modules = {
        "robot_lab": types.ModuleType("robot_lab"),
        "robot_lab.tasks": types.ModuleType("robot_lab.tasks"),
        "robot_lab.tasks.go2": types.ModuleType("robot_lab.tasks.go2"),
        "robot_lab.tasks.go2.mdp": types.ModuleType("robot_lab.tasks.go2.mdp"),
        "robot_lab.tasks.go2.mdp.events": events,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    inherited = []
    leaf = []

    def original(env, ids, pose, velocity, asset):
        leaf.append((ids, pose, velocity, asset))

    def current(env, ids, pose, velocity, asset, original):
        inherited.append(ids)
        original(env, ids, pose, velocity, asset)

    terrain = NS(terrain_types=torch.tensor([20, 0]))
    env = NS(device="cpu", scene=NS(terrain=terrain))
    pose = {"x": (-.5, .5), "y": (-.5, .5), "z": (0., 0.1), "yaw": (-3., 3.)}
    velocity, asset = {"x": (0., 0.)}, object()
    runtime.reset_root(env, torch.tensor([0, 1]), pose, velocity, asset,
                       continuous_original=current, original=original)

    assert len(inherited) == len(leaf) == 1
    ids, route_pose, passed_velocity, passed_asset = reset_calls[0]
    assert ids.tolist() == [0]
    assert route_pose["x"] == route_pose["y"] == (0., 0.)
    assert route_pose["z"] == (0., .1)
    assert route_pose["yaw"] == pytest.approx((-np.deg2rad(5), np.deg2rad(5)))
    assert passed_velocity is velocity and passed_asset is asset
    # Root spawn is one metre inside the rear cell edge before body extent.
    assert -3.5 - (-4.5) == pytest.approx(1.)


def test_curriculum_excludes_new_columns_from_legacy_and_writes_levels_directly():
    levels = torch.tensor([5, 5, 5, 0])
    columns = torch.tensor([0, 20, 21, 22])
    origins = torch.zeros((10, 40, 3))
    origins[..., 0] = torch.arange(10)[:, None]
    terrain = NS(terrain_levels=levels, terrain_types=columns,
                 terrain_origins=origins, env_origins=torch.zeros((4, 3)))
    command = NS(
        continuous_success=torch.tensor([False, True, False, False]),
        continuous_boundary=torch.tensor([False, False, True, False]),
        continuous_tracking_sum=torch.zeros((4, 3)),
        continuous_tracking_count=torch.ones(4),
        continuous_totals=torch.zeros((6, 10, 5), dtype=torch.long),
        pit=NS(active=torch.zeros(4, dtype=torch.bool)),
    )
    env = NS(
        device="cpu", scene=NS(terrain=terrain, env_origins=torch.zeros((4, 3))),
        episode_length_buf=torch.ones(4, dtype=torch.long),
        command_manager=NS(get_term=lambda name: command),
        termination_manager=NS(terminated=torch.zeros(4, dtype=torch.bool)),
    )
    delegated = []

    def old_course(_env, ids):
        delegated.extend(ids.tolist())
        return torch.tensor(12.)

    def boundary_course(_env, ids, original):
        return original(_env, ids)

    result = runtime.curriculum(
        env, torch.arange(4), original=old_course, continuous_original=boundary_course)
    assert delegated == [0]
    assert levels.tolist() == [5, 6, 4, 0]
    assert terrain.env_origins[1:, 0].tolist() == [6., 4., 0.]
    assert env.scene.env_origins[1:, 0].tolist() == [6., 4., 0.]
    assert command.continuous_totals.sum((0, 1)).tolist() == [3, 1, 1, 0, 1]
    assert result.item() == pytest.approx(5.)


def test_checkpoint_cumulative_statistics_restore_exactly(monkeypatch):
    monkeypatch.setattr(runtime, "geometry_source_sha256", lambda: "abc")
    source = NS(continuous_totals=torch.arange(300).reshape(6, 10, 5))
    payload = runtime.checkpoint_payload(source)
    target = NS(device="cpu", continuous_totals=torch.zeros((6, 10, 5), dtype=torch.long))
    receipt = runtime.restore_checkpoint(payload, target)
    torch.testing.assert_close(target.continuous_totals, source.continuous_totals, rtol=0, atol=0)
    assert receipt["continuous_totals"] == "restored exactly"

    bad = dict(payload, continuous_geometry_sha256="changed")
    with pytest.raises(ValueError, match="geometry source changed"):
        runtime.restore_checkpoint(bad, target)
