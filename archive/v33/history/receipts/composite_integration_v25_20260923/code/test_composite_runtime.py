"""CPU contract tests for the Isaac-facing route adapter's command and gate."""

import math
from types import SimpleNamespace

import pytest
import torch

from composite_route import COMPACT_9X9
from composite_runtime import (WHEEL_RADIUS_M, CompositeRuntime,
                               terrain_contact_forces, terrain_wheel_support)
from train import validate_source_checkpoint_profiles


def _sample_hull(_joint_pos, root_state):
    xy = root_state[:, None, :2]
    corners = torch.tensor(((-.25, -.2), (-.25, .2), (.25, -.2), (.25, .2)))
    return torch.cat((xy + corners, torch.zeros(len(root_state), 4, 1)), dim=-1)


def _fixture():
    data = SimpleNamespace(
        body_pos_w=torch.zeros(1, 4, 3), root_pos_w=torch.zeros(1, 3),
        root_state_w=torch.zeros(1, 13), root_lin_vel_w=torch.zeros(1, 3),
        root_ang_vel_w=torch.zeros(1, 3), projected_gravity_b=torch.tensor(((0., 0., -1.),)),
        joint_pos=torch.zeros(1, 16), default_joint_pos=torch.zeros(1, 16),
    )
    ground_contacts = [SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=torch.tensor([[[[0., 0., 10.]]]]))) for _ in range(4)]
    pit = SimpleNamespace(wheel_ids=torch.arange(4), ground_contacts=ground_contacts,
                          base_contact_ids=torch.tensor((0,)),
                          contact=SimpleNamespace(data=SimpleNamespace(
                              net_forces_w_history=torch.zeros(1, 1, 1, 3))),
                          geometry=SimpleNamespace(
                              world_points=_sample_hull,
                              wheel_points=torch.tensor((True, True, False, False))))
    origins = torch.zeros(1, 3)
    terrain = SimpleNamespace(terrain_levels=torch.zeros(1, dtype=torch.long),
                              terrain_types=torch.tensor((20,)),
                              terrain_origins=torch.zeros(22, 42, 3), env_origins=origins)
    env = SimpleNamespace(device="cpu", num_envs=1, common_step_counter=0,
                          step_dt=.02, episode_length_buf=torch.ones(1, dtype=torch.long),
                          max_episode_length=1250,
                          scene=SimpleNamespace(env_origins=origins, terrain=terrain),
                          actuator_guard=SimpleNamespace(violation=torch.zeros(1, dtype=torch.bool)),
                          termination_manager=SimpleNamespace(
                              terminated=torch.zeros(1, dtype=torch.bool)))
    command = SimpleNamespace(terrain_type2idx={"composite": 2},
                              terrain_idxs=torch.tensor((2,)), pit=pit,
                              robot=SimpleNamespace(data=data),
                              commands=torch.zeros(1, 3),
                              last_is_limit_vel=torch.zeros(1, dtype=torch.bool),
                              time_left=torch.zeros(1), learner=torch.tensor((1,)))
    return CompositeRuntime(env, command), data


def _advance(runtime, data, root_local_x, speed):
    world_x = root_local_x - COMPACT_9X9.spawn_x_m
    data.root_pos_w[0, 0] = world_x
    data.root_state_w[0, 0] = world_x
    data.body_pos_w[0, :, 0] = world_x + torch.tensor((-.25, .25, -.25, .25))
    top = COMPACT_9X9.slope_top_m
    for wheel in range(4):
        x = float(data.body_pos_w[0, wheel, 0]) + COMPACT_9X9.spawn_x_m
        if x < COMPACT_9X9.approach_end_x_m:
            height = 0.
        elif x < COMPACT_9X9.slope_end_x_m:
            height = (x - COMPACT_9X9.approach_end_x_m) * math.tan(math.radians(10.))
        elif x < COMPACT_9X9.transition_end_x_m:
            height = top
        elif x < COMPACT_9X9.final_start_x_m:
            height = top + (int((x - COMPACT_9X9.transition_end_x_m)
                                / COMPACT_9X9.stair_tread_m) + 1) * COMPACT_9X9.stair_height_m
        else:
            height = COMPACT_9X9.final_height_m
        data.body_pos_w[0, wheel, 2] = height + WHEEL_RADIUS_M
    data.root_lin_vel_w[0, 0] = speed
    runtime.env.common_step_counter += 1
    runtime.update_command()


def test_route_requires_slope_stairs_whole_body_and_two_second_stop():
    route, data = _fixture()
    route.held_speed[0] = .5
    route.needs_speed[0] = False
    _advance(route, data, 2.8, .5)
    assert route.course.seen_slope[0]
    assert not route.course.seen_stair[0]
    assert route.command.commands[0, 0] == .5

    _advance(route, data, 5.45, .5)
    assert route.course.seen_stair[0]
    _advance(route, data, 6.25, .5)
    assert not route.course.full_body_platform[0]  # rear collision surface is short
    assert not route.stop_on_platform[0]
    _advance(route, data, 7.1, 0.)
    assert route.course.full_body_platform[0]
    assert route.stop_on_platform[0]
    assert route.command.commands[0, 0] == 0
    assert not route.success[0]
    for _ in range(99):
        _advance(route, data, 7.1, 0.)
    assert route.success[0]
    route.env.termination_manager.terminated[0] = True
    route.record_episodes(torch.tensor((0,)))
    assert route.course.totals[1].tolist() == [1, 1, 1, 1, 1, 0]
    assert route.env.scene.terrain.terrain_levels[0] == 0


def test_bounded_per_environment_grade_needs_two_complete_routes_or_failures():
    route, _ = _fixture()
    ids = torch.tensor((0,))
    terrain = route.env.scene.terrain
    for completed in (True, True):
        route.reset(ids)
        route.course.success[0] = completed
        route.record_episodes(ids)
    assert terrain.terrain_levels[0] == 1
    assert route.level_totals[1, 0, 3] == 1
    for _ in range(2):
        route.reset(ids)
        route.physical[0] = True
        route.record_episodes(ids)
    assert terrain.terrain_levels[0] == 0
    assert route.level_totals[1, 1, 4] == 1
    saved = route.state_dict()
    restored, _ = _fixture()
    restored.load_state_dict(saved, environment_state_restored=False)
    torch.testing.assert_close(restored.level_totals, route.level_totals)
    assert not restored.course.success[0]


def test_slow_time_ineligible_timeout_is_recorded_without_unfair_downgrade():
    route, _ = _fixture()
    ids = torch.tensor((0,))
    route.env.scene.terrain.terrain_levels[0] = 2
    route.env.episode_length_buf[0] = route.env.max_episode_length - 1
    route.held_speed[0] = .2
    for _ in range(2):
        route.reset(ids)
        route.record_episodes(ids)
    assert route.env.scene.terrain.terrain_levels[0] == 2
    assert route.level_totals[1, 2, 5] == 2
    route.held_speed[0] = .6
    for _ in range(2):
        route.reset(ids)
        route.record_episodes(ids)
    assert route.env.scene.terrain.terrain_levels[0] == 1
    assert route.level_totals[1, 2, 4] == 1


def test_route_loses_continuous_dwell_on_motion_and_never_uses_four_metre_exit():
    route, data = _fixture()
    _advance(route, data, 5.45, 0.)
    assert not route.success[0]
    assert not route.stop_on_platform[0]
    route.course.seen_slope[0] = True
    _advance(route, data, 7.1, 0.)
    for _ in range(45):
        _advance(route, data, 7.1, 0.)
    assert 0 < route.course.dwell_s[0] < 2
    _advance(route, data, 7.1, .2)
    assert route.course.dwell_s[0] == 0
    assert not route.success[0]


def test_horizontal_riser_push_and_wrong_height_never_count_as_support():
    route, data = _fixture()
    for sensor in route.command.pit.ground_contacts:
        sensor.data.force_matrix_w[0, 0, 0] = torch.tensor((30., 0., 0.))
    _advance(route, data, 2.8, 0.)
    assert not route.course.seen_slope[0]
    for sensor in route.command.pit.ground_contacts:
        sensor.data.force_matrix_w[0, 0, 0] = torch.tensor((30., 0., 8.))
    data.body_pos_w[0, :, 2] += .08
    route.env.common_step_counter += 1
    route.refresh()
    assert not route.course.seen_slope[0]
    _advance(route, data, 2.8, 0.)
    assert route.course.seen_slope[0]

    route.course.seen_stair[0] = False
    # The wheel centre has crossed a riser, but the radius still overlaps it.
    riser_x = COMPACT_9X9.transition_end_x_m + COMPACT_9X9.stair_tread_m
    data.body_pos_w[0, :, 0] = riser_x + .03 - COMPACT_9X9.spawn_x_m
    data.body_pos_w[0, :, 2] = (COMPACT_9X9.slope_top_m
                                  + 2 * COMPACT_9X9.stair_height_m + WHEEL_RADIUS_M)
    route.env.common_step_counter += 1
    route.refresh()
    assert not route.course.seen_stair[0]


def test_wheel_surface_witness_uses_each_environment_level_and_inside_tread():
    wheels = torch.zeros(2, 4, 3)
    wheels[..., 0] = COMPACT_9X9.transition_end_x_m + .16
    wheels[..., 1] = COMPACT_9X9.size_y_m / 2
    wheels[0, :, 2] = COMPACT_9X9.slope_top_m + .08 + WHEEL_RADIUS_M
    top_l9 = (COMPACT_9X9.slope_end_x_m - COMPACT_9X9.approach_end_x_m) * math.tan(math.radians(20.))
    wheels[1, :, 2] = top_l9 + .12 + WHEEL_RADIUS_M
    forces = torch.zeros_like(wheels)
    forces[..., 2] = 6.
    assert terrain_wheel_support(wheels, forces, torch.tensor((0, 9))).all()
    wheels[1, 0, 0] = COMPACT_9X9.transition_end_x_m + .03
    assert not terrain_wheel_support(wheels, forces, torch.tensor((0, 9)))[1, 0]


def test_outside_detects_collision_wheel_before_root_crosses_tile_border():
    route, data = _fixture()
    _advance(route, data, .6, 0.)
    root_tile_x = data.root_pos_w[0, 0] + COMPACT_9X9.spawn_x_m
    assert root_tile_x > COMPACT_9X9.tile_border_m
    # The fixture's first two collision samples are wheel surfaces; its body
    # samples still have x=root+.25 inside the tile.
    assert root_tile_x + .25 > COMPACT_9X9.tile_border_m
    assert root_tile_x - .25 < COMPACT_9X9.tile_border_m
    assert route.outside[0]
    assert not route.success[0]


def test_lateral_wheel_surface_crossing_is_also_outside():
    route, data = _fixture()
    data.root_pos_w[0, 1] = .6 - COMPACT_9X9.size_y_m / 2
    data.root_state_w[0, 1] = data.root_pos_w[0, 1]
    _advance(route, data, 2.8, 0.)
    assert route.outside[0]


def test_contact_sensor_contract_rejects_ambiguous_force_matrix():
    route, _ = _fixture()
    ids = torch.tensor((0,))
    sensors = route.command.pit.ground_contacts
    assert terrain_contact_forces(sensors, ids, 1).shape == (1, 4, 3)
    sensors[0].data.force_matrix_w = torch.zeros(1, 2, 1, 3)
    with pytest.raises(ValueError, match="terrain-filtered"):
        terrain_contact_forces(sensors, ids, 1)


def test_route_training_samples_only_time_feasible_speeds():
    route, _ = _fixture()
    ids = torch.tensor((0,))
    for _ in range(200):
        route.reset(ids)
        route.resample(ids)
        assert route.held_speed[0] >= route.minimum_training_speed_mps
        assert route.held_speed[0] < 1.0
    assert route.minimum_training_speed_mps > .2


@pytest.mark.parametrize("field", ("hip_request_profile", "mixed_stair_profile"))
def test_v25_rejects_unintegrated_v24_checkpoint_profiles(field):
    validate_source_checkpoint_profiles({})
    validate_source_checkpoint_profiles({field: "off"})
    with pytest.raises(ValueError, match=field):
        validate_source_checkpoint_profiles({field: "v24_active_profile"})
