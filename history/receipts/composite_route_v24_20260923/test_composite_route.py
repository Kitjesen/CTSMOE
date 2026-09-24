"""CPU geometry and acceptance checks; no Isaac/Omniverse runtime required."""

from types import SimpleNamespace

import numpy as np
import pytest

from composite_route import (
    COMPACT_9X9,
    IDEAL_12X9,
    build_route_mesh,
    composite_terrain_12x9,
    composite_terrain_9x9,
    profile_stations,
    world_to_tile_xy,
)
from route_acceptance import AcceptanceSpec, RouteJudge, route_section


@pytest.mark.parametrize("route", [IDEAL_12X9, COMPACT_9X9])
def test_geometry_exact_elevation_and_non_degenerate_faces(route):
    meshes, origin = build_route_mesh(route)
    assert len(meshes) == 1
    mesh = meshes[0]
    stations = profile_stations(route)
    assert np.array_equal(stations[[0, -1], 0], [0, route.size_x_m])
    assert np.array_equal(origin, [route.spawn_x_m, route.size_y_m / 2, 0])
    assert np.array_equal(mesh.bounds[:, :2], [[0, 0], [route.size_x_m, route.size_y_m]])

    # Slope joins both flat platforms at identical elevations; its measured
    # gradient is 10 degrees and the staircase has exactly four vertical risers.
    assert np.isclose(stations[1, 1], 0)
    assert np.isclose(stations[2, 1], route.slope_top_m)
    assert np.isclose(stations[3, 1], stations[2, 1])
    assert np.isclose(
        np.degrees(np.arctan2(stations[2, 1] - stations[1, 1],
                              stations[2, 0] - stations[1, 0])),
        10.0,
    )
    dx, dz = np.diff(stations[:, 0]), np.diff(stations[:, 1])
    risers = (dx == 0) & (dz > 0)
    assert risers.sum() == 4
    assert np.allclose(dz[risers], 0.08)
    assert np.isclose(stations[-2, 0], route.final_start_x_m)
    assert np.isclose(stations[-2, 1], route.final_height_m)
    assert np.isclose(stations[-1, 1], route.final_height_m)
    assert route.final_usable_length_m >= 1.5

    assert np.all(mesh.area_faces > 0)
    assert (mesh.face_normals[:, 2] >= -1e-12).all()
    # The risers face the robot coming from -X rather than becoming ramps.
    assert (mesh.face_normals[:, 0] < -0.9).sum() == 4 * 2 * 2


def test_both_isaac_function_contracts_are_size_specific():
    for function, route in ((composite_terrain_12x9, IDEAL_12X9),
                            (composite_terrain_9x9, COMPACT_9X9)):
        meshes, origin = function(0.6, SimpleNamespace(size=(route.size_x_m, route.size_y_m)))
        assert len(meshes) == 1
        assert origin[0] == route.spawn_x_m
        with pytest.raises(ValueError, match="size"):
            function(0.6, SimpleNamespace(size=(1.0, 1.0)))
        with pytest.raises(ValueError, match="Difficulty"):
            function(float("nan"), SimpleNamespace(size=(route.size_x_m, route.size_y_m)))


def test_world_coordinate_adapter_uses_spawn_origin_not_robot_yaw():
    origin_world = np.asarray((100.0, -20.0))
    points = np.asarray(((100.0, -20.0), (105.0, -20.0)))
    local = world_to_tile_xy(points, origin_world, COMPACT_9X9)
    assert np.allclose(local, ((1.1, 4.5), (6.1, 4.5)))


def _snapshot(route, x, *, support=True, spread=0.08):
    wheels = np.asarray(((x - spread, 4.2), (x + spread, 4.2),
                         (x - spread, 4.8), (x + spread, 4.8)))
    body = np.asarray(((x - spread, 4.1), (x + spread, 4.1),
                       (x - spread, 4.9), (x + spread, 4.9)))
    return dict(
        wheel_xy=wheels, body_hull_xy=body,
        wheel_support=np.full(4, support),
        translation_speed_mps=0.04, yaw_rate_rps=0.02, body_tilt_rad=0.03,
        dt_s=0.05,
    )


@pytest.mark.parametrize("route", [IDEAL_12X9, COMPACT_9X9])
def test_full_traversal_needs_section_contact_all_body_and_two_second_dwell(route):
    judge = RouteJudge(route)
    # The old 4 m radius is inside a transition or stair; it cannot pass here.
    assert route_section(4.0, route) != "final_platform"
    assert not judge.update(**_snapshot(route, 4.0))["success"]
    final_x = route.final_start_x_m + 0.2
    for _ in range(41):
        status = judge.update(**_snapshot(route, final_x))
    assert not status["success"]  # jumping straight to the platform is not the route
    assert status["continuous_support_s"] == 0

    judge.reset()
    judge.update(**_snapshot(route, (route.approach_end_x_m + route.slope_end_x_m) / 2))
    judge.update(**_snapshot(route, (route.transition_end_x_m + route.final_start_x_m) / 2))
    assert judge.seen_slope_support and judge.seen_stair_support
    too_far_back = _snapshot(route, route.final_start_x_m + 0.10)
    assert not judge.update(**too_far_back)["all_wheels_and_body_on_final_platform"]
    for _ in range(39):
        status = judge.update(**_snapshot(route, final_x))
    assert not status["success"]
    status = judge.update(**_snapshot(route, final_x))
    assert status["success"]
    assert status["continuous_support_s"] == pytest.approx(2.0)
    assert status["section"] == "final_platform"


def test_support_interruption_motion_and_physical_failure_reset_or_block_dwell():
    route = COMPACT_9X9
    judge = RouteJudge(route, AcceptanceSpec(dwell_required_s=0.2))
    judge.update(**_snapshot(route, 3.0))
    judge.update(**_snapshot(route, 5.1))
    target = route.final_start_x_m + 0.2
    ready = _snapshot(route, target)
    judge.update(**ready)
    moving = dict(ready, translation_speed_mps=0.5)
    assert judge.update(**moving)["continuous_support_s"] == 0
    judge.update(**ready)
    lost_contact = dict(ready, wheel_support=np.asarray((True, True, True, False)))
    assert judge.update(**lost_contact)["continuous_support_s"] == 0
    for _ in range(3):
        assert not judge.update(**ready)["success"]
    assert judge.update(**ready)["success"]

    judge.reset()
    judge.update(**_snapshot(route, 3.0))
    judge.update(**_snapshot(route, 5.1))
    assert not judge.update(**dict(ready, physical_failure=True))["success"]
    for _ in range(5):
        assert not judge.update(**ready)["success"]


def test_lateral_border_prevents_neighbor_tile_credit():
    route = COMPACT_9X9
    judge = RouteJudge(route)
    judge.update(**_snapshot(route, 3.0))
    judge.update(**_snapshot(route, 5.1))
    far_side = _snapshot(route, route.final_start_x_m + 0.2)
    far_side["wheel_xy"][:, 1] = 8.9
    assert not judge.update(**far_side)["all_wheels_and_body_on_final_platform"]


def test_gpu_batch_dwell_matches_cpu_contract():
    torch = pytest.importorskip("torch")
    from route_acceptance import update_dwell_torch

    route = COMPACT_9X9
    xy = _snapshot(route, route.final_start_x_m + 0.2)
    wheel = torch.tensor(np.stack((xy["wheel_xy"], xy["wheel_xy"])), dtype=torch.float32)
    body = torch.tensor(np.stack((xy["body_hull_xy"], xy["body_hull_xy"])), dtype=torch.float32)
    output = update_dwell_torch(
        route=route, acceptance=AcceptanceSpec(dwell_required_s=0.1),
        prior_dwell_s=torch.tensor([0.05, 0.05]), wheel_xy=wheel,
        body_hull_xy=body, wheel_support=torch.ones((2, 4), dtype=torch.bool),
        seen_slope_support=torch.tensor([True, True]),
        seen_stair_support=torch.tensor([True, False]),
        translation_speed_mps=torch.zeros(2), yaw_rate_rps=torch.zeros(2),
        body_tilt_rad=torch.zeros(2), physical_failure=torch.zeros(2, dtype=torch.bool),
        dt_s=0.05,
    )
    assert output["success"].tolist() == [True, False]
    assert output["dwell_s"].tolist() == pytest.approx([0.1, 0])
