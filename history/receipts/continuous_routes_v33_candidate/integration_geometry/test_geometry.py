import numpy as np
import pytest

from integration_geometry import FAMILIES, TerrainSpec, generate
from integration_geometry.route_geometry import mesh_for


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", range(10))
def test_every_curriculum_level_is_one_valid_cell(family, level):
    mesh, metadata = generate(TerrainSpec(family, level, seed=17))

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.body_count == 1
    assert mesh.area_faces.min() > 1e-10
    assert np.allclose(mesh.bounds[:, :2], [[-4.5, -4.5], [4.5, 4.5]])
    assert metadata["spawn"] == [-3.5, 0.0, 0.0]
    assert metadata["finish_x"] == 3.3
    assert metadata["exit_bound_x"] == 4.4
    assert metadata["final_obstacle_x"] <= 2.2
    assert metadata["collision"]["mesh_contract"].startswith("single closed")
    x, z = mesh.vertices[: len(mesh.vertices) // 2, [0, 2]].T
    assert np.allclose(z[np.isclose(x, -3.5)], 0.0)
    assert np.allclose(z[np.isclose(x, 3.3)], 0.0)

    # Every undirected edge has exactly two incident faces: no seam walls or
    # hidden open sheets can exist inside this one boundary shell.
    assert np.all(mesh.edges_unique_inverse.size > 0)
    assert np.array_equal(np.bincount(mesh.edges_unique_inverse), np.full(len(mesh.edges_unique), 2))


@pytest.mark.parametrize("family", FAMILIES)
def test_level_progression_is_low_to_high(family):
    ranges = []
    for level in range(10):
        _, metadata = generate(TerrainSpec(family, level, seed=5))
        ranges.append(metadata["collision"]["surface_height_range_m"])
    assert np.all(np.diff(ranges) > 0), ranges


@pytest.mark.parametrize("family", FAMILIES)
def test_collision_metadata_is_measured_from_actual_mesh(family):
    mesh, metadata = generate(TerrainSpec(family, 9, seed=3))
    collision = metadata["collision"]
    top = mesh.vertices[: len(mesh.vertices) // 2, 2]
    assert collision["surface_min_height_m"] == pytest.approx(float(top.min()))
    assert collision["surface_max_height_m"] == pytest.approx(float(top.max()))
    assert collision["surface_height_range_m"] == pytest.approx(float(np.ptp(top)))
    assert collision["triangle_count"] == len(mesh.faces)
    assert 0 < collision["max_surface_triangle_slope_deg"] < 90
    assert metadata["route_events"]
    claimed_slopes = [abs(event[key]) for event in metadata["route_events"]
                      for key in ("actual_slope_deg", "actual_cross_slope_deg") if key in event]
    assert max(claimed_slopes) <= collision["max_surface_triangle_slope_deg"] + 1e-5


def test_level_nine_hits_requested_feasible_targets():
    expected = {
        "continuous_curbs": 0.4,
        "continuous_ramp_stairs": 0.9,
        "continuous_split_height": 0.2,
        "continuous_cross_bumps": 0.15,
        "continuous_split_route": 0.6,
    }
    for family, target in expected.items():
        _, metadata = generate(TerrainSpec(family, 9))
        collision = metadata["collision"]
        if family == "continuous_cross_bumps":
            bump_events = [event for event in metadata["route_events"] if event["name"] == "smooth_bump"]
            assert all(event["height_m"] == pytest.approx(target) for event in bump_events)
        else:
            assert collision["surface_height_range_m"] == pytest.approx(target, abs=2e-3)

    for family in ("continuous_ramp_stairs", "continuous_split_route"):
        _, metadata = generate(TerrainSpec(family, 9))
        ramp = next(event for event in metadata["route_events"] if "ramp_start" in event["name"])
        assert ramp["actual_slope_deg"] == pytest.approx(45.0)


def test_seeded_variable_blocks_are_reproducible_and_variable():
    _, first = generate(TerrainSpec("continuous_blocks", 9, seed=11))
    _, again = generate(TerrainSpec("continuous_blocks", 9, seed=11))
    _, other = generate(TerrainSpec("continuous_blocks", 9, seed=12))
    assert first["route_events"] == again["route_events"]
    assert first["route_events"] != other["route_events"]
    heights = [event["height_m"] for event in first["route_events"] if event["name"].startswith("block_up")]
    assert len(set(round(height, 6) for height in heights)) == 2


def _surface_z(mesh, x, y):
    top = mesh.vertices[: len(mesh.vertices) // 2]
    matches = np.isclose(top[:, 0], x) & np.isclose(top[:, 1], y)
    assert matches.sum() == 1
    return float(top[matches, 2][0])


def test_split_height_reaches_both_wheel_tracks():
    mesh, metadata = mesh_for("continuous_split_height", 9)
    assert _surface_z(mesh, -1.0, 0.25) - _surface_z(mesh, -1.0, -0.25) == pytest.approx(0.2)
    assert _surface_z(mesh, 1.0, -0.25) - _surface_z(mesh, 1.0, 0.25) == pytest.approx(0.2)
    assert metadata["design_targets_at_level_9"]["wheel_track_y_m"] == [-0.25, 0.25]


def test_cross_bumps_are_centred_on_alternating_wheel_tracks():
    mesh, metadata = mesh_for("continuous_cross_bumps", 9)
    bumps = [event for event in metadata["route_events"] if event["name"] == "smooth_bump"]
    assert [event["y_m"] for event in bumps] == [0.25, -0.25, 0.25]
    assert all(event["height_m"] == pytest.approx(0.15) for event in bumps)
    bank_difference = np.tan(np.deg2rad(14.0)) * 0.5
    actual_difference = _surface_z(mesh, -1.0, 0.25) - _surface_z(mesh, -1.0, -0.25)
    assert actual_difference == pytest.approx(bank_difference + 0.15)


def test_riser_contract_is_explicitly_chamfered():
    _, metadata = mesh_for("continuous_ramp_stairs", 9)
    collision = metadata["collision"]
    assert collision["riser_run_m"] == 0.03
    assert collision["has_exact_vertical_risers"] is False
    assert "not an exact vertical" in collision["riser_model"]


@pytest.mark.parametrize("family", FAMILIES)
def test_stable_runtime_api(family):
    mesh, metadata = mesh_for(family, 4)
    assert mesh.is_watertight and mesh.volume > 0
    assert metadata["config"]["family"] == family
    assert metadata["config"]["level"] == 4


@pytest.mark.parametrize("bad", [-1, 10, 1.5, True])
def test_invalid_levels_are_rejected(bad):
    with pytest.raises(ValueError):
        TerrainSpec("continuous_curbs", bad)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        TerrainSpec("unknown", 0)
