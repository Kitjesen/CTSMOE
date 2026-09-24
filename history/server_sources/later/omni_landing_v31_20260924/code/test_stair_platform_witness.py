import pytest
import torch
import hashlib
from types import SimpleNamespace

from mixed_stair_logic import conservative_center_hull_support
from stair_platform_witness import conservative_contact_provider, platform_world_z, validated_wheel_envelope


class _Scene(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def test_plateau_height_uses_measured_mesh_relative_to_spawn_origin():
    receipt = {"selected": True, "origin_unchanged": True,
               "origin_after": [0, 0, -1.35],
               "geometry_after": {"positive_x": {"surface_runs": [
                   {"z_m": -1.35}, {"z_m": -0.45}]}}}
    result = platform_world_z(torch.tensor([12.0]), {(4, 8): receipt}, [4], [8], [0], "cpu")
    torch.testing.assert_close(result, torch.tensor([12.9]))


def test_rejects_unverified_platform_origin():
    receipt = {"selected": True, "origin_unchanged": False}
    with pytest.raises(ValueError):
        platform_world_z(torch.zeros(1), {(1, 8): receipt}, [1], [8], [0], "cpu")


def test_wheel_envelope_requires_four_cylinders(tmp_path):
    links = "".join(
        f'<link name="{side}_foot_Link"><collision><origin xyz="0 -0.01631 0"/>'
        '<geometry><cylinder radius="0.093" length="0.05117"/></geometry>'
        '</collision></link>' for side in ("fr", "fl", "rr", "rl")
    )
    path = tmp_path / "robot.urdf"
    path.write_text(f"<robot>{links}</robot>")
    _, envelope = validated_wheel_envelope(str(path))
    assert 0.13 < envelope < 0.14
    path2 = tmp_path / "bad.urdf"
    invalid = links.replace('radius="0.093"', 'radius="0.5"')
    path2.write_text(f"<robot>{invalid}</robot>")
    with pytest.raises(ValueError):
        validated_wheel_envelope(str(path2))


def test_provider_to_curriculum_contract_uses_inferred_collision_geometry(tmp_path):
    links = "".join(
        f'<link name="{side}_foot_Link"><collision><origin xyz="0 -0.01631 0"/>'
        '<geometry><cylinder radius="0.093" length="0.05117"/></geometry>'
        '</collision></link>' for side in ("fr", "fl", "rr", "rl")
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f"<robot>{links}</robot>")
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    wheel = torch.tensor([[[3.6, -.15, 13.035], [3.6, .15, 13.035],
                           [3.9, -.15, 13.035], [3.9, .15, 13.035]]])
    hull = torch.tensor([[[3.45, -.1, 13.5], [3.45, .1, 13.5],
                          [4.0, -.1, 13.5], [4.0, .1, 13.5]]])
    geometry = SimpleNamespace(
        urdf_sha256=digest, spacing=.04,
        wheel_points=torch.tensor([False] * 4 + [True] * 4),
        world_points=lambda joint_pos, root_pose: torch.cat((hull, wheel), 1),
    )
    receipt = {"selected": True, "origin_unchanged": True,
               "origin_after": [0, 0, -1.35],
               "geometry_after": {"positive_x": {"surface_runs": [
                   {"z_m": -1.35}, {"z_m": -0.45}]}}}
    command = SimpleNamespace(pit=SimpleNamespace(geometry=geometry, wheel_ids=[0, 1, 2, 3]),
                              _mixed_receipts={(4, 8): receipt})
    robot = SimpleNamespace(
        body_names=[f"{side}_foot_Link" for side in ("fr", "fl", "rr", "rl")],
        data=SimpleNamespace(body_pos_w=wheel, joint_pos=torch.zeros(1, 12),
                             root_state_w=torch.zeros(1, 13)),
    )
    terrain = SimpleNamespace(terrain_levels=torch.tensor([4]), terrain_types=torch.tensor([8]))
    sensors = {f"pit_contact_{side}": SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.tensor([[[[0., 0., 50.]]]])))
        for side in ("fr", "fl", "rr", "rl")}
    scene = {"robot": robot, **sensors}
    scene = _Scene(**scene, terrain=terrain, env_origins=torch.tensor([[0., 0., 12.]]))
    env = SimpleNamespace(
        device="cpu", scene=scene, stair_axis=torch.tensor([0]),
        cfg=SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(
            spawn=SimpleNamespace(asset_path=str(urdf))))),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    data = conservative_contact_provider(env, torch.tensor([0]))
    assert data["kind"] == "conservative_center_hull"
    assert data["validated_collision_geometry"] is True
    assert data["wheel_centers_world_xyz"].shape == (1, 4, 3)
    assert data["body_collision_hull_world_xy"].shape == (1, 4, 2)
    torch.testing.assert_close(data["platform_world_z_m"], torch.tensor([12.9]))
    assert bool(conservative_center_hull_support(
        data["wheel_centers_world_xyz"][..., :2], data["body_collision_hull_world_xy"],
        data["wheel_centers_world_xyz"][..., 2], data["platform_world_z_m"],
        data["terrain_forces_n"], torch.tensor([3.0]), torch.tensor([4.5]),
        torch.tensor([1.0]), data["wheel_radius_m"])[0])
