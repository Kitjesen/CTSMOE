"""Check placement against simple analytic geometry and articulated transforms."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from reset_geometry import CollisionGeometry, placement_delta, primitive_surface


@pytest.fixture
def geometry(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="fixture">
    <link name="base_link"><collision><geometry><box size="1 1 0.2"/></geometry></collision></link>
    <link name="wheel_foot_Link"><collision><origin xyz="0 0 -1"/>
      <geometry><cylinder radius="0.1" length="0.2"/></geometry></collision></link>
    <joint name="hinge" type="revolute"><parent link="base_link"/><child link="wheel_foot_Link"/>
      <origin xyz="0.5 0 0"/><axis xyz="0 1 0"/></joint></robot>''')
    return CollisionGeometry(path, ['hinge'])


def test_joint_and_world_rotation(geometry):
    q = torch.tensor([[0.], [math.pi/2]])
    roots = torch.tensor([[0., 0., 2., 1., 0., 0., 0.], [3., 4., 2., 2**-.5, 0., 0., 2**-.5]])
    pos, rot = geometry.transforms(q, roots)
    child = geometry.link_names.index('wheel_foot_Link')
    torch.testing.assert_close(pos[:, child], torch.tensor([[.5, 0., 2.], [3., 4.5, 2.]]))
    tip = pos[:, child] + (rot[:, child] @ torch.tensor([0., 0., -1.]))
    torch.testing.assert_close(tip, torch.tensor([[.5, 0., 1.], [3., 3.5, 2.]]))


def test_lowering_uses_every_surface_and_can_raise(geometry):
    q = torch.zeros(2, 1)
    pose = torch.tensor([[0., 0., 2., 1., 0., 0., 0.], [0., 0., .5, 1., 0., 0., 0.]])
    points = geometry.world_points(q, pose)
    floor = torch.zeros(points.shape[:-1])
    dz = placement_delta(points, floor, .03)
    torch.testing.assert_close(dz, torch.tensor([-.87, .63]))
    torch.testing.assert_close((points[..., 2]+dz[:, None]-floor).amin(1), torch.full((2,), .03))
    # A peak underneath the base interior must win over the wheel query.
    base = ~geometry.wheel_points
    floor[:, base] = 1.6
    dz_peak = placement_delta(points, floor, .03)
    assert bool((dz_peak > dz).all())
    assert bool((points[..., 2]+dz_peak[:, None] >= floor+.03-1e-6).all())


def test_ray_miss_is_rejected(geometry):
    points = geometry.world_points(torch.zeros(1, 1), torch.tensor([[0., 0., 2., 1., 0., 0., 0.]]))
    floor = torch.zeros(points.shape[:-1])
    floor[0, 0] = torch.inf
    with pytest.raises(ValueError, match='missed'):
        placement_delta(points, floor, .03)


def test_box_faces_include_interior():
    points = primitive_surface(ET.fromstring('<geometry><box size="1 1 1"/></geometry>'), .1)
    assert np.any(np.all(np.isclose(points, [0, 0, -.5]), axis=1))


def test_unsupported_geometry_fails_closed():
    with pytest.raises(ValueError, match='Unsupported'):
        primitive_surface(ET.fromstring('<geometry><mesh filename="x.stl"/></geometry>'), .02)
