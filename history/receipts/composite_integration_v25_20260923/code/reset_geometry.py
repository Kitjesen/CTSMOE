"""URDF collision-surface FK for reset placement; independent of stale PhysX link buffers."""

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array(((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                     (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr)))


def origin(element):
    item = element.find('origin')
    xyz = [float(v) for v in item.get('xyz', '0 0 0').split()] if item is not None else [0., 0., 0.]
    rpy = [float(v) for v in item.get('rpy', '0 0 0').split()] if item is not None else [0., 0., 0.]
    return np.asarray(xyz), rotation_rpy(rpy)


def primitive_surface(geometry, spacing):
    """Sample every collision surface, including cylinder end caps and box interiors."""
    if spacing <= 0:
        raise ValueError('Surface spacing must be positive')
    shape = list(geometry)[0]
    if shape.tag == 'box':
        size = np.array([float(v) for v in shape.get('size').split()])
        axes = [np.linspace(-v/2, v/2, math.ceil(v/spacing)+1) for v in size]
        surfaces = []
        for fixed in range(3):
            for sign in (-1, 1):
                grids = [axis if i != fixed else np.array([sign*size[i]/2]) for i, axis in enumerate(axes)]
                surfaces.append(np.stack(np.meshgrid(*grids, indexing='ij'), -1).reshape(-1, 3))
        return np.unique(np.concatenate(surfaces), axis=0)
    if shape.tag == 'cylinder':
        radius, length = float(shape.get('radius')), float(shape.get('length'))
        angle = np.linspace(0, 2*math.pi, max(16, math.ceil(2*math.pi*radius/spacing)), endpoint=False)
        z = np.linspace(-length/2, length/2, math.ceil(length/spacing)+1)
        theta, height = np.meshgrid(angle, z, indexing='ij')
        surfaces = [np.stack((radius*np.cos(theta), radius*np.sin(theta), height), -1).reshape(-1, 3)]
        for cap in (-length/2, length/2):
            for r in np.linspace(0, radius, math.ceil(radius/spacing)+1):
                surfaces.append(np.stack((r*np.cos(angle), r*np.sin(angle), np.full_like(angle, cap)), -1))
        return np.unique(np.concatenate(surfaces), axis=0)
    raise ValueError(f'Unsupported reset collision shape: {shape.tag}')


def quaternion_matrix(quat):
    q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


class CollisionGeometry:
    """Read the actual primitive collision model and articulated joint transforms once."""

    def __init__(self, urdf_path, joint_names, device='cpu', spacing=0.02):
        path = Path(urdf_path)
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.spacing = spacing
        tree = ET.parse(path).getroot()
        self.link_names = [link.get('name') for link in tree.findall('link')]
        link_index = {name: i for i, name in enumerate(self.link_names)}
        joint_index = {name: i for i, name in enumerate(joint_names)}
        children = {joint.find('child').get('link') for joint in tree.findall('joint')}
        roots = set(self.link_names)-children
        if roots != {'base_link'}:
            raise ValueError(f'Expected one Thunder root, found {roots}')
        self.root = link_index['base_link']
        self.joints = []
        seen = {'base_link'}
        pending = list(tree.findall('joint'))
        while pending:
            ready = [joint for joint in pending if joint.find('parent').get('link') in seen]
            if not ready:
                raise ValueError('URDF joint tree is disconnected')
            for joint in ready:
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                xyz, rotation = origin(joint)
                kind = joint.get('type')
                if kind not in ('fixed', 'revolute', 'continuous'):
                    raise ValueError(f'Unsupported joint type: {kind}')
                index = None if kind == 'fixed' else joint_index[joint.get('name')]
                axis = np.zeros(3) if index is None else np.array([float(v) for v in joint.find('axis').get('xyz').split()])
                if index is not None:
                    axis = axis/np.linalg.norm(axis)
                tensor = lambda value: torch.tensor(value, device=device, dtype=torch.float32)  # noqa: E731
                axis_t = tensor(axis)
                skew = tensor(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
                self.joints.append((link_index[parent], link_index[child], index, tensor(xyz), tensor(rotation),
                                    skew, axis_t[:, None]*axis_t[None, :]))
                seen.add(child)
                pending.remove(joint)
        points, owners = [], []
        for link in tree.findall('link'):
            for collision in link.findall('collision'):
                xyz, rotation = origin(collision)
                samples = primitive_surface(collision.find('geometry'), spacing) @ rotation.T + xyz
                points.append(samples)
                owners.extend([link_index[link.get('name')]]*len(samples))
        self.local_points = torch.tensor(np.concatenate(points), device=device, dtype=torch.float32)
        self.owners = torch.tensor(owners, device=device, dtype=torch.long)
        self.wheel_points = torch.tensor([self.link_names[i].endswith('_foot_Link') for i in owners], device=device)

    def transforms(self, joint_pos, root_pose):
        positions, rotations = [None]*len(self.link_names), [None]*len(self.link_names)
        positions[self.root], rotations[self.root] = root_pose[:, :3], quaternion_matrix(root_pose[:, 3:7])
        identity = torch.eye(3, device=root_pose.device)
        for parent, child, index, xyz, local_rot, skew, outer in self.joints:
            positions[child] = positions[parent] + torch.matmul(rotations[parent], xyz)
            rotation = torch.matmul(rotations[parent], local_rot)
            if index is not None:
                angle = joint_pos[:, index, None, None]
                motion = angle.cos()*identity + angle.sin()*skew + (1-angle.cos())*outer
                rotation = torch.bmm(rotation, motion)
            rotations[child] = rotation
        return torch.stack(positions, 1), torch.stack(rotations, 1)

    def world_points(self, joint_pos, root_pose):
        positions, rotations = self.transforms(joint_pos, root_pose)
        return (rotations[:, self.owners] @ self.local_points[None, :, :, None]).squeeze(-1) + positions[:, self.owners]


def placement_delta(points, ground_z, clearance):
    """Smallest vertical translation keeping every sampled collision point above terrain."""
    if points.shape[:-1] != ground_z.shape or clearance < 0:
        raise ValueError('Invalid collision/terrain placement input')
    if not bool(torch.isfinite(ground_z).all() & torch.isfinite(points).all()):
        raise ValueError('Reset terrain query missed or returned non-finite coordinates')
    return (ground_z + clearance - points[..., 2]).amax(dim=1)
