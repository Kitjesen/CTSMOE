"""Independent URDF FK and inertia reflection audit for the proposed joint mapping."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

URDF_SHA256 = '83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3'
S = np.diag([1., -1., 1.])


def vec(node, key='xyz'):
    return np.fromstring(node.get(key, '0 0 0'), sep=' ') if node is not None else np.zeros(3)


def rotate(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def transform(origin):
    out = np.eye(4)
    out[:3, 3] = vec(origin)
    r, p, y = vec(origin, 'rpy')
    out[:3, :3] = rotate(np.array([0., 0., 1.]), y) @ rotate(np.array([0., 1., 0.]), p) @ rotate(np.array([1., 0., 0.]), r)
    return out


def leg_fk(root, leg, q):
    current = np.eye(4)
    for name, angle in zip(('hip', 'thigh', 'calf', 'foot'), q):
        joint = root.find(f"joint[@name='{leg}_{name}_joint']")
        current = current @ transform(joint.find('origin'))
        axis = vec(joint.find('axis'))
        world_axis = current[:3, :3] @ axis
        rotation = np.eye(4); rotation[:3, :3] = rotate(axis, angle)
        current = current @ rotation
    link = root.find(f"link[@name='{leg}_foot_Link']")
    local = np.append(vec(link.find('collision/origin')), 1.)
    return (current @ local)[:3], world_axis


def audit(urdf, seed=73, samples=256):
    urdf = Path(urdf)
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if digest != URDF_SHA256:
        raise ValueError('URDF changed; mapping must be re-audited')
    root = ET.parse(urdf).getroot()
    rng = np.random.default_rng(seed)
    errors, axis_errors, rolling_errors = {}, {}, {}
    for right, left in (('fr', 'fl'), ('rr', 'rl')):
        pe, ae, re = [], [], []
        for _ in range(samples):
            q = rng.uniform([-0.4, -2.5, -2.5, -3.], [0.4, 2.5, 2.5, 3.])
            p, a = leg_fk(root, right, q)
            mp, ma = leg_fk(root, left, -q)
            pe.append(np.linalg.norm(mp - S @ p))
            # Angular velocity is an axial vector; all 16 mirrored qdot signs are negative.
            ae.append(np.linalg.norm(-ma - (-S @ a)))
            re.append(np.linalg.norm(np.cross(-ma, [0., 0., -.093]) - S @ np.cross(a, [0., 0., -.093])))
        errors[right + '_' + left] = float(max(pe))
        axis_errors[right + '_' + left] = float(max(ae))
        rolling_errors[right + '_' + left] = float(max(re))
    base = root.find("link[@name='base_link']/inertial")
    com = vec(base.find('origin'))
    inertia = base.find('inertia')
    matrix = np.array([
        [float(inertia.get('ixx')), float(inertia.get('ixy')), float(inertia.get('ixz'))],
        [float(inertia.get('ixy')), float(inertia.get('iyy')), float(inertia.get('iyz'))],
        [float(inertia.get('ixz')), float(inertia.get('iyz')), float(inertia.get('izz'))]])
    mass_errors = {}
    for part in ('hip', 'thigh', 'calf', 'foot'):
        for right, left in (('fr', 'fl'), ('rr', 'rl')):
            links = [root.find(f"joint[@name='{leg}_{part}_joint']/child").get('link') for leg in (right, left)]
            masses = [float(root.find(f"link[@name='{link}']/inertial/mass").get('value')) for link in links]
            mass_errors[right + '_' + left + '_' + part] = abs(masses[0] - masses[1])
    return {'urdf_sha256': digest, 'samples_per_pair': samples, 'seed': seed,
            'wheel_center_reflection_max_error_m': errors, 'wheel_axial_reflection_max_error': axis_errors,
            'wheel_tangent_reflection_max_error': rolling_errors,
            'base_com_y_m': float(com[1]), 'base_com_reflection_error_m': float(np.linalg.norm(com - S @ com)),
            'base_inertia_reflection_max_error_kg_m2': float(np.abs(matrix - S @ matrix @ S).max()),
            'paired_link_mass_difference_kg': mass_errors,
            'exact_dynamics_symmetry_proven': False,
            'policy': 'Mapping available for soft student consistency; no equivalent-return PPO augmentation.'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('urdf'); p.add_argument('--output')
    a = p.parse_args(); result = json.dumps(audit(a.urdf), indent=2)
    if a.output:
        Path(a.output).write_text(result + '\n', encoding='utf-8')
    print(result)
