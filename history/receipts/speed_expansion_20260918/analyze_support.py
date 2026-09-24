"""Quantify the saved high-stair rollouts; no simulation or training mutation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'recordings/straight_high13501_20260918'
manifest = json.loads((SOURCE / 'trace/manifest.json').read_text())
urdf = ET.parse(SOURCE / 'runtime_thunder_v4.urdf').getroot()
links = {link.attrib['name']: link for link in urdf.findall('link')}


def rotate(q, vector):
    """Active quaternion rotation, source ordering wxyz."""
    vector = np.broadcast_to(vector, q[..., 1:].shape)
    cross = 2 * np.cross(q[..., 1:], vector)
    return vector + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def longest_interval(times, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    segment = max(groups, key=len)
    return {'from_s': float(times[segment[0]]), 'to_s': float(times[segment[-1]]),
            'sample_span_s': float(times[segment[-1]] - times[segment[0]])}


results = []
for level in (7, 9):
    path = SOURCE / f'trace/stairs_up_L{level}.json'
    records = json.loads(path.read_text())
    times = np.array([row['time'] for row in records])
    body_pos = np.array([row['body_pos'] for row in records])
    body_quat = np.array([row['body_quat'] for row in records])
    forces = np.array([row['contact_forces_w'] for row in records])
    velocity = np.array([row['velocity'] for row in records])
    w, x, y, z = np.array([row['root_quat'] for row in records]).T
    roll = np.rad2deg(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    wheels = {}
    for leg in ('fl', 'fr', 'rl', 'rr'):
        name = f'{leg}_foot_Link'
        index = manifest['body_names'].index(name)
        contact_index = manifest['probe']['contact_body_names'].index(name)
        collision = links[name].find('collision')
        origin = collision.find('origin')
        xyz = np.fromstring(origin.attrib['xyz'], sep=' ')
        rpy = np.fromstring(origin.attrib['rpy'], sep=' ')
        # All four wheel cylinders use a rotation around local X only.
        assert np.allclose(rpy[1:], 0)
        axis_local = np.array([0, -np.sin(rpy[0]), np.cos(rpy[0])])
        cylinder = collision.find('geometry/cylinder')
        radius = float(cylinder.attrib['radius'])
        half_length = float(cylinder.attrib['length']) / 2
        q = body_quat[:, index]
        center = body_pos[:, index] + rotate(q, xyz)
        axis = rotate(q, axis_local)
        extent = half_length * np.abs(axis) + radius * np.sqrt(np.clip(1-axis**2, 0, 1))
        wheels[leg] = {'center': center, 'min': center-extent, 'max': center+extent,
                       'force': forces[:, contact_index]}
    vertices = np.array([list(map(float, line.split()[1:4])) for line in
                         (SOURCE / f'trace/terrain_stairs_up_L{level}.obj').read_text().splitlines()
                         if line.startswith('v ')])
    floor = vertices[:, 2].min()
    mid_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2
    centerline = vertices[np.isclose(vertices[:, 1], mid_y, atol=.001)]
    window = (times >= 12) & (times <= 16)
    wheel_stats = {}
    for leg, wheel in wheels.items():
        wheel_stats[leg] = {
            'center_x_span_m_12to16s': float(np.ptp(wheel['center'][window, 0])),
            'mean_center_world_xyz_m_12to16s': wheel['center'][window].mean(axis=0).tolist(),
            'mean_forward_extent_x_m_12to16s': float(wheel['max'][window, 0].mean()),
            'mean_force_world_xyz_N_12to16s': wheel['force'][window].mean(axis=0).tolist(),
            'vertical_support_over5N_fraction_12to16s': float((wheel['force'][window, 2] > 5).mean()),
        }
    moving = times >= 2
    right_unsupported = moving & (wheels['fr']['force'][:, 2] < 5) & (wheels['rr']['force'][:, 2] < 5)
    left_supported = (wheels['fl']['force'][:, 2] > 5) & (wheels['rl']['force'][:, 2] > 5)
    result = {'level': level, 'floor_world_z_m': float(floor),
              'mean_body_vx_mps_12to16s': float(velocity[window, 0].mean()),
              'wheel_stats': wheel_stats,
              'longest_left_pair_only_vertical_support': longest_interval(times, right_unsupported & left_supported),
              'centerline_vertices': centerline.tolist(),
              'samples': []}
    for time in (6, 8, 12, 14, 16, 17, 17.12, 17.28):
        i = int(np.argmin(np.abs(times-time)))
        result['samples'].append({'time': float(times[i]), 'roll_deg': float(roll[i]),
                                   'pitch_deg': float(pitch[i]), 'vx': float(velocity[i, 0]),
                                   'wheel_forces_xyz_N': {leg: wheel['force'][i].tolist() for leg, wheel in wheels.items()},
                                   'wheel_bottom_height_above_floor_m': {leg: float(wheel['min'][i, 2]-floor) for leg, wheel in wheels.items()}})
    results.append(result)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for leg, wheel in wheels.items():
        axes[0].plot(times, wheel['min'][:, 2]-floor, label=leg.upper())
        axes[1].plot(times, wheel['center'][:, 0]-wheel['center'][0, 0], label=leg.upper())
        axes[2].plot(times, wheel['force'][:, 2], label=leg.upper())
    axes[0].set_ylabel('Wheel bottom height (m)')
    axes[1].set_ylabel('Wheel forward displacement (m)')
    axes[2].set_ylabel('Wheel vertical contact force (N)')
    axes[2].set_ylim(-5, 700)
    axes[3].plot(times, roll, label='Roll')
    axes[3].plot(times, pitch, label='Pitch')
    axes[3].set_ylabel('Base angle (deg)')
    axes[3].set_xlabel('Simulation time (s); forward command begins at 2 s')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', ncol=4)
        ax.axvspan(12, 16, alpha=.08, color='red')
    clip = next(clip for clip in manifest['clips'] if clip['terrain']['level'] == level)
    fig.suptitle(f'Student 13501 | L{level}, {clip["terrain"]["actual_step_height_m"]*100:.1f} cm stairs | vx=0.35 m/s\n'
                 'One nominal rollout; contact forces sampled at 50 Hz, last physics substep', fontsize=13)
    fig.savefig(HERE / f'support_L{level}.png', dpi=150)
    plt.close(fig)

output = {'source_policy_sha256': manifest['policy_sha256'], 'completed_updates': 13501,
          'limitations': 'One nominal rollout per level, no causal ablation. Cylinder bounds are world-axis bounds, not exact contact points. Force samples omit intermediate 200 Hz substeps.',
          'results': results}
(HERE / 'support_analysis.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps([{k: v for k, v in r.items() if k not in ('samples', 'centerline_vertices')} for r in results], indent=2))
