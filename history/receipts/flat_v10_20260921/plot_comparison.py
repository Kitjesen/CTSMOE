"""Plot valid forward trajectories; never include frames after episode end."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
directory = args.directory
manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
if manifest.get('smoke_only'):
    raise ValueError('Smoke trajectories cannot be shown as policy evaluation')
data = np.load(directory/'forward_traces.npz')
legs = [i for i, name in enumerate(manifest['joint_names']) if not name.endswith('_foot_joint')]
nominal = np.asarray(manifest['default_q'])[legs]
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
joint_fig, joint_axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
for column, case_id in enumerate(data['ids']):
    case = next(c for c in manifest['cases'] if c['id'] == case_id)
    label = manifest['history'][case['policy']]['label']
    n = min(len(data['joint_pos']), case['end_step']+1)
    t = (np.arange(n)+1)*manifest['dt']
    q = data['joint_pos'][:n, column][:, legs]-nominal
    values = [np.sqrt(np.mean(q*q, axis=-1)), abs(q).max(axis=-1),
              data['root_clearance'][:n, column], data['velocity'][:n, column, 0]]
    for ax, value in zip(axes, values):
        ax.plot(t, value, label=label, ls='--' if case['teacher'] else '-')
    for j, ax in enumerate(joint_axes.flat):
        take = t <= 2.
        ax.plot(t[take], q[take, j], label=label, ls='--' if case['teacher'] else '-')
        ax.set_title(manifest['joint_names'][legs[j]].replace('_joint', ''))
axes[0].axhline(.25, ls=':', color='black', label='First 1s RMS criterion: 0.25')
axes[1].axhline(.6, ls=':', color='black', label='First 1s joint criterion: 0.60')
axes[2].axhline(.46, ls=':', color='black', label='Target clearance: 0.46m')
axes[3].axhline(.6, ls=':', color='black', label='Command: 0.6m/s')
for ax, ylabel in zip(axes, ['Leg pose RMS (rad)', 'Max joint deviation (rad)',
                           'Body clearance (m)', 'Forward speed (m/s)']):
    ax.set_ylabel(ylabel)
    ax.axvspan(0., 1., alpha=.04, color='black')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=3)
axes[-1].set_xlabel('Time after command (s)')
fig.suptitle('Matched flat forward start: actual joint posture and tracking\nDefault pose, 0.6m/s command, no noise/push, fixed 10ms delay')
fig.tight_layout()
fig.savefig(directory/'forward_comparison.png', dpi=140)
for ax in joint_axes.flat:
    ax.axhline(0., color='black', lw=.7)
    ax.grid(alpha=.25)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Actual minus default (rad)')
joint_axes.flat[0].legend(fontsize=7)
joint_fig.suptitle('First 2s: all 12 leg joints; matched flat forward start')
joint_fig.tight_layout()
joint_fig.savefig(directory/'startup_joints.png', dpi=140)
plt.close('all')
print('Saved forward_comparison.png and startup_joints.png')
