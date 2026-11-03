"""Read-only V14 fixed-case review figures; never changes original receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / 'review300'
OUT.mkdir(exist_ok=True)
suite = ROOT / 'evaluation300' / 'flat25'
manifest = json.loads((suite / 'manifest.json').read_text())
analysis = json.loads((suite / 'analysis_v14.json').read_text())
legs = np.asarray(manifest['request_joint_ids'])
nominal = np.asarray(manifest['default_q'])
dt = float(manifest['dt'])
selected = [next(r for r in analysis['cases'] if r['label'] == label
                 and r['mode'] == 'direct' and r['direction'] == 'forward')
            for label in ('early_31831', 'V14_32131')]
with np.load(suite / 'traces.npz') as archive:
    arrays = {key: archive[key] for key in
              ('joint_pos', 'target', 'requested', 'root_clearance', 'velocity', 'command')}
summary = []
fig_compare, axes_compare = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for case, color in zip(selected, ('#2171b5', '#d94801')):
    label, i = case['label'], case['id']
    n = case['end_step'] + 1
    t = (np.arange(n) + 1) * dt
    q = arrays['joint_pos'][:n, i][:, legs] - nominal[legs]
    target = arrays['target'][:n, i][:, legs] - nominal[legs]
    request = arrays['requested'][:n, i] - nominal[legs]
    fig, axes = plt.subplots(3, 4, figsize=(17, 9), sharex=True)
    for col, axis in enumerate(axes.flat):
        axis.plot(t, request[:, col], color='#d73027', linestyle=':', label='Policy request')
        axis.plot(t, target[:, col], color='#fdae61', linestyle='--', label='Limited target')
        axis.plot(t, q[:, col], color='#2166ac', label='Actual angle')
        axis.axhline(0, color='gray', linewidth=.7)
        axis.set_title(manifest['joint_names'][legs[col]])
        axis.set_xlim(0, 3)
        axis.grid(alpha=.2)
        axis.set_ylabel('Offset from default (rad)')
        axis.set_xlabel('Time from direct start (s)')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{label}: matched flat +0.6 m/s, first valid episode; no noise', fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / f'{label}_forward_12_joints.png', dpi=150)
    plt.close(fig)
    rms = np.sqrt(np.mean(q*q, axis=1))
    axes_compare[0].plot(t, rms, color=color, label=label)
    axes_compare[1].plot(t, arrays['root_clearance'][:n, i], color=color)
    axes_compare[2].plot(t, arrays['velocity'][:n, i, 0], color=color)
    summary.append({'label': label, 'case': i, 'first_peak_rms': case['first_peak_rms'],
                    'last_xy_error': case['last_xy_error'], 'last_yaw_error': case['last_yaw_error'],
                    'per_joint_first_second_peak_abs': {
                        manifest['joint_names'][j]: float(np.abs(q[:round(1/dt), col]).max())
                        for col, j in enumerate(legs)}})
axes_compare[0].set_ylabel('12-leg posture RMS (rad)')
axes_compare[0].legend()
axes_compare[1].set_ylabel('Body ground clearance (m)')
axes_compare[1].axhline(.46, color='gray', linestyle='--')
axes_compare[2].set_ylabel('Body forward speed (m/s)')
axes_compare[2].axhline(.6, color='gray', linestyle='--')
axes_compare[2].set_xlabel('Time from direct start (s)')
for axis in axes_compare:
    axis.grid(alpha=.2)
    axis.set_xlim(0, 25)
fig_compare.suptitle('V14 after 300 updates: same-batch flat forward comparison')
fig_compare.tight_layout()
fig_compare.savefig(OUT / 'flat_forward_comparison.png', dpi=150)
plt.close(fig_compare)
(OUT / 'forward_joint_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
