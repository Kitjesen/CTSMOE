"""Measured per-wheel steps and startup joint trajectories on unchanged fixed cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from omni_lift import LiftState, RULES


def wheel_metrics(clearance, force, commands, actual, dt):
    contact = force > 5.
    state = LiftState(1)
    state.previous_contact[:] = torch.from_numpy(contact[:1])
    for height, touching, goal, velocity in zip(clearance, contact, commands, actual):
        state.update(torch.from_numpy(touching[None]), torch.from_numpy(height[None]),
                     torch.ones(1, dtype=torch.bool), torch.from_numpy(goal[None]),
                     torch.from_numpy(velocity[None]), torch.zeros(1, dtype=torch.bool), dt)
    return dict(qualifying_lift_landings_per_wheel=state.landing_count[0].tolist(),
                max_clearance_m_per_wheel=clearance.max(0).tolist(),
                contact_fraction_per_wheel=contact.mean(0).tolist(),
                less_than_two_support_fraction=float((contact.sum(-1) < 2).mean()))


def main(directory):
    m = json.loads((directory/'manifest.json').read_text())
    dt = m['dt']
    analysis = json.loads((directory/'analysis.json').read_text())
    rows, figures = [], []
    with np.load(directory/'traces.npz') as data:
        for row in analysis['cases']:
            if row['terrain'] != 'flat':
                continue
            i, n = row['id'], row['end_step']+1
            start = 100 if row['mode'] == 'settle' else 0
            if n <= start:
                continue
            velocity = np.concatenate((data['velocity'][start:n, i, :2], data['angular_velocity'][start:n, i, 2:3]), axis=-1)
            values = wheel_metrics(data['flat_wheel_clearance'][start:n, i], data['wheel_ground_force_z'][start:n, i],
                                   data['command'][start:n, i], velocity, dt)
            values.update(label=row['label'], direction=row['direction'], mode=row['mode'],
                          case_id=i, physical=row['physical'], tracking_pass=row['tracking_pass'])
            rows.append(values)
            if (directory.name == 'flat25' and row['mode'] == 'direct'
                    and row['label'] in ('source31831', 'candidate')
                    and row['direction'] in ('forward', 'left', 'right', 'yaw_left', 'yaw_right')):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                ids = m['request_joint_ids']
                limit = min(n, 250)
                default = np.asarray(m['default_q'])
                fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
                for joint, ax in zip(ids, axes.flat):
                    t = np.arange(limit)*dt
                    ax.plot(t, data['joint_pos'][:limit, i, joint]-default[joint], label='actual')
                    ax.plot(t, data['target'][:limit, i, joint]-default[joint], label='target', alpha=.65)
                    request_index = ids.index(joint)
                    ax.plot(t, data['requested'][:limit, i, request_index]-default[joint], label='request', alpha=.4)
                    ax.axhline(0, color='k', linestyle=':', linewidth=.6)
                    ax.set_title(m['joint_names'][joint])
                    ax.set_ylabel('offset (rad)')
                axes[0, 0].legend()
                fig.suptitle(f"{row['label']} / {row['direction']} / direct command; first 5 seconds")
                fig.tight_layout()
                filename = f"joints_{row['label']}_{row['direction']}.png"
                fig.savefig(directory/filename, dpi=130)
                plt.close(fig)
                figures.append(filename)
    with (directory/'OMNI_LIFT_METRICS.json').open('x') as f:
        json.dump(dict(scope='fixed cases, not random success rate; identical LiftState and thresholds to training',
                       rules=RULES.__dict__, dt=dt, cases=rows, figures=figures), f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
