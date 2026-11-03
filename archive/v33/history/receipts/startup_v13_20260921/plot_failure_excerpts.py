"""Plot measured failure excerpts; substep maxima are not endpoint velocities."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent / 'evaluation300' / 'failure_excerpts_r1'


def main():
    meta = json.loads((HERE / 'flat25.json').read_text())
    data = np.load(HERE / 'flat25.npz', allow_pickle=False)
    index = meta['original_case_ids'].index(62)
    joint = meta['joint_names'].index('rr_foot_joint')
    count = meta['cases'][index]['end_step'] + 1
    times = (np.arange(count) + 1) * meta['dt']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    # The action contract uses FR, FL, RR, RL for its last four wheel actions.
    request = data['raw_action'][:count, index, 14] * 5.0
    axes[0].plot(times, request, label='Requested RR wheel speed (before delay)')
    axes[0].plot(times, data['joint_vel'][:count, index, joint], label='20ms endpoint actual speed')
    axes[0].plot(times, data['speed_peak'][:count, index, joint], label='Absolute 5ms-substep maximum', alpha=.65)
    axes[0].axhline(45, color='red', ls='--', label='Current extreme-speed termination')
    axes[0].set_ylabel('rad/s')
    axes[1].plot(times, data['torque'][:count, index, joint], label='Endpoint applied torque')
    axes[1].plot(times, data['torque_peak'][:count, index, joint], label='Absolute substep peak', alpha=.65)
    axes[1].set_ylabel('N m')
    axes[2].plot(times, data['wheel_ground_force'][:count, index, 2], label='RR endpoint ground-force norm')
    axes[2].set_ylabel('N (not impulse)')
    axes[2].set_xlabel('Episode seconds; constant lateral command 0.4 m/s')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(loc='upper left', fontsize=8)
    fig.suptitle('V13 student left/direct: no high wheel-speed request, but a brief actual-speed spike\n'
                 '50Hz endpoints cannot establish the force/velocity order inside the failing control step')
    fig.tight_layout()
    fig.savefig(HERE / 'left_rr_wheel_full.png', dpi=150)
    axes[2].set_xlim(10.1, 10.72)
    fig.savefig(HERE / 'left_rr_wheel_last600ms.png', dpi=150)
    plt.close(fig)

    meta = json.loads((HERE / 'normal.json').read_text())
    data = np.load(HERE / 'normal.npz', allow_pickle=False)
    index = meta['original_case_ids'].index(213)
    joint = meta['joint_names'].index('rl_foot_joint')
    count = meta['cases'][index]['end_step'] + 1
    times = (np.arange(count) + 1) * meta['dt']
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for wheel, label in enumerate(('FR', 'FL', 'RR', 'RL')):
        axes[0].plot(times, data['wheel_clearance'][:count, index, wheel], label=label)
    axes[0].set_ylabel('Wheel-bottom ray gap (m)')
    axes[1].plot(times, data['raw_action'][:count, index, 15] * 5, label='RL requested speed before delay')
    axes[1].plot(times, data['joint_vel'][:count, index, joint], label='RL actual endpoint speed')
    axes[1].set_ylabel('rad/s')
    axes[2].plot(times, data['wheel_ground_force'][:count, index, 3], label='RL endpoint ground-force norm')
    axes[2].set_ylabel('N')
    axes[2].set_xlabel('Episode seconds')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Original evaluation case 213: wave-reset correction was disabled\n'
                 'Keep as diagnosis of the old evaluation protocol, not a training-parity result')
    fig.tight_layout()
    fig.savefig(HERE / 'wave_disabled_reset_failure.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
