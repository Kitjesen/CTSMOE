"""Joint-by-joint startup evidence, with command-aligned windows and no pooled wheel angles."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyze(folder):
    m = json.loads((folder / 'manifest.json').read_text())
    d = np.load(folder / 'forward.npz')
    default = np.asarray(m['default_q'])
    dt = m['dt']
    legs = list(range(12))
    request_order = [m['request_joint_ids'].index(j) for j in legs]
    rows, cases = [], []
    for i, case in enumerate(m['cases']):
        # Extraction retains source IDs; array columns follow manifest case order.
        n = case['end_step'] + 1
        start = int(np.flatnonzero(np.abs(d['command'][:n, i, 0]) > .1)[0])
        q = d['joint_pos'][:n, i, :12] - default[:12]
        target = d['target'][:n, i, :12] - default[:12]
        request = d['requested'][:n, i][:, request_order] - default[:12]
        delta = np.diff(d['target'][:n, i, :12], axis=0, prepend=d['initial_q'][i, :12][None])
        label = Path(case['checkpoint']).stem
        role = 'teacher' if case['teacher'] else 'student'
        rms = np.sqrt(np.mean(q**2, axis=1))
        summary = dict(label=label, role=role, mode=case['mode'], source_case=case['id'],
                       command_onset_s=start*dt, physical=case['physical'])
        for secs in (1, 2):
            window = slice(start, min(start + round(secs/dt), n))
            summary[f'rms_peak_{secs}s'] = float(rms[window].max())
            summary[f'joint_peak_{secs}s'] = float(np.abs(q[window]).max())
        summary.update(rms_last5_mean=float(rms[-250:-1].mean()),
                       height_min_2s=float(d['root_clearance'][start:start+100, i].min()),
                       vx_peak_2s=float(d['velocity'][start:start+100, i, 0].max()),
                       vx_last5_mean=float(d['velocity'][-250:-1, i, 0].mean()),
                       target_step_peak_1s=float(abs(delta[start:start+50]).max()),
                       first_target_step_peak=float(abs(delta[start]).max()),
                       requested_peak_1s=float(abs(request[start:start+50]).max()),
                       target_peak_1s=float(abs(target[start:start+50]).max()))
        cases.append(summary)
        for j in legs:
            for seconds in (1, 2):
                stop = min(n, start + round(seconds/dt))
                peak = start + int(np.abs(q[start:stop, j]).argmax())
                rows.append(dict(label=label, role=role, mode=case['mode'], joint=m['joint_names'][j],
                    window_s=seconds, actual_peak_rad=float(abs(q[peak, j])),
                    actual_signed_rad=float(q[peak, j]), peak_after_command_s=(peak-start+1)*dt,
                    request_at_peak_rad=float(request[peak, j]), target_at_peak_rad=float(target[peak, j]),
                    target_step_peak_rad=float(abs(delta[start:stop, j]).max()),
                    request_peak_rad=float(abs(request[start:stop, j]).max()),
                    torque_peak_nm=float(d['torque_peak'][start:stop, i, j].max()),
                    speed_peak_rad_s=float(d['speed_peak'][start:stop, i, j].max()),
                    actual_last5_mean_rad=float(q[-250:-1, j].mean())))
    (folder/'forward_metrics.json').write_text(json.dumps(cases, indent=2))
    with (folder/'per_joint.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    picks = [(i, c) for i, c in enumerate(m['cases']) if not c['teacher'] and c['mode']=='direct']
    assert len(picks) == 2
    old, current = picks[0][0], picks[1][0]
    t = (np.arange(150)+1)*dt
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    leg_labels = ['左前', '右前', '左后', '右后']
    joint_labels = ['髋', '大腿', '小腿']
    for panel, (j, ax) in enumerate(zip(order, axes.flat)):
        ax.plot(t, d['joint_pos'][:150, old, j]-default[j], color='#999999', label='31,831轮实际')
        ax.plot(t, d['requested'][:150, current, request_order[j]]-default[j], ':', color='#b766bb', label='候选原始请求')
        ax.plot(t, d['target'][:150, current, j]-default[j], '--', color='#e59028', label='候选限制后目标')
        ax.plot(t, d['joint_pos'][:150, current, j]-default[j], color='#176c9c', label='候选实际角度')
        ax.axhline(0, color='black', lw=.6)
        ax.set_title(f'{leg_labels[panel//3]}{joint_labels[panel%3]} / {m["joint_names"][j]}')
        ax.grid(alpha=.2)
        extent = max(.4, float(np.abs(d['requested'][:150, current, request_order[j]]-default[j]).max()),
                     float(np.abs(d['joint_pos'][:150, :, j]-default[j]).max())) * 1.1
        ax.set_ylim(-extent, extent)
    axes[0, 0].legend(fontsize=8)
    completed = int(Path(picks[1][1]['checkpoint']).stem.split('_')[1])+1
    fig.suptitle(f"直行起步12腿关节：{completed:,}轮对比31,831轮\n0.6 m/s直接阶跃；纵轴为相对默认角度(rad)，各图量程不同；横轴为起步后秒数", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(folder/'joint_startup.png', dpi=135)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    t = (np.arange(1250)+1)*dt
    for idx, case in picks:
        label = str(int(Path(case['checkpoint']).stem.split('_')[1])+1)+'轮'
        q = d['joint_pos'][:, idx, :12]-default[:12]
        axes[0].plot(t, np.sqrt((q*q).mean(-1)), label=label)
        axes[1].plot(t, d['root_clearance'][:, idx], label=label)
        axes[2].plot(t, d['velocity'][:, idx, 0], label=label)
    for ax, ylabel in zip(axes, ['12腿关节偏差 RMS (rad)', '机身离地高度 (m)', '前向速度 (m/s)']):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend()
    axes[1].axhline(.46, ls='--', color='grey')
    axes[2].axhline(.6, ls='--', color='grey')
    axes[2].set_xlabel('收到速度命令后 (s)')
    fig.suptitle('直行完整25秒：姿态、机身高度和速度；固定案例，不是随机成功率')
    fig.tight_layout()
    fig.savefig(folder/'forward_summary.png', dpi=135)
    plt.close(fig)
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    analyze(parser.parse_args().folder)
