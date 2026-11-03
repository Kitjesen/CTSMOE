"""Per-joint assessment from valid first-episode trajectories, never pooled RMS."""
import csv
import json
from pathlib import Path

import numpy as np


def window_metrics(actual, target, requested, initial_target, dt):
    """Angles are already relative to default; initial_target precedes this window."""
    actual, target, requested = map(np.asarray, (actual, target, requested))
    if actual.ndim != 1 or not len(actual) or actual.shape != target.shape or actual.shape != requested.shape:
        raise ValueError('Non-empty matched one-joint time series required')
    if dt <= 0 or not all(np.isfinite(x).all() for x in (actual, target, requested)):
        raise ValueError('Invalid time step or non-finite trajectory')
    peak = int(np.argmax(np.abs(actual)))
    delta = np.diff(target, prepend=initial_target)
    return dict(actual_rms_rad=float(np.sqrt(np.mean(actual**2))),
                actual_mean_signed_rad=float(actual.mean()), actual_mae_rad=float(np.abs(actual).mean()),
                actual_p95_abs_rad=float(np.percentile(np.abs(actual), 95)),
                actual_peak_abs_rad=float(np.abs(actual[peak])),
                actual_at_peak_rad=float(actual[peak]), peak_sample=peak,
                target_at_actual_peak_rad=float(target[peak]), request_at_actual_peak_rad=float(requested[peak]),
                target_peak_abs_rad=float(np.abs(target).max()), request_peak_abs_rad=float(np.abs(requested).max()),
                tracking_mae_rad=float(np.abs(actual-target).mean()),
                target_step_peak_rad=float(np.abs(delta).max()),
                target_rate_peak_rad_s=float(np.abs(delta).max()/dt))


def write_joint_assessment(directory, manifest, data):
    directory = Path(directory)
    dt = float(manifest['dt'])
    nominal = np.asarray(manifest['default_q'])
    legs = list(manifest['request_joint_ids'])
    if len(legs) != 12 or len(set(legs)) != 12:
        raise ValueError('Expected the 12 controlled leg joints')
    rows, wheels = [], []
    for case in manifest['cases']:
        i, n = case['id'], case['end_step']+1
        label = manifest['history'][case['policy']]['label']
        onset = int(round(2./dt)) if case['mode'] == 'settle' else 0
        windows = [('startup_0_1', onset, onset+round(1/dt)),
                   ('startup_0_2', onset, onset+round(2/dt)),
                   ('after_start_2_5', onset+round(2/dt), onset+round(5/dt)),
                   ('full_episode', 0, n), ('last_1s', max(0, n-round(1/dt)), n)]
        if manifest.get('protocol_id') == 'flat25_80m':
            windows += [('episode_5_15', round(5/dt), round(15/dt)),
                        ('episode_15_25', round(15/dt), round(25/dt))]
        for window, start, stop in windows:
            end = min(n, stop)
            if end <= start:
                continue
            for column, j in enumerate(legs):
                actual = data['joint_pos'][start:end, i, j]-nominal[j]
                target = data['target'][start:end, i, j]-nominal[j]
                requested = data['requested'][start:end, i, column]-nominal[j]
                previous = (data['target'][start-1, i, j] if start else data['initial_q'][i, j])-nominal[j]
                stats = window_metrics(actual, target, requested, previous, dt)
                peak_sample = stats.pop('peak_sample')
                rows.append(dict(label=label, teacher=bool(case['teacher']), case=i,
                    terrain=case['terrain'], level=case['level'], direction=case['direction'],
                    mode=case['mode'], joint=manifest['joint_names'][j], window=window,
                    samples=end-start, window_complete=end == stop,
                    peak_episode_s=(start+peak_sample+1)*dt,
                    peak_after_command_s=(start+peak_sample+1-onset)*dt,
                    actual_peak_abs_deg=float(np.degrees(stats['actual_peak_abs_rad'])),
                    torque_peak_nm=float(data['torque_peak'][start:end,i,j].max()),
                    speed_peak_rad_s=float(data['speed_peak'][start:end,i,j].max()),
                    hard_excess_rad=float(data['hard_excess'][start:end,i,j].max()),
                    **stats))
        # Wheel rotation is unbounded: default-angle deviation is not a pose metric.
        for j, name in enumerate(manifest['joint_names']):
            if j not in legs:
                wheels.append(dict(label=label, case=i, terrain=case['terrain'], mode=case['mode'],
                    joint=name, torque_peak_nm=float(data['torque_peak'][:n,i,j].max()),
                    speed_peak_rad_s=float(data['speed_peak'][:n,i,j].max()),
                    speed_mean_abs_rad_s=float(np.abs(data['joint_vel'][:n,i,j]).mean())))
    for name, values in (('joint_assessment', rows), ('wheel_assessment', wheels)):
        with (directory/(name+'.json')).open('x', encoding='utf-8') as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2)
        with (directory/(name+'.csv')).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ['# V13逐关节评估', '',
             '全部数值来自有效回合时程；失败后帧不参与统计。RMS按单个关节沿时间计算。',
             '角度偏差相对默认站姿，不是目标跟踪误差；攀爬阶段偏离默认姿态不自动判错。',
             '4个轮关节单列转速/力矩，不用累计旋转角评价站姿。',
             'CSV包含所有地形、方向、策略、窗口及请求/限位后目标/实际角度。窗口不足会标记，不能当完整通过。', '']
    for label in dict.fromkeys(r['label'] for r in rows):
        selected = [r for r in rows if r['label']==label and r['terrain']=='flat'
                    and r['direction']=='forward' and r['mode']=='direct' and r['window']=='startup_0_1']
        if not selected:
            continue
        lines += [f'## {label}：平地前进直接起步首秒', '',
                  '|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|',
                  '|---|---:|---:|---:|---:|']
        for r in selected:
            lines.append(f"|{r['joint']}|{r['actual_peak_abs_rad']:.3f} / {r['actual_peak_abs_deg']:.1f}|"
                         f"{r['request_at_actual_peak_rad']:+.3f} / {r['target_at_actual_peak_rad']:+.3f} / {r['actual_at_peak_rad']:+.3f}|"
                         f"{r['actual_rms_rad']:.3f}|{r['target_step_peak_rad']:.3f}|")
        lines += ['', '各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：', '',
                  '|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|', '|---|---|---:|---:|---|']
        for j in legs:
            candidates = [r for r in rows if r['label']==label and r['joint']==manifest['joint_names'][j]
                          and r['window']=='startup_0_1']
            r = max(candidates, key=lambda x:x['actual_peak_abs_rad'])
            lines.append(f"|{r['joint']}|{r['terrain']}/{r['level']}/{r['direction']}|"
                         f"{r['actual_peak_abs_rad']:.3f}|{r['target_peak_abs_rad']:.3f}|{r['window_complete']}|")
        lines += ['']
    (directory/'JOINT_ASSESSMENT.md').write_text('\n'.join(lines), encoding='utf-8')
    return rows


def plot_joint_terrain_peaks(directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['window']=='startup_0_1']
    for label in dict.fromkeys(r['label'] for r in selected):
        rr = [r for r in selected if r['label']==label]
        joints = list(dict.fromkeys(r['joint'] for r in rr))
        terrains = list(dict.fromkeys(r['terrain'] for r in rr))
        values = np.array([[max(r['actual_peak_abs_rad'] for r in rr
                               if r['joint']==j and r['terrain']==t) for j in joints] for t in terrains])
        fig, ax = plt.subplots(figsize=(14, max(3, len(terrains)*.6+2)))
        im = ax.imshow(values, aspect='auto', vmin=0., vmax=max(1.2, float(values.max())), cmap='YlOrRd')
        ax.set_xticks(range(len(joints)), [j.replace('_joint','') for j in joints], rotation=35, ha='right')
        ax.set_yticks(range(len(terrains)), terrains)
        for y, x in np.ndindex(values.shape):
            ax.text(x, y, f'{values[y,x]:.2f}', ha='center', va='center', fontsize=9)
        ax.set_title(label+' | first-second actual pose deviation by joint\nMaximum across fixed cases in each terrain; rad. Not a skill pass/fail score.')
        fig.colorbar(im, ax=ax, label='Absolute deviation from default (rad)')
        fig.tight_layout()
        fig.savefig(Path(directory)/(label+'_joint_terrain_peaks.png'), dpi=130)
        plt.close(fig)
