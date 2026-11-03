"""Compare the three student policies on the same fixed 25-second case."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    directory = root / 'pilot300_flat25'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    data = np.load(directory / 'forward_traces.npz')
    legs = [i for i, name in enumerate(manifest['joint_names']) if not name.endswith('_foot_joint')]
    nominal = np.asarray(manifest['default_q'])[legs]
    labels = {'V6_43202': ('V6 姿态参考', '#209562'),
              'V9_44711': ('V9 续训起点', '#3576af'),
              'V12_45011': ('V12 新增300轮', '#d94a3b')}
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')
    for column, case_id in enumerate(data['ids']):
        case = next(c for c in manifest['cases'] if c['id'] == case_id)
        label = manifest['history'][case['policy']]['label']
        if label not in labels:
            continue
        n = min(len(data['joint_pos']), case['end_step'] + 1)
        t = (np.arange(n) + 1) * manifest['dt']
        q = data['joint_pos'][:n, column][:, legs] - nominal
        values = [np.sqrt(np.mean(q*q, axis=-1)), data['root_clearance'][:n, column],
                  data['velocity'][:n, column, 0]]
        for row, value in enumerate(values):
            for col, limit in enumerate((5., 25.)):
                take = t <= limit
                axes[row, col].plot(t[take], value[take], label=labels[label][0], color=labels[label][1], lw=1.7)
    for col, limit in enumerate((5., 25.)):
        axes[0, col].axvspan(0, 1, color='black', alpha=.055, label='收到命令后首秒')
        axes[1, col].axhline(.46, color='#666666', ls='--', lw=1, label='高度目标 0.46 m')
        axes[2, col].axhline(.6, color='#666666', ls='--', lw=1, label='前进指令 0.6 m/s')
        for row, ylabel in enumerate(('12腿关节相对默认姿态 RMS (rad)', '机身相对地面高度 (m)', '实际前向速度 (m/s)')):
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].set_xlim(0, limit)
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=9)
        axes[2, col].set_xlabel('收到前进命令后的时间（秒）')
    axes[0, 0].set_title('起步前5秒：V12首秒偏差仍大')
    axes[0, 1].set_title('完整25秒：V12后段姿态改善')
    fig.suptitle('平地学生策略固定案例对照：起步未修好，后段已有改善\n默认站姿、0.6 m/s前进、无噪声/推扰、固定10 ms延迟；不是随机成功率', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    output = root / 'PILOT300_STUDENT_COMPARISON.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
