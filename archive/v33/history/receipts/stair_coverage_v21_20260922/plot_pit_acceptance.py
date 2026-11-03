"""Plot measured first-episode stopping conditions; preserve terminal censoring."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
ENV = 28
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for mode, label, color in [('stochastic', '训练采样动作', '#d74c35'),
                           ('mean', '高台改用平均动作（零更新诊断）', '#267fbd')]:
    d = json.loads((HERE/f'pit_probe_{mode}_FINALIZED.json').read_text(encoding='utf-8'))
    z = np.load(HERE/f'pit_probe_{mode}_traces.npz')
    j = z['env_ids'].tolist().index(ENV)
    age = z['episode_age'][:, j]
    reset = np.where(np.diff(age) <= 0)[0]
    stop = int(reset[0]+1) if len(reset) else len(age)
    t = age[:stop]*.02
    ys = [z['xy'][:stop, j], z['angular'][:stop, j],
          z['force'][:stop, j].min(-1), z['hold'][:stop, j]*.02]
    for ax, y in zip(axes, ys, strict=True):
        ax.plot(t, y, label=label, color=color, linewidth=1.1)
    rows = [x for x in d['episodes'] if x['env'] == ENV]
    if not rows or rows[0]['level'] != 2:
        raise ValueError('Illustrated first episode must be25cm')
    print(mode, 'env', ENV, 'speed', rows[0]['speed'], 'success', rows[0]['success'])
for ax, threshold, title in zip(axes, [.15, .3, 5, 2],
        ['机身水平速度 (m/s)', '机身总角速度 (rad/s)', '四轮中最小向上接触力 (N)', '完整连续达标计时 (s)'], strict=True):
    ax.axhline(threshold, color='#666666', linestyle='--', linewidth=1)
    ax.set_ylabel(title, fontproperties=FONT)
    ax.grid(alpha=.2)
axes[0].legend(prop=FONT, loc='upper right')
axes[1].set_ylim(bottom=0)
axes[2].set_ylim(bottom=-5)
axes[3].set_ylim(-.05, 2.1)
axes[-1].set_xlim(0, 12)
axes[-1].set_xlabel('首次回合时间（秒）；回合结束后不延长曲线', fontproperties=FONT)
fig.suptitle('25厘米高台停车判定：34823轮模型，环境28首回合\n只对比动作采样；观测噪声和训练随机参数保留，未更新模型',
             fontproperties=FONT, fontsize=16)
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(HERE/'PIT_ACCEPTANCE_TIMELINE.png', dpi=160)
