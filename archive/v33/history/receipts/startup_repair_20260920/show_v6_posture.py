"""Visualize measured V6 rigid-body origins, not a rendered robot or terrain."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

b = Path(__file__).parent / 'all_scene_v6'
m = json.loads((b / 'manifest.json').read_text(encoding='utf-8'))
with np.load(b / 'traces.npz') as f:
    d = {k: f[k] for k in ('body_pos', 'body_quat', 'gravity', 'joint_pos', 'velocity', 'root_clearance')}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus'] = False
names = [n.lower() for n in m['body_names']]
base = names.index('base_link')
legs = [i for i, n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default = np.array(m['default_q'])


def case(terrain, level):
    return next(c for c in m['cases'] if c['policy'] == 1 and c['terrain'] == terrain
                and c['level'] == level and c['mode'] == 'direct' and c['velocity'] == [.6, 0., 0.])


flat, stair = case('flat', 0), case('stairs_up', 9)
tilts = np.degrees(np.arccos(np.clip(-d['gravity'][..., 2], -1, 1)))
peak = int(np.argmax(tilts[:stair['end_step'] + 1, stair['id']]))
snapshots = [(flat, 24, '平地起步：0.50秒'), (flat, 199, '平地行进：4.00秒'),
             (stair, peak, f'9级上楼：倾斜最大瞬间 {(peak + 1) * m["dt"]:.2f}秒')]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
colors = ['#2473b5', '#e38a24', '#548b52', '#a45393']
report = []
for ax, (c, step, title) in zip(axes[0], snapshots):
    i = c['id']
    points = d['body_pos'][step, i].copy()
    points -= points[base]
    w, x, y, z = d['body_quat'][step, i, base]
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    px = np.cos(yaw) * points[:, 0] + np.sin(yaw) * points[:, 1]
    pz = points[:, 2]
    for leg, color, label in zip(('fl', 'fr', 'rl', 'rr'), colors, ('左前', '右前', '左后', '右后')):
        ids = [names.index(f'{leg}_{part}_link') for part in ('hip', 'thigh', 'calf', 'foot')]
        ax.plot(px[ids], pz[ids], '-o', color=color, ms=4, lw=2, label=label)
        ax.plot(px[ids[-1]], pz[ids[-1]], 'o', ms=12, mfc='none', mec=color)
    frame = [names.index(f'{leg}_hip_link') for leg in ('fl', 'fr', 'rr', 'rl', 'fl')]
    ax.plot(px[frame], pz[frame], color='#555555', lw=3)
    ax.plot(0, 0, 'kx', ms=7)
    ax.set(xlim=(-.85, .85), ylim=(-.8, .55), aspect='equal',
           xlabel='相对机身前后位置（米，右为前）', ylabel='相对机身高度（米）', title=title)
    ax.grid(alpha=.2)
    q = d['joint_pos'][step, i, legs] - default[legs]
    ax.text(.03, .96, f'机身倾斜 {tilts[step, i]:.1f}°\n腿姿偏差RMS {np.sqrt(np.mean(q*q)):.3f} rad',
            va='top', transform=ax.transAxes, fontsize=10)
    report.append(dict(terrain=c['terrain'], time_s=(step+1)*m['dt'],
                       tilt_deg=float(tilts[step, i]), joint_rms_rad=float(np.sqrt(np.mean(q*q)))))
axes[0, 0].legend(loc='lower left', ncol=2, fontsize=8)
for c, label, color in ((flat, '平地 0.6 m/s', '#2473b5'), (stair, '9级上楼 0.6 m/s', '#e38a24')):
    i, n = c['id'], c['end_step'] + 1
    t = (np.arange(n)+1)*m['dt']
    q = d['joint_pos'][:n, i][:, legs] - default[legs]
    values = (np.sqrt(np.mean(q*q, axis=1)), tilts[:n, i], d['velocity'][:n, i, 0])
    for ax, value in zip(axes[1], values):
        ax.plot(t, value, label=label, color=color)
        ax.grid(alpha=.2)
        ax.set_xlabel('收到命令后的时间（秒）')
for ax, title in zip(axes[1], ('腿姿偏离默认角 RMS（rad）', '机身总倾斜角（度）', '机身前向速度（m/s）')):
    ax.set_title(title)
axes[1, 2].axhline(.6, color='gray', ls='--', lw=1)
axes[1, 0].legend(fontsize=9)
fig.suptitle('V6学生：实际记录的起步、平地与上楼姿态\n上排为刚体原点连线的侧视示意，非机器人外观录像；左右腿会重叠，未绘制地面', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .92), h_pad=3)
fig.savefig(b / 'v6_posture_explained.png', dpi=140)
plt.close(fig)
(b / 'v6_posture_snapshots.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
