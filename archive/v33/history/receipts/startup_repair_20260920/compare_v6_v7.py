"""Compare completed fixed-case V6/V7 evaluations without mixing training levels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

b = Path(__file__).parent
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus': False})
all_data = {}
for version in ('v6', 'v7'):
    directory = b / ('all_scene_' + version)
    m = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    analysis = json.loads((directory / 'analysis.json').read_text(encoding='utf-8'))
    with np.load(directory / 'traces.npz') as f:
        keys = ['initial_q', 'initial_root', 'joint_pos', 'root_clearance', 'velocity', 'gravity']
        if version == 'v7':
            keys += ['pit_front_gap', 'wheel_ground_force', 'platform_stop', 'base_contact']
        data = {k: f[k] for k in keys}
    all_data[version] = (m, analysis, data)
m6, a6, d6 = all_data['v6']
m7, a7, d7 = all_data['v7']
assert [{k: c[k] for k in ('policy', 'terrain', 'level', 'mode', 'velocity')} for c in m6['cases']] == [
    {k: c[k] for k in ('policy', 'terrain', 'level', 'mode', 'velocity')} for c in m7['cases']]
initial_deltas = {k: float(np.max(np.abs(d6[k] - d7[k]))) for k in ('initial_q', 'initial_root')}


def selected(analysis, terrain, level):
    return next(c for c in analysis['cases'] if c['policy'] == 1 and c['terrain'] == terrain
                and c['level'] == level and c['mode'] == 'direct' and c['velocity'] == [.6, 0., 0.])


fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for row, (terrain, level, title) in enumerate((('flat', 0, '平地'), ('stairs_up', 9, '9级上楼'), ('pit', 21, '1.2米高台'))):
    for version, color in (('v6', '#2674ac'), ('v7', '#db7139')):
        m, analysis, data = all_data[version]
        c = selected(analysis, terrain, level)
        i, n = c['id'], c['end_step'] + 1
        legs = [j for j, name in enumerate(m['joint_names']) if not name.endswith('_foot_joint')]
        q = data['joint_pos'][:n, i][:, legs] - np.array(m['default_q'])[legs]
        t = (np.arange(n) + 1) * m['dt']
        values = (np.sqrt(np.mean(q*q, axis=1)), data['root_clearance'][:n, i], data['velocity'][:n, i, 0])
        for ax, value in zip(axes[row], values):
            ax.plot(t, value, color=color, label=version.upper())
            ax.grid(alpha=.2)
            ax.set_xlabel('秒')
    axes[row, 0].set_ylabel(title)
    axes[row, 2].axhline(.6, ls='--', color='gray', lw=.8)
for ax, title in zip(axes[0], ('腿姿偏差 RMS（rad）', '机身下方射线净空（m）', '前向速度（m/s）')):
    ax.set_title(title)
axes[0, 0].legend()
fig.suptitle('V6（96轮修复）与V7（1000轮修复）：同为默认姿态起步、0.6 m/s\n高台曲线在失败时结束；楼梯射线净空随台阶参考面切换，不等同于机身升降', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(b / 'all_scene_v7/v6_v7_comparison.png', dpi=140)
plt.close(fig)
pit = selected(a7, 'pit', 21)
i, n = pit['id'], pit['end_step'] + 1
pit_context = dict(front_gap_last_m=float(d7['pit_front_gap'][n-1, i]),
                   peak_base_contact_n=float(d7['base_contact'][:n, i].max()),
                   stop_on_platform_ever=bool(d7['platform_stop'][:n, i].any()))
names = dict(flat='平地姿态＋跟踪', stairs_up='上楼梯', stairs_down='下楼梯', slope_up='上坡',
             slope_down='下坡', wave='波浪', rough_slope='粗糙坡', obstacles='障碍块', pit='高台')
lines = ['# V7千轮后评估：44,106轮', '',
         '训练和自动评估已完成；未自动继续长训练。三种策略共213个固定名义案例，学生每版71例。V6和V7案例配置一致；初始状态最大差：'+json.dumps(initial_deltas), '',
         '|项目|V6学生通过|V7学生通过|V7额外侧偏≤0.5m且通过4米|', '|---|---:|---:|---:|']
for terrain, name in names.items():
    g6 = next(g for g in a6['groups'] if g['policy'] == 1 and g['terrain'] == terrain)
    g7 = next(g for g in a7['groups'] if g['policy'] == 1 and g['terrain'] == terrain)
    straight = f"{g7['straight']}/{g7['count']}" if terrain not in ('flat', 'pit') else '不适用'
    lines.append(f"|{name}|{g6['passed']}/{g6['count']}|{g7['passed']}/{g7['count']}|{straight}|")
lines += ['', '平地通过需姿态和末秒跟踪同时合格；其余原地形表中通过表示前进4米且无物理失败，高台表示攀上并站稳。固定小样本不能解释为随机成功率。', '',
          '## 关键结果', '',
          '- V7平地0.6m/s首秒姿态RMS 0.217 rad，V6为0.116；峰值0.323对0.132 rad。高度仍稳定，但姿态偏差与该样本跟踪误差比V6大。',
          '- V7上楼6/6通过，但只有3/6同时满足侧偏≤0.5m。L9、0.6m/s侧偏0.515m，V6为0.222m。不能把6/6当作姿态与路线全面合格。',
          '- V7学生高台0/9：0.6、0.9、1.2m各三个速度全部触发illegal_contact，均未攀上；教师同为0/9，两个接触失败、七个超时。不是只发生在学生蒸馏，也不是登顶停车条件导致。',
          '- V7的1.2m、0.6m/s学生样本1.8秒终止，前进1.079m，倾斜最大4.13°，尚未形成有效攀爬。地形/阶段识别、奖励与学到的动作切换的因果关系仍需对照试验；不能凭现象指定唯一原因。',
          '- 不直接把V7替换为完整正式策略，也不因千轮已完成就自动加长相同训练。保留V6作为平地姿态基准。', '',
          '高台测量上下文：'+json.dumps(pit_context), '',
          '图见 v6_v7_comparison.png；逐场景指标见 analysis.json，全部地形图见 terrain_comparison.png。']
(b / 'all_scene_v7/V7_THOUSAND_EVALUATION.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(initial_deltas=initial_deltas, pit_context=pit_context,
    flat_failed=[{k:c[k] for k in ('mode','direction','pose_pass','tracking_pass','last_xy_error','last_yaw_error')}
                 for c in a7['cases'] if c['policy']==1 and c['terrain']=='flat' and not c['pass']],
    terrain_failures=[{k:c[k] for k in ('terrain','level','velocity','terms','dx_m')}
                     for c in a7['cases'] if c['policy']==1 and c['terrain'] in ('wave','slope_up') and not c['pass']]), indent=2))
