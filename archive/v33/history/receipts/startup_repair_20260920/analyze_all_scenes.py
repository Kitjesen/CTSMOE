"""Summarize matched first episodes; preserve failures and expose sample counts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
m = json.loads((a.directory/'manifest.json').read_text(encoding='utf-8'))
if m.get('smoke_only', False):
    raise ValueError('Smoke traces are not a behavior evaluation')
version = m['history'][1]['fields']['startup_repair_profile'].removeprefix('omni_start_').upper()
with np.load(a.directory/'traces.npz') as f:
    d = {k:f[k] for k in f.files}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus':False})
names = dict(flat='平地',stairs_up='上楼梯',stairs_down='下楼梯',slope_up='上坡',
             slope_down='下坡',wave='波浪',rough_slope='粗糙坡',obstacles='障碍块',pit='高台')
legs = [i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
wheels = [i for i in range(len(m['joint_names'])) if i not in legs]
base_id = m['body_names'].index('base_link')
default = np.array(m['default_q'])
rows = []
for c in m['cases']:
    i,n = c['id'],c['end_step']+1
    for key in ('joint_pos','root_clearance','torque_peak','speed_peak','velocity','command'):
        assert np.isfinite(d[key][:n,i]).all(),(key,i)
    q = d['joint_pos'][:n,i][:,legs]-default[legs]
    rms = np.sqrt(np.mean(q*q,axis=-1))
    start = 100 if c['mode']=='settle' else 0
    first = rms[start:min(n,start+50)]
    last = slice(max(start,n-50),n)
    xy = np.linalg.norm(d['velocity'][:n,i,:2]-d['command'][:n,i,:2],axis=-1)
    yaw = np.abs(d['angular_velocity'][:n,i,2]-d['command'][:n,i,2])
    torque = d['torque_peak'][:n,i]
    speed = d['speed_peak'][:n,i]
    tilt = np.degrees(np.arccos(np.clip(-d['gravity'][:n,i,2],-1,1)))
    physical = c['physical']
    early = any(t not in ('time_out','stair_route_exit','budget') for t in c['terms'])
    row = dict(**c,label=m['history'][c['policy']]['label'],
        first_rms=float(first.mean()) if len(first) else None,
        max_rms=float(rms.max()),height_min=float(d['root_clearance'][:n,i].min()),
        height_max=float(d['root_clearance'][:n,i].max()),
        xy_mae=float(xy[start:].mean()) if n>start else None,
        last_xy_error=float(xy[last].mean()) if n>start else None,
        last_yaw_error=float(yaw[last].mean()) if n>start else None,
        tilt_max_deg=float(tilt.max()),
        leg_torque_peak_nm=float(torque[:,legs].max()),wheel_torque_peak_nm=float(torque[:,wheels].max()),
        leg_speed_peak_rad_s=float(speed[:,legs].max()),wheel_speed_peak_rad_s=float(speed[:,wheels].max()),
        hard_position_excess_rad=float(d['hard_excess'][:n,i][:,legs].max()),
        leg_saturation_fraction=float(d['saturation'][:n,i][:,legs].mean()),
        base_contact_peak_n=float(d['base_contact'][:n,i].max()))
    if c['terrain']=='flat':
        row['pose_pass']=bool(not physical and not early and len(first)==50 and row['first_rms']<.25
            and row['max_rms']<.35 and row['height_min']>.34 and row['height_max']<.60
            and np.linalg.norm(d['velocity'][:n,i,:2],axis=-1).max()<max(.5,np.linalg.norm(c['velocity'][:2])+.5))
        row['tracking_pass']=bool(n-start>=100 and row['last_xy_error']<.15 and row['last_yaw_error']<.2)
        row['pass']=row['pose_pass'] and row['tracking_pass']
    elif c['terrain']=='pit':
        row['pass']=bool(c['climb'] and c['stable'] and not physical)
    else:
        row['pass']=bool(c['passed_4m'] and not physical)
    rows.append(row)

groups=[]
for policy in range(len(m['history'])):
    for terrain in names:
        rr=[r for r in rows if r['policy']==policy and r['terrain']==terrain]
        groups.append(dict(policy=policy,terrain=terrain,count=len(rr),passed=sum(r['pass'] for r in rr),
            straight=sum(r.get('straight_4m',False) for r in rr),physical=sum(r['physical'] for r in rr),
            mean_xy_mae=float(np.mean([r['xy_mae'] for r in rr if r['xy_mae'] is not None])),
            peak_leg_torque=max(r['leg_torque_peak_nm'] for r in rr),
            peak_wheel_torque=max(r['wheel_torque_peak_nm'] for r in rr),
            peak_leg_speed=max(r['leg_speed_peak_rad_s'] for r in rr),
            peak_wheel_speed=max(r['wheel_speed_peak_rad_s'] for r in rr),
            max_hard_excess=max(r['hard_position_excess_rad'] for r in rr)))
(a.directory/'analysis.json').write_text(json.dumps(dict(cases=rows,groups=groups),indent=2),encoding='utf-8')

colors=['#c95845','#2674ac','#4a8d66']
labels=['训练前学生',version+'学生',version+'教师']
fig,ax=plt.subplots(figsize=(13,5))
x=np.arange(len(names))
for policy in range(len(m['history'])):
    gg=[g for g in groups if g['policy']==policy]
    offset=(policy-1)*.25
    bars=ax.bar(x+offset,[g['passed']/g['count']*100 for g in gg],.23,color=colors[policy],label=labels[policy])
    for bar,g in zip(bars,gg):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{g['passed']}/{g['count']}",ha='center',fontsize=8)
ax.set_xticks(x,list(names.values()))
ax.set_ylim(0,118)
ax.set_ylabel('固定测试通过比例（%）')
ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.02))
ax.set_title('九类地形统一初始状态对照\n平地：姿态+跟踪；其他原地形：前进4米；高台：攀上+站稳。非随机成功率。')
fig.tight_layout()
fig.savefig(a.directory/'terrain_comparison.png',dpi=150)
plt.close(fig)

fig,axes=plt.subplots(9,3,figsize=(14,22))
for j,terrain in enumerate(names):
    for policy in range(len(m['history'])):
        candidates=[r for r in rows if r['terrain']==terrain and r['policy']==policy
                    and r['mode']=='direct' and r['direction']=='forward' and r['velocity'][0]==.6]
        r=max(candidates,key=lambda c:c['level'])
        i,n=r['id'],r['end_step']+1
        t=(np.arange(n)+1)*m['dt']
        q=d['joint_pos'][:n,i][:,legs]-default[legs]
        for ax,y in zip(axes[j],(d['velocity'][:n,i,0],d['root_clearance'][:n,i],np.sqrt((q*q).mean(-1)))):
            ax.plot(t,y,color=colors[policy],label=labels[policy],lw=1.4)
            ax.grid(alpha=.2)
        axes[j,0].set_ylabel(names[terrain]+f" L{r['level']}")
    axes[j,0].axhline(.6,color='gray',ls='--',lw=.8)
    if terrain=='flat':
        axes[j,1].axhline(.46,color='gray',ls='--',lw=.8)
for ax,title in zip(axes[0],('前向速度 m/s','机身垂直射线净空 m','腿相对默认姿态 RMS rad')):
    ax.set_title(title)
axes[0,0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('秒（终止后不延伸曲线）')
fig.suptitle('每类最高测试档、0.6m/s代表时程；坡面/攀爬姿态偏离不按平地标准判失败',y=.998)
fig.tight_layout(rect=(0,0,1,.985))
fig.savefig(a.directory/'terrain_timeseries.png',dpi=120)
plt.close(fig)

for terrain,level in (('flat',0),('pit',21)):
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True)
    for policy in (0,1):
        r=next(r for r in rows if r['terrain']==terrain and r['level']==level and r['policy']==policy
               and r['mode']=='direct' and r['direction']=='forward' and r['velocity'][0]==.6)
        i,n=r['id'],r['end_step']+1
        t=(np.arange(n)+1)*m['dt']
        for ax,joint in zip(axes.flat,legs):
            ax.plot(t,d['joint_pos'][:n,i,joint]-default[joint],color=colors[policy],label=labels[policy]+'实际')
            if policy==1:
                request_index=m['request_joint_ids'].index(joint)
                ax.plot(t,d['target'][:n,i,joint]-default[joint],color='#7755aa',ls='--',label=version+'目标')
                ax.plot(t,d['requested'][:n,i,request_index]-default[joint],color='#999999',alpha=.7,label=version+'原始请求')
            ax.set_title(m['joint_names'][joint],fontsize=10)
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f"{names[terrain]} L{level}、0.6m/s：12腿关节相对默认角的时程（rad）")
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(a.directory/f'{terrain}_joint_timeseries.png',dpi=130)
    plt.close(fig)

lines=[f'# 九地形 {version} 固定策略验证','',m['protocol'],'',
       f'|地形|源学生通过|{version}学生通过|{version}教师通过|{version}学生物理失败|',
       '|---|---:|---:|---:|---:|']
for terrain,name in names.items():
    gg=[next(g for g in groups if g['terrain']==terrain and g['policy']==k) for k in range(3)]
    lines.append('|'+name+'|'+'|'.join(f"{g['passed']}/{g['count']}" for g in gg)+f"|{gg[1]['physical']}|")
lines += ['', '指标注意：非平地关节姿态偏离包含必要攀爬动作，不能用默认角偏差单独判坏。扭矩/轮速采用每个20ms步内四个5ms子步的峰值；关节时程为50Hz。净空按机身正下方射线测量，在台阶边缘会发生参考地面切换，不能把曲线跳变全部当机身跳动。',
          '', f'这些是固定名义条件的场景覆盖测试，没有覆盖观测噪声、推扰、随机摩擦和全部速度范围；不能当作实机部署验收。本报告比较源策略与清单记录的{version}检查点，不推断其他在训策略的表现。']
(a.directory/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(groups,ensure_ascii=False,indent=2))
