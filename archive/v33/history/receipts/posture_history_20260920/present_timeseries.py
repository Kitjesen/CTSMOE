"""Readable Chinese joint/body plots from retained first-episode evidence."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

p=Path(__file__).parent
font=Path('C:/Windows/Fonts/msyh.ttc')
assert font.is_file()
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                     'axes.unicode_minus':False,'font.size':11})
m=json.loads((p/'manifest.json').read_text())
with np.load(p/'traces.npz') as archive:
    data={k:archive[k] for k in ('joint_pos','joint_vel','target','torque','root_clearance',
                                 'velocity','command','body_quat')}
names=m['joint_names']
default=np.array(m['default_q'])
legs=[i for i,n in enumerate(names) if not n.endswith('_foot_joint')]
base=m['body_names'].index('base_link')

def select(update,mode='direct'):
    c=next(c for c in m['cases'] if m['history'][c['policy']]['label']==str(update) and c['mode']==mode)
    n=min(c['end_step']+1,200)
    return c['id'],n,(np.arange(n)+1)*m['dt']

fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
for update,color in ((31001,'#2176ae'),(42001,'#d14935')):
    i,n,t=select(update)
    delta=data['joint_pos'][:n,i][:,legs]-default[legs]
    axes[0].plot(t,np.sqrt(np.mean(delta**2,axis=1)),color=color,label=f'{update:,}轮')
    axes[1].plot(t,data['root_clearance'][:n,i],color=color)
    axes[2].plot(t,data['velocity'][:n,i,0],color=color)
axes[0].set_ylabel('12腿关节姿态偏差\nRMS（rad）')
axes[0].legend(loc='upper right')
axes[1].set_ylabel('机身相对地面高度（m）')
axes[1].axhline(.46,color='#6b7280',linestyle='--',label='目标高度0.46 m')
axes[1].legend(loc='upper right')
axes[2].set_ylabel('机身前向速度（m/s）')
axes[2].axhline(.6,color='#6b7280',linestyle='--',label='前进指令0.6 m/s')
axes[2].legend(loc='upper right')
axes[2].set_xlabel('起步后时间（秒）')
for ax in axes:
    ax.grid(alpha=.2)
    ax.set_xlim(0,4)
fig.suptitle('同一平地起步：早期策略与姿态修复中的策略\n立即给0.6 m/s；无噪声、无推扰、固定10 ms延迟；各取首次回合',fontsize=15)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(p/'body_timeseries_cn.png',dpi=150)
plt.close(fig)

i,n,t=select(42001)
fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True)
side_names={'fl':'左前','fr':'右前','rl':'左后','rr':'右后'}
kind_names={'hip':'髋关节','thigh':'大腿','calf':'小腿'}
for row,side in enumerate(('fl','fr','rl','rr')):
    for col,kind in enumerate(('hip','thigh','calf')):
        ax=axes[row,col]
        j=names.index(f'{side}_{kind}_joint')
        ax.plot(t,np.rad2deg(data['joint_pos'][:n,i,j]),color='#d14935',label='实际角度')
        ax.plot(t,np.rad2deg(data['target'][:n,i,j]),color='#2176ae',linestyle='--',label='下发目标角度')
        ax.axhline(np.rad2deg(default[j]),color='#6b7280',linestyle=':',label='默认站姿角度')
        ax.set_title(side_names[side]+' · '+kind_names[kind])
        ax.set_ylabel('角度（°）')
        ax.grid(alpha=.2)
        ax.set_xlim(0,4)
        if row==3:
            ax.set_xlabel('起步后时间（秒）')
axes[0,0].legend(fontsize=9)
fig.suptitle('42,001轮学生：平地立即起步，12个腿关节时程\n目标为动作处理/保护之后的下发角；左右关节坐标正负不同，请各自对照默认角',fontsize=15)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(p/'joint_angles_cn.png',dpi=150)
plt.close(fig)

fig,axes=plt.subplots(4,3,figsize=(15,10),sharex=True)
for row,side in enumerate(('fl','fr','rl','rr')):
    for col,kind in enumerate(('hip','thigh','calf')):
        ax=axes[row,col]
        j=names.index(f'{side}_{kind}_joint')
        ax.plot(t,data['joint_vel'][:n,i,j],color='#6f4bb2')
        ax.set_title(side_names[side]+' · '+kind_names[kind])
        ax.set_ylabel('角速度（rad/s）')
        ax.grid(alpha=.2)
        if row==3:
            ax.set_xlabel('起步后时间（秒）')
fig.suptitle('42,001轮学生：12个腿关节角速度\n20 ms控制步采样；不能用作5 ms物理步峰值上限证明',fontsize=15)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(p/'joint_velocity_cn.png',dpi=150)
plt.close(fig)

with (p/'42001_startup_full_timeseries.csv').open('w',newline='',encoding='utf-8-sig') as f:
    header=['time_s','height_m','body_vx_mps','body_vy_mps','body_vz_mps','command_vx_mps']
    header += [f'{name}_{kind}' for name in names for kind in ('actual_rad','target_rad','velocity_rad_s','torque_Nm')]
    w=csv.writer(f)
    w.writerow(header)
    for step in range(n):
        row=[float(t[step]),float(data['root_clearance'][step,i]),
             *data['velocity'][step,i].tolist(),float(data['command'][step,i,0])]
        for j in range(len(names)):
            row += [float(data[k][step,i,j]) for k in ('joint_pos','target','joint_vel','torque')]
        w.writerow(row)
result={'checkpoint':42001,'steps':n,'dt_s':m['dt'],
        'min_height_m':float(data['root_clearance'][:n,i].min()),
        'max_height_m':float(data['root_clearance'][:n,i].max()),
        'max_body_vx_mps':float(data['velocity'][:n,i,0].max()),
        'artifacts':['body_timeseries_cn.png','joint_angles_cn.png','joint_velocity_cn.png',
                     '42001_startup_full_timeseries.csv']}
(p/'presentation_receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
