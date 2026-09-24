"""Show all leg targets and measured angles, without smoothing telemetry."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--audit',default='candidate96')
parser.add_argument('--tag',default='joint_targets')
args=parser.parse_args()
directory=p/args.audit
m=json.loads((directory/'manifest.json').read_text())
with np.load(directory/'traces.npz') as f:
    q,target=f['joint_pos'],f['target']
legs=[j for j,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axs=plt.subplots(4,3,figsize=(15,11),sharex=True)
for ax,j in zip(axs.flat,legs):
    for policy,color in ((0,'#d14935'),(1,'#2176ae')):
        c=next(c for c in m['cases'] if c['policy']==policy and c['mode']=='direct' and c['direction']=='forward')
        n=c['end_step']+1
        t=(np.arange(n)+1)*m['dt']
        label='基线' if policy==0 else '专项候选'
        ax.plot(t,np.rad2deg(q[:n,c['id'],j]),color=color,label=label+'实际',linewidth=1.6)
        ax.plot(t,np.rad2deg(target[:n,c['id'],j]),color=color,linestyle='--',alpha=.6,label=label+'目标',linewidth=.9)
    ax.axhline(np.rad2deg(m['default_q'][j]),color='gray',linestyle=':',label='默认角度')
    ax.set_title(m['joint_names'][j])
    ax.set_xlim(0,4)
    ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
for ax in axs[:,0]:
    ax.set_ylabel('角度（°）')
for ax in axs[-1]:
    ax.set_xlabel('起步后时间（秒）')
fig.suptitle('平地立即收到0.6 m/s前进命令：12腿关节目标与实际角度\n目标为经过现有执行器限制后的控制目标；原始时程未平滑')
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(p/(args.tag+'_comparison.png'),dpi=150)
