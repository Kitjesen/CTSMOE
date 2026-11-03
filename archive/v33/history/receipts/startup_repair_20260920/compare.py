"""Report fixed-input startup repair evidence and preserved terrain traversals."""
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
parser.add_argument('--regression',default='regression96')
parser.add_argument('--tag',default='repair')
args=parser.parse_args()
audit=p/args.audit
m=json.loads((audit/'manifest.json').read_text())
summary=json.loads((audit/'summary.json').read_text())
reg=json.loads((p/args.regression/'manifest.json').read_text())
with np.load(audit/'traces.npz') as f:
    d={k:f[k] for k in ('joint_pos','target','root_clearance','velocity')}
legs=[j for j,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default=np.array(m['default_q'])[legs]
font=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False})
fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
for policy,color in ((0,'#d14935'),(1,'#2176ae')):
    label=('基线' if policy==0 else '专项候选')+m['history'][policy]['label']+'轮'
    c=next(c for c in m['cases'] if c['policy']==policy and c['mode']=='direct' and c['direction']=='forward')
    i=c['id']
    n=c['end_step']+1
    t=(np.arange(n)+1)*m['dt']
    q=d['joint_pos'][:n,i][:,legs]-default
    axes[0].plot(t,np.sqrt(np.mean(q*q,axis=1)),color=color,label=label)
    axes[1].plot(t,d['root_clearance'][:n,i],color=color)
    axes[2].plot(t,d['velocity'][:n,i,0],color=color)
axes[0].set_ylabel('腿姿RMS（rad）')
axes[0].legend()
axes[1].set_ylabel('机身高度（m）')
axes[1].axhline(.46,color='gray',linestyle='--')
axes[2].set_ylabel('前向速度（m/s）')
axes[2].axhline(.6,color='gray',linestyle='--')
axes[2].set_xlabel('起步后时间（秒）')
for ax in axes:
    ax.grid(alpha=.2)
    ax.set_xlim(0,4)
fig.suptitle('相同条件平地直接起步对照：前进指令0.6 m/s\n无噪声/推扰、固定10ms延迟；未增加推理时姿态修正或新的硬裁剪')
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(p/(args.tag+'_comparison.png'),dpi=150)
result={}
for policy,label in ((0,'before'),(1,'candidate')):
    rows=[r for r in summary if r['policy']==policy]
    result[label]=dict(direct_pass=sum(r['startup_pass'] for r in rows if r['mode']=='direct'),
                       settled_pass=sum(r['startup_pass'] for r in rows if r['mode']=='settle'),
                       combined_pass=sum(r['combined_pass'] for r in rows),
                       forward=next(r for r in rows if r['mode']=='direct' and r['direction']=='forward'))
for label in ('before','current'):
    cases=[c for c in reg['cases'] if c['policy']==label]
    stair=[c for c in cases if c['terrain']!='pit']
    pit=[c for c in cases if c['terrain']=='pit']
    result['regression_'+label]=dict(stair_pass=sum(c['passed_4m'] for c in stair),
        stair_straight=sum(c['straight_4m'] for c in stair),stair_count=len(stair),
        pit_climb=sum(c['climb'] for c in pit),pit_stable=sum(c['stable'] for c in pit),pit_count=len(pit),
        physical_fail=sum(c['physical'] for c in cases))
result['nominal_startup_gate']=result['candidate']['combined_pass']==20
result['terrain_no_count_regression']=all(result['regression_current'][k]>=result['regression_before'][k]
    for k in ('stair_pass','stair_straight','pit_climb','pit_stable'))
(p/(args.tag+'_comparison.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
