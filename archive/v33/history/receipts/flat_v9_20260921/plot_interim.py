import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
b=Path(__file__).parent/'interim_44001'
m=json.loads((b/'manifest.json').read_text())
d=np.load(b/'forward_traces.npz')
legs=[i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
q0=np.asarray(m['default_q'])[legs]
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False})
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
colors=['#777777','#d75a47','#2677b5','#58934f']
for j,case_id in enumerate(d['ids']):
 c=next(c for c in m['cases'] if c['id']==case_id)
 label=m['history'][c['policy']]['label']
 n=min(300,c['end_step']+1)
 t=(np.arange(n)+1)*m['dt']
 q=d['joint_pos'][:n,j][:,legs]-q0
 vals=(np.sqrt((q*q).mean(-1)),d['root_clearance'][:n,j],d['velocity'][:n,j,0])
 for ax,y in zip(axes,vals):
  ax.plot(t,y,label=label,color=colors[j],ls='--' if c['teacher'] else '-')
axes[1].axhline(.46,color='black',ls=':',label='目标高度0.46m')
axes[2].axhline(.6,color='black',ls=':',label='前进指令0.6m/s')
for ax,title in zip(axes,['腿相对默认角RMS (rad)','机身高度 (m)','前向速度 (m/s)']):
 ax.set_ylabel(title)
 ax.grid(alpha=.25)
 ax.legend(fontsize=9,ncol=3)
axes[-1].set_xlabel('起步后时间 (s)')
fig.suptitle('V9续训290轮时的中途对照：相同默认站姿、前进0.6m/s\n无噪声/推扰，固定10ms延迟；不是千轮最终验收')
fig.tight_layout()
fig.savefig(b/'forward_comparison.png',dpi=140)
print(str(b/'forward_comparison.png'))
