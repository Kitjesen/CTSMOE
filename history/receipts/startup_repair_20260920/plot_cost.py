import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p=Path(__file__).parent
rows=[r for r in json.loads((p/'cost_probe/cost_summary.json').read_text()) if r['policy']=='before']
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
x=np.arange(len(rows))
approach=np.array([r['approach_request_penalty'] for r in rows])
after=np.array([r['after_climb_seen_penalty'] for r in rows])
total=np.array([r['integrated_request_penalty'] for r in rows])
middle=total-approach-after
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x,approach,label='接近高台阶段',color='#edb763')
ax.bar(x,middle,bottom=approach,label='攀爬及过渡阶段',color='#eb785a')
ax.bar(x,after,bottom=approach+middle,label='已经爬上后的阶段',color='#b94243')
ax.axhline(15,color='#2176ae',linestyle='--',label='爬上+站稳事件奖励合计15（未含其他奖励）')
for i,t in enumerate(total):
    ax.text(i,t+1,f'{t:.1f}',ha='center')
ax.set_xticks(x,[str(r['speed'])+' m/s' for r in rows])
ax.set_ylabel('新增惩罚累计扣分的绝对值')
ax.set_title('把第三轮新惩罚回算到第二轮成功爬1.2米的轨迹\n主要扣分集中在爬上后的阶段，不能直接套用平地权重')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
fig.tight_layout()
fig.savefig(p/'pit_cost_breakdown.png',dpi=150,bbox_inches='tight')
