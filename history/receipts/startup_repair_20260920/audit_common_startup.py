"""Compare shared startup behavior across terrain; no terrain-blind pass labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

b = Path(__file__).parent/'all_scene_v6'
m = json.loads((b/'manifest.json').read_text(encoding='utf-8'))
with np.load(b/'traces.npz') as f:
    d = {k:f[k] for k in ('requested','joint_pos','target','velocity','body_pos')}
names = dict(flat='平地',stairs_up='上楼梯',stairs_down='下楼梯',slope_up='上坡',
             slope_down='下坡',wave='波浪',rough_slope='粗糙坡',obstacles='障碍块',pit='高台')
legs = [i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default = np.array(m['default_q'])
request_ids = m['request_joint_ids']
order = [request_ids.index(j) for j in legs]
q0 = default[legs]


def case(policy, terrain):
    rr = [c for c in m['cases'] if c['policy']==policy and c['terrain']==terrain
          and c['mode']=='direct' and c['direction']=='forward' and c['velocity']==[.6,0.,0.]]
    return max(rr,key=lambda c:c['level'])


def cosine(a, z):
    denom = np.linalg.norm(a)*np.linalg.norm(z)
    return float(np.dot(a.ravel(),z.ravel())/denom) if denom>1e-12 else None


rows=[]
for policy in range(3):
    reference=case(policy,'flat')
    ref=d['requested'][:,reference['id']][:,order]-q0
    for terrain in names:
        c=case(policy,terrain)
        i,n=c['id'],c['end_step']+1
        request=d['requested'][:n,i][:,order]-q0
        q=d['joint_pos'][:n,i][:,legs]-q0
        first=min(50,n,reference['end_step']+1)
        row=dict(policy=policy,terrain=terrain,level=c['level'],
            first_step_request_cosine_to_flat=cosine(request[0],ref[0]),
            first_second_request_cosine_to_flat=cosine(request[:first],ref[:first]),
            first_second_joint_rms=float(np.sqrt((q[:first]**2).mean(-1)).mean()),
            half_second_actual_offset=q[24].tolist() if n>=25 else None,
            half_second_requested_offset=request[24].tolist() if n>=25 else None,
            first_second_body_z_min=float(d['body_pos'][:first,i,0,2].min()),
            first_second_body_z_max=float(d['body_pos'][:first,i,0,2].max()))
        rows.append(row)
(b/'common_startup.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus':False})
fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
for policy,ax in enumerate(axes):
    rr=[r for r in rows if r['policy']==policy]
    values=np.array([r['half_second_actual_offset'] for r in rr])
    im=ax.imshow(values,cmap='RdBu_r',vmin=-.8,vmax=.8,aspect='auto')
    ax.set_yticks(range(9),list(names.values()))
    ax.set_title(('原策略学生' if policy==0 else 'V6学生')+'：收到0.6m/s指令后0.5秒的实际角偏差')
    ax.set_xticks(range(12),[m['joint_names'][i].replace('_joint','') for i in legs],rotation=30)
    for j in range(9):
        for k in range(12):
            ax.text(k,j,f'{values[j,k]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(values[j,k])>.5 else 'black')
fig.colorbar(im,ax=axes.tolist(),label='相对默认关节角（rad）',fraction=.025,pad=.02)
fig.suptitle('跨地形共同起步行为检查；相似动作不自动等于错误，仍需结合障碍与支撑阶段判断')
fig.subplots_adjust(top=.9,bottom=.1,right=.9,hspace=.3)
fig.savefig(b/'common_startup_joint_map.png',dpi=140)
plt.close(fig)

for r in rows:
    print(r['policy'],r['terrain'],round(r['first_second_joint_rms'],3),
          round(r['first_second_request_cosine_to_flat'],3))
