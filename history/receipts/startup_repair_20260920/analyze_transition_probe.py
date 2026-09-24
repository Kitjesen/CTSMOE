"""Validate telemetry and compare encoders on the exact same observed state."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p=Path(__file__).parent
d=p/'transition_probe'
m=json.loads((d/'manifest.json').read_text())
with np.load(d/'traces.npz') as f:
    a={k:f[k] for k in f.files}
names=[f'{s}_{j}_joint' for s in ('fr','fl','rr','rl') for j in ('hip','thigh','calf')]
ids=[m['joint_names'].index(s) for s in names]
nominal=np.array(m['default_q'])[ids]
rows=[]
for c in m['cases']:
    i,n=c['id'],c['end_step']+1
    start=100 if c['mode']=='settle' else 0
    stop=start+100
    for k,v in a.items():
        assert np.isfinite(v[:n,i]).all(),(k,i)
    assert a['active'][:n,i].all()
    expected=np.zeros((n,3))
    expected[start:min(n,stop)]=c['velocity']
    np.testing.assert_allclose(a['command'][:n,i],expected,atol=1e-6)
    source='teacher_action' if c['teacher'] else 'student_action'
    np.testing.assert_allclose(a['raw_action'][:n,i],a[source][:n,i],atol=1e-6)
    weights=a['expert_weights'][:n,i]
    np.testing.assert_allclose(weights.sum(-1),1.,atol=1e-5)
    assert (weights>=0).all()
    # Recorded pre-state must equal the previous post-state; confirms time alignment.
    np.testing.assert_allclose(a['pre_joint_pos'][1:n,i],a['joint_pos'][:n-1,i],atol=1e-5)
    q=a['joint_pos'][:n,i][:,ids]-nominal
    rms=np.sqrt(np.mean(q*q,axis=-1))
    student=.25*a['student_action'][:n,i,:12]
    teacher=.25*a['teacher_action'][:n,i,:12]
    movement=slice(start,min(start+50,n))
    tail=slice(max(stop,n-50),n)
    motiontail=slice(max(start,stop-25),min(stop,n))
    row=dict(label=m['history'][c['policy']]['label'],mode=c['mode'],direction=c['direction'],
        actual_first_second_rms=float(rms[movement].mean()),
        same_state_action_gap_first_second_rad=float(np.sqrt(((student[movement]-teacher[movement])**2).mean())),
        first_student_request_rms_rad=float(np.sqrt((student[start]**2).mean())),
        first_teacher_request_rms_rad=float(np.sqrt((teacher[start]**2).mean())),
        first_expert_weights=weights[start].tolist(),
        dominant_expert_switches_first_second=int((weights[movement].argmax(-1)[1:]!=weights[movement].argmax(-1)[:-1]).sum()),
        max_gate_change_first_second=float(np.abs(np.diff(weights[movement],axis=0)).sum(-1).max()),
        min_height_m=float(a['root_clearance'][:n,i].min()),
        max_height_m=float(a['root_clearance'][:n,i].max()),
        move_xy_error=float(np.linalg.norm(a['velocity'][motiontail,i,:2]-np.array(c['velocity'][:2]),axis=-1).mean()),
        move_yaw_error=float(np.abs(a['angular_velocity'][motiontail,i,2]-c['velocity'][2]).mean()),
        stop_xy_speed=float(np.linalg.norm(a['velocity'][tail,i,:2],axis=-1).mean()),
        stop_yaw_speed=float(np.abs(a['angular_velocity'][tail,i,2]).mean()),
        stop_pose_rms=float(rms[tail].mean()),
        stop_height_mean=float(a['root_clearance'][tail,i].mean()),terms=c['terms'])
    row['stop_diagnostic_pass']=bool(row['stop_xy_speed']<.1 and row['stop_yaw_speed']<.1
        and row['stop_pose_rms']<.2 and .4<row['stop_height_mean']<.52)
    rows.append(row)
(d/'analysis.json').write_text(json.dumps(rows,indent=2))
(d/'validation.json').write_text(json.dumps(dict(valid=True,cases=len(rows),
    checks=['finite','command_phase','executed_action_source','expert_probability','pre_post_state_alignment'])))
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True)
for column,mode in enumerate(('direct','settle')):
    for policy,color in ((0,'#16836f'),(2,'#c04739')):
        c=next(x for x in m['cases'] if x['policy']==policy and x['mode']==mode and x['direction']=='forward')
        i,n=c['id'],c['end_step']+1
        t=(np.arange(n)+1)*m['dt']
        t0=t-m['dt']
        label=m['history'][policy]['label']
        rms=np.sqrt(((a['joint_pos'][:n,i][:,ids]-nominal)**2).mean(-1))
        axs[0,column].plot(t,rms,color=color,label=label)
        axs[1,column].plot(t,a['root_clearance'][:n,i],color=color)
        axs[2,column].plot(t,a['velocity'][:n,i,0],color=color)
    axs[2,column].plot(t0,a['command'][:n,i,0],'--',color='gray',label='前进命令')
    for ax in axs[:,column]:
        ax.grid(alpha=.2)
        ax.axvline(2 if mode=='direct' else 4,color='gray',ls=':')
    axs[0,column].set_title('立即启动' if mode=='direct' else '站立2秒后启动')
    axs[0,column].legend()
    axs[2,column].set_xlabel('时间（秒）')
axs[0,0].set_ylabel('实际腿姿 RMS（rad）')
axs[1,0].set_ylabel('机身离地高度（m）')
axs[2,0].set_ylabel('前向速度（m/s）')
for row, limits in zip(axs, ((0.,.6),(.26,.48),(-.15,1.2))):
    for ax in row:
        ax.set_ylim(*limits)
fig.suptitle('独立启停复测：前进0.6 m/s持续2秒，然后三轴指令归零\n名义平地、固定10ms延迟、无噪声推扰；未更新策略')
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(p/'transition_start_stop.png',dpi=150)
print(json.dumps([r for r in rows if r['direction']=='forward'],indent=2))
for label in sorted(set(r['label'] for r in rows)):
    group=[r for r in rows if r['label']==label]
    print(label,'stop pass',sum(r['stop_diagnostic_pass'] for r in group),'/',len(group))
