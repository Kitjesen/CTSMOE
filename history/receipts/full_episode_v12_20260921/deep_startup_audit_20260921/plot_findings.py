"""Plot measured startup responses and exact delayed actuator requests."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

HERE = Path(__file__).resolve().parent
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                     "font.size": 11, "figure.dpi": 140})
m = json.loads((HERE/"results_r2/manifest.json").read_text())
z = np.load(HERE/"results_r2/traces.npz")
p = np.load(HERE/"results_r2/physics.npz")
first = np.load(HERE/"results_r1/traces.npz")
legs = [i for i,n in enumerate(m["joint_names"]) if not n.endswith("_foot_joint")]
default = np.asarray(m["default_q"])
t = np.arange(1,401)*.02
fig,axes = plt.subplots(2,2,figsize=(14,9))
series = [(z,2,"V12 直接给 0.6 m/s","#c83e39"),
          (first,3,"V12 仅降为 3 rad/s","#8a7c93"),
          (z,8,"V12 命令用 0.2 秒渐变","#d49327"),
          (z,10,"V12 同步重置目标＋0.2 秒渐变","#168064"),
          (z,0,"V6 同条件参考","#2876b7")]
for data,ci,label,color in series:
    q=data['joint_pos'][:,ci][:,legs]-default[legs]
    axes[0,0].plot(t,np.sqrt((q*q).mean(-1)),label=label,color=color,lw=1.8)
    axes[0,1].plot(t,data['height'][:,ci],color=color,lw=1.8)
axes[0,0].set(ylabel="12 腿关节相对默认姿态 RMS（rad）",title="仅压低目标变化率，没有消除大偏姿")
axes[0,0].legend(fontsize=9,loc='upper right')
axes[0,1].axhline(.46,color='#777777',ls='--',lw=1,label='目标高度')
axes[0,1].set(ylabel="机身离平地高度（m）",title="缓和命令切入后，机身不再明显下蹲")
for ax in axes[0]:
    ax.set_xlim(0,4)
    ax.set_xlabel('起步后时间（秒）')
for ci,label,color in [(2,'原流程','#c83e39'),(10,'同步重置目标＋命令渐变','#168064')]:
    use=p['calf_step']>=0
    tp=np.arange(use.sum())*.005
    joint=m['actuator_init']['calf']['joint_indices'].index(m['joint_names'].index('fl_calf_joint'))
    axes[1,0].step(tp,p['calf_delayed'][use,ci,joint],where='post',label=label,color=color,lw=1.9)
    tau=np.stack([np.abs(p[g+'_applied'][p[g+'_step']>=0,ci]).max(-1) for g in ('hip','thigh','calf')]).max(0)
    axes[1,1].plot(tp,tau,label=label,color=color,lw=1.9)
axes[1,0].axhline(default[m['joint_names'].index('fl_calf_joint')],color='#777',ls='--',lw=1,label='默认小腿角 −1.96 rad')
axes[1,0].set(xlim=(0,.06),xlabel='物理子步时间（秒）',ylabel='左前小腿送入 PD 的绝对目标角（rad）',title='确认了独立的重置目标错误：最初 10 ms 仍在追 0 rad')
axes[1,0].legend(fontsize=9)
axes[1,1].set(xlim=(0,.2),xlabel='物理子步时间（秒）',ylabel='当步所有腿关节最大绝对力矩（N·m）',title='联合诊断方案压住了起步的力矩冲击')
axes[1,1].legend(fontsize=9)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('冻结同一 V12 策略的起步诊断：没有训练、没有改变奖励\n固定平地案例；上排 50 Hz 状态，下排 200 Hz 执行器记录',fontsize=16)
fig.tight_layout(rect=(0,0,1,.93))
fig.savefig(HERE/'STARTUP_CAUSES.png')
print(HERE/'STARTUP_CAUSES.png')
