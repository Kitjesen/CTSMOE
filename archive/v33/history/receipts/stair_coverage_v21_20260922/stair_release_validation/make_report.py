"""Generate review charts and a clearly non-released ONNX interface bundle."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
review = ROOT/'review'
result = json.loads((review/'BEHAVIOR_REVIEW.json').read_text())
summary = json.loads((review/'SUMMARY.json').read_text())
selected = json.loads((review/'SELECTED_CASES.json').read_text())
with np.load(review/'SELECTED_TRACES.npz') as z:
    traces = dict(z)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(4, 4, figsize=(19, 12), constrained_layout=True)
for col, row in enumerate(selected[:4]):
    key = row['condition']+'_'+str(row['id'])+'__'
    dt = row['dt']
    q = traces[key+'joint_pos']
    t = (np.arange(len(q))+1)*dt
    leg = [i for i,n in enumerate(row['joint_names']) if not n.endswith('foot_joint')]
    default = np.asarray(row['default_q'])
    rms = np.sqrt(((q[:,leg]-default[leg])**2).mean(-1))
    terrain = '上楼' if row['terrain']=='stairs_up' else '下楼'
    label = '干净' if row['condition']=='nominal' else '扰动'
    axes[0,col].set_title(f"{label} {terrain} L{row['level']} · {row['speed']} m/s\n固定例 {row['id']} · {'达标' if row['complete'] else '未达标'}")
    axes[0,col].plot(t,rms)
    axes[1,col].plot(t,traces[key+'velocity'][:,0],label='实际vx')
    axes[1,col].axhline(row['speed'],color='grey',linestyle='--',label='运动指令幅值')
    base = row['body_names'].index('base_link')
    axes[2,col].plot(t,traces[key+'body_pos'][:,base,1])
    axes[2,col].axhline(.5,color='red',linestyle='--')
    axes[2,col].axhline(-.5,color='red',linestyle='--')
    axes[3,col].plot(t,traces[key+'world_velocity'][:,2])
    platform = np.flatnonzero(traces[key+'stair_platform_hull'])
    for ax in axes[:,col]:
        ax.grid(alpha=.25)
        if len(platform):
            ax.axvline(t[platform[0]],color='green',linestyle=':',label='全身进入平台')
        if row['physical']:
            ax.axvline(t[-1],color='red',linestyle=':')
    axes[3,col].set_xlabel('回合时间 / s')
for row,label in zip(range(4),['12腿关节偏差RMS / rad','前向速度 / m/s','相对地块中心侧偏 / m','世界竖直速度 / m/s']):
    axes[row,0].set_ylabel(label)
fig.suptitle('34823 ONNX连续楼梯：6个台阶＋末端平台；绿色竖线为全身进入平台\n角度偏差同时包含攀爬所需姿态，不能直接等同起步异常',fontsize=15)
fig.savefig(ROOT/'STAIR_TRACES.png',dpi=150)
plt.close(fig)

failed = next(x for x in selected if x['physical'])
key = failed['condition']+'_'+str(failed['id'])+'__'
fig, axes = plt.subplots(4,3,figsize=(16,11),constrained_layout=True)
count = len(traces[key+'joint_pos'])
t=(np.arange(count)+1)*failed['dt']
for j,(ax,joint_id) in enumerate(zip(axes.flat,failed['request_joint_ids'])):
    default=failed['default_q'][joint_id]
    ax.plot(t,traces[key+'requested'][:,j]-default,label='原始请求',alpha=.7)
    ax.plot(t,traces[key+'target'][:,joint_id]-default,label='限制后目标')
    ax.plot(t,traces[key+'joint_pos'][:,joint_id]-default,label='实际角度')
    ax.set_title(failed['joint_names'][joint_id])
    ax.set_ylabel('相对默认角 / rad')
    ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8)
fig.suptitle('未通过例：L9上楼1.2 m/s，观测噪声＋初始偏航；2.62秒触发转速终止',fontsize=15)
fig.savefig(ROOT/'FAILED_CASE_12_JOINTS.png',dpi=150)
plt.close(fig)

table=['|速度|上楼：干净|上楼：扰动|下楼：干净|下楼：扰动|','|---|---:|---:|---:|---:|']
for speed in (.3,.6,1.2):
    values=[]
    for terrain,condition in [('stairs_up','nominal'),('stairs_up','perturbed'),('stairs_down','nominal'),('stairs_down','perturbed')]:
        g=next(g for g in result['groups'] if (g['terrain'],g['condition'],g['speed'])==(terrain,condition,speed))
        values.append(f"{g['complete']}/{g['total']}")
    table.append('|'+str(speed)+' m/s|'+'|'.join(values)+'|')
parities=[json.loads(p.read_text()) for p in (review/'results_verified').rglob('ONNX_PARITY.json')]
max_error=max(p['max_absolute_error'] for p in parities)
active_steps=sum(round(c['seconds']/.02) for c in result['cases'])
report=f'''# 34823连续楼梯与ONNX验收

结论：**楼梯专项行为门槛未全通过，不能按“无问题”放行实机。ONNX候选文件已完成转换和接口校验。**

143/144例曾全身进入末端平台并形成至少3轮支撑，97/144例完成对应的侧偏和继续/停车要求。物理失败1例。平台到达是“曾到达”，不代表之后一直保持稳定；完整通过数才包含后续要求。

这次实际地形是**连续6个台阶＋末端平台**。射线实测上楼单阶12.5/17/23.5cm，累计升高0.75/1.02/1.41m；下楼单阶12/17.5/25cm，累计下降0.72/1.05/1.50m。它不是单级高台。这是封顶平台专项几何，外圈阶数比原地形少；此前原始楼梯学生上楼5/6、下楼6/6，以及L9上楼1.2m/s超时的证据保留，未用本测试替代。

144固定首回合包含两种方向、三级、三种速度、立即给速/先站2秒、平台继续/停车、干净/扰动两组。扰动组开启观测噪声，并设初始偏航±5°及侧移±0.1m，未启用推扰或质量摩擦随机化。它不是随机成功率估计，不能外推为实机成功率。

{'\n'.join(table)}

每格12例由三级×两种启动×两种平台指令组成。通过需要真实平台全身/支撑、无物理终止、全程侧偏≤0.5m；停车例还要求线速度<0.15m/s、总角速度<0.3rad/s、倾角<15°、连续支撑2秒。25秒超时保持原规则。

## 未通过项

- 34例侧偏超过0.5m，最大1.55m；属于路线保持不足，不等同没有下完台阶。扰动组低速下楼0/12完整达标，但12/12均曾支撑于末端平台。
- 14例未取得2秒连续安静支撑，含物理终止样本；上述类别有交叉。72个停车例中58个取得安静支撑2秒，计入侧偏等条件后47/72完整达标。
- 1例L9上楼、1.2m/s、直接给速、扰动条件在2.62秒触发速度保护；轮速峰值49.87rad/s，超过30rad/s名义值，也超过45rad/s极端终止阈值。腿速18.75rad/s，腿力矩115.51Nm、轮17Nm。终止原因字段为speed，position/torque均false。
- 全144例中5例腿实际速度超过15.001rad/s，18例轮超过30.001rad/s；仿真允许短瞬态，不代表可忽略实机限速。实际力矩峰值腿120Nm、轮17Nm。
- 运动命令后的首秒，最大关节RMS偏差0.468rad、单关节最大1.026rad。高速样本在这个窗口内可能已经接触首级台阶，因此这些数值不能单独证明平地起步退化；应结合接触阶段和逐关节时程解释。
- 全身进入平台后的正向竖直速度峰值0.223m/s，比旧大上冲场景温和，但不据此宣称末端恢复已完全解决。

## ONNX与输入输出校验

- ONNX opset17，history[N,530]＋single_obs[N,53]→actions[N,16]，动态batch。
- 实际仿真由ONNX Runtime输出驱动，逐步与原检查点student相同输入对比，最大绝对误差{max_error:.8g}；有效首回合控制样本合计{active_steps}。数值误差不能解释当前行为失败。
- 1704组真实观测、历史拼接对比误差0；41829组目标后处理对比误差0；单机器人batch1与batch72最大差1.91e-6。
- 物理源码、模型、双优化器所在检查点哈希均保持，新增优化器更新0。原训练没有恢复。
- 实机加载程序、URDF到驱动器符号/减速比映射和实机保护链没有经过本次接入验收。当前候选不等于已兼容用户的实机程序。

## 产物与下一步

逐例指标见`review/BEHAVIOR_REVIEW.json`，1728行逐关节汇总见`review/PER_JOINT.csv`，关键时程见两张PNG。远端保留全部144例原始时程和每类观测样本。完整原始目录为`/root/autodl-tmp/cts-omni-repair-v19-20260922/stair_release_validation_attempt02_20260922/results_verified`。

优先补齐低速下楼的方向保持/停车支撑，并限制高速起步的异常请求与超速。应做单因素对照，保留已通过的0.3/0.6m/s干净上楼能力；本轮仅诊断和候选导出，不自动更改训练或放宽门槛。

技术执行记录：首次转换在JIT数值比较阶段遇到autograd/inference执行计划冲突，改为始终关闭梯度的推理转换；学生单模型检查使用明确的单学生源契约；NumPy维度序列化改成原生int。前三个技术入口均无优化器更新，原始失败证据保留。随后完成两组72例。时程分析和接口比对改为一次性解压数组，避免重复读取压缩成员；未改动仿真结果。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')

package=ROOT/'Thunder_CTSMoE_V21_34823_ONNX_CANDIDATE_NOT_RELEASED'
package.mkdir(exist_ok=False)
for name in ('thunder_cts_moe_34823.onnx','CONVERSION.json','actuator_contract.json','training_context.json','README.md'):
    shutil.copy2(ROOT/'export_candidate'/name,package/name)
def strict_json(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:strict_json(v) for k,v in value.items()}
    if isinstance(value,list):
        return [strict_json(x) for x in value]
    return value
contract=json.loads((ROOT/'export_candidate/runtime_contract.json').read_text())
(package/'runtime_contract.json').write_text(json.dumps(strict_json(contract),indent=2,allow_nan=False))
for name in ('BEHAVIOR_REVIEW.json','SUMMARY.json','IO_VERIFICATION.json','PER_JOINT.csv'):
    shutil.copy2(review/name,package/name)
for name in ('reference_io.py','REPORT.md','STAIR_TRACES.png','FAILED_CASE_12_JOINTS.png'):
    shutil.copy2(ROOT/name,package/name)
status=dict(release_status='HOLD_BEHAVIOR_FAILED', completed_updates=34823,
    behavior_complete=97,behavior_cases=144,hardware_deployed=False,optimizer_updates=0,
    input_contract_verified=True,checkpoint_sha256='521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5',
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.iterdir())})
(package/'MANIFEST.json').write_text(json.dumps(status,indent=2))
with zipfile.ZipFile(package.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(package.iterdir()):
        archive.write(path,package.name+'/'+path.name)
print(json.dumps(dict(package=str(package),zip=str(package.with_suffix('.zip')),sha256=hashlib.sha256(package.with_suffix('.zip').read_bytes()).hexdigest())))
