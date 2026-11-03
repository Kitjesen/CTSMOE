"""Present the preserved zero-update 5 ms stair replay; no training changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "physics_review"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
meta = json.loads((DATA / "SELECTED_LOAD_CASES.json").read_text())
audit = json.loads((DATA / "LOAD_AUDIT.json").read_text())
with np.load(DATA / "SELECTED_LOAD_TRACES.npz") as archive:
    traces = dict(archive)

titles = ["上楼：23.5 cm台阶，0.6 m/s\n干净观测，通过",
          "下楼：25 cm台阶，0.6 m/s\n干净观测，通过",
          "上楼：23.5 cm台阶，1.2 m/s\n噪声及初态偏置，超速终止"]
joint_labels = {"fr_calf_joint": "右前小腿", "fl_calf_joint": "左前小腿"}
fig, axes = plt.subplots(3, 3, figsize=(15.5, 10.5), sharex="col")
events = []
for column, case in enumerate(meta):
    key = f"{case['condition']}_{case['id']}__"
    a = traces[key + "acceleration"][:, case["leg_ids"]]
    peak_index, local_joint = np.unravel_index(abs(a).argmax(), a.shape)
    joint = case["leg_ids"][local_joint]
    name = case["joint_names"][joint]
    t = traces[key + "time"]
    peak_time = t[peak_index]
    window = (t >= peak_time - .10) & (t <= peak_time + .10)
    axes[0, column].set_title(titles[column] + "\n" + joint_labels[name])
    axes[0, column].plot(t[window], a[window, local_joint], color="#cc493a", marker=".")
    axes[1, column].plot(t[window], traces[key + "torque"][window, joint],
                         color="#1676a8", marker=".", label="施加扭矩")
    axes[1, column].plot(t[window], traces[key + "computed_torque"][window, joint],
                         color="#888888", linestyle="--", label="限幅前计算扭矩")
    axes[2, column].plot(t[window], traces[key + "velocity"][window, joint],
                         color="#226d4f", marker=".")
    for limit in (-120, 120):
        axes[1, column].axhline(limit, color="#ba4b41", linestyle=":", linewidth=1)
    for limit in (-15, 15):
        axes[2, column].axhline(limit, color="#ba4b41", linestyle=":", linewidth=1)
    for row in range(3):
        axes[row, column].axvline(peak_time, color="#666666", alpha=.55, linewidth=1)
        axes[row, column].grid(alpha=.2)
        axes[row, column].set_xlim(peak_time - .10, min(peak_time + .10, t[-1]))
    axes[2, column].set_xlabel("首次回合时间（s）")
    axes[0, column].annotate(f"峰值 {abs(a[peak_index, local_joint]):.0f}",
                            xy=(.04, .93), xycoords="axes fraction", va="top")
    events.append({"condition": case["condition"], "case_id": case["id"],
                   "joint": name, "time_s": float(peak_time),
                   "acceleration_rad_s2": float(a[peak_index, local_joint]),
                   "speed_before_rad_s": float(traces[key + "velocity"][peak_index-1, joint]),
                   "speed_after_rad_s": float(traces[key + "velocity"][peak_index, joint]),
                   "applied_torque_at_peak_nm": float(traces[key + "torque"][peak_index, joint])})
axes[0, 0].set_ylabel("实际关节角加速度（rad/s²）")
axes[1, 0].set_ylabel("关节扭矩（N·m）")
axes[2, 0].set_ylabel("实际关节角速度（rad/s）")
axes[1, 0].legend(loc="lower left", fontsize=8)
fig.suptitle("楼梯瞬态负载：逐5毫秒原始数据，没有平滑\n"
             "虚线为当前仿真施加扭矩±120 N·m、名义腿速±15 rad/s；不是加速度安全阈值",
             fontsize=14)
fig.tight_layout(rect=(0, .035, 1, .925))
fig.text(.5, .018, "三个固定样本；所有144例复放与原始轨迹完全一致，0次模型更新。"
         "加速度峰值不等于电机主动加速指令。", ha="center", fontsize=10)
fig.savefig(ROOT / "STAIR_LOAD_TRANSIENTS.png", dpi=160)
plt.close(fig)
(DATA / "PEAK_EVENTS.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

whole = [r["phases"]["whole"] for r in audit["rows"]]
assert len(whole) == 144
assert sum(r["torque_over120_samples"] for r in whole) == 0
assert all(x["baseline_outcomes_identical"] for x in audit["capture"].values())
table = []
for title, case in zip(["上楼0.6，干净", "下楼0.6，干净", "上楼1.2，噪声/偏置"], meta):
    p = case["phases"]["climb"]
    table.append(f"|{title}|{p['torque_peak_nm']:.2f}|{p['torque_max_joint_rms_nm']:.2f}|"
                 f"{p['acceleration_peak_rad_s2']:.1f}|{p['acceleration_p99_rad_s2']:.1f}|"
                 f"{p['torque_ge114_longest_single_joint_s']*1000:.0f}|")
report = """# 连续楼梯：姿态偏移与实际动态负载的区分

森哥，上下楼梯需要离开默认站姿。默认角偏差本身不能判定坏动作，也不能替代扭矩、转速和接触瞬态评价。本次结论是：施加腿扭矩限幅有效，仍存在明显的短时关节加速度尖峰；高速扰动样本出现了真实超速失败。

## 本次数据

34823轮学生ONNX策略，144个固定案例，6个连续台阶加末端平台；上/下楼、3/6/9级、0.3/0.6/1.2 m/s、直接/静置起步、末端继续/停止，分别在干净观测和观测噪声+固定初态偏置下测试。它不是随机成功率。保持原模型、奖励、执行器和物理参数，优化器更新0次；5ms补录的全部关节角、速度、目标、扭矩、动作和机身轨迹与原评估逐项完全一致，终止结果一致，冻结文件哈希不变。

加速度是仿真实际关节速度在5ms内的变化率，不是策略的加速度指令。另用5ms速度差分复核，最大浮点差0.000977 rad/s²。只统计首次回合，排除重置后的数据。

跨阶窗口从最前轮水平外缘到达第一阶开始，到全身碰撞包络进入末端平台结束。这是几何阶段划分，不是精确接触分类。

## 三个固定案例的跨阶负载

|案例|腿扭矩峰值 N·m|最大单关节扭矩RMS N·m|角加速度峰值 rad/s²|角加速度P99 rad/s²|单关节≥114 N·m最长连续时间 ms|
|---|---:|---:|---:|---:|---:|
""" + "\n".join(table) + """

RMS列先分别计算12个关节在该案例跨阶窗口的RMS，再取最大；不是12关节的总扭矩。P99列将该窗口的12腿关节加速度绝对值合并计算。扭矩峰值和加速度峰值可能发生在不同关节、不同时间。

## 已确定的事实

- 全部144例中，实际施加腿扭矩最大120 N·m，超过120.001的物理步为0。58例曾达到至少114 N·m；最久单关节连续高负载25 ms。没有发现持续数秒贴120上限的情况，但这不代表冲击已经很小。
- 0.6 m/s上楼case28在5.605秒，右前小腿速度5ms内从+7.718变成−8.987 rad/s，对应−3341 rad/s²。该物理步施加扭矩仅+6.30 N·m；随后3个物理步施加扭矩达到120，限幅前计算约139、140、133 N·m。不能把这一加速度尖峰解释为电机主动输出了超120的扭矩。
- 0.6 m/s下楼case64在1.655秒左前小腿速度先快速上升，再在1.660秒由+10.155变成−3.666 rad/s，最大负加速度约−2764。两例20ms角度记录中对应小腿均远离硬角度端点。速度突变与后续PD反应清楚可见，但还不足以唯一归因到某个碰撞部位或控制参数。
- 扭矩限幅限制的是电机在仿真中的主动施加力矩，不是地面接触/约束反力。加速度尖峰可能包含外部接触冲量、机构耦合与控制响应，不能由峰值数字直接反推出超额电机输出。
- 1.2 m/s噪声/初态偏置上楼case33的腿速达到18.748 rad/s；轮速49.871 rad/s，超过当前轮速极端终止线45，在2.62秒物理终止。它不是由于姿态RMS大而判失败。全矩阵有5例腿速短时超过15.001，只有这一例触发物理终止。
- 当前没有经过实机校验的关节加速度许用阈值，不能将3341或4937直接宣布为电机/结构安全超限。它们是需要进一步压制和定位的尖峰，不能当作平滑、低冲击运动已经达标。

## 对训练判断的修正

正常跨阶屈伸应保留。0.6 m/s的两个干净样本在初始第一秒默认角RMS峰值约0.057 rad，末段均值约0.066/0.092 rad；大默认角偏差集中在跨阶中段，不能据此推断它们已经学成全程扭曲行走。

前一份姿态解释把攀爬时额外默认角惩罚减弱列为审查点；这项配置本身不是已经证实的错误根因。后续修复应优先针对实际速度突变、接触瞬态、持续高负载、原始目标越限与平台恢复，不能仅为了降低12关节默认角RMS而把攀爬姿态压回固定站姿。

本次没有修改奖励或启动续训。ONNX导出数值校验仍有效；现有候选未通过整体行为验收的结论不变，其中包括偏航、末端停车和这一例实际超速问题，而非因为楼梯中段默认角RMS大。

## 产物

- `STAIR_LOAD_TRANSIENTS.png`：三个案例的5ms加速度、扭矩和速度尖峰放大图。
- `physics_review/LOAD_PER_JOINT.csv`：144例、每个腿关节、完整回合/跨阶/平台后分阶段指标。
- `physics_review/LOAD_AUDIT.json`：完整案例统计与复放一致性证明。
- `physics_review/SELECTED_LOAD_TRACES.npz`：三个案例原始5ms时程。
"""
(ROOT / "STAIR_DYNAMIC_LOAD_REVIEW.md").write_text(report, encoding="utf-8")
print("Wrote STAIR_DYNAMIC_LOAD_REVIEW.md and STAIR_LOAD_TRANSIENTS.png")
