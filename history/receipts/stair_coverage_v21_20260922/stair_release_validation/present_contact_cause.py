"""Plot the measured contact/velocity/torque chronology, without simulation edits."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "contact_review"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
fig, axes = plt.subplots(4, 3, figsize=(15.6, 12), sharex="col")
specs = [
    ("nominal", 28, "fr", 5.605, 5.605, "上楼0.6 m/s：右前轮落到平台", "fr_foot_Link"),
    ("nominal", 64, "fl", 1.655, 1.660, "下楼0.6 m/s：左前小腿连杆碰撞", "fl_calf_Link"),
    ("perturbed", 33, "fr", 2.620, 2.620, "上楼1.2 m/s：右前轮撞台阶立面", "fr_foot_Link"),
]
for col, (condition, case, side, impact, peak, title, force_body) in enumerate(specs):
    meta = json.loads((DATA / condition / "CONTACT_CAPTURE.json").read_text())
    assert meta["baseline_outcomes_identical"] and not any(meta["baseline_max_abs_difference"].values())
    with np.load(DATA / condition / "CONTACT_CAUSE_5MS.npz") as archive:
        z = dict(archive)
    local = meta["case_ids"].index(case)
    joint = meta["joint_names"].index(side + "_calf_joint")
    body = meta["contact_names"].index(force_body)
    time = (np.arange(len(z["q"])) + 1) * .005
    end = peak if case == 33 else peak + .10
    window = (time >= impact - .06) & (time <= end + 1e-7)
    t = time[window]
    contact = np.linalg.norm(z["net_contact"][window, local, body], axis=1)
    axes[0, col].plot(t, contact, color="#9b4ab5", marker=".")
    axes[0, col].set_title(title)
    axes[1, col].plot(t, z["dq"][window, local, joint], color="#15764c", marker=".")
    axes[2, col].plot(t, z["torque"][window, local, joint], color="#166aab", marker=".")
    axes[3, col].plot(t, z["q"][window, local, joint], label="实际角", color="#166aab", marker=".")
    axes[3, col].plot(t, z["target"][window, local, joint], label="限制后目标", color="#d2553c")
    for row in range(4):
        axes[row, col].axvline(impact, color="#777777", linestyle="--", linewidth=1)
        axes[row, col].grid(alpha=.2)
        axes[row, col].set_xlim(impact - .06, end)
    axes[3, col].set_xlabel("首次回合时间（s）")
    axes[0, col].annotate(f"{contact.max():.0f} N", (.04, .9), xycoords="axes fraction")
axes[0, 0].set_ylabel("指定部件接触力大小（N）")
axes[1, 0].set_ylabel("对应小腿角速度（rad/s）")
axes[2, 0].set_ylabel("对应小腿施加扭矩（N·m）")
axes[3, 0].set_ylabel("对应小腿关节角（rad）")
axes[3, 0].legend(fontsize=8)
fig.suptitle("加速度尖峰的事件链：先发生接触冲击，再出现速度突变与控制响应\n"
             "5毫秒原始数据；虚线标出接触冲击时刻；最后一列在物理终止处结束", fontsize=15)
fig.tight_layout(rect=(0, .035, 1, .935))
fig.text(.5, .015, "144例复放轨迹与原测试完全一致，0次训练更新。接触力是仿真冲量/5ms，不是电机扭矩。",
         ha="center", fontsize=10)
fig.savefig(ROOT / "STAIR_CONTACT_CAUSE.png", dpi=160)
plt.close(fig)
print("Wrote STAIR_CONTACT_CAUSE.png")
