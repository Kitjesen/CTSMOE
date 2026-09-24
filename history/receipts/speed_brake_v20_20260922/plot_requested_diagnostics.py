"""Plot immutable selected traces from the 33865 zero-update evaluation."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "requested_evaluation_33865"
DATA = ROOT / "diagnostic_excerpts"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})


def load(name):
    manifest = json.loads((DATA / f"{name}_manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(DATA / f"{name}_selected.npz")
    return manifest, arrays


def kinematic_valid(arrays, manifest, case):
    """Do not draw auto-reset kinematics as part of the terminated trajectory."""
    valid = arrays["active"][:, case].astype(bool)
    item = manifest["cases"][case]
    if item.get("reset_metadata", {}).get("reset_occurred_after_capture", False):
        valid[item["end_step"]] = False
    return valid


bm, b = load("brake25")
sm, s = load("stairend")
fig, ax = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
fig.suptitle("V20 · 33,865轮：姿态改善后仍存在的运动缺口\n固定案例时程；不是随机成功率；全部数据来自已完成的零更新评估", fontsize=17)

for case, label, color in [(0, "33,174轮学生", "#999999"), (2, "33,865轮学生", "#d65f27"), (6, "33,865轮教师", "#238b8e")]:
    valid = kinematic_valid(b, bm, case)
    t = np.arange(len(valid))[valid] * bm["dt"]
    ax[0, 0].plot(t, b["velocity"][valid, case, 0], label=label, color=color)
    valid = b["active"][:, case].astype(bool)
    t = np.arange(len(valid))[valid] * bm["dt"]
    ax[0, 1].plot(t, np.max(np.abs(b["speed_peak"][valid, case, 12:]), axis=1), label=label, color=color)
valid = b["active"][:, 2].astype(bool)
t = np.arange(len(valid))[valid] * bm["dt"]
ax[0, 0].step(t, b["command"][valid, 2, 0], where="post", color="#303030", linestyle="--", label="前进速度命令")
ax[0, 0].set(title="后退急停：实际已达到约 −1.84 m/s", ylabel="机身前向速度 (m/s)", xlim=(13.8, 16.2))
ax[0, 1].axhline(30, color="#777777", linestyle="--", label="名义轮速30 rad/s（非瞬态终止线）")
ax[0, 1].set(title="学生在停车后0.04秒内超速终止：46.1 rad/s", ylabel="四轮物理子步绝对峰值 (rad/s)", xlim=(13.8, 16.2))
for plot in ax[0]:
    plot.axvline(15, color="#303030", alpha=.5)
    plot.legend(fontsize=8)

for case, label, color in [(3, "左移 +0.2", "#238b8e"), (4, "右移 −0.2，直接给速", "#d65f27"), (5, "右移 −0.2，稳定后给速", "#7556a4")]:
    valid = kinematic_valid(b, bm, case)
    t = np.arange(len(valid))[valid] * bm["dt"]
    ax[1, 0].plot(t, b["velocity"][valid, case, 1], label=label, color=color)
ax[1, 0].axhline(.2, linestyle="--", color="#238b8e", alpha=.6)
ax[1, 0].axhline(-.2, linestyle="--", color="#d65f27", alpha=.6)
ax[1, 0].set(title="0.2 m/s低速侧移：跟踪不连续，右移末段近零", ylabel="实际侧向速度 (m/s)", xlim=(0, 5), ylim=(-.3, .23))
ax[1, 0].legend(fontsize=9)

case = 2
valid = kinematic_valid(b, bm, case)
t = np.arange(len(valid))[valid] * bm["dt"]
q = b["joint_pos"][valid, case, :12] - np.asarray(bm["default_q"])[:12]
ax[1, 1].plot(t, np.sqrt(np.mean(q**2, axis=1)), label="12腿关节姿态偏差RMS", color="#238b8e")
ax[1, 1].plot(t, np.max(np.abs(q), axis=1), label="最大单关节偏差", color="#d65f27")
ax[1, 1].set(title="同一急停案例：姿态指标不能代替轮速验收", ylabel="相对默认角度 (rad)", xlim=(13.8, 16.2))
ax[1, 1].legend(fontsize=9)

for case, label, color in [(0, "33,174轮学生", "#999999"), (3, "33,865轮学生", "#238b8e")]:
    valid = s["active"][:, case].astype(bool)
    t = np.arange(len(valid))[valid] * sm["dt"]
    ax[2, 0].plot(t, s["world_velocity"][valid, case, 2], label=label, color=color)
ax[2, 0].set(title="上楼宽窗口含最后爬阶：正Vz不等于登顶跳动", ylabel="机身世界坐标竖直速度 (m/s)", xlim=(2, 5.1))
ax[2, 0].axvline(4.12, color="#238b8e", linestyle=":", label="候选全身进入平台")
ax[2, 0].legend(fontsize=9)

case = 2
valid = s["active"][:, case].astype(bool)
t = np.arange(len(valid))[valid] * sm["dt"]
ax[2, 1].step(t, s["command"][valid, case, 0], where="post", label="前向命令 (m/s)", color="#238b8e")
ax[2, 1].step(t, s["stair_platform_hull"][valid, case].astype(float), where="post", label="全身几何包络进入平台 (0/1)", color="#7556a4")
ax[2, 1].step(t, s["stair_platform_supported"][valid, case].astype(float), where="post", label="有效平台支撑判定 (0/1)", color="#d65f27")
ax[2, 1].set(title="6级上楼停车分支：未触发停车即到达安全边界", xlim=(8, 13.5), ylim=(-.05, 1.1))
ax[2, 1].legend(fontsize=8)
for plot in ax.flat:
    plot.set_xlabel("控制步起点时间 (s)，状态在该步结束采集")
    plot.grid(alpha=.2)
fig.savefig(ROOT / "DIAGNOSTIC_OVERVIEW.png", dpi=155)
plt.close(fig)

wheel = np.abs(b["speed_peak"][:, 2, 12:])
wheel[~b["active"][:, 2].astype(bool)] = np.nan
step, wheel_id = np.unravel_index(np.nanargmax(wheel), wheel.shape)
detail = {
    "source": "diagnostic_excerpts/brake25_selected.npz, original case74",
    "peak_step": int(step),
    "step_start_s": float(step * bm["dt"]),
    "step_end_s": float((step + 1) * bm["dt"]),
    "wheel": bm["joint_names"][12 + wheel_id],
    "absolute_physics_peak_rad_s": float(wheel[step, wheel_id]),
    "physical_end_reason": bm["cases"][2]["end_reason"],
    "note": "Physical-substep peak is retained; terminal post-reset state must not establish pre-impact causality.",
}
(ROOT / "SELECTED_TRACE_DETAIL.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(detail, ensure_ascii=False))
