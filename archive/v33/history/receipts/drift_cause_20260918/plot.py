"""Visualize matched frozen-policy paths and their measured components."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).parent
baseline = json.loads((root / "baseline19001_v2.json").read_text())
b = np.load(root / "baseline19001_v2_traces.npz")
h = np.load(root / "heading19001_v2_traces.npz")
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for row, (terrain, speed, title) in enumerate((("wave", 0.3, "9级波浪，0.3 m/s"),
                                               ("stairs_up", 0.9, "9级上楼，0.9 m/s"))):
    j = next(i for i, idx in enumerate(b["case_indices"]) if baseline["cases"][idx]["terrain"] == terrain
             and baseline["cases"][idx]["speed"] == speed)
    idx = b["case_indices"][j]
    k = list(h["case_indices"]).index(idx)
    bv = b["values"][:, j]
    hv = h["values"][:, k]
    bv, hv = bv[bv[:, 1] > 0], hv[hv[:, 1] > 0]
    ax = axes[row, 0]
    ax.axhline(0, color="#667085", linestyle="--", linewidth=1, label="目标直线")
    ax.plot(bv[:, 2], bv[:, 3], color="#c34343", label="原始命令")
    ax.plot(hv[:, 2], hv[:, 3], color="#177c70", label="仅加入航向反馈")
    ax.set(title=title + "：行进路径", xlabel="前进距离（米）", ylabel="横向偏移（米，右侧为负）")
    ax.legend(loc="best")
    ax = axes[row, 1]
    ax.plot(bv[:, 0], bv[:, 3], color="#202b3d", linewidth=2, label="实际侧偏")
    ax.plot(bv[:, 0], bv[:, 8] + bv[:, 9], color="#c34343", label="朝向偏转后的前进贡献")
    ax.plot(bv[:, 0], bv[:, 10], color="#377bb8", label="横向运动贡献")
    ax.set(title="原始命令：侧偏来源分解", xlabel="时间（秒）", ylabel="累计横向位移（米）")
    ax.legend(loc="best")
for ax in axes.flat:
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("同一19,001轮策略的独立诊断：保持网络权重不变", fontsize=15)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.text(0.5, 0.015, "固定名义条件、保留动作延迟；每5毫秒采样；图示为选定案例，不代表全部场景。", ha="center", fontsize=9)
fig.savefig(root / "drift_diagnosis.png", dpi=160)
plt.close(fig)
