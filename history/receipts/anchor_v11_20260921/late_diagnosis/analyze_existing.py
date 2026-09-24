"""Describe archived failures from actual telemetry; never infer a PPO gradient."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
m = json.loads((a.directory / "manifest.json").read_text())
with np.load(a.directory / "traces.npz") as f:
    d = {k: f[k] for k in f.files}
legs = [j for j, name in enumerate(m["joint_names"]) if not name.endswith("foot_joint")]
wheels = [j for j in range(len(m["joint_names"])) if j not in legs]
requests = m["request_joint_ids"]
default = np.asarray(m["default_q"])
base = m["body_names"].index("base_link")
results = []
for col, c in enumerate(m["cases"]):
    n = c["end_step"] + 1
    t = (np.arange(n) + 1) * m["dt"]
    q = d["joint_pos"][:n, col][:, legs] - default[legs]
    target = d["target"][:n, col][:, legs] - default[legs]
    req = d["requested"][:n, col][:, [requests.index(j) for j in legs]] - default[legs]
    rms = np.sqrt(np.mean(q*q, axis=1))
    body = d["body_pos"][:n, col, base]
    label = m["history"][c["policy"]]["label"]
    windows = {}
    for name, mask in {"first_1s": t <= 1., "last_5s": t > max(0., t[-1]-5.),
                       "last_1s": t > max(0., t[-1]-1.)}.items():
        torque = d["torque_peak"][:n, col][mask]
        speed = d["speed_peak"][:n, col][mask]
        force = d["wheel_ground_force"][:n, col][mask]
        r = dict(samples=int(mask.sum()), mean_rms=float(rms[mask].mean()),
                 peak_rms=float(rms[mask].max()),
                 x_progress=float(np.ptp(body[mask, 0])),
                 mean_velocity=d["velocity"][:n, col][mask].mean(0).tolist(),
                 peak_leg_torque=float(torque[:, legs].max()),
                 peak_wheel_torque=float(torque[:, wheels].max()),
                 peak_leg_speed=float(speed[:, legs].max()),
                 peak_wheel_speed=float(speed[:, wheels].max()),
                 leg_saturation_fraction=float(d["saturation"][:n, col][mask][:, legs].mean()),
                 hard_excess=float(d["hard_excess"][:n, col][mask][:, legs].max()),
                 mean_support_wheels=float((force > 5.).sum(1).mean()),
                 mean_wheel_force=force.mean(0).tolist(),
                 base_contact_peak=float(d["base_contact"][:n, col][mask].max()),
                 phase_weight_mean=float(d["phase_weight"][:n, col][mask].mean()),
                 joint_mean_offset=q[mask].mean(0).tolist(),
                 joint_target_mean=target[mask].mean(0).tolist(),
                 joint_request_mean=req[mask].mean(0).tolist(),
                 mean_target_actual_abs_error=np.abs(target[mask]-q[mask]).mean(0).tolist())
        windows[name] = r
    k = int(rms.argmax())
    order = np.argsort(np.abs(q[k]))[-3:][::-1]
    peak = dict(time_s=float(t[k]), rms=float(rms[k]), velocity=d["velocity"][k, col].tolist(),
                joints=[dict(name=m["joint_names"][legs[j]], actual=float(q[k, j]),
                             target=float(target[k, j]), requested=float(req[k, j])) for j in order])
    results.append(dict(case=c, label=label, windows=windows, peak=peak))
    if c["teacher"] or (c["terrain"] == "flat" and c["direction"] != "forward"):
        continue
    slug = f'{label}_{c["terrain"]}_L{c["level"]}_v{c["velocity"][0]}'
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, req[:, j], label="raw request", alpha=.65)
        ax.plot(t, target[:, j], label="limited target", alpha=.7)
        ax.plot(t, q[:, j], label="actual", lw=1.5)
        ax.axhline(0., color="black", lw=.5)
        ax.set_title(m["joint_names"][legs[j]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Offset from default (rad)")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{label}: {c['terrain']} L{c['level']}, vx={c['velocity'][0]} m/s\n"
                 "First episode only; raw request / limited target / actual")
    fig.tight_layout()
    fig.savefig(a.directory / f"{slug}_joints.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(t, body[:, 0]-body[0, 0], label="x progress")
    axes[0].plot(t, body[:, 1]-body[0, 1], label="lateral displacement")
    axes[0].set_ylabel("Distance (m)")
    axes[1].plot(t, d["velocity"][:n, col, 0], label="vx body")
    axes[1].plot(t, d["command"][:n, col, 0], ls="--", label="vx command")
    axes[1].set_ylabel("Speed (m/s)")
    axes[2].plot(t, rms, label="leg pose RMS")
    axes[2].plot(t, d["root_clearance"][:n, col], label="body clearance (m)")
    axes[2].set_ylabel("rad / m")
    axes[3].plot(t, d["torque_peak"][:n, col][:, legs].max(1), label="leg torque peak")
    axes[3].plot(t, d["torque_peak"][:n, col][:, wheels].max(1), label="wheel torque peak")
    axes[3].set_ylabel("Torque (Nm)")
    for j in range(4):
        axes[4].plot(t, d["wheel_ground_force"][:n, col, j], label=f"wheel sensor {j}")
    axes[4].set_ylabel("Ground contact (N)")
    axes[4].set_xlabel("Time (s)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle(f"{label}: {c['terrain']} L{c['level']}, vx={c['velocity'][0]} m/s")
    fig.tight_layout()
    fig.savefig(a.directory / f"{slug}_motion.png", dpi=130)
    plt.close(fig)
(a.directory / "diagnosis.json").write_text(json.dumps(dict(joint_names=[m["joint_names"][j] for j in legs],
    cases=results, note="Fixed first-episode telemetry. Phase weight is recorded, not causal attribution."), indent=2))
for row in results:
    if not row["case"]["teacher"]:
        w = row["windows"]["last_5s"]
        print(row["label"], row["case"]["terrain"], row["case"]["direction"],
              json.dumps({k: w[k] for k in ("mean_rms", "x_progress", "peak_leg_torque",
                  "peak_wheel_torque", "mean_support_wheels", "leg_saturation_fraction")}))
