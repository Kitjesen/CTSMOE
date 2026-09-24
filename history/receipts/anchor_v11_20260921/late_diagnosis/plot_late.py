"""Plot the actual late-failure repeat and diagnostic controller interventions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

b = Path(__file__).parent / "result"
m = json.loads((b / "manifest.json").read_text())
with np.load(b / "traces.npz") as f:
    d = {k: f[k] for k in ("joint_pos", "velocity", "root_clearance", "latent_v11_delta",
                          "gate_v11", "action_v11_student", "action_v11_teacher")}
legs = [j for j, name in enumerate(m["joint_names"]) if not name.endswith("foot_joint")]
nominal = np.asarray(m["default_q"])[legs]
series = [(0, "V6 student", "#1976b8", "-"),
          (2, "V11 student", "#d64532", "-"),
          (3, "V11 teacher", "#8166a9", "--"),
          (6, "V11 student to V6 at 2.8s", "#198754", "-"),
          (9, "V11 student to teacher at 2.8s", "#ac8515", "--")]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for case_id, label, color, style in series:
    c = m["cases"][case_id]
    n = c["end_step"]+1
    t = (np.arange(n)+1)*m["dt"]
    q = d["joint_pos"][:n, case_id][:, legs]-nominal
    for ax, v in zip(axes, (np.sqrt(np.mean(q*q, axis=1)), d["root_clearance"][:n, case_id],
                           d["velocity"][:n, case_id, 0])):
        ax.plot(t, v, color=color, ls=style, label=label)
axes[0].axhline(.25, color="black", ls=":", label="Posture reference: 0.25 rad")
axes[1].axhline(.46, color="black", ls=":", label="Height target: 0.46m")
axes[2].axhline(.6, color="black", ls=":", label="Velocity command: 0.6m/s")
for ax, label in zip(axes, ("Actual leg posture RMS (rad)", "Body clearance (m)", "Forward speed (m/s)")):
    ax.set_ylabel(label)
    ax.axvline(2.8, color="gray", ls=":", alpha=.7)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.25)
axes[-1].set_xlabel("Time after command (s)")
fig.suptitle("V11 late flat failure repeats; diagnostic interventions recover it\n"
             "Matched default starts; 0.6m/s, no noise/push, fixed 10ms delay. No training updates.")
fig.tight_layout()
fig.savefig(b / "late_response.png", dpi=150)
plt.close(fig)

case_id = 2
n = m["cases"][case_id]["end_step"]+1
pre_t = np.arange(n)*m["dt"]
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(pre_t, np.linalg.norm(d["latent_v11_delta"][:n, case_id], axis=-1), color="#d64532")
axes[0].set_ylabel("Student-teacher latent L2")
axes[1].plot(pre_t, np.linalg.norm(d["action_v11_student"][:n, case_id]-d["action_v11_teacher"][:n, case_id], axis=-1))
axes[1].set_ylabel("Raw action L2 difference")
for expert in range(8):
    axes[2].plot(pre_t, d["gate_v11"][:n, case_id, expert], label=f"Expert {expert+1}")
axes[2].set_ylim(.11, .15)
axes[2].set_ylabel("Mixture weight")
axes[2].legend(ncol=4, fontsize=8)
axes[-1].set_xlabel("Observation time before physics step (s)")
for ax in axes:
    ax.axvspan(2.2, 3.3, color="#d64532", alpha=.07)
    ax.grid(alpha=.25)
fig.suptitle("V11 teacher and student evaluated on the SAME student rollout states\n"
             "Latent/action difference grows; mixture weights do not switch abruptly. Correlation is not full root-cause proof.")
fig.tight_layout()
fig.savefig(b / "student_drift.png", dpi=150)
plt.close(fig)
print("Saved late_response.png and student_drift.png")
