# Slope diagnosis, 2026-09-20

Frozen student checkpoint: completed update 37283, model_37282.pt. No production training settings were changed. Production remains in mixed9_4096_60000_flat_speed_20260920 (PID 158275). All probes exited after closing result files.

128 first episodes: 32 with training randomization/observation noise, plus 96 controlled factorial cases. Both use mean student actions, body vx=0.6 m/s, initial yaw=0, maximum 25 s; success is world-x relative to terrain origin >=4 m before physical termination. No action exploration noise. Physics traces at 200 Hz and control/contact traces at 50 Hz are retained in training/ and controlled/. These are simulation screenings, not real-robot or exhaustive capability certification.

Training-randomized sample: 30 degrees 7/8 passes, 35 degrees 3/8, 40 degrees 0/8, 45 degrees 0/8. Totals: 10 passes, 16 timeouts, 2 speed terminations, 6 base-contact terminations. Thus termination relaxation alone does not explain or fix most non-completions.

Controlled factors: nominal mass/gains/default joint posture; fixed 10 ms actuator delay; no observation noise; wheel static/dynamic friction 0.5 or 1.2 (terrain material 1, average combine); heading controller off or yaw-rate=clip(-1.5*yaw,-1,1); pushes off/on. Three matched initial x offsets (-0.1,0,+0.1 m) per cell. Push vectors are shared by repeat across cells, every four seconds, within the training x/y +/-0.4 m/s and angular +/-0.6 rad/s ranges. Other startup/reset randomization is disabled. Fixed controls isolate factors but are not the full training distribution.

| Slope | Wheel friction | Heading | Push | Pass |
|---|---:|---|---|---:|
| 35 deg | 1.2 | off/on | off/on | 12/12 total |
| 40 deg | 0.5 | off | off | 0/3 |
| 40 deg | 1.2 | off | off | 3/3 |
| 40 deg | 1.2 | off | on | 0/3 |
| 40 deg | 1.2 | on | off | 2/3 |
| 40 deg | 1.2 | on | on | 1/3 |
| 45 deg | either | either | either | 0/24 total |

Supported findings:

1. Friction and disturbance robustness materially limit 40-degree performance in this small paired screen. Heading feedback alone is not a reliable fix (2/3 vs 3/3 without it under high friction/no push). Low friction makes steep climbing physically more demanding; with average material combine, nominal contact coefficients are about 0.75 vs 1.1. Exact tilted-wheel contact kinematics and friction response remain simulator-dependent.
2. 45 degrees still stalls in favorable conditions. High friction, no push, heading feedback: 3/3 time out at x=2.64–2.69 m, y=-0.15 to +0.20 m; last-five-second body vx=0.003–0.043 m/s. Average number of wheels carrying >5 N is 2.37–2.56. Contact-point tangential speed proxy is 0.19–0.29 m/s. Wheels keep turning at mean absolute joint speed 4.23–6.75 rad/s. This demonstrates ineffective support/traction/propulsion after substantial initial progress, beyond a heading-only problem.
3. The stalled state is not sustained actuator torque saturation: those 45-degree cases have last-five-second mean wheel absolute torque 4.57–5.00 Nm, wheel clipping on 3.2–4.3% of joint physics samples, leg clipping 0–0.033%. Transient peak torques can still reach 120/17 Nm. These aggregates do not prove that instantaneous motor authority, joint limits, target slew limits, or individual rewards are irrelevant.
4. Wheel overspeed is impulsive in the reproduced cases. At 7.36 s one wheel jumps from +6.73 to -91.63 rad/s in a 5 ms step while applied motor torque is about -1.64 Nm; another jumps from -7.31 to +52.06 rad/s with about -0.36 Nm. This points to contact/articulation-coupled impulses rather than prolonged high motor torque, but does not by itself prove a PhysX numerical bug. The two peaks are 0.295 s and 3.34 s after preceding pushes. Push is not the sole direct trigger.
5. The production foot-regulation reward uses a gravity-down contact-point approximation. On a slope this can penalize ideal rolling because the true contact normal differs. It is a concrete modeling approximation, but its performance impact is not measured here and should not be described as the main cause.

Slip estimate uses wheel link-origin linear velocity plus angular velocity cross (-radius*raycast surface normal), projected onto the surface tangent, only when net contact force >5 N. Radius 0.093 m. It omits finite wheel width, cylinder-axis projection and exact contact manifold, so values are a slip proxy, not measured tire slip. Support-count correlation alone does not establish a causal mechanism.

Recommended next changes, not implemented in this audit: a slope-specific curriculum that first establishes 30–40 degree forward progress on adequate friction, then adds lower friction and stronger disturbances separately; an explicit low-progress diagnostic and recovery-learning objective for 45-degree stalls; contact/solver replay before any further relaxation of velocity termination. Do not lower the 4 m criterion to report a higher level. Evaluate any support/reward change using frozen paired slope cases and retain mixed-terrain regression checks.

Implementation note: first probe attempt failed event-manager signature validation before simulation; v2 uses explicit parameter signatures. Trace outputs come only from v2. Test commands and code are retained in launch.sh and probe.py. No model updates occurred in probes.
