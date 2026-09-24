# Measured braking postprocessor

Run with a directory containing the existing `evaluate_measured.py` outputs:

```text
python focus_metrics.py PATH_TO_BRAKE25 --output-directory PATH_TO_METRICS
```

The output is one `BRAKE_METRICS_<manifest-history-label>.json` per policy,
including source, candidate and any separately labelled teachers. Each report
contains the actual manifest checkpoint, case IDs and a compact event list.
The callable API is `analyze_braking(manifest, traces)` or
`write_reports(directory, output_directory=None)`. Only NumPy and the Python
standard library are required. No training, simulator or network call occurs.

The input schema was read from the frozen
`omni_repair_v19_20260922/evaluate_measured.py`. Case indices are `case.id`,
policy metadata comes from `history[case.policy]`, the base body is resolved by
`body_names.index('base_link')`, and the 12 requested leg joints use
`request_joint_ids` in its recorded order. Wheel IDs are resolved from the four
actual `*_foot_joint` names. The processor does not assume requested joint order
equals articulation order. It reads only the fields it uses from the NPZ.

Events are actual nonzero-to-all-zero command transitions, with `1e-8` zero
tolerance. An initial zero command is not braking. The nominal brake25 clock
is not used to fabricate missing transitions. Every case ends at its inclusive
`end_step` or an earlier inactive prefix. Data after failure is excluded.

The command at index `k` was issued before the simulation step; kinematics at
`k` were recorded after it. A stop begins at time `k*dt`, with its measured entry
state at `k-1`. Kinematic event metrics include that entry state. Torque/speed
substep peaks and requested/target angles include only the zero-command steps,
so a peak from the preceding movement interval is not attributed to braking.

`stable_stop_time_s` is the earliest observed low-speed state that is followed
by at least 0.5 measured seconds with XY norm strictly below 0.1 m/s and absolute
body-z angular velocity strictly below 0.15 rad/s. The separate
`stable_stop_confirmation_s` is when that duration is confirmed. Missing
stability is `null`; observed timestamps do not assume an unobserved crossing
between samples. Zero windows end at the next nonzero command or valid case end.

`entry_speed_ratio` uses signed body-XY velocity projected onto commanded XY,
and signed body-z angular velocity divided by commanded yaw; for mixed commands
it is the minimum active-channel ratio. Components are retained separately.
`reached_command_before_stop` requires at least 80% of available samples in the
last second to match the preceding command, and the entry sample must also
match. Each active axis uses 20% relative tolerance; zero axes use 0.1 m/s or
0.15 rad/s absolute tolerance. The observed window duration is retained.
`qualified_stop` additionally requires observed stability and excludes a
terminal failure during the zero window or an unexplained inactive truncation.
Low actual entry speed can therefore produce an observed stop but cannot count
as successfully braking from the requested high speed.

Path length and displacement use base-link world-axis XY relative to its fixed
terrain origin. Omega-z integrals are explicitly body-frame angular-velocity
integrals, not world yaw angles. XY rebound is a body-frame dot product with
the entry velocity direction, which is approximate because the body rotates;
pure-yaw rebound is separately reported from signed body-z angular velocity.
Vertical velocity is the body-z component, not world vertical velocity. Height
is reported both as ground clearance and base height relative to terrain origin.
The new evaluator may additionally record `world_velocity`; when present its
true world-z peak is reported under a separate key. That key is explicitly
`null` for legacy traces without the field; body-z is never substituted.

Posture uses tilt from projected gravity plus 12-joint deviations from the
manifest default pose. Actual/requested/target angle peaks preserve recorded
request-joint order. Torque uses `torque_peak` when present and otherwise sampled
applied `torque`; the report states which. Leg and wheel torque/speed peaks are
kept separate. These are measured-case metrics, not evidence of training or
real-world braking performance.
