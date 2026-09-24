# V14 `target_step_limit` audit

## Result

The 51 candidate `target_step_limit` rejections in `gate300.json` are not
evidence of policy target-rate violations.  They are produced by comparing the
last action target with a terminal sample that has already been overwritten by
reset-target synchronization.

The recorded runtime contract is `12 rad/s * 0.02 s = 0.24 rad` per control
step.  The `0.2401 rad` gate therefore has an appropriate numerical margin for
this evaluation.  After masking reset-overwritten terminal samples, every
remaining observable target step is at most `0.24000000953674316 rad`.

This finding does **not** change the V14 checkpoint decision to accept.  The
gate has 52 independent non-target rejection reasons, including four
`new_physical_failure` reasons and 23 `hard_position_limit` reasons.  The gate
must remain rejected/stopped on those independent reasons.

## Root cause

The target limiter itself is correctly placed before actuator delay and torque
clipping:

- `guard_actions.py:24` computes the limit from target rate and `env.step_dt`.
- `guard_actions.py:26-30` limits each processed action relative to the prior
  processed target.
- `actuators.py:6-7` places actuator delay and DCMotor torque clipping after
  that target path.  Torque clipping cannot write the post-guard user target.

The measurement crosses a termination/reset boundary:

- `evaluate_early.py:70-75` suppresses the full environment reset after
  `retain_terminal` is enabled at `evaluate_early.py:213`.
- `reset_targets.py:55-64` wraps `_reset_idx` and still calls
  `synchronize_reset_targets` after the suppressed reset.
- `reset_targets.py:37-41` copies measured `joint_pos` into the position target.
- `evaluate_early.py:272` returns from `env.step` only after that termination
  handling, then records `robot.data.joint_pos_target` at
  `evaluate_early.py:293-304`.
- `enrich_analysis.py:15-19` includes `end_step` via `end_step + 1` and treats
  the reset-synchronized target as an adjacent action target.

Thus the reported jump is neither a real action-rate step nor a torque-clipping
effect.  It is the difference between the preceding limited action target and
the terminal joint position copied into the target buffer.

## Full-artifact evidence

| Suite | Cases | Raw `>0.2401` | Source | Candidate | Peak at `end_step` and has term | Observable `>0.2401` after mask |
|---|---:|---:|---:|---:|---:|---:|
| terrain83 | 332 | 83 | 40 | 43 | 83 / 83 | 0 |
| flat25 | 80 | 22 | 14 | 8 | 22 / 22 | 0 |

The candidate count is exactly the 51 `target_step_limit` reasons in
`gate300.json`: 43 terrain83 cases plus eight flat25 cases (three student and
five teacher).

Every case with a termination term also supplies direct reset-write evidence:

| Suite | Termination cases | Terminal `target == joint_pos` | Maximum absolute error |
|---|---:|---:|---:|
| terrain83 | 145 | 145 / 145 | 0 rad |
| flat25 | 80 | 80 / 80 | 0 rad |

Representative candidate events:

- terrain83 case 310, `V14_32131_teacher`, obstacles L9 at 1.2 m/s:
  `rr_thigh_joint` changes from `0.343629` to `1.019851 rad` at the terminal
  `stair_route_exit` frame, a reported `0.676222 rad` jump.
- terrain83 case 286, `V14_32131_teacher`, slope-up L9 at 1.2 m/s:
  `rl_thigh_joint` changes from `-1.392931` to `-0.844364 rad` at the terminal
  `time_out` frame, a reported `0.548567 rad` jump.
- terrain83 case 209, `V14_32131`, slope-down L9 at 1.2 m/s:
  `rl_calf_joint` changes from `0.602012` to `1.082489 rad` at the terminal
  `actuator_violation` frame, a reported `0.480477 rad` jump.
- flat25 case 72, `V14_32131_teacher`, settle-left:
  `rr_calf_joint` changes from `-2.257274` to `-1.901663 rad` at the terminal
  `time_out` frame, a reported `0.355612 rad` jump.

The frozen manifests establish the expected step limit:

- `evaluation300/terrain83/manifest.json:10776` records `dt = 0.02`, and line
  10814 records `leg_target_rate_rad_s = 12.0`.
- `evaluation300/flat25/manifest.json:2834` records `dt = 0.02`, and line 2872
  records `leg_target_rate_rad_s = 12.0`.

## Evidence limit

For a terminated episode, the trace does not contain the final action target
that was applied before reset synchronization.  That value has been replaced
by `joint_pos` before `evaluate_early.py` reads it.  Masking the overwritten
sample proves only that all still-observable non-terminal target steps are at
or below `0.2401 rad`.  It does not recover or certify the missing final action
target.

The offline mask is therefore a diagnostic correction, not authority to change
the gate to accepted.  The durable evaluator fix is to record the post-guard,
pre-reset applied target before reset writes back into the articulation target
buffer, and to gate that preserved sequence.  The frozen evaluator and gate
files were not modified in this audit.

## Reproduction

From this directory, run:

```powershell
python audit_target_steps.py
pytest -q test_target_step_mask.py
ruff check target_step_mask.py audit_target_steps.py test_target_step_mask.py
```

`audit_target_steps.py` reads both frozen `traces.npz` artifacts, manifests,
analyses, and `gate300.json`.  `target_step_mask.py` holds the independent
boundary rule.  The tests cover:

1. a termination term excludes the overwritten terminal sample;
2. a case without termination terms retains its final sample;
3. the first retained target is still compared with `initial_q`.

Verified locally on 2026-09-21: the audit completed successfully, all three
tests passed, and Ruff reported no issues.
