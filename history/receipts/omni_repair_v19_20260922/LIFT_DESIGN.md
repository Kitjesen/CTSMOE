# V19 flat omni wheel-lift design

This module is a training-side proposal built against the frozen V18 code. It
does not modify the frozen source and does not claim a training improvement.

## Scope and measured state

The terms are active only where `MixedCommand.terrain_idxs` exactly equals the
`flat` terrain index. Each wheel uses its real `*_foot_Link` world position and
a contact sensor filtered to `/World/ground/terrain/mesh`. On flat terrain the
wheel-bottom clearance is:

`h_i = wheel_link_z_i - env_origin_z - 0.093 m`

This deliberately does not estimate clearance from joint targets, ray-scanner
relief, a gait clock, or the robot posture. The existing wheel-bottom contact
velocity/slip penalty remains untouched.

## Command and tracking gate

An event can start only for either:

- lateral command `|vy_cmd| >= 0.20 m/s`; or
- pure yaw `|yaw_cmd| >= 0.30 rad/s` with commanded XY norm `<= 0.15 m/s`.

The corresponding measured body-frame velocity must have the requested sign,
reach at least 35% of command magnitude, and remain within
`max(error_floor, 0.75 * |command|)`. The error floors are 0.25 m/s laterally
and 0.30 rad/s in yaw. Normal forward/reverse commands and standing are zero.

## State transition and formulas

There is no desired foot sequence. A flight starts from a measured
contact-to-no-contact transition. It remains valid only while the environment
is flat, command and actual tracking remain eligible, and at least two wheels
have measured ground support. Thus a four-wheel jump invalidates all four
flights, and losing support or tracking cannot be repaired just before landing.

A no-contact-to-contact transition emits at most one landing event when:

- peak wheel-bottom clearance is 3--6 cm;
- measured air time is 0.08--0.45 s; and
- all validity gates held for the flight.

Lift quality is event based, so an airborne wheel cannot collect reward each
step:

`q(h) = clamp(min((h - .03)/.015, (.06 - h)/.015), 0, 1)`

At a valid landing, `r_lift = sum_i q(h_peak_i)` and
`r_land = number_of_valid_landing_events`. Reviewed initial manager weights are
`0.50` and `1.00`, fixed during a run: there is no automatic reward scaling.
The unintegrated maximum is therefore 1.50 per valid ideal wheel cycle, while
ordinary velocity tracking remains dense. With the expected `dt=0.02`, this is
`1.50 * dt = 0.03` per ideal cycle, comparable to the reviewed 0.02--0.05 L1
incremental cost. The sparse addition avoids a direct clock target.

The state resets from the episode reset mask. Reset contact is copied into the
previous-contact state, preventing a reset from fabricating a landing event.

## Integration and diagnostics

Call `configure(cfg)` after `configure_mixed`: V19 requires `mixed9_v1` and the
complete existing `pit_contact_{fr,fl,rr,rl}` ground-filtered sensor set. A
missing sensor is an integration error; this module does not create a fallback.
It adds `omni_wheel_lift` and `omni_wheel_landing` reward terms. `snapshot(env)`
reports gate populations, current valid flights, current-episode event counts,
the cumulative actual landing-event count since environment initialization,
support count, thresholds, formulas and weights. It also accumulates five
mutually exclusive command categories: `left`, `right`, `yawL`, `yawR`, and
`other`. Each category records tracking-gate environment
steps, measured wheel takeoffs, valid-flight environment steps, valid wheel
landings, and reward integral `dt * (0.50*r_lift + 1.00*r_land)`. These counters
survive episode reset; flight state and per-episode landing counters do not.
Nonfinite force, clearance,
command, or actual velocity raises instead of silently becoming a no-event. It
is a live training diagnostic, not held-out evidence.

Main risks are contact-force threshold chatter, command bins too close to the
gate, sparse credit at low policy quality, and the assumption that flat terrain
height equals the environment origin. Before a full run, inspect event rate,
support count, lift-height histograms and lateral/yaw tracking separately.
Isaac's reward manager multiplies term output by control-step `dt`. The reviewed
weights address the first-order scale mismatch, but sparse credit can still be
too weak if valid events are rare. Do not change weights during training or
raise them from the formula alone: the zero-update preflight must first prove at
least one valid landing event and report per-category counts. That proves the
measurement/reward chain executes; it does not prove the policy learned the
behavior. Any later calibration must compare event incidence with dense
velocity reward, extra actual-angle L1 cost, and regression metrics.
Changing the 3--6 cm band should follow observed Thunder wheel clearances; it
must not be tuned from reward alone. A held-out evaluation still needs to check
side speed/yaw error, wheel clearance, slip, base vertical motion and four-wheel
jump rate.
