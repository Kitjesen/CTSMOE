# V20 flat speed command patch

Prepared from the frozen `omni_repair_v19_20260922/recovery01/code` receipt.
Only the new `command_patch` directory was written. No SSH, training launch,
pipeline change, or modification of the frozen V19 source was performed.

## Runtime changes

Deploy `omni_commands.py` together with `omni_lift.py` into the new V20 code
snapshot. Continue using the unchanged V19 `flat_boundary.py` dependency.

| Nonzero flat command magnitude | V19 | V20 |
| --- | --- | --- |
| longitudinal m/s | 0.2–1.0 | 0.1–2.0 |
| lateral m/s | 0.2–0.6 | 0.2–1.0 |
| yaw rad/s | 0.3–1.0 | 0.2–1.5 |

Magnitudes remain uniformly sampled before boundary handling. Direction/class
probabilities remain `(0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05,
0.05, 0.10)` in the existing `NAMES` order. Pure lateral, pure yaw and full zero
commands retain exact zero axes. Five 5-second segments, 25-second configured
episodes, the deterministic 20% cohort per learner group, alternating
move/stop/move/stop/move, and a 50% probability of a 0.2-second ramp on each
episode's first segment are retained. The 50% is a Bernoulli probability, not an
exact half-count guarantee in every finite batch. Existing physical/guard
terminations can still end episodes before the configured horizon.

The flat override intentionally does not change the inherited global/other
terrain ranges. The frozen runtime receipt advertises global `vx=[-1,1]`,
`vy=[-1,1]`, `yaw=[-1.5,1.5]`; those values never described the V19 flat sampler
and must not be used as the V20 flat-range evidence. `cfg.omni_command_sampling`
and `omni_snapshot().sampling_contract` now record the actual flat sampler
contract. The snapshot separately records current nonzero goal magnitudes and
current nonzero command magnitudes so the initial ramp is visible.

`omni_lift.py` has exactly one changed line: `yaw_command_min: 0.30 -> 0.20`,
authorized after identifying the newly uncovered 0.20–0.30 pure-yaw band. Lift
and landing weights remain 0.50 and 1.00; tracking and other eligibility rules
are unchanged. The existing lift snapshot already serializes `RULES`, including
this threshold. Its inherited `omni_lift_v19` profile string is unchanged because
the authorized reward patch was restricted to that threshold.

## Boundary compatibility

The frozen `recovery01/PREFLIGHT_RUN.json` records the true contiguous flat band
as `[-99,-27]` to `[99,0]` meters, rather than one terrain cell. The unchanged
1.2-meter planning margin leaves 195.6 by 24.6 meters. Maximum longitudinal
motion over a full segment is 10 meters; maximum straight diagonal path length
is `sqrt(2^2+1^2)*5 = 11.18034` meters. Both fit from the band's center at all
headings. Tests exercise 2049 headings at maximum longitudinal, diagonal and
low/high yaw-turn speeds, retaining all requested amplitudes.

Near corners, a heading and its sign-reflection can both be infeasible. The
existing boundary policy remains: eight magnitude-preserving reflections, up
to four replacement draws, then an explicitly counted 0.2 m/s cardinal recovery.
The patch does not lower the requested maximum or clip sampled speeds. These
existing replacements and direction reflections mean final executed class
frequencies are not identical to the base sampler probabilities. The 20% sequence
cohort also changes overall stop frequency by design.

`BOUNDARY_PROBE.json` compares fixed positions/headings and sampler seeds for
6000 moving-only draws in each stress scenario:

| Positions | V19 initial infeasible | V20 initial infeasible | V19 / V20 cardinal recoveries |
| --- | ---: | ---: | ---: |
| Uniform inside planning rectangle | 7 (0.117%) | 25 (0.417%) | 0 / 0 |
| 1 cm inside a planning edge | 231 (3.85%) | 469 (7.817%) | 1 / 1 |
| 1 cm inside a planning corner | 2328 (38.8%) | 2387 (39.783%) | 78 / 89 |

Every final sampled path stayed in the planning rectangle, and none became a
stop in these moving-only stress cases. These are CPU twist-geometry tests, not
estimates of robot occupancy, training frequencies, dynamic braking success,
or held-out policy performance. The existing path checker uses 51 time samples;
continuous physical tracking, slip, inertia and guard timeouts need the normal
V20 simulator qualification. A deliberate corner case test confirms that an
infeasible 2 m/s goal remains explicitly unresolved rather than silently slowed.

## Verification

- `python -m pytest -q -p no:cacheprovider command_patch --durations=3`: **28 passed**.
- `ruff check command_patch`: **All checks passed**.
- `python command_patch/boundary_probe.py`: completed; output is `BOUNDARY_PROBE.json`.
- The first test run had one strict floating-point equality assertion at ramp
  midpoint; the assertion was corrected to tolerance and all 28 tests passed.

Tests execute the new sampler, inherited boundary functions and lift state
machine on real CPU PyTorch tensors. The command lifecycle uses a small fake
base class, so the tests do not establish IsaacLab integration or trained-policy
performance. Tests append the frozen V19 `code` directory to the import path for
the unchanged boundary dependency; do not deploy `conftest.py` as runtime code.
