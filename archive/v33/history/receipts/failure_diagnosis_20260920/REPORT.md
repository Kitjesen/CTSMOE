# Wave and slope screening, 2026-09-20

Frozen student: model_35500.pt (completed_updates 35501). Production training was left running and unchanged. Probe v4 produced results.json; its process was terminated after results were saved because Isaac shutdown hung. Earlier probe attempts produced no valid results: mixed-task reset expected pit_valid_spawn even with an empty pit membership. The isolated probe explicitly initializes that all-false flag.

Protocol: 144 first episodes; fixed body-frame vx=0.6 m/s, vy=0 and yaw-rate=0; no heading feedback. Wave levels 3/6/9; slope levels 5/7/9 (nominal 30/40/45 degrees). Eight cases per terrain/level/action-noise condition. Training observation noise and domain randomization retained. Cases between noise groups are independently randomized, not paired ablations. Success means world x relative to tile origin reaches 4 m before physical termination; it does not certify lane keeping or deployment safety. Time limit 25 s.

| Terrain | Pass / 48 | Base contact | Overspeed | Timeout without 4 m |
|---|---:|---:|---:|---:|
| Wave | 41 | 6 | 1 | 0 |
| Uphill | 9 | 7 | 11 | 21 |
| Downhill | 46 | 0 | 2 | 0 |

No position-limit or applied-torque-limit termination occurred. Absence of torque-limit termination does not establish absence of actuator saturation or adequate motor authority.

Uphill: 30 degrees passed 7/8 without action exploration and 2/8 with it. Both 40 and 45 degrees passed 0/16. At 40 degrees, five deterministic timeouts ended near x=2.09–2.69 m, with vx absolute error 0.45–0.49 m/s. At 45 degrees, deterministic timeouts included severe sideways drift (absolute y reached about 2.4–2.5 m) and negative world-x progress. These observations establish a propulsion/direction-control shortfall, not a causal attribution to one reward. All overspeed cases' largest measured joint speed was a wheel (about 47–112 rad/s uphill). This does not prove whether impact, slip or another joint triggered the guard first.

Wave: six of seven failures were base-contact termination (base_link contact force threshold 1 N); one early 0.10 s failure had wheel peak 56.44 rad/s. Higher waves therefore show contact/clearance/support robustness problems in this screening. Contact instrumentation alone does not distinguish belly scraping from overturning. The historical birth-height bug cannot explain all present failures; a residual early-reset failure remains worth investigating.

Downhill: 46/48 passed; two overspeed events at 0.20 s and 4.26 s, wheel peaks 77.52 and 104.49 rad/s. No evidence here for a general downhill propulsion bottleneck. Passing 4 m is not sufficient evidence of good braking, tracking or stability.

Runtime constraints checked: base-height penalty 0; mature vertical-speed penalty 0. Actual torque budgets legs 120 Nm / wheels 17 Nm; guarded leg target rate 12 rad/s (0.24 rad per 20 ms). Transient speed termination: same joint above 1.02x budget for 60 ms, or above 1.5x immediately; budgets legs 15 and wheels 30 rad/s. Do not infer that removing this guard fixes the underlying dynamics.

Ordinary wave/slope curriculum uses maximum displacement and command-dependent target distance. Downgrading is not equivalent to a physical failure. Friction randomization is static/dynamic 0–2, but this probe logged only per-robot material means, not per-wheel effective traction. Friction causality is unproven.

Next useful diagnostics: paired fixed-material uphill tests with wheel contact slip, base heading, per-joint clipping and actual applied torque histories; wave failure contact/body-pose traces; matched reset replay for early overspeed. No production constraints or rewards were changed by this investigation.
