# Student 11501 upstairs diagnostic

One nominal rollout per level, seed 42; student only; 2 s initial stand then constant vx=0.35 m/s, vy=yaw=0. No observation noise, parameter randomization or external pushes. Original actuator delay remains. Stop at first environment termination, 25 s limit, or horizontal distance >4 m from terrain origin. This tests the distance criterion; it does not estimate a success rate under training conditions.

| Level | Actual riser (cm) | Elapsed incl. 2 s stand | Reached 4 m | Maximum body tilt | Sampled leg clipping |
|---|---:|---:|---|---:|---:|
| L3 | 13.0 | 16.26 s | True | 30.1 deg | 0.0000% |
| L4 | 14.5 | 17.16 s | True | 32.1 deg | 0.0000% |
| L5 | 15.5 | 16.38 s | True | 35.9 deg | 0.0000% |
| L6 | 18.0 | 19.26 s | True | 40.8 deg | 0.0000% |

Torque figures sample the final physics substep at each 50 Hz control step. Clipping means |computed-applied torque|>0.001 Nm; unsampled peaks may exist. All frames and first terminations are retained.

Conclusion: the existing student can complete the distance criterion up through this particular L6 staircase under nominal straight commands. This does not isolate the relative contribution of mixed commands, random initial states, observation noise, dynamics randomization or pushes to training difficulty.

Suggested next comparison: same checkpoint and matched seeds, crossing nominal/full randomization with straight/original mixed commands. Measure per-level 4 m completion, episode duration, base/calf contact, stalls and sampled clipping. If mixed commands dominate failures, prioritize an upstairs-only command curriculum; if randomization dominates, identify the failing randomization factor before adjusting its range. Reward changes and gain sweeps should be separate ablations.
