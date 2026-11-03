# Uphill approach V16 — bounded experiment, 2026-09-22

User authorized the suggested higher-speed/run-up uphill training after asking why 45-degree evaluation triggered overspeed. Preserve every checkpoint and frozen V15 runtime. No hardware deployment.

## Observed failure, not an inferred motor command

Corrected 332-case evaluation, V15 student completed32131, slope_up level9 (45deg), vx=0.3m/s, case201: actuator speed termination at16.66s. Right-front wheel substep absolute peak55.9287rad/s. Guard causes speed=true, torque=false, position=false. Extreme wheel threshold45rad/s applies to any5ms sample; sustained threshold30.6rad/s requires12consecutive5ms samples (0.06s). Applied torque remained limited to17Nm wheel/120Nm legs.

Before the peak the robot was retreating and yawing, with intermittent support. At16.66s the raw right-front wheel target was0.3845rad/s; the20ms-end measured speed was-2.9211rad/s despite the55.9287rad/s intra-step peak. Thus this is not proof that the policy requested55.9rad/s. Existing20ms contact/state records cannot distinguish delayed-PD oscillation, constraint/contact impulses and free-wheel acceleration within the step. Preserve that uncertainty until5ms replay.

## Planned serial execution

1. Verify coordinator8153 and training child8285 identity, then transfer coordination only. Child700-update segment continues to completed32831 unchanged. Preserve original coordinator status and record handoff.
2. Complete corrected332nine-terrain +80full-flat paired evaluation. Replay the old332-case batch with5ms joint velocity/target/torque/contact/root records.
3. At completed32831 run180 fixed uphill cases: student+teacher,35/40/45deg, command0.3/0.6/1.0/1.2/1.5m/s, front collision gap0.1/1.5m, lateral offsets-0.1/0/+0.1m. Record measured entry speed and compare the same x=4m exit in both gap groups. This exit is not whole-body crest completion and fixed counts are not random success probabilities.
4. Zero-update3000-step/60-simulated-second preflight with4096environments for the new approach profile. Forbid optimizer/save calls; compare model and optimizer hashes; check finite states, torque budget, per-reset collision placement and command scope. Physical failure counts remain diagnostic: this runtime preflight does not certify successful45deg locomotion.
5. If runtime preflight passes,16integration updates then984updates (1000 total) to completed33831. Keep rewards, actuator thresholds, remaining terrain tasks and4096allocation unchanged. Evaluate all412cases and180uphill cases again. Remaining4x2000updates end completed41831, total10000from31831 including the prior1000. Preserve existing continuation stop criteria: technical/numerical/applied-torque failures; or the same learner below10/20flat on two consecutive reviews. No old early300-update performance rejection is reinstated.

## New training scope and limits

Only slope_up environments with env_id modulo4 nonzero receive the approach profile (approximately75%; exact count recorded); remaining slope_up environments keep original mixed commands. Other eight terrain classes are untouched by the new hooks. Dedicated random generator avoids consuming the global random stream for new draws.

Approach group: per-episode held positive0.6/1.0/1.2/1.5m/s, vy=wz=0, one of four outward initial headings, collision-front gap0.1or1.5m. All collision points must fit in the central3m platform with0.1m edge margin and0.03m initial clearance. No2m backward translation, no teleport onto the opposing slope, no initial velocity boost. Actual heading may drift; heading feedback is not part of this first factor comparison.

Episode cap stays25s. The approach group alone truncates on outward base projection4m, preventing training on the neighboring terrain atlas tile under an uphill label. No claim of crest parking/deceleration: the existing outer platform is too short to validate that phase; a longer isolated track would be needed. Original terrain-level logic is retained.

## Validation before scheduling

Ruff passed. Four focused tests passed: parent commands/time-left/global RNG remain unchanged outside the selected uphill cohort; both gap placements across four headings; oversized collision rejection before simulator writes; complete180-case pairing required. CPU geometry probe used the evaluated URDF hash83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3 and128 poses across the existing0.5–1.5default-joint reset range; minimum platform margin0.0999999m, clearance0.0299999m. These checks are not a substitute for queued Isaac preflight or learned behavior evaluation.
