# Mixed training audit, 2026-09-19 18:38–18:41 CST

Read-only audit; live training unchanged. Source: audit_progress_1838.json; run mixed9_4096_60000_climbfirst_20260919, updates 31866–32668. Compare first and last 100 updates; episode metrics use differences of cumulative counters, not lifetime averages. Curriculum difficulty and speed mix are not held fixed, so these are warning signals rather than a controlled policy regression test.

## Findings

- Pit frontier reached 0.60 m; recent five frontier batches failed promotion. Latest bin counts (episodes, successful climb, failed): (78,72,6), (71,67,4), (65,63,2), (79,75,4). Failure threshold, not aggregate climb rate, prevents further promotion.
- Recent 100 updates: student pit 3952/4311 climbs (91.67%), 351 failed (8.14%); teacher 12518/12793 (97.85%), 265 failed (2.07%). Includes lower-height review. Student physical causes: 221 base contacts, 126 overspeed, 0 position, 0 torque; four lane exits. Cause counters can overlap. Contact location and phase require trajectories; counters do not identify wall vs floor contact.
- Student stair forward-exit rates early/recent: up 96.44%/92.96%, down 96.96%/94.32%; forward velocity MAE up 0.1134/0.1664 m/s, down 0.1182/0.1443 m/s. These are training route-exit metrics, not independent staircase-completion tests.
- Terrain first100/last100 mean: wave 6.2825/5.9416, slope up 4.8372/4.5402, slope down 5.3856/5.7971, rough slope 5.7865/5.7826, stairs up 6.1808/6.1189, stairs down 6.3161/5.9806, obstacles 5.5186/5.9034, flat 5.4664/5.8383. Top-level random recycling and small terrain allocations affect means.
- Allocation: pit 2048, stairs 1639 (90.01% together); wave32, slopeup64, slopedown63, rough31. Imbalance is a plausible interference/coverage risk, not proven causal attribution.
- Mean action noise standard deviation first/last100: 0.37424/0.58331; learning rate 0.00011646/0.00013779. Noise increases may harm parking and stochastic rollouts. Current causal link requires deterministic/noisy matched evaluations. Earlier 25cm tests found deterministic standing success despite noisy failure; do not extrapolate to60cm.
- Strict parking success remains zero. Continuous pit stability reward is nonzero (~0.027 global logged episode reward); sparse success reward stays zero. Current three-second post-climb limit plus uninterrupted two-second gate narrows the stabilization window.
- Course only advances, no fallback when current frontier becomes too difficult. Repeated 64+ sample batch testing without held-out confirmation can eventually advance on favorable batches. Fixed-seed held-out height/speed evaluation is missing from promotion gate.
- Guard last100 has zero applied torque violations but 19123 joint-physics-step overspeed samples, zero sustained threshold samples, and extreme peak220.925rad/s at update32643 left-rear wheel. Includes all terrains and resets; cannot attribute to pit or normal steady locomotion. Extreme overspeed termination exists in code; termination is not a hard speed clamp. Need per-environment event context and reset/landing separation.
- Self collision remains disabled (run.shared_self_collisions=false); simulated climb may exploit configurations invalid on hardware. Platform-stop command uses perfect simulator state. Neither gap is resolved by higher terrain level.
- ~6.888s/update: collection6.381s, learning0.507s. Losses finite; these alone do not prove effective policy improvement.

## Recommended investigation order

1. Frozen matched teacher/student, deterministic/noisy 0.50/0.55/0.60m evaluation; log base collision location, support, clearance, joint speed, stopping state and action noise.
2. Attribute extreme-speed events to terrain/phase before changing constraints.
3. Fixed-height stair/wave/slope regression suite before redistributing environments or changing rewards.
4. Test task/phase-specific exploration and longer parking evaluation, then decide whether a curriculum fallback and terrain-balanced sampling are justified.

No training parameters, rewards, termination rules or running processes were changed by this audit.
