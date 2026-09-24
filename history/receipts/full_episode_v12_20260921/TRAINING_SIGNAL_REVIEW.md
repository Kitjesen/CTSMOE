# V12 training signal review (read-only evidence)

`pilot300_metrics.jsonl` contains 284 records, updates 44728..45011. No GPU was run and production files were not changed.

## Measured facts

- Preflight: 3000 steps, 4096 environments, 60 simulated seconds; 25 s episodes, ordinary resampling 5 s, `startup_window=null`, `MixedCommand`, `step_dt=0.02`.
- State totals: special teacher 2,775,000 + special student 915,000 = 3,690,000; ordinary teacher 912,000 + ordinary student 315,000 = 1,227,000; all flat states = 4,917,000; all-terrain states = 4096 × 3000 = 12,288,000.
- Preflight startup counts: teacher 8,793, student 2,809. These are diagnostic counts, not the PPO eligibility denominator.
- Pilot cumulative sample counters: first record teacher/student = 2,736/1,003; final record = 11,844/4,010. Across 284 records, increments average 41.70 teacher and 14.12 student per record; teacher increments are zero in 84/284 records and student increments zero in 185/284.
- Across the 284 pilot updates there are 284 × 4096 × 24 = 27,918,336 rollout env-states. The 15,854 eligible anchor env-states occupy **0.056787%** of that total (repeated time samples, not unique environments). A cumulative counter divided by one rollout is not an eligibility rate. Student anchor has no support and exactly zero loss in **185/284 updates (65.14%)**; teacher in 84/284. The first pilot update contributes 2,736 teacher / 1,003 student samples; startup resets are therefore important to the count distribution. This does not establish sufficient or insufficient total causal influence by itself.
- Anchor first/last 20-record means: `anchor_main` 0.032573→0.019571; teacher component 0.042775→0.028896; student component 0.022371→0.010245; `anchor_student` 0.019683→0.007875. `value` 0.446193→0.115783 and `surrogate` 0.023031→0.002574. These unconditional trends include zero-support updates. Conditioned on at least one eligible student sample, the first/last20 student anchor loss means are **0.043741 → 0.039377**, only about a 10% decrease; it is misleading to read the unconditional 0.019683 → 0.007875 drop as a 60% improvement in matching behavior. Teacher supported-update loss means are 0.071292 → 0.041281. These different sampled trajectories are not a matched behavior test and do not identify the gate failure cause.
- V12 already uses inherited V11 fixed coefficients: main `0.2109336283`, student `0.0022221143`. The receipt marks calibration disabled; the stored ratio 0.1 is V11 historical calibration metadata, not a measured V12 current gradient ratio.

## Mask/loss interpretation

The mask is documented as default-reset startup cohort on flat during the first 1 s; shared weights affect all terrains. After the first second there is no stated anchor-mask constraint. The actual masked error expression is `normalized_error[mask].square().mean()`. Do not describe this as a `mask.mean()` denominator or as loss weight scaling by eligibility fraction. Fewer eligible samples reduce estimator support and sample count; they do not automatically reduce the configured coefficient proportionally.

## Command sampling boundary

The receipt reports normal command limits `vx=[-1,1]`, `vy=[-1,1]`, `yaw=[-1.5,1.5]`, with flat scheduled/effective `vx=[-2,2]`. Exact float equality at `vx=0.6` is not a useful frequency target under continuous sampling. The useful support region is `0.45<=vx<=0.75`, `abs(vy)<0.05`, `abs(yaw)<0.05`; current metrics do not count this region, so disappearance or persistence is unproven.

## Causal boundary and next diagnostic

The pilot log establishes declining logged anchor losses and sparse cumulative eligibility, but does not establish that either caused the gate rejection. A valid next single-factor diagnostic, if authorized, is to keep the inherited coefficients, 25 s episode, and normal sampler fixed while adding/recording per-update support-region command counts and separate teacher/student mask deltas. Restoring V11 coefficients is invalid because V12 already uses them. Any V11 comparison must use matched update windows and report per-update counter differences rather than cumulative counters.

Detailed independently recalculated statistics: `training_signal_statistics.json`. Cumulative counters belong to this 284-update process, not the preceding 16-update smoke process.
