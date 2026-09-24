# V10 verification

- Local Ruff: changed production files, new tests, pipeline, generated evaluation/analysis and artifact scripts passed.
- Local pytest: outlier/continuous/startup/phase suites: 26 passed; stance/stair-speed-logic/guard-math suites: 22 passed.
- Remote pytest in isolated V10 code: outlier/continuous/startup/phase suites: 26 passed (one existing pynvml deprecation warning).
- Remote deployed source hashes matched source_hashes.json before launch.
- Revised analysis ran successfully on complete real V9 interim trajectories in a separate directory; original artifacts were not overwritten. Per-joint startup evidence is exported and samples stop at each case's end_step.
- Counterfactual reward calculations are saved separately and are not reported as improvements in policy behavior.
- The launch shell was corrected to LF before successful launch. No training was launched by the earlier CRLF shell error.
- Runtime smoke and final behavior acceptance are separate: the pipeline checks 16 actual training updates, resume invariants, active flat counts, and an 83-case three-step evaluation smoke before the remaining 984 updates. Behavior is judged only from full evaluation after the thousand updates.

## Observed runtime result

- V9 completed at 44,711 and both full evaluations completed before V10 began.
- V10 smoke completed 16 updates to 44,727. Every required resume check passed; no phase-foundation reset occurred. Actual reward telemetry: flat_outlier_v1, top_k=3, tail_weight=0.5, 1,229 teacher flat environments and 410 student flat environments.
- The 83-case three-step evaluation smoke completed; manifest is explicitly smoke_only=true and is not a success-rate evaluation.
- Formal V10 continuation observed training at 44,731 (20 total V10 updates), 6.728 seconds/update, error=null. Model, both optimizers, terrain assignment, exact difficulty and course state restored. Target 45,711; automatic 332-case evaluation follows, with no automatic further training.
- deployment_receipt.json, run_at_launch.json, status_at_launch.json and eval_smoke_manifest.json contain the evidence. Snapshot training averages are not used as claims of improved startup behavior.
