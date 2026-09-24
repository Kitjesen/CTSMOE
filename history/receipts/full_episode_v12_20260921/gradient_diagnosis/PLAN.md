# V12 static-gradient diagnosis plan

Status: prepared only. Do not run until the parent review selects the execution path. This protocol performs zero optimizer steps, writes no checkpoint, and never changes the fixed anchor coefficients.

## Exact reusable sources

- `receipts/anchor_v11_20260921/distribution_audit/launcher.py`: reuse its inherited `ReceiptRunner.learn` replacement, model/optimizer state hashing, mutation guard, native `alg.act` / `process_env_step` rollout, exclusive output rules, and final no-mutation checks. Its old guard forbids `compute_returns`; V12 must intentionally permit `compute_returns` because fresh on-policy advantages are required.
- `receipts/anchor_v11_20260921/distribution_audit/audit_stats.py`: reuse bounded streaming aggregation and environment patch restoration patterns. Do not reuse its sampled-record gradient claim because it does not contain PPO advantages or old action log probabilities.
- `receipts/anchor_v11_20260921/late_diagnosis/probe.py`: reuse default-reset construction, fixed command/control telemetry, exact observation capture (`policy`, `single_obs`, `critic`), per-joint actions/requests/targets, actuator peaks, phase state, wheel support, base contact, and model counterfactual action extraction.
- `receipts/anchor_v11_20260921/late_diagnosis/gradient_audit.py`: reuse `state_hash`, `assert_no_accumulated_gradients`, `gradient_vector`, norm/cosine helpers, production action scaling, model reconstruction, and per-window reporting. Its own receipt correctly says it has no PPO direction; that limitation must remain explicit.
- `anchor_cts.py:126-305`: authoritative production equations and parameter boundaries for PPO, entropy, main V6 anchor, CTS latent/load-balance, and student V6 anchor.

## Minimal safe protocol

Use the rejected V12 checkpoint `runs/full_episode_v12_pilot300_20260921/model_45010.pt` at completed update 45011 and the existing V6 reference. Instantiate the exact 4096-environment V12 training configuration. Do not randomize initial episode lengths. Run 12 consecutive native horizons of 24 control steps, totaling 288 steps or 5.76 seconds. This stays below the 320-step budget and crosses the original 5-second command resampling boundary once.

For every 24-step horizon, collect with the production stochastic `alg.act`, `env.step`, and `process_env_step`. Add diagnostic tensors to rollout storage before collection: episode age, original environment id, teacher/student group, flat-special-cohort membership, command, phase/support validity, foot contacts, base contact, and per-joint action/request/target state. Call production `compute_returns(next_obs)` to obtain fresh bootstrap values, returns, normalized advantages, stored actions, and old log probabilities. This storage mutation is required and is not an optimizer or model mutation.

Before clearing that horizon, reconstruct the production objectives with `torch.autograd.grad`:

1. Main actor: PPO clipped surrogate plus the production entropy term from the current on-policy storage. Report teacher and student rows separately and the production sum. The critic value objective is reported separately because it does not share the actor parameters used by the action anchor.
2. Student primary: CTS latent loss plus production load-balance loss on student rows.
3. V6 anchor: production masked physical-scale action loss with the saved main/student coefficients. In 0–1 seconds, use the stored `anchor_mask` and reference action. In 1–5 seconds, recompute the V6 reference only for the same surviving cohort and label this direction counterfactual because training does not apply the anchor after one second.

Compare only gradients defined on identical ordered parameter vectors: PPO versus main anchor on the shared actor (and teacher encoder where applicable), and latent/load-balance versus student anchor on the student encoder. Report raw norm, weighted norm, dot product, cosine, zero-gradient status, clipping fraction, and sample count.

The two registered windows are `0 <= age < 1` and `1 <= age < 5`. Startup rows must have the real stored `anchor_mask`. Sustained rows must be in the same flat special cohort, alive, moving (`command.norm > 0.05`), and have finite support telemetry. Preserve counts before any balancing; optionally compute a second equal-count stratified view, while keeping the full native view authoritative.

For every window and learner group, report raw rows, unique environments, survival, command directions, support-valid rows, nonzero advantages, and nonzero gradients. Low or missing counts indicate sample scarcity. Negative cosine or a materially negative dot product is called conflict only when both gradients are finite/nonzero and the registered count floor is met. Otherwise report “insufficient evidence,” never “no conflict.”

Per-joint evidence must retain all 16 action dimensions: four hip, four thigh, four calf position actions and four wheel velocity actions. Report action-reference error, requested/limited target error, pose deviation, torque/speed peaks, ground force, wheel clearance, phase ready/stable time, and base contact for exactly the same selected samples.

## Mutation and truth gates

- Hash model, PPO optimizer, student optimizer, checkpoint, and fixed coefficients before and after.
- Patch `optimizer.step`, `optimizer_stu_enc.step`, `alg.update`, runner save, and checkpoint writers to raise. Permit only `compute_returns` and rollout-storage clearing.
- Use `torch.autograd.grad`; require all parameter `.grad` fields to remain `None`.
- Require `optimizer_updates == 0`, no `model_*.pt` or `student_*.pt`, identical before/after hashes, and exactly 288 control steps.
- Never call the old V11 retained-observation direction a PPO gradient. Never describe the 1–5 second V6 direction as an applied training objective. Never auto-adjust coefficients or launch training from the diagnosis.

`prepare.py` only validates the rejected V12 checkpoint, gate, fixed coefficients, V6 reference, and source hashes, then writes an exclusive `PREPARED.json`. It does not import Isaac, open a GPU, or execute this protocol.
