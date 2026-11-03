# Thunder boundaries around upstream Go2 lab3 CTS

The runner is `compat.runner:ThunderMoeCtsRunner`, a subclass of the unmodified
`rsl_rl.moe_cts.runner.MoeCtsRunner` at Go2 commit
`6c95d9feef4814c25eb6536e61bb38e1d1d718d4`. It uses upstream actor/teacher/student
modules, PPO, distillation, storage, logging and rollout loop. The old copied
`teacher/moe_cts.py` is not imported. Go2 mirrored PPO augmentation is disabled.

The environment must return a TensorDict with `policy` (530), `single_obs` (53),
`critic` (309) and declare `cfg.newpose_history_layout = 'frame_major_current_last'`.
Also set `cfg.compute_final_obs=True` and
`cfg.newpose_timeout_bootstrap='terminal_observation'` (infinite horizon).
The ten full frames include the current frame and previous nine; each reset
repeats the first post-reset frame. Do not use IsaacLab term-major group history
on the six individual student observation terms. `compat.history.FrameHistory`
is available when the environment does not already implement this contract.

## Checkpoint and progress

The verified old 19538 checkpoint is SHA256
`8db552f69ab35105f737e74be9477c82e2d539a5d3984b2c32e41c1e992c226e`.
The old model tensor names and optimizer parameter groups match the upstream
model; only outer container keys are renamed. Conversion to Newton delegates
to `teacher.checkpoint.migrate_newton_checkpoint`: remove old dynamic-friction
input 272 and add 36 privileged inputs. Student/shared actor tensors and both
Adam histories are retained, with corresponding teacher/critic moments resized.
Teacher/critic predictions can change; this is not continuous physics or a
forward-equivalent teacher migration. Old terrain levels/types, common step
counter, Python/NumPy/Torch/CUDA RNG payloads are retained exactly in
`environment_state`, tagged with their original update count and schema, but
are **not restored** into Newton by assumption. CUDA device/state-format
differences are reported explicitly. Native observations, action semantics and trajectories still need
official-environment regression before further formal training.

Production `learn()` now refuses a fresh course: environment count, explicit
terrain/physics schema, terrain-column assignments and state update count must
match, and course/RNG restoration plus native acceptance must be verified.
The current PhysX-to-Newton bridge has no accepted restoration/mapping, so it
**cannot start formal continuation**. `load(..., load_optimizer=False)` remains
available for fresh-course diagnostics and reports that mode explicitly.
There is no runner-configuration flag that bypasses this requirement. The CPU
regression uses a test-only subclass restricted to its synthetic environment.

Timeouts retain the old `gamma*V(s_T)` reward correction. Lab3 captures final
observations before reset, but its history term is read without advancing the
history. The boundary rebuilds final history from the upstream transition's
pre-step history plus the terminal single frame, evaluates the correct teacher
or student critic, excludes simultaneous true terminations, and removes the
`time_outs` key before calling the original algorithm's process method. This
prevents upstream's default `V(s_t)` correction from being applied a second time.
Missing terminal data is an error; it never silently falls back to reset data.

Upstream stores `iter` as a zero-based last update index. This adapter records
`completed_updates`, saves `iter` as the next starting index, counts only
successful upstream `alg.update()` calls, restores the adaptive learning-rate
scalar, corrects automatic checkpoint filenames, and rejects budgets exceeding
the cumulative 20000 target. Raw upstream checkpoints without an explicit
Thunder schema/count are rejected; no count is inferred from a filename.

## Deployment

Go2's default exporter expects term-major history and `9+3*num_actions` single
features. Thunder uses `9+12+16+16=53`, so JIT and ONNX exports override only
history extraction. JIT accepts a complete **batch x 530** history and returns
**batch x 16** actions; CPU parity was tested with batch size 32. ONNX currently
has fixed input shape **1 x 530** and output shape **1 x 16**, because the reused
upstream serializer does not request a dynamic batch dimension. Neither export
maintains history internally. Both use the upstream encoder and actor.

## Validation

From the integration root with vendored RSL on PYTHONPATH:

The optional `--onnx` parity check additionally needs **onnxruntime** in the
isolated Newton environment. It is a test dependency, not required to train or
save checkpoints. Upstream's exporter also needs its normal `onnx`/`onnxscript`
dependencies. Install these in the new environment only; do not alter the
original formal-training environment to run this check.

```text
python -m compat.verify_bridge --source legacy/diagnostics/newpose_mixed_17650_20000_train_20261101/model_19538.pt --old-policy legacy/diagnostics/newpose_mixed_17650_20000_train_20261101/policy_19538.pt --report results/compat_cpu.json --exercise-update --onnx
```

The optional optimizer check uses synthetic observations with no simulator.
Its single update and checkpoint are temporary and never formal progress.
The test verifies actual 19538 student inference against the old JIT, all
unchanged tensors, both optimizers, reset/history order, export parity,
learning-rate restore, count round-trip and budget rejection. Native physics
acceptance is separate. Until its report exists, these are test definitions,
not completed test results.

To exercise the terminal-observation callback through the actual upstream
runner/storage/PPO with a two-step synthetic timeout, run
`python -m compat.verify_timeout_runner --source <model_19538.pt> --report <result.json>`.
This additionally excludes simultaneous true terminations and checks that no
duplicate upstream bootstrap occurs. Its optimizer update is synthetic and
does not save a checkpoint or advance the formal run.
