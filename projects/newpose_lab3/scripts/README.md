# Reused Go2 lab3.0 entry points

`launch.py` registers Thunder and executes the pinned upstream `scripts/rsl_rl/train.py`
or `play.py` with `runpy`. It does not copy either training loop, the video writer or the
MoE-CTS algorithm. The upstream checkout defaults to `upstream/go2_rl_lab`; `--go2-root`
can select another checkout of the same pinned revision.

Both modes print a plan unless `--execute` is supplied. Fresh training requires an explicit
update count and loads no checkpoint. `--resume --checkpoint PATH` resumes only a new native
run; the compatibility runner enforces `completed_updates + updates <= 20000`.
The default runner config has zero updates, log standard deviation, no observation
normalizer, no extra wrapper reward/action clipping, and no Go2 mirror mapping.
The imported checkpoint restores its saved optimizer learning rate.
RoboGauge remains disabled because the upstream benchmark describes Go2. Playback uses the
compatibility runner's 530-to-16 frame-major exports; the wrapper corrects the upstream
metadata helper's zero-based iteration assumption to the explicit completed-update count.
No cumulative environment-step estimate is fabricated when migration changes environment count.

```bash
# Review the plan first. This does not import the simulator or load the checkpoint.
python scripts/launch.py train --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --num-envs 32 --updates 2 --seed 42 --run-name migration_smoke

# Resume a new native run, restoring course/RNG then starting new episodes.
python scripts/launch.py train --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --resume --checkpoint /ABS/model_2.pt --num-envs 32 --updates 2 --seed 42

# Official Newton GL recording through Go2 env.video_recorders; 600 steps at 50 Hz = 12 s.
python scripts/launch.py play --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --checkpoint /ABS/model_2.pt --num-envs 4 --video --video-length 600 --execute

# Fixed staircase row; use stairs_down for the descent counterpart and --level 9 for L9.
python scripts/launch.py play --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --checkpoint /ABS/model_2.pt --num-envs 4 --terrain stairs_up --level 6 \
  --video --video-length 600 --execute
```

Add `--legacy` for the old 19538 checkpoint and frozen environment. Legacy continuation is
blocked; a legacy checkpoint must not become a fresh run by changing its filename.

Playback terrain selectors are consumed by the selected fresh or legacy play config: `stairs_up` maps to
the inward/upward stair tile (`pyramid_stairs_inv`), `stairs_down` to `pyramid_stairs`, and
`flat` to a plane tile. `--level` chooses a fixed row, not a random upper initialization bound.
These options are rejected for training and do not overwrite the ten-level mixed curriculum.
Candidate fields can be set for either mode using repeated `--env-override FIELD=JSON`.
The whitelist is `gradual_boxes`, `height_weight`, `height_options`, `hip_extra_weight`;
the registered Gym factory applies them after configuration composition and before construction.

Keep the vendored `source/rsl_rl` 3.3.0 package first on the import path. Do not replace it
with the newer distribution required by generic upstream Isaac Lab training scripts.
The wrapper intentionally supports only the standalone Newton runtime; it does not import
Kit bindings. Playback copies and SHA-verifies the checkpoint into a fresh `results/play_*`
directory first; the upstream entry point writes its exports and videos beside that copy.
Training logs use `logs/`, and generated URDF conversion output uses `generated/`.
Native resume checks the saved source/configuration fingerprint, URDF, environment count,
terrain origins, both optimizer states and update budget. It restores course levels/types and
RNG before resetting episodes; it does not restore an in-flight physical trajectory.
Old PhysX-to-Newton continuation still fails closed.

This wiring does not establish checkpoint transfer equivalence, successful locomotion,
or stair acceptance. The environment, history, 309-dimensional critic, reset, both optimizer
updates and exported policy must pass native checks before a long training run.

`probe_native.py` instantiates the integrated eight-terrain environment without a runner or
optimizer. Start with `--steps 16 --num-envs 32`; use `--steps 1002` to verify real timeout and
reset/history behavior. If an environment has not naturally timed out, the long probe injects
only its episode counter near the end and records this separately. It preserves 20-second
episodes and collision terminations; injected timeouts are not evidence of surviving 20 seconds.

```bash
python scripts/probe_native.py --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --output /ABS/fresh-api-probe --num-envs 32 --steps 16 --visualizer none --device cuda:0
python scripts/probe_native.py --urdf /ABS/thunder/urdf/thunder_v4.urdf \
  --output /ABS/fresh-reset-probe --num-envs 32 --steps 1002 --visualizer none --device cuda:0
```

The output directory must be new. Optional `--policy /ABS/policy_19538.pt` accepts the old
stateless 530-to-16 TorchScript interface after action/default-pose checks. `trajectory.npz`
contains actual history, single-frame observations, teacher observations, actions, resets,
terrain assignments, forces, joints and weighted reward rates at 50 Hz. These samples cannot
rule out high-frequency physics-step torque oscillations or capture every impact peak.
Because the official environment autoresets before returning observations, post-reset contact
values are censored. `terminal_pre_reset.npz` captures each terminating environment before
`_reset_idx` changes its root/joints or clears sensor history, including per-body history peaks,
each termination term and leg-contact latch/duration before and after the complete reset.
`interface_passed` is separate from `rollout_valid`: three consecutive one-step resets fail
the overall probe even if dimensions, finite values and reset/history plumbing all pass.
