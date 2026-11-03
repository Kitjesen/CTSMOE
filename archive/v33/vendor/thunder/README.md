# DOSO Thunder Training

Private, standalone Isaac Lab extension for the DOSO Thunder v4 wheeled-legged robot.

The repository contains only the Thunder v4 runtime assets and the `terrain_inward` velocity tasks. It does not depend on, vendor, or preserve the Git history of RobotLab.

## Supported stack

- Isaac Lab 2.3.2
- Isaac Sim 5.1
- Python 3.11
- RSL-RL 3.1.2
- CusRL 1.1.9.post0

## Install

Run from the repository root with the Python environment used by Isaac Lab:

```bash
${ISAACLAB_ROOT}/isaaclab.sh -p -m pip install -e source/doso_train
```

## Tasks

- `DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0`
- `DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0`

The rough task is registered to the frozen V0 configuration in
`source/doso_train/doso_train/tasks/terrain_inward/rough_v0_env_cfg.py`.
Training recipes and checkpoint lineage are grouped under
`scripts/experiments/rough/` and `docs/experiments/rough/`; generated
`logs/`, `outputs/`, checkpoints, TensorBoard events, and videos are not
versioned.

## Training history

See [docs/experiments/README.md](docs/experiments/README.md) for run timelines, parameter changes, checkpoints, measured effects, and pending evaluations.

For the current rough line, start at
[docs/experiments/rough/README.md](docs/experiments/rough/README.md). The
versioned launchers resolve the repository root from their own location and
accept `DOSO_ROOT`, `ISAAC_PYTHON`, and `DEVICE` overrides.

## Train and play

```bash
${ISAACLAB_ROOT}/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0

${ISAACLAB_ROOT}/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0 \
  --checkpoint /path/to/model.pt

${ISAACLAB_ROOT}/isaaclab.sh -p scripts/reinforcement_learning/cusrl/train.py \
  --task DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0

${ISAACLAB_ROOT}/isaaclab.sh -p scripts/reinforcement_learning/cusrl/play.py \
  --task DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0 \
  --checkpoint /path/to/run
```

## Validation status

Completed on the current host: static checks, the complete configuration contract, editable installation, Gym registration, all six configuration entry points, four CLI help checks, and wheel asset verification.

GPU environment reset, rollout, training, and checkpoint playback remain pending. The current cloud instance has no usable NVIDIA device and its system `libcuda.so.1` is invalid, so CUDA/Vulkan/PhysX execution cannot be treated as verified. Configuration loading was validated with the CUDA compatibility library at `/usr/local/cuda-12.8/compat`.

Run the complete contract with `python -m pytest -q tests/test_package_contract.py`. It includes the Kit configuration contract by default. On the current cloud instance, prepend `/usr/local/cuda-12.8/compat` to `LD_LIBRARY_PATH`; use `DOSO_SKIP_KIT_CONTRACT=1` only for package-only checks.

The initial `0.1.0` configuration intentionally preserves the peak actuator limits from the source task. Real-hardware derating belongs in a separate follow-up change after the baseline is reproducible.

## Provenance and licensing

This repository is private and intended for internal use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/` for source and asset provenance. The repository as a whole is not licensed under Apache-2.0 or BSD-3-Clause.
