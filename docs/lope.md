# Lope on the CTS+MoE / Isaac Lab 3 stack

Lope is a 12-DoF fixed-foot quadruped added beside Thunder. Thunder's URDF, actuator contract, task id, and formal-run settings are unchanged. This task does not load a Thunder or Go2 checkpoint.

## Launch

From the repository root. The command prints a plan. Add `--execute` only when Isaac Lab 3 / Newton is installed and a run is intended. This adaptation does not start training.

```bash
python scripts/rsl_rl/launch_lope.py train \
  --urdf source/lope_lab3/assets/lope/urdf/lope.urdf \
  --num-envs 32 --updates 2 --seed 42 --run-name lope-smoke
```

Task id: `Lope-Newton3-Fresh-v0`.

The gym registration points at:

- `lope_lab3.tasks.lope.env_cfg:LopeFreshEnvCfg`
- `lope_lab3.tasks.lope.env_cfg:LopeFreshPlayEnvCfg`
- `lope_lab3.tasks.lope.rsl_rl_cfg:LopeFreshMoeCtsRunnerCfg`

Playback example, still plan-only without `--execute`:

```bash
python scripts/rsl_rl/launch_lope.py play \
  --urdf source/lope_lab3/assets/lope/urdf/lope.urdf \
  --checkpoint /ABS/model_N.pt --terrain stairs_up --level 3
```

`--terrain` accepts `mixed`, `flat`, `rough`, `slope_up`, `slope_down`, `stairs_up`, `stairs_down`, `obstacles`, and `wave`. `--level` is a row in `0..7`. Default environment count is 4096. The cumulative update cap is 3000, which is `SmallDogRoughRunnerCfg.max_iterations` on `headless-v67`. A fresh plan passes no `--checkpoint`.

## Asset

Copied from [Kitjesen/lope](https://github.com/Kitjesen/lope) branch `cursor/headless-urdf-meshes-only-51bf` at commit `74e51939d35e31cfa6dc03951af3911fbe0fe565`.

- URDF: `source/lope_lab3/assets/lope/urdf/lope.urdf` (robot name `lope`)
- SHA-256: `56cd5b1dc81545cc324e8b741e947ba0ccc90d8343b999286ef2ce550caa3a2b`
- Meshes: the 35 files that URDF references, including `meshes/lope_visual_preview/`
- Mass summed from the URDF: 13.333464609253825 kg
- Third-party note: `source/lope_lab3/assets/lope/THIRD_PARTY.md`

Training numbers were read from `headless-v67` commit `86008cc53e119038959f93fa48b9f1b9287dfb6f`:

- `training/v67_cts/joint_motion_config.json`
- `training/v67_cts/assets/smalldog_v67/config/model_contract.json`
- `training/runtime/source/robot_lab/robot_lab/assets/smalldog.py`
- `training/runtime/source/robot_lab/robot_lab/tasks/smalldog/env_cfg.py`
- `training/runtime/source/robot_lab/robot_lab/tasks/smalldog/rough_cfg.py`

The recovery run `training/recovery_headless/run_configuration.json` flattens every joint to 17 N·m. That model is rejected.

## Actuators

`lope.urdf` still says `effort="17"` on every revolute joint. That field is the rejected flat cap. Isaac Lab 3 `DCMotorCfg` enforces the RS02 model from `smalldog.py`:

| Group | effort / solver limit | stall `saturation_effort` | velocity | stiffness | damping | armature |
| --- | --- | --- | --- | --- | --- | --- |
| hip and thigh | 7.0 N·m | 17.0 N·m | 42.9350995991 rad/s | 40 | 0.8 | 0.0042 |
| calf, 2:1 | 14.0 N·m | 34.0 N·m | 21.4675497995 rad/s | 60 | 1.5 | 0.0168 |

`friction` is 0. There is no `ImplicitActuatorCfg` and no wheel joint. `self_collision=True` and `merge_fixed_joints=False`, so the four fixed feet stay separate. Each foot collision is a sphere of radius 0.032 m. `soft_joint_pos_limit_factor` is 0.9. Action targets are clipped to that same soft window, centered on each URDF interval, as `SmallDogEnvCfg` did.

Standing pose and spawn height, from `joint_motion_config.json`:

- hips 0
- thighs ±0.6715534335840251 rad (right negative, left positive)
- calves ±1.1 rad (right positive, left negative)
- spawn `initial_root_height_m` = 0.2674918208774965
- height target `base_height_target_m` = 0.2561583499877814

Action scale: hip 0.15 rad, thigh 0.2 rad, calf 0.2 rad. The action vector is 12 position targets in order FR, FL, RR, RL and hip, thigh, calf. URDF position limits are unchanged, including the mirrored thigh and calf windows.

## Observations, symmetry, rewards

Policy frame is 45 values: body angular velocity, projected gravity, velocity command, 12 relative joint positions, 12 relative joint velocities, 12 previous actions. History is 10 frames, oldest first, so the policy input is 450. Scales match the robot_lab Go2/Lope terms (0.25, 1, 1, 1, 0.05, 1).

Critic is 263 values: linear velocity scaled by 2, the policy core, joint acceleration scaled by 1e-4, applied torque scaled by 0.01, four foot normal-force magnitudes scaled by 1e-3, and 187 height samples. The scan is the 1.6 m by 1.0 m yaw-aligned grid at 0.1 m, which is the Go2/Lope scanner, then `(sensor_z - hit_z - 0.5)` clipped to [-1, 1] and multiplied by 2.5. There is no wheel velocity, wheel torque, or wheel material block.

Symmetry swaps FR with FL and RR with RL, then multiplies every revolute component by -1. That map fixes the standing pose. Go2's hip-only sign does not, because Lope's left thigh and calf axes are the negation of the right axes. Foot-force magnitudes are permuted with no extra sign. Hips also use the minus: the two front hip axes are both -X and the two rear hip axes are both +X, so a sagittal reflection reverses roll rate.

Rewards use the pinned `go2_rl_lab` terms with Lope's documented overrides:

- linear tracking standard deviation 0.2, yaw 0.35
- base height weight -1 at 0.2561583499877814 m, with no -1 to -10 schedule (`SmallDogEnvCfg` sets that curriculum to `None`)
- vertical linear velocity weight -2, also without the 1500-update ramp
- feet air time weight 0.25, threshold 0.12 s
- flat orientation weight -2
- undesired thigh/calf contacts, threshold 5 N
- hip and thigh/calf pose penalties with stand-still scale 1
- joint torque weight -1e-4, power -2e-5, acceleration -1e-7

Termination is timeout, leaving the terrain or the assigned tile, base contact above 1 N, and `bad_orientation` at 0.85 rad. There is no wheel contact reward and no Thunder leg-contact duration term.

Domain randomization follows the Lope flat/rough overrides: base mass add ±0.3 kg, other links scale 0.9–1.1, friction 0.7–1.2 / 0.6–1.0, restitution 0–0.05, actuator gains scale 0.9–1.1, joint reset scale 0.99–1.01. Push and motor zero-offset stay off. The center-of-mass box is Lope's ±1 cm in x/y and ±0.5 cm in z, applied through this repo's Newton bucket hook because MJWarp needs the offset before the model is finalized.

## Commands, terrain, timing

Physics is 0.005 s and decimation is 4 (50 Hz). Lope's Go2 parent uses that same pair. The episode is 15 s, from `SmallDogEnvCfg`, not Thunder's 25 s. Rollout length is 24. The network is the pinned 8-expert CTS policy with latent size 32 and scalar noise. Reward clip is ±10. Save interval is 500.

Initial commands, from `rough_cfg.py`: x -0.15..0.35 m/s, y ±0.1 m/s, yaw ±0.3 rad/s, resampling 15 s, 10% standing. `ForwardSpeedCurriculum` raises only the positive-x ceiling by 0.1 m/s after a batch of 128 eligible forward episodes with pass ratio 0.7, and it stops at 1.0 m/s. The later R6 config (ceiling 1.5 m/s, initial terrain level 10, 10000 updates) is a warm-started gait extension and is not this from-scratch task.

Terrain is the rough_cfg mix on 6 m tiles, 8 rows, initial row 0:

- flat 30%
- rough noise 0.5–2.5 cm
- slopes 0–35°, stored as the rise/run gradient `(0, tan(35°))`. Isaac Lab's `pyramid_sloped_terrain` sets height to `slope * run`. It does not treat `slope_range` as radians. `tan(35°)` is about 0.7002. The old 0.22 value was the historical Lope rough cap and is not used.
- stairs `step_height_range=(0.05, 0.25)` m (5–25 cm), step width 0.25 m
- obstacles 1–10 cm
- waves 1–8 cm

Official Go2 mixed terrain at the pinned commit uses stairs `(0.05, 0.257)` m and slope gradient `(0.1, 0.568)` (about 29.6°). Thunder uses stairs `(0.05, 0.25)` m and `tan(40°)`. Lope stairs match Thunder's 5–25 cm ceiling. Lope slopes stop at 35°, not 40°. The historical Lope rough cap of 1–10 cm and 0.22 is not used.

`stairs_up` is `MeshPyramidStairsTerrainCfg` and `stairs_down` is the inverted mesh, the same class assignment as `rough_cfg.py`. Isaac Lab 3's stair docstring describes the non-inverted pyramid as a high center platform. Both stair classes use the 5–25 cm step range. Command limits stay at Lope's rough-task speeds; this is not the Go2 flat speed range near -4.5..4.2 m/s.

Newton settings that have no Lope historical value (implicitfast integrator, 128 contacts, 0.01 m margin, 2.5e6 triangle pairs) follow this repository's fresh Newton task so the URDF can load on the same backend. They were not retuned in simulation.

## What was not run

This environment has no Isaac Lab, Newton, or a GPU training stack. Static tests cover the URDF, RS02 numbers, the 12-action mirror, the speed gate, and the plan printed by `launch_lope.py`. They do not step the simulator, and no checkpoint was created.
