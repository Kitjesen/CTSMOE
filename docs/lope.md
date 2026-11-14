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

`--terrain` accepts `mixed`, `flat`, `wave`, `slope_up`, `slope_down`, `rough_slope`, `stairs_up`, `stairs_down`, `obstacles`, and `small_squares`. `--level` is a row in `0..9`. Default environment count is 16384. The cumulative update cap is 100000, matching `Go2MoECTSRunnerCfg.max_iterations`. A fresh plan passes no `--checkpoint`. Smoke checks still pass `--num-envs 32 --updates 2`.

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

Policy frame is 45 values: body angular velocity, projected gravity, velocity command, 12 relative joint positions, 12 relative joint velocities, 12 previous actions. History is 10 frames, oldest first, so the policy input is 450. Angular velocity is scaled by 0.25, joint velocity by 0.05, and the command's x component by 0.5, matching the official mixed observation scales `(0.5, 1, 1)`. The critic linear velocity and command use that same `(0.5, 1, 1)` scale. Joint order stays FR, FL, RR, RL.

Critic is 263 values: linear velocity scaled by 2, the policy core, joint acceleration scaled by 1e-4, applied torque scaled by 0.01, four foot normal-force magnitudes scaled by 1e-3, and 187 height samples. The scan is the 1.6 m by 1.0 m yaw-aligned grid at 0.1 m, which is the Go2/Lope scanner, then `(sensor_z - hit_z - 0.5)` clipped to [-1, 1] and multiplied by 2.5. There is no wheel velocity, wheel torque, or wheel material block.

Symmetry swaps FR with FL and RR with RL, then multiplies every revolute component by -1. That map fixes the standing pose. Go2's hip-only sign does not, because Lope's left thigh and calf axes are the negation of the right axes. Foot-force magnitudes are permuted with no extra sign. Hips also use the minus: the two front hip axes are both -X and the two rear hip axes are both +X, so a sagittal reflection reverses roll rate.

Rewards use the pinned `go2_rl_lab` terms with Lope's documented overrides:

- linear tracking standard deviation 0.5 and weight 2, yaw standard deviation 0.5 and weight 1
- base height weight from -1 to -10 over 5000 updates, target 0.2561583499877814 m (not Go2's 0.38 m)
- vertical linear velocity weight from -2 to 0 over 1500 updates
- Newton joint acceleration weight -3e-7, power -1.5e-5, torque -6e-5
- undesired-contact threshold 110 N
- hip and leg pose penalties with stand-still scale 10
- no feet-air-time term and no separate flat-orientation term
- undesired thigh/calf contacts, threshold 5 N
- hip and thigh/calf pose penalties with stand-still scale 1
- joint torque weight -1e-4, power -2e-5, acceleration -1e-7

Termination is timeout, leaving the terrain or the assigned tile, base contact above 1 N, and `bad_orientation` at 0.85 rad. There is no wheel contact reward and no Thunder leg-contact duration term.

Domain randomization follows the official mixed task: base mass add ±1 kg, other links scale 0.9–1.1, friction 0–2, restitution 0–0.5, actuator gains scale 0.9–1.1, joint reset scale 0.5–1.5, and motor zero offset ±0.035 rad. Push linear velocity is the official ±0.4 m/s times the height scale; push angular rate stays ±0.6 rad/s. The Newton center-of-mass buckets use the official offsets, on Lope's `base_link`.

## Commands, terrain, timing

Physics is 0.005 s and decimation is 4 (50 Hz). The episode is the official 25 s. Formal training uses 16384 environments and a 100000-update cap. Rollout length is 24. The network is the official 8-expert CTS policy with latent size 32 and scalar noise. Reward clip is ±10. Save interval is 500.

Commands use `RLGymVelCommand` and `TerrainLevelsVelGym`. The scale factor is `0.2561583499877814 / 0.38` = `0.6741009210204774`. Linear ranges and yaw magnitudes both use that factor. Limit-velocity entries stay official direction codes, including the stage at iteration 25000 that drops lateral limit sampling. Official initial ranges are x/y ±0.5 m/s and yaw ±1 rad/s, so Lope starts at about ±0.337 m/s and ±0.674 rad/s. The iteration-45000 stage is officially x −4.5..4.2 m/s, y ±1, yaw ±3; Lope uses about x −3.033..2.831 m/s, y ±0.674, yaw ±2.022. Terrain caps, including flat's official (−4.5, 4.2), are scaled the same way. Push and reset linear velocities are scaled; push angular rates stay at the official ±0.6 rad/s. The raw 4.5 m/s figure is not used as a Lope command.

Terrain copies `MIXED_MESH_TERRAIN_CFG`: 9 m tiles, 10 rows, 20 columns, border 25 m, initial level up to row 5. The wheels sub-terrain is removed and flat's proportion is 0.20. `stairs_up` is the official inverted pyramid.

- slopes 0–35°, stored as the rise/run gradient `(0, tan(35°))` ≈ `(0, 0.7002075382097097)`. This replaces both the official `(0.1, 0.568)` and Thunder's `tan(40°)`.
- stairs `step_height_range=(0.05, 0.25)` m. Official tread is 0.31 m; Lope keeps 0.25 m so the step is not scaled all the way down to about 0.21 m.
- obstacle heights and wave amplitudes are the official values times the height scale. Obstacle width stays 1–2 m.

The height-reward crop is the official `(±0.2, ±0.15)` m box times the same scale, about `(±0.135, ±0.101)` m. The 1.6 m by 1.0 m lookahead scan is unchanged, so the critic stays 263 values. Bad orientation terminates at 2.0 rad.

Newton settings that have no Lope historical value (implicitfast integrator, 128 contacts, 0.01 m margin, 2.5e6 triangle pairs) follow this repository's fresh Newton task so the URDF can load on the same backend. They were not retuned in simulation.

## What was not run

This environment has no Isaac Lab, Newton, or a GPU training stack. Static tests cover the URDF, RS02 numbers, the 12-action mirror, the speed gate, and the plan printed by `launch_lope.py`. They do not step the simulator, and no checkpoint was created.
