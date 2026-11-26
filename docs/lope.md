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

`friction` is 0. There is no `ImplicitActuatorCfg` and no wheel joint. `self_collision=False` (avoiding permanent Newton `margin=0.01` phantom contact between the fixed battery links and chassis-thigh shapes) and `merge_fixed_joints=False`, so the four fixed feet stay separate. Each foot collision is a sphere of radius 0.032 m. `soft_joint_pos_limit_factor` is 0.9. Action targets are clipped to that same soft window, centered on each URDF interval, as `SmallDogEnvCfg` did.

Standing pose keeps the solved thigh and calf magnitudes and adducts the hips by 0.08 rad. Front hip axes are −X and rear hip axes are +X, so the signs below move every foot toward the midline. Stance width is 0.25688538375184233 m. The root is higher than the old 0.2674918208774965 m spawn so the original 11.333470889715092 mm sole gap stays.

- `fr_hip_joint` −0.08, `fl_hip_joint` +0.08, `rr_hip_joint` +0.08, `rl_hip_joint` −0.08
- thighs ±0.6715534335840251 rad (right negative, left positive)
- calves ±1.1 rad (right positive, left negative)
- spawn height 0.2740510271690917 m
- height target 0.2627175562793766 m

Action scale: hip 0.15 rad, thigh 0.2 rad, calf 0.2 rad. The action vector is 12 position targets in order FR, FL, RR, RL and hip, thigh, calf. URDF position limits are unchanged, including the mirrored thigh and calf windows.

## Observations, symmetry, rewards

Policy frame is 45 values: body angular velocity, projected gravity, velocity command, 12 relative joint positions, 12 relative joint velocities, 12 previous actions. History is 10 frames, oldest first, so the policy input is 450. Angular velocity is scaled by 0.25, joint velocity by 0.05, and the command's x component by 0.5, matching the official mixed observation scales `(0.5, 1, 1)`. The noisy training frame uses that same command scale, not 1.0 on command x. The critic linear velocity and command use that same `(0.5, 1, 1)` scale. Joint order stays FR, FL, RR, RL.

Critic is 275 values: linear velocity scaled by `(0.5, 1, 1)`, the policy core, joint acceleration scaled by 1e-4, applied torque scaled by 0.01, 16 foot-force values scaled by 1e-3, and 187 height samples. The 16 values are the official `foot_contact_force_norm` of `net_forces_w_history`: the max over the 3-step history, then those three steps, each a group of four feet in order FR, FL, RR, RL. The contact sensor is `NewtonContactSensorCfg` with `history_length=3`. The scan is the 1.6 m by 1.0 m yaw-aligned grid at 0.1 m, which is the Go2/Lope scanner. Heights come from `go2_rl_lab.utils.height_scan.ray_caster_height_scan` with offset 0.5, which replaces a non-finite ray hit with the minimum finite hit in that environment, or sensor z minus 1 m if every ray misses, then clipped to [-1, 1] and multiplied by 2.5. There is no wheel velocity, wheel torque, or wheel material block.

Symmetry swaps FR with FL and RR with RL, then multiplies every revolute component by -1. That map fixes the standing pose. Go2's hip-only sign does not, because Lope's left thigh and calf axes are the negation of the right axes. Each of the four foot-force groups is permuted the same way, with no extra sign. Hips also use the minus: the two front hip axes are both -X and the two rear hip axes are both +X, so a sagittal reflection reverses roll rate.

Rewards use the pinned `go2_rl_lab` terms with Lope's documented overrides:

- linear tracking standard deviation 0.5 and weight 2, yaw standard deviation 0.5 and weight 1
- base height weight from -1 to -10 over 5000 updates, target 0.2627175562793766 m (not Go2's 0.38 m)
- vertical linear velocity weight from -2 to 0 over 1500 updates
- Newton joint acceleration weight -3e-7, power -1.5e-5, torque -6e-5
- undesired-contact threshold 110 N
- hip and leg pose penalties with stand-still scale 10
- no feet-air-time term and no separate flat-orientation term

Termination is the 25 s timeout, `bad_orientation_locomotion` (2 rad tilt while walking, disabled while recovering from a fallen spawn), and `fallen_timeout` (2 s down while walking, 6 s down when spawned fallen). `illegal_contact` is not registered, and `undesired_contacts_locomotion` masks thigh/calf contact penalties while recovering from a fallen spawn. There is no wheel contact reward and no Thunder leg-contact duration term.

Domain randomization follows the official mixed task: base mass add ±1 kg, other links scale 0.9–1.1, friction 0–2, restitution 0–0.5, actuator gains scale 0.9–1.1, joint reset scale 0.5–1.5, and motor zero offset ±0.035 rad. Push linear velocity is the official ±0.4 m/s. Push angular rate stays ±0.6 rad/s. The Newton center-of-mass buckets on Lope's `base_link` are scaled to its 0.26 m × 0.20 m chassis (`x` ±0.05 m, `y` ±0.03 m, `z` −0.02..0.03 m).

## Commands, terrain, timing

Physics is 0.005 s and decimation is 4 (50 Hz). The episode is the official 25 s. Formal training uses 16384 environments and a 100000-update cap. Rollout length is 24. The network is the official 8-expert CTS policy with latent size 32, scalar noise, and `entropy_coef=0.005`. Reward clip is ±10. Save interval is 500.

Commands use `RLGymVelCommand` and `TerrainLevelsVelGym` with the official unscaled numbers. Initial ranges are x/y ±0.5 m/s and yaw ±1 rad/s. The range curriculum is ±1 / yaw ±2 at iteration 5000, x ±2 at 15000, x ±3 at 25000, x ±4 at 35000, and x −4.5..4.2, y ±1, yaw ±3 at 45000. Terrain caps use those same official speeds, with no wheels entry. Clip rules stay x limit 2.0 (y ±0.5, yaw ±1) and yaw limit 2.0 (x ±2, y ±1). Zero-command probability goes from 0 to 0.1 over iterations 0–1500. Resampling time is 5 s. Push linear velocity is ±0.4 m/s. The height ratio is not applied to any command or push speed.

The height-reward crop stays a Lope-sized box, ±0.135 m by ±0.101 m. That crop is body geometry, not a speed scale.

Fallen resets start at 0% for the first 3000 locomotion warmup updates, then rise to 25% by update 20000: left side, right side, back, or belly, with a random yaw and the root 6–12 cm lower. Locomotion remains the main task. Instant base-contact termination is not used, and `bad_orientation_locomotion` is masked while a fallen-spawn episode is still recovering. A `fallen_timeout` ends the episode after the base has stayed down (`projected gravity z > -0.7`) for 6 s (recovery) or 2 s (walking). Timeouts still end a normal 25 s episode. `recovery_upright` (weight −2) penalizes `max(0, 0.9 - upright_score)` continuously so standing smoothly reaches 0, and `stand_up_bonus` (weight 50) pays once per fallen-spawn episode when it first reaches `upright_score >= 0.9`.

Terrain copies `MIXED_MESH_TERRAIN_CFG`: 9 m tiles, 10 rows, 20 columns, border 25 m, initial level up to row 1. The wheels sub-terrain is removed and flat's proportion is 0.20. `stairs_up` is the official inverted pyramid.

- slopes 0–35°, stored as the rise/run gradient `(0, tan(35°))` ≈ `(0, 0.7002075382097097)`. This replaces both the official `(0.1, 0.568)` and Thunder's `tan(40°)`.
- stairs `step_height_range=(0.05, 0.25)` m. Official tread is 0.31 m; Lope keeps 0.25 m so the step is not scaled all the way down to about 0.21 m.
- obstacle height is the official unscaled `(0.05, 0.275)` m and wave amplitude is the official unscaled `(0.1, 0.28)` m. Obstacle width stays `(1.0, 2.0)` m. These sizes are not multiplied by a height ratio.

The height-reward crop is the official `(±0.2, ±0.15)` m box times the same scale, about `(±0.135, ±0.101)` m. The 1.6 m by 1.0 m lookahead scan is unchanged. The critic is 275 values: 88 proprioceptive values, including the 16-wide official foot-force history, plus 187 height samples.

Newton settings that have no Lope historical value (implicitfast integrator, nconmax 128, 0.01 m margin, 2.5e6 triangle pairs) follow this repository's fresh Newton task so the URDF can load on the same backend. `njmax` is 512 rather than 384 or the official 300 to provide headroom for multi-link terrain contacts. The other solver fields were not retuned in simulation.

## What was not run

This environment has no Isaac Lab, Newton, or a GPU training stack. Static tests cover the URDF, RS02 numbers, the 12-action mirror, the speed gate, and the plan printed by `launch_lope.py`. They do not step the simulator, and no checkpoint was created.
