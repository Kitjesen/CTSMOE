"""Matched frozen-policy wave probe with explicit source and execution contracts."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
for name in ('root', 'code', 'checkpoint', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--profile', choices=('rs0402_slew12_v3', 'rs0402_transient_v4'), required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(a.code), str(a.root/'upstream/source/rsl_rl'),
               str(a.root/'upstream/source/robot_lab'), str(a.root/'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
# Isaac app startup must follow import-path and EULA bootstrap, before simulation imports.
from isaaclab.app import AppLauncher  # noqa: E402
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from actuator_guard import ActuatorGuard
    from policy_contract import checkpoint_profile, export_contract
    from train import apply_command_curriculum_start
    from wave_reset import configure_wave_reset

    torch.set_num_threads(2)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    source_profile = checkpoint_profile(checkpoint)
    if source_profile != 'rs0402_slew12_v3':
        raise ValueError('Expected the same pre-change policy for both probe arms')
    cases = [dict(learner=learner, level=level, speed=speed, repeat=repeat)
             for learner in ('teacher', 'student') for level in (0, 3, 6, 9)
             for speed in (.3, .6, .9) for repeat in range(3)]

    class FirstEpisodeEnv(Go2Env):
        retain_terminal = False

        def _reset_idx(self, ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)

    cfg = make_config(len(cases), a.root/'cache/usd', actuator_profile=a.profile)
    cfg.seed, cfg.log_dir = 42, str(a.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, 'intermediate')
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    for name in vars(cfg.events):
        if name not in ('reset_base', 'reset_robot_joints'):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.func = reset_root_state_uniform
    cfg.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.reset_base.params['velocity_range'] = dict(cfg.events.reset_base.params['pose_range'])
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    configure_wave_reset(cfg, 'wave_surface_v1')
    raw = FirstEpisodeEnv(cfg=cfg)
    guard = ActuatorGuard(raw, a.profile)
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = torch.tensor([c['level'] for c in cases], device=raw.device)
    terrain.terrain_types.zero_()  # First original column is wave.
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    raw.scene.env_origins[:] = terrain.env_origins
    command = raw.command_manager.get_term('base_velocity')
    command._init_terrain_infos()
    command._update_env_command_ranges()
    assert bool((command.terrain_idxs == command.terrain_type2idx['wave']).all())
    speeds = torch.tensor([c['speed'] for c in cases], device=raw.device)

    def resample(ids):
        command.commands[ids] = 0
        command.commands[ids, 0] = speeds[ids]
        command.time_left[ids] = 1e9

    command._resample = resample
    env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
    agent = MoECTSRunnerCfg().to_dict()
    agent['robogauge'] = {'enabled': False}
    runner = OnPolicyRunnerCTS(env, agent, log_dir=str(a.output/'loader'), device=raw.device)
    policy = runner.alg.policy
    policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    runner.eval_mode()
    robot = raw.scene['robot']
    initial = robot.data.root_state_w.clone()
    start_joints = robot.data.joint_pos.clone()
    raw.retain_terminal = True
    active = torch.ones(len(cases), dtype=torch.bool, device=raw.device)
    ended = [None]*len(cases)
    max_speed = torch.zeros_like(robot.data.joint_vel)
    max_torque = torch.zeros_like(max_speed)
    update_scene = raw.scene.update

    def measure(dt):
        update_scene(dt)
        max_speed[:] = torch.maximum(max_speed, robot.data.joint_vel.abs()*active[:, None])
        max_torque[:] = torch.maximum(max_torque, robot.data.applied_torque.abs()*active[:, None])

    raw.scene.update = measure
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(1250):
            actions = policy.act_inference(obs)
            ids = torch.arange(36, device=raw.device)
            critic = policy.critic_obs_normalizer(policy.get_critic_obs(obs[ids]))
            actions[ids] = policy.actor(torch.cat((policy.teacher_encoder(critic),
                policy.single_obs_normalizer(obs['single_obs'][ids])), -1))
            actions[~active] = 0
            obs, _, _, _ = env.step(actions)
            distance = robot.data.root_pos_w[:, 0]-initial[:, 0]
            physical = raw.termination_manager.terminated
            timeout = raw.termination_manager.time_outs
            success = (distance >= 4.) & ~physical
            finish = active & (physical | timeout | success)
            for i in finish.nonzero().flatten().tolist():
                ended[i] = dict(seconds=(step+1)*raw.step_dt, forward_m=float(distance[i]),
                    success=bool(success[i]), physical=bool(physical[i]), timeout=bool(timeout[i]),
                    causes={name: bool(value[i]) for name, value in guard.violation_causes.items()})
            active[finish] = False
            if step % 200 == 0:
                print('WAVE_PROBE', a.profile, step, int(active.sum()), flush=True)
            if not bool(active.any()):
                break
    for i, case in enumerate(cases):
        if ended[i] is None:
            # The probe has an explicit 25-second observation budget even if the
            # environment's own timeout configuration is longer.
            ended[i] = dict(success=False, physical=False, timeout=False,
                            observation_budget_exhausted=True, seconds=25.)
        case.update(ended[i])
        case.update(max_joint_speed=max_speed[i].cpu().tolist(), max_applied_torque=max_torque[i].cpu().tolist())
    result = dict(source_profile=source_profile, execution_contract=export_contract(a.profile),
                  completed_updates=checkpoint['completed_updates'], initial_root=initial.cpu().tolist(),
                  initial_joints=start_joints.cpu().tolist(), cases=cases,
                  scope='Same frozen model; nominal dynamics, fixed forward command; first episode, 4m or 25s. Not training success certification.')
    (a.output/'results.json').write_text(json.dumps(result, indent=2))
    (a.output/'SUCCESS.json').write_text(json.dumps({'state': 'complete'}))
    env.close()
finally:
    app.close()
