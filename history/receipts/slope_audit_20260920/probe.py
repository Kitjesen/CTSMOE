"""Frozen-policy slope diagnosis. Writes traces only; does not alter training."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

p = argparse.ArgumentParser()
for name in ('root', 'code', 'checkpoint', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--mode', choices=('training', 'controlled'), required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(a.code), str(a.root/'upstream/source/rsl_rl'),
               str(a.root/'upstream/source/robot_lab'), str(a.root/'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import numpy as np
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from actuator_guard import ActuatorGuard
    from train import apply_command_curriculum_start
    from training_profiles import configure_base_height_penalty
    from stair_speed import configure_stair_speed
    from wave_reset import configure_wave_reset
    from mixed_task import configure_mixed

    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    controlled = a.mode == 'controlled'
    cases = ([dict(level=l, mu=mu, heading=h, push=k, repeat=i)
              for l in (5,6,7,9) for mu in (.5,1.2) for h in (False,True)
              for k in (False,True) for i in range(3)] if controlled else
             [dict(level=l, mu=None, heading=False, push=True, repeat=i)
              for l in (5,6,7,9) for i in range(8)])
    class FirstEpisode(Go2Env):
        retain_terminal = False
        def _reset_idx(self, ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)

    cfg = make_config(len(cases), a.root/'cache/usd', actuator_profile=saved['actuator_profile'])
    cfg.seed, cfg.log_dir = 9127, str(a.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, saved['command_curriculum_start'])
    configure_base_height_penalty(cfg, saved['base_height_penalty'])
    configure_stair_speed(cfg, saved['stair_command_profile'])
    configure_wave_reset(cfg, saved['wave_reset_profile'])
    configure_mixed(cfg, slope_profile=saved['slope_profile'], pit_training_profile=saved['pit_training_profile'])
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    cfg.rewards.lin_vel_z_l2.weight = 0
    cfg.events.reset_base.params['pose_range']['yaw'] = (0.,0.)
    push_original = cfg.events.randomize_push_robot.func
    push_log = []
    rng = torch.Generator().manual_seed(117)
    kicks = (torch.rand((8,3,6), generator=rng)*2-1)*torch.tensor([.4,.4,0.,.6,.6,.6])
    if controlled:
        for name, term in vars(cfg.events).items():
            if name not in ('reset_base', 'reset_robot_joints', 'randomize_push_robot'):
                setattr(cfg.events, name, None)
        cfg.events.reset_robot_joints.params['position_range'] = (1.,1.)
        for name in cfg.events.reset_base.params['pose_range']:
            cfg.events.reset_base.params['pose_range'][name] = (0.,0.)
        for name in cfg.events.reset_base.params['velocity_range']:
            cfg.events.reset_base.params['velocity_range'][name] = (0.,0.)
        reset_original = cfg.events.reset_base.func
        def reset_pose(env, env_ids, pose_range, velocity_range, asset_cfg):
            reset_original(env, env_ids, pose_range, velocity_range, asset_cfg)
            pose = env.scene['robot'].data.root_state_w[env_ids,:7].clone()
            pose[:,0] += torch.tensor([-.1 + .1*cases[i]['repeat'] for i in env_ids.tolist()], device=env.device)
            env.scene['robot'].write_root_pose_to_sim(pose, env_ids=env_ids)
        cfg.events.reset_base.func = reset_pose
        for act in cfg.scene.robot.actuators.values():
            act.min_delay = act.max_delay = 2
        for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
            group.enable_corruption = False

    def traced_push(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg('robot')):
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        if controlled:
            ids = ids[torch.tensor([c['push'] for c in cases], device=env.device)[ids]]
        before = env.scene['robot'].data.root_vel_w[ids].clone()
        if controlled:
            tick = min(7, max(0, int(env.episode_length_buf.max())//200-1))
            vel = before.clone()
            vel += kicks[tick, torch.tensor([cases[i]['repeat'] for i in ids.tolist()])].to(env.device)
            env.scene['robot'].write_root_velocity_to_sim(vel, env_ids=ids)
        else:
            push_original(env, ids, velocity_range, asset_cfg)
        push_log.append(dict(step=int(env.episode_length_buf.max()), ids=ids.cpu().tolist(),
                             before=before.cpu().tolist(), after=env.scene['robot'].data.root_vel_w[ids].cpu().tolist()))
    cfg.events.randomize_push_robot.func = traced_push
    raw = FirstEpisode(cfg=cfg)
    guard = ActuatorGuard(raw)
    raw.common_step_counter = saved['common_step_counter']
    terrain = raw.scene.terrain
    terrain.terrain_types.fill_(saved['terrain_column_names'].index('slope_up'))
    terrain.terrain_levels[:] = torch.tensor([c['level'] for c in cases], device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    raw.scene.env_origins[:] = terrain.env_origins
    command = raw.command_manager.get_term('base_velocity')
    command.refresh_membership()
    heading = torch.tensor([c['heading'] for c in cases], device=raw.device)
    def update():
        command.commands[:] = 0
        command.commands[:,0] = .6
        yaw = wrap_to_pi(euler_xyz_from_quat(raw.scene['robot'].data.root_quat_w)[2])
        command.commands[:,2] = torch.where(heading, (-1.5*yaw).clamp(-1.,1.), 0.)
    def resample(ids):
        update()
        command.time_left[ids] = 1e9
    command._resample, command._update_command = resample, update
    raw.pit_valid_spawn = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
    agent = MoECTSRunnerCfg().to_dict()
    agent['robogauge'] = {'enabled':False}
    runner = OnPolicyRunnerCTS(env, agent, log_dir=str(a.output/'loader'), device=raw.device)
    policy = runner.alg.policy
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    runner.eval_mode()
    robot = raw.scene['robot']
    materials = robot.root_physx_view.get_material_properties()
    if controlled:
        mu = torch.tensor([c['mu'] for c in cases])
        materials[:,:,0] = mu[:,None]
        materials[:,:,1] = mu[:,None]
        materials[:,:,2] = 0.
        robot.root_physx_view.set_material_properties(materials, torch.arange(raw.num_envs))
    wheel_b = [robot.body_names.index(s+'_foot_Link') for s in ('fr','fl','rr','rl')]
    wheel_j = [robot.joint_names.index(s+'_foot_joint') for s in ('fr','fl','rr','rl')]
    sensor = raw.scene['contact_forces']
    wheel_s = [sensor.body_names.index(s+'_foot_Link') for s in ('fr','fl','rr','rl')]
    base_s = sensor.body_names.index('base_link')
    scanner = raw.scene['height_scanner']
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    origins = raw.scene.env_origins.clone()
    raw.retain_terminal = True
    active = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    peaks = torch.zeros_like(robot.data.joint_vel)
    torque_peaks = torch.zeros_like(peaks)
    clips = torch.zeros_like(peaks)
    physics_trace = []
    measure_original = guard.measure
    def measure():
        measure_original()
        peaks[:] = torch.maximum(peaks, robot.data.joint_vel.abs())
        torque_peaks[:] = torch.maximum(torque_peaks, robot.data.applied_torque.abs())
        clips[:] += ((robot.data.computed_torque-robot.data.applied_torque).abs() > .1).float()
        physics_trace.append(torch.cat((robot.data.joint_vel, robot.data.applied_torque,
                                        robot.data.computed_torque),dim=1).clone())
    guard.measure = measure
    traces = []
    obs = env.get_observations()
    results = [None]*raw.num_envs
    with torch.inference_mode():
        for step in range(1250):
            peaks.zero_(); torque_peaks.zero_(); clips.zero_()
            actions = policy.act_inference(obs)
            actions[~active] = 0
            obs,_,_,_ = env.step(actions)
            d = robot.data
            xyz = d.root_pos_w-origins
            rpy = torch.stack(euler_xyz_from_quat(d.root_quat_w),dim=-1)
            rpy = wrap_to_pi(rpy)
            starts = d.body_link_pos_w[:,wheel_b].clone()
            starts[:,:,2] += 3.
            directions = torch.zeros_like(starts); directions[:,:,2] = -1
            hits,_,normals,_ = raycast_mesh(starts, directions, mesh, return_normal=True)
            normals *= torch.where(normals[:,:,2:3] < 0, -1., 1.)
            patch_velocity = d.body_link_lin_vel_w[:,wheel_b] + torch.cross(d.body_ang_vel_w[:,wheel_b], -.093*normals, dim=-1)
            tangent = patch_velocity-(patch_velocity*normals).sum(-1,keepdim=True)*normals
            slip = tangent.norm(dim=-1)
            forces = sensor.data.net_forces_w[:,wheel_s].norm(dim=-1)
            base_force = sensor.data.net_forces_w[:,base_s].norm(dim=-1)
            # Slip is a geometric contact-point proxy, interpreted only with contact forces >5 N.
            trace = torch.cat((xyz, rpy, d.root_lin_vel_b, d.root_lin_vel_w, command.commands,
                               forces, slip, base_force[:,None], d.joint_pos, d.joint_pos_target,
                               d.joint_vel, d.applied_torque, d.computed_torque,
                               peaks, torque_peaks, clips/4., active[:,None]),dim=1)
            traces.append(trace.clone())
            physical = raw.termination_manager.terminated
            passed = (xyz[:,0] >= 4.) & ~physical
            ended = active & (physical | passed | raw.termination_manager.time_outs)
            for i in ended.nonzero().flatten().tolist():
                results[i] = dict(**cases[i], step=step, seconds=(step+1)*raw.step_dt,
                                 passed=bool(passed[i]), physical=bool(physical[i]),
                                 xyz=xyz[i].cpu().tolist(),
                                 terms=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])],
                                 causes={k:bool(v[i]) for k,v in guard.violation_causes.items()})
            active[ended] = False
            if step % 250 == 0:
                print('PROGRESS',step,int(active.sum()),flush=True)
            if not bool(active.any()):
                break
    assert all(x is not None for x in results)
    fields = [('xyz',3),('rpy',3),('vel_b',3),('vel_w',3),('command',3),('wheel_force',4),('slip_proxy',4),('base_force',1)]
    fields += [(name,len(robot.joint_names)) for name in ('q','target','qvel','torque','computed','peak_qvel','peak_torque','clip_fraction')]
    fields += [('active',1)]
    data = dict(checkpoint=str(a.checkpoint),completed_updates=saved['completed_updates'],mode=a.mode,
                cases=results, fields=fields,joints=robot.joint_names,wheel_joints=wheel_j,pushes=push_log,
                material_properties=materials.cpu().tolist(),
                ground_material=str(cfg.scene.terrain.physics_material),
                protocol='frozen student mean action; vx=.6; first episode; controlled mode nominal parameters, fixed 10ms actuator delay, no observation noise; training mode original randomization and observation noise; no action exploration noise')
    (a.output/'results.json').write_text(json.dumps(data,indent=2))
    np.savez_compressed(a.output/'traces.npz',control=torch.stack(traces).cpu().numpy(),physics=torch.stack(physics_trace).cpu().numpy())
    print('DONE',flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    # Isaac's normal shutdown callback has hung in prior diagnostic runs; all outputs are closed above.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
