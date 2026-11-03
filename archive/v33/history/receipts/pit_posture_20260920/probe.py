"""Record all three first student episodes on the actual 1 m pit geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

p = argparse.ArgumentParser()
for key in ('root', 'code', 'checkpoint', 'output'):
    p.add_argument('--'+key, type=Path, required=True)
p.add_argument('--posture-profile', choices=('off','phase_stance_v1'), default='phase_stance_v1')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(a.code), str(a.root/'upstream/source/rsl_rl'),
               str(a.root/'upstream/source/robot_lab'), str(a.root/'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher  # noqa: E402
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import numpy as np
    import torch
    import trimesh
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
    from mixed_task import configure_mixed, mixed_pit_terrain

    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cases = [dict(id=i, speed=v, level=17, height_m=1.) for i,v in enumerate((.3,.6,.9))]
    class FirstEpisode(Go2Env):
        retain_terminal = False
        def _reset_idx(self, ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    cfg = make_config(len(cases), a.root/'cache/usd', actuator_profile=saved['actuator_profile'])
    cfg.seed, cfg.log_dir = 123, str(a.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, saved['command_curriculum_start'])
    configure_base_height_penalty(cfg, saved['base_height_penalty'])
    configure_stair_speed(cfg, saved['stair_command_profile'])
    configure_wave_reset(cfg, saved['wave_reset_profile'])
    configure_mixed(cfg, slope_profile=saved['slope_profile'], pit_training_profile=saved['pit_training_profile'])
    from pit_posture import configure as configure_posture, measure as posture_measure
    configure_posture(cfg,a.posture_profile)
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    for name in vars(cfg.events):
        if name not in ('reset_base', 'reset_robot_joints'):
            setattr(cfg.events, name, None)
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        group.enable_corruption = False
    cfg.events.reset_robot_joints.params['position_range'] = (1.,1.)
    for values in ('pose_range', 'velocity_range'):
        for name in cfg.events.reset_base.params[values]:
            cfg.events.reset_base.params[values][name] = (0.,0.)
    cfg.rewards.lin_vel_z_l2.weight = 0.
    from types import SimpleNamespace
    meshes, origin = mixed_pit_terrain(17.5/22, SimpleNamespace(size=cfg.scene.terrain.terrain_generator.size))
    mesh = trimesh.util.concatenate(meshes)
    mesh.apply_translation(-origin)
    mesh.export(a.output/'terrain.obj')
    urdf = Path(cfg.scene.robot.spawn.asset_path)
    (a.output/'runtime.urdf').write_bytes(urdf.read_bytes())
    raw = FirstEpisode(cfg=cfg)
    guard = ActuatorGuard(raw)
    raw.common_step_counter = saved['common_step_counter']
    terrain = raw.scene.terrain
    terrain.terrain_levels.fill_(17)
    terrain.terrain_types.fill_(20)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    raw.scene.env_origins[:] = terrain.env_origins
    command = raw.command_manager.get_term('base_velocity')
    command.refresh_membership()
    c = command.pit
    c.fixed_levels = terrain.terrain_levels.clone()
    speeds = torch.tensor([x['speed'] for x in cases], device=raw.device)
    def resample(ids):
        c.held_speed[ids] = speeds[ids]
        c.speed_bins[ids] = ((speeds[ids]-.2)/.2).long().clamp(0,3)
        c.commands[ids] = 0.
        c.commands[ids,0] = speeds[ids]
        c.time_left[ids] = 1e9
    c._resample = resample
    env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
    agent = MoECTSRunnerCfg().to_dict()
    agent['robogauge'] = {'enabled':False}
    runner = OnPolicyRunnerCTS(env, agent, log_dir=str(a.output/'loader'), device=raw.device)
    policy = runner.alg.policy
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    runner.eval_mode()
    c.learner.fill_(1)
    assert bool(c.active.all()) and bool((terrain.terrain_levels==17).all())
    assert bool(c.started_in_pit.all())
    raw.retain_terminal = True
    robot = raw.scene['robot']
    origins = raw.scene.env_origins.clone()
    active = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    traces = {k:[] for k in ('body_pos','body_quat','joint_pos','joint_vel','command','velocity','climb','stable','supported',
                            'joint_target','torque','posture_weight','posture_ok','height')}
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(1250):
            actions = policy.act_inference(obs)
            actions[~active] = 0.
            obs,_,_,_ = env.step(actions)
            d = robot.data
            values = dict(body_pos=d.body_pos_w-origins[:,None,:], body_quat=d.body_quat_w,
                          joint_pos=d.joint_pos, joint_vel=d.joint_vel, command=command.commands,
                          velocity=d.root_lin_vel_b, climb=c.climb_seen, stable=c.success, supported=c.supported,
                          joint_target=d.joint_pos_target,torque=d.applied_torque)
            state=posture_measure(c)
            if state is None:
                values.update(posture_weight=torch.zeros_like(c.hold),posture_ok=torch.zeros_like(c.success),height=d.root_pos_w[:,2]-origins[:,2])
            else:
                values.update(posture_weight=state['weight'],posture_ok=state['acceptable'],height=state['height_m'])
            for name,value in values.items():
                traces[name].append(value.clone())
            ended = active & (raw.termination_manager.terminated | raw.termination_manager.time_outs)
            for i in ended.nonzero().flatten().tolist():
                cases[i].update(end_step=step, seconds=(step+1)*raw.step_dt,
                    climb=bool(c.climb_seen[i] & c.platform_clear[i] & ~c.physical[i] & ~c.outside[i]),
                    stable=bool(c.success[i]), physical=bool(c.physical[i]),
                    terms=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])])
            active[ended] = False
            if step%250 == 0:
                print('PROGRESS',step,int(active.sum()),flush=True)
            if not bool(active.any()):
                break
    assert all('end_step' in case for case in cases)
    np.savez_compressed(a.output/'traces.npz', **{k:torch.stack(v).cpu().numpy() for k,v in traces.items()})
    manifest = dict(checkpoint=str(a.checkpoint), completed_updates=saved['completed_updates'],cases=cases,
        joint_names=robot.joint_names, body_names=robot.body_names, dt=raw.step_dt,
        default_joint_pos=robot.data.default_joint_pos[0].cpu().tolist(),posture_profile=a.posture_profile,
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        protocol='First episodes, frozen student mean action, nominal parameters, no observation noise/pushes, original actuator delay; external zero command once whole body clears platform; first terminal retained')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',json.dumps(cases),flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    # Isaac shutdown previously hung after completed probes; files are closed above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
