"""Audit shared student posture across all terrain families, retaining first episodes."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

p=argparse.ArgumentParser()
for key in ('root','code','checkpoint','output'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.code),str(a.root/'upstream/source/rsl_rl'),str(a.root/'upstream/source/robot_lab'),str(a.root/'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaaclab.app import AppLauncher  # noqa: E402
app=AppLauncher(headless=True,device='cuda:0').app
try:
    import numpy as np
    import torch
    from isaaclab.utils.warp import raycast_mesh
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
    from pit_posture import configure as configure_posture

    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    levels={'flat':(0,), 'stairs_up':(0,6),'stairs_down':(0,6),'slope_up':(0,5),
            'slope_down':(0,5),'wave':(0,6),'rough_slope':(0,5),'obstacles':(0,6),'pit':(17,)}
    cases=[dict(terrain=k,level=level,mode=m,repeat=r) for k,ls in levels.items() for level in ls
           for m in ('stand','move','start_stop') for r in range(2)]
    class FirstEpisode(Go2Env):
        retain_terminal=False
        def _reset_idx(self,ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    cfg=make_config(len(cases),a.root/'cache/usd',actuator_profile=saved['actuator_profile'])
    cfg.seed,cfg.log_dir=1942,str(a.output)
    cfg.episode_length_s=12.
    apply_command_curriculum_start(cfg.commands.base_velocity,saved['command_curriculum_start'])
    configure_base_height_penalty(cfg,saved['base_height_penalty'])
    configure_stair_speed(cfg,saved['stair_command_profile'])
    configure_wave_reset(cfg,saved['wave_reset_profile'])
    configure_mixed(cfg,slope_profile=saved['slope_profile'],pit_training_profile=saved['pit_training_profile'])
    configure_posture(cfg,saved.get('pit_posture_profile','off'))
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum,name,None)
    for name in vars(cfg.events):
        if name not in ('reset_base','reset_robot_joints','wave_surface'):
            setattr(cfg.events,name,None)
    for group in (cfg.observations.policy,cfg.observations.single_obs,cfg.observations.critic):
        group.enable_corruption=False
    cfg.events.reset_robot_joints.params['position_range']=(1.,1.)
    for key in ('pose_range','velocity_range'):
        for name in cfg.events.reset_base.params[key]:
            cfg.events.reset_base.params[key][name]=(0.,0.)
    cfg.rewards.lin_vel_z_l2.weight=0.
    raw=FirstEpisode(cfg=cfg)
    guard=ActuatorGuard(raw)
    raw.common_step_counter=saved['common_step_counter']
    terrain=raw.scene.terrain
    terrain.terrain_types[:]=torch.tensor([saved['terrain_column_names'].index(c['terrain']) for c in cases],device=raw.device)
    terrain.terrain_levels[:]=torch.tensor([c['level'] for c in cases],device=raw.device)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    raw.scene.env_origins[:]=terrain.env_origins
    command=raw.command_manager.get_term('base_velocity')
    command.refresh_membership()
    pit=command.pit
    pit.fixed_levels=terrain.terrain_levels.clone()
    mode=torch.tensor([('stand','move','start_stop').index(c['mode']) for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        moving=(mode==1)|((mode==2)&(elapsed>=2)&(elapsed<8))
        vx=.6*moving.float()
        vx=torch.where(pit.active & pit.stop_on_platform,0.,vx)
        command.commands.zero_()
        command.commands[:,0]=vx
        pit.commands.zero_()
        pit.commands[:,0]=vx
        pit.held_speed[:]=vx
    def resample(ids):
        command.time_left[ids]=1e9
        update()
    command._resample,command._update_command=resample,update
    raw.pit_valid_spawn=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device)
    env=RslRlVecEnvWrapper(raw,clip_actions=MoECTSRunnerCfg().clip_actions)
    agent=MoECTSRunnerCfg().to_dict()
    agent['robogauge']={'enabled':False}
    runner=OnPolicyRunnerCTS(env,agent,log_dir=str(a.output/'loader'),device=raw.device)
    policy=runner.alg.policy
    policy.load_state_dict(saved['model_state_dict'],strict=True)
    runner.eval_mode()
    robot=raw.scene['robot']
    origins=raw.scene.env_origins.clone()
    wheel=[robot.body_names.index(s+'_foot_Link') for s in ('fr','fl','rr','rl')]
    contact=raw.scene['contact_forces']
    contact_ids=[contact.body_names.index(s+'_foot_Link') for s in ('fr','fl','rr','rl')]
    scanner=raw.scene['height_scanner']
    mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    raw.retain_terminal=True
    active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    traces={k:[] for k in ('q','target','qvel','torque','xyz','quat','velocity','command','root_clearance','wheel_ground','wheel_force','active')}
    obs=env.get_observations()
    with torch.inference_mode():
        for step in range(600):
            actions=policy.act_inference(obs)
            actions[~active]=0.
            obs,_,_,_=env.step(actions)
            d=robot.data
            start=torch.cat((d.root_pos_w[:,None,:],d.body_pos_w[:,wheel]),dim=1).clone()
            start[:,:,2]+=10.
            direction=torch.zeros_like(start)
            direction[:,:,2]=-1.
            hits=raycast_mesh(start,direction,mesh)[0]
            values=dict(q=d.joint_pos,target=d.joint_pos_target,qvel=d.joint_vel,torque=d.applied_torque,
                xyz=d.root_pos_w-origins,quat=d.root_quat_w,velocity=d.root_lin_vel_b,
                command=command.commands,root_clearance=d.root_pos_w[:,2]-hits[:,0,2],
                wheel_ground=hits[:,1:,2]-origins[:,None,2],
                wheel_force=contact.data.net_forces_w[:,contact_ids].norm(dim=-1),active=active)
            for k,v in values.items():
                traces[k].append(v.clone())
            reached_limit=(d.root_pos_w[:,0]-origins[:,0])>=4.
            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|reached_limit)
            for i in ended.nonzero().flatten().tolist():
                cases[i].update(end_step=step,seconds=(step+1)*raw.step_dt,
                    physical=bool(guard.violation[i]|pit.body_contact[i]),
                    terms=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])]
                    +(['audit_4m_boundary'] if bool(reached_limit[i]) else []))
            active[ended]=False
            if step%200==0:
                print('PROGRESS',step,int(active.sum()),flush=True)
            if not bool(active.any()):
                break
    for c in cases:
        if 'end_step' not in c:
            c.update(end_step=step,seconds=(step+1)*raw.step_dt,physical=False,terms=['audit_budget'])
    np.savez_compressed(a.output/'traces.npz',**{k:torch.stack(v).cpu().numpy() for k,v in traces.items()})
    manifest=dict(checkpoint=str(a.checkpoint),completed_updates=saved['completed_updates'],cases=cases,
        joints=robot.joint_names,default_q=robot.data.default_joint_pos[0].cpu().tolist(),
        protocol='96 first student mean-action episodes; nominal physics, no sensor noise/push; normal root offsets and joint position scale fixed, stairs retain four-axis heading draws, pit retains pose/joint spawn draws, wave surface reset preserved; stand 0, move +0.6, start_stop 0 for 2s then +0.6 until 8s then 0; pit external stop remains; repeat differences include actuator delay; existing route exit truncation or positive world-x 4m audit boundary prevents leaving the tile')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    # Close artifacts before exiting; normal Isaac shutdown has hung after prior probes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
