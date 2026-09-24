"""Matched stair first episodes before/after stronger posture rewards, plus 1.2 m pit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

p=argparse.ArgumentParser()
for key in ('root','output'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())
selection=json.loads((a.root/'posture_history_fine_selection.json').read_text())
sys.path[:0]=[receipt['code'],str(a.root/'upstream/source/rsl_rl'),
              str(a.root/'upstream/source/robot_lab'),str(a.root/'thunder/source/doso_train')]
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
    from mixed_task import configure_mixed, mixed_reset_root
    from pit_posture import configure as configure_pit_posture
    from surface_posture import configure as configure_surface_posture
    from stance_strength import configure as configure_stance_strength

    torch.set_num_threads(2)
    latest=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=False)
    cases=[dict(policy=index,terrain='flat',level=0,mode=mode,speed=.6,settle_s=0.,id=index*4+j)
           for index in range(len(selection)) for j,mode in enumerate(('stand','direct','settle','ramp'))]
    class FirstEpisode(Go2Env):
        retain_terminal=False
        def _reset_idx(self,ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    def reset_start(env,env_ids,pose_range,velocity_range,asset_cfg):
        mixed_reset_root(env,env_ids,pose_range,velocity_range,asset_cfg)
        ids=env_ids[~env.command_manager.get_term('base_velocity').pit.active[env_ids]]
        robot=env.scene['robot']
        state=robot.data.root_state_w[ids].clone()
        state[:,3:7]=0.
        state[:,3]=1.
        robot.write_root_state_to_sim(state,env_ids=ids)
        env.stair_axis[ids]=0

    cfg=make_config(len(cases),a.root/'cache/usd',actuator_profile=latest['actuator_profile'])
    cfg.seed,cfg.log_dir=123,str(a.output)
    cfg.episode_length_s=8.
    apply_command_curriculum_start(cfg.commands.base_velocity,latest['command_curriculum_start'])
    configure_base_height_penalty(cfg,latest['base_height_penalty'])
    configure_stair_speed(cfg,latest['stair_command_profile'])
    configure_wave_reset(cfg,latest['wave_reset_profile'])
    configure_mixed(cfg,slope_profile=latest['slope_profile'],pit_training_profile=latest['pit_training_profile'])
    configure_pit_posture(cfg,latest['pit_posture_profile'])
    configure_surface_posture(cfg,latest['surface_posture_profile'])
    configure_stance_strength(cfg,latest['stance_strength_profile'])
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum,name,None)
    for name in vars(cfg.events):
        if name not in ('reset_base','reset_robot_joints'):
            setattr(cfg.events,name,None)
    for group in (cfg.observations.policy,cfg.observations.single_obs,cfg.observations.critic):
        group.enable_corruption=False
    cfg.events.reset_base.func=reset_start
    cfg.events.reset_robot_joints.params.update(position_range=(1.,1.),velocity_range=(0.,0.))
    for key in ('pose_range','velocity_range'):
        for name in cfg.events.reset_base.params[key]:
            cfg.events.reset_base.params[key][name]=(0.,0.)
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay=actuator.max_delay=2
    cfg.rewards.lin_vel_z_l2.weight=0.
    urdf=Path(cfg.scene.robot.spawn.asset_path)
    (a.output/'runtime.urdf').write_bytes(urdf.read_bytes())
    raw=FirstEpisode(cfg=cfg)
    guard=ActuatorGuard(raw)
    raw.common_step_counter=latest['common_step_counter']
    terrain=raw.scene.terrain
    terrain.terrain_types[:]=torch.tensor([latest['terrain_column_names'].index(c['terrain']) for c in cases],device=raw.device)
    terrain.terrain_levels[:]=torch.tensor([c['level'] for c in cases],device=raw.device)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    raw.scene.env_origins[:]=terrain.env_origins
    command=raw.command_manager.get_term('base_velocity')
    command.refresh_membership()
    pit=command.pit
    pit.fixed_levels=terrain.terrain_levels.clone()
    modes=torch.tensor([('stand','direct','settle','ramp').index(c['mode']) for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        vx=torch.where(modes==0,0.,torch.where(modes==2,.6*(elapsed>=2),
                      torch.where(modes==3,.6*elapsed.clamp(0.,1.),.6)))
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
    policy.load_state_dict(latest['model_state_dict'],strict=True)
    runner.eval_mode()
    policies=[]
    metadata=[]
    for index,item in enumerate(selection):
        state=torch.load(a.root/item['path'],map_location='cpu',weights_only=False)
        model=copy.deepcopy(policy)
        model.load_state_dict(state['model_state_dict'],strict=True)
        model.eval()
        policies.append(model)
        fields={k:v for k,v in state.items() if isinstance(v,(str,int,float,bool))}
        metadata.append(dict(**item,fields=fields))
        del state
    print('LOADED',len(policies),flush=True)
    pit.learner.fill_(1)
    command.learner.fill_(1)
    raw.retain_terminal=True
    robot=raw.scene['robot']
    origins=raw.scene.env_origins.clone()
    initial=robot.data.root_pos_w.clone()
    active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    traces={k:[] for k in ('body_pos','body_quat','joint_pos','joint_vel','target','torque','command','velocity','root_clearance','climb','stable','active')}
    scanner=raw.scene['height_scanner']
    mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    obs=env.get_observations()
    with torch.inference_mode():
        for step in range(400):
            issued=command.commands.clone()
            actions=torch.cat([model.act_inference({k:v[i*4:(i+1)*4] for k,v in obs.items()}) for i,model in enumerate(policies)],dim=0)
            actions[~active]=0.
            obs,_,_,_=env.step(actions)
            d=robot.data
            start=d.root_pos_w[:,None,:].clone()
            start[:,:,2]+=10.
            direction=torch.zeros_like(start)
            direction[:,:,2]=-1.
            hits=raycast_mesh(start,direction,mesh)[0]
            values=dict(body_pos=d.body_pos_w-origins[:,None,:],body_quat=d.body_quat_w,
                        joint_pos=d.joint_pos,joint_vel=d.joint_vel,target=d.joint_pos_target,
                        torque=d.applied_torque,command=issued,velocity=d.root_lin_vel_b,
                        root_clearance=d.root_pos_w[:,2]-hits[:,0,2],
                        climb=pit.climb_seen,stable=pit.success,active=active)
            for k,v in values.items():
                traces[k].append(v.clone())
            dx=d.root_pos_w[:,0]-initial[:,0]
            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|((~pit.active)&(dx>=4.)))
            for i in ended.nonzero().flatten().tolist():
                physical=bool(guard.violation[i]|pit.body_contact[i])
                terms=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])]
                cases[i].update(end_step=step,seconds=(step+1)*raw.step_dt,physical=physical,terms=terms,
                                passed_4m=bool(dx[i]>=4. and not physical),dx_m=float(dx[i]),
                                climb=bool(pit.climb_seen[i]&pit.platform_clear[i]&~pit.physical[i]&~pit.outside[i]),
                                stable=bool(pit.success[i]),checkpoint=selection[cases[i]['policy']]['path'])
            active[ended]=False
            if step%200==0:
                print('PROGRESS',step,int(active.sum()),flush=True)
            if not bool(active.any()):
                break
    for c in cases:
        if 'end_step' not in c:
            c.update(end_step=step,seconds=(step+1)*raw.step_dt,physical=False,terms=['budget'],passed_4m=False)
    arrays={k:torch.stack(v).cpu().numpy() for k,v in traces.items()}
    for c in cases:
        n=c['end_step']+1
        c['max_side_m']=float(abs(arrays['body_pos'][:n,c['id'],robot.body_names.index('base_link'),1]).max())
        c['straight_4m']=c['passed_4m'] and c['max_side_m']<=.5
    np.savez_compressed(a.output/'traces.npz',**arrays)
    manifest=dict(checkpoint=receipt['checkpoint'],completed_updates=latest['completed_updates'],
                  history=metadata,
                  cases=cases,joint_names=robot.joint_names,body_names=robot.body_names,dt=raw.step_dt,
                  default_q=robot.data.default_joint_pos[0].cpu().tolist(),
                  urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  protocol='Historical weights evaluated in identical current nominal physics; flat L0, zero initial velocities/default joints, yaw0, no noise/push, fixed10ms actuator delay; four first episodes per checkpoint: stand, immediate0.6, stand2s then0.6, ramp0to0.6 over1s; current guards;8s or4m boundary; not historical deployment reproduction')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',json.dumps(cases),flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    # All traces are persisted; prior Isaac shutdowns hung after completed jobs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
