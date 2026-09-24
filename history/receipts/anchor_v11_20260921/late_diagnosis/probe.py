"""Matched flat first-episode diagnosis of the V11 late posture excursion."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

p=argparse.ArgumentParser()
for key in ('root','output','selection'):
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--code',type=Path,required=True)
p.add_argument('--smoke',action='store_true',help='Three physics control steps; not a behavior evaluation')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
selection=json.loads(a.selection.read_text())
sys.path[:0]=[str(a.code),str(a.root/'upstream/source/rsl_rl'),
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
    from isaaclab.envs.mdp.events import reset_joints_by_scale
    from pit_posture import configure as configure_pit_posture
    from surface_posture import configure as configure_surface_posture
    from stance_strength import configure as configure_stance_strength

    torch.set_num_threads(2)
    latest=torch.load(a.root/selection[-1]['path'],map_location='cpu',weights_only=False)
    labels=[item['label'] for item in selection]
    assert len(selection)==3, labels
    assert labels[:2]==['V6_43202','V9_44711'], labels
    assert labels[2].startswith('V11_'), labels
    commands=[('forward_0.6',[.6,0.,0.]),('reverse_0.6',[-.6,0.,0.]),
              ('left_0.4',[0.,.4,0.]),('yaw_0.6',[0.,0.,.6]),
              ('diagonal_forward',[.4,.3,0.]),('diagonal_reverse',[-.4,-.3,0.])]
    controls=[('v6_student',None,None),('v9_student',None,None),
              ('v11_student',None,None),('v11_teacher',None,None)]
    controls += [('v11_student','v6_student',t) for t in (1.,2.,2.8)]
    controls += [('v11_student','v11_teacher',t) for t in (1.,2.,2.8)]
    cases=[]
    for name,velocity in commands:
        for controller,target,t in controls:
            policy={'v6_student':0,'v9_student':1}.get(controller,2)
            cases.append(dict(policy=policy,terrain='flat',level=0,mode='direct',
                              direction=name,velocity=velocity,command=velocity,
                              controller_source=controller,intervention_target=target,
                              intervention_time=t,teacher=controller=='v11_teacher'))
    for target in ('v6_student','v11_teacher'):
        for t in (2.6,3.):
            cases.append(dict(policy=2,terrain='flat',level=0,mode='direct',
                              direction='forward_0.6',velocity=[.6,0.,0.],command=[.6,0.,0.],
                              controller_source='v11_student',intervention_target=target,
                              intervention_time=t,teacher=False))
    assert len(cases)==64
    for i,c in enumerate(cases):
        c['id']=i
    class FirstEpisode(Go2Env):
        retain_terminal=False
        def _reset_idx(self,ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    def reset_start(env,env_ids,pose_range,velocity_range,asset_cfg):
        robot=env.scene['robot']
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids],
                                      torch.zeros_like(robot.data.default_joint_pos[env_ids]),env_ids=env_ids)
        mixed_reset_root(env,env_ids,pose_range,velocity_range,asset_cfg)
        pit=env.command_manager.get_term('base_velocity').pit
        pit_ids=env_ids[pit.active[env_ids]]
        if len(pit_ids):
            pose=robot.data.default_root_state[pit_ids,:7].clone()
            origins=env.scene.env_origins[pit_ids]
            pose[:,:3]+=origins
            pose[:,3:]=0.
            pose[:,3]=1.
            points=pit.geometry.world_points(robot.data.default_joint_pos[pit_ids],pose)
            pose[:,2]+=origins[:,2]+.03-points[...,2].amin(1)
            assert bool((points[...,0].amax(1)<origins[:,0]+1.4).all())
            robot.write_root_pose_to_sim(pose,pit_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(pit_ids),6,device=env.device),pit_ids)
            env.pit_valid_spawn[pit_ids]=True
        ids=env_ids[~env.command_manager.get_term('base_velocity').pit.active[env_ids]]
        robot=env.scene['robot']
        state=robot.data.root_state_w[ids].clone()
        state[:,3:7]=0.
        state[:,3]=1.
        robot.write_root_state_to_sim(state,env_ids=ids)
        env.stair_axis[ids]=0

    cfg=make_config(len(cases),a.root/'cache/usd',actuator_profile=latest['actuator_profile'])
    cfg.seed,cfg.log_dir=123,str(a.output)
    cfg.episode_length_s=25.
    apply_command_curriculum_start(cfg.commands.base_velocity,latest['command_curriculum_start'])
    configure_base_height_penalty(cfg,latest['base_height_penalty'])
    configure_stair_speed(cfg,latest['stair_command_profile'])
    configure_wave_reset(cfg,latest['wave_reset_profile'])
    configure_mixed(cfg,slope_profile=latest['slope_profile'],pit_training_profile=latest['pit_training_profile'])
    configure_pit_posture(cfg,latest['pit_posture_profile'])
    configure_surface_posture(cfg,latest['surface_posture_profile'])
    configure_stance_strength(cfg,latest['stance_strength_profile'])
    from startup_repair import configure as configure_startup
    from phase_transition import measure as phase_measure
    assert latest['startup_repair_profile'] in ('omni_start_v9', 'omni_start_v10')
    configure_startup(cfg,latest['startup_repair_profile'])
    cfg.terminations.startup_window=None

    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum,name,None)
    for name in vars(cfg.events):
        if name not in ('reset_base','reset_robot_joints'):
            setattr(cfg.events,name,None)
    for group in (cfg.observations.policy,cfg.observations.single_obs,cfg.observations.critic):
        group.enable_corruption=False
    cfg.events.reset_base.func=reset_start
    cfg.events.reset_robot_joints.func=reset_joints_by_scale
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
    settle=torch.tensor([2. if c['mode']=='settle' else 0. for c in cases],device=raw.device)
    velocities=torch.tensor([c['velocity'] for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        command.commands[:]=velocities*(elapsed>=settle)[:,None]
        command.commands[pit.active & pit.stop_on_platform]=0.
        pit.commands[:]=command.commands
        pit.held_speed[:]=command.commands[:,0]
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
    controller_ids={'v6_student':0,'v9_student':1,'v11_student':2,'v11_teacher':3}
    switch_times=torch.tensor([c['intervention_time'] if c['intervention_time'] is not None else float('inf') for c in cases],device=raw.device)
    initial_controller=torch.tensor([controller_ids[c['controller_source']] for c in cases],device=raw.device)
    target_controller=torch.tensor([controller_ids.get(c['intervention_target'],-1) for c in cases],device=raw.device)
    def act(model,obs,teacher):
        if not teacher:
            return model.act_inference(obs)
        critic=model.critic_obs_normalizer(model.get_critic_obs(obs))
        single=model.single_obs_normalizer(obs['single_obs'])
        return model.actor(torch.cat((model.teacher_encoder(critic),single),dim=-1))
    def counterfactual(model,obs):
        single=model.single_obs_normalizer(obs['single_obs'])
        actor_obs=model.actor_obs_normalizer(model.get_actor_obs(obs))
        latent_s,weights=model.student_moe_encoder(actor_obs)
        critic=model.critic_obs_normalizer(model.get_critic_obs(obs))
        latent_t=model.teacher_encoder(critic)
        student=model.actor(torch.cat((latent_s,single),dim=-1))
        teacher=model.actor(torch.cat((latent_t,single),dim=-1))
        torch.testing.assert_close(student,model.act_inference(obs))
        torch.testing.assert_close(teacher,act(model,obs,True))
        return student,teacher,latent_s,latent_t,weights
    print('LOADED',len(policies),flush=True)
    pit.learner.fill_(1)
    command.learner.fill_(1)
    raw.retain_terminal=True
    robot=raw.scene['robot']
    origins=raw.scene.env_origins.clone()
    initial=robot.data.root_pos_w.clone()
    initial_q=robot.data.joint_pos.clone()
    initial_root=robot.data.root_state_w.clone()
    relative=initial_root.clone()
    relative[:,:3]-=origins
    torch.testing.assert_close(relative,relative[:1].expand_as(relative),atol=2e-4,rtol=0)
    torch.testing.assert_close(initial_q,initial_q[:1].expand_as(initial_q),atol=1e-6,rtol=0)
    print('MATCHED_INITIAL_STATES',len(cases),flush=True)
    active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    traces={k:[] for k in ('body_pos','body_quat','joint_pos','joint_vel','target','torque','command','velocity','root_clearance','climb','stable','active','angular_velocity','raw_action','requested','torque_peak','speed_peak','hard_excess','saturation','gravity','base_contact','ground_relief','ground_scan_valid','wheel_clearance','wheel_ground_force','pit_front_gap','platform_stop','phase_weight','phase_ready','phase_stable_s')}
    traces.update({k:[] for k in ('action_v6_student','action_v9_student','action_v9_teacher',
                                      'action_v11_student','action_v11_teacher','latent_v11_student',
                                      'latent_v11_teacher','latent_v11_delta','gate_v11',
                                      'actual_controller_id')})
    scanner=raw.scene['height_scanner']
    mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    step_stats={k:torch.zeros_like(robot.data.joint_pos) for k in
                ('torque_peak','speed_peak','hard_excess','saturation')}
    original_measure=guard.measure
    def measure_substep():
        original_measure()
        d=robot.data
        step_stats['torque_peak'].copy_(torch.maximum(step_stats['torque_peak'],d.applied_torque.abs()))
        step_stats['speed_peak'].copy_(torch.maximum(step_stats['speed_peak'],d.joint_vel.abs()))
        excess=torch.maximum(guard.hard[...,0]-d.joint_pos,d.joint_pos-guard.hard[...,1]).clamp_min(0)*guard.leg
        step_stats['hard_excess'].copy_(torch.maximum(step_stats['hard_excess'],excess))
        step_stats['saturation'].add_((d.computed_torque-d.applied_torque).abs().gt(.001)/raw.cfg.decimation)
    guard.measure=measure_substep
    flat=torch.tensor([c['terrain']=='flat' for c in cases],device=raw.device)
    obs=env.get_observations()
    obs_trace_names={key:f'obs_{key}' for key in obs.keys()}
    assert len(set(obs_trace_names.values()))==len(obs_trace_names)
    traces.update({name:[] for name in obs_trace_names.values()})
    with torch.inference_mode():
        for step in range(3 if a.smoke else int(round(6./raw.step_dt))):
            issued=command.commands.clone()
            before_obs={k:v.clone() for k,v in obs.items()}
            v6_student,_,_,_,_=counterfactual(policies[0],obs)
            v9_student,v9_teacher,_,_,_=counterfactual(policies[1],obs)
            v11_student,v11_teacher,z_s,z_t,gate=counterfactual(policies[2],obs)
            switched=step*raw.step_dt+1e-8>=switch_times
            actual_controller=torch.where(switched,target_controller,initial_controller)
            actions=v11_student.clone()
            candidates=(v6_student,v9_student,v11_student,v11_teacher)
            for controller_id,candidate in enumerate(candidates):
                mask=actual_controller==controller_id
                actions[mask]=candidate[mask]
            actions[~active]=0.
            for value in step_stats.values():
                value.zero_()
            obs,_,_,_=env.step(actions)
            d=robot.data
            start=torch.cat((d.root_pos_w[:,None,:],d.body_pos_w[:,pit.wheel_ids]),dim=1).clone()
            start[:,:,2]+=10.
            direction=torch.zeros_like(start)
            direction[:,:,2]=-1.
            hits=raycast_mesh(start,direction,mesh)[0]
            ray_z=scanner.data.ray_hits_w[...,2]
            valid=torch.isfinite(ray_z).all(-1)
            relief=torch.where(valid,ray_z.amax(-1)-ray_z.amin(-1),torch.full_like(ray_z[:,0],float('nan')))
            wheel_z=d.body_pos_w[:,pit.wheel_ids,2]
            ground_force=torch.stack([sensor.data.force_matrix_w[:,0,0].norm(dim=-1)
                                      for sensor in pit.ground_contacts],dim=1)
            front_gap=origins[:,0]+1.5-(d.body_pos_w[:,pit.wheel_ids,0].amax(-1)+.093)
            phase=phase_measure(raw)
            pre_obs={obs_trace_names[k]:v for k,v in before_obs.items()}
            values=dict(**pre_obs,action_v6_student=v6_student,action_v9_student=v9_student,
                        action_v9_teacher=v9_teacher,action_v11_student=v11_student,
                        action_v11_teacher=v11_teacher,latent_v11_student=z_s,latent_v11_teacher=z_t,
                        latent_v11_delta=z_s-z_t,gate_v11=gate,actual_controller_id=actual_controller,
                        phase_weight=phase['weight'],phase_ready=phase['ready'],phase_stable_s=phase['stable_s'],ground_relief=relief,ground_scan_valid=valid,
                        wheel_clearance=wheel_z-hits[:,1:,2]-.093,
                        wheel_ground_force=ground_force,
                        pit_front_gap=torch.where(pit.active,front_gap,torch.full_like(front_gap,float('nan'))),
                        platform_stop=pit.stop_on_platform,body_pos=d.body_pos_w-origins[:,None,:],body_quat=d.body_quat_w,
                        joint_pos=d.joint_pos,joint_vel=d.joint_vel,target=d.joint_pos_target,
                        torque=d.applied_torque,command=issued,velocity=d.root_lin_vel_b,
                        angular_velocity=d.root_ang_vel_b,raw_action=actions,
                        requested=raw.action_manager.get_term('joint_pos').requested_actions,
                        gravity=d.projected_gravity_b,
                        base_contact=pit.contact.data.net_forces_w_history[:,:,pit.base_contact_ids].norm(dim=-1).amax(dim=(1,2)),
                        **step_stats,
                        root_clearance=d.root_pos_w[:,2]-hits[:,0,2],
                        climb=pit.climb_seen,stable=pit.success,active=active)
            for k,v in values.items():
                traces[k].append(v.clone())
            dx=d.root_pos_w[:,0]-initial[:,0]
            budget=flat & ((step+1)*raw.step_dt>=6.)
            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|budget)
            for i in ended.nonzero().flatten().tolist():
                physical=bool(guard.violation[i]|pit.body_contact[i])
                terms=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])]
                cases[i].update(end_step=step,seconds=(step+1)*raw.step_dt,physical=physical,terms=terms,
                                passed_4m=bool(dx[i]>=4. and not physical),dx_m=float(dx[i]),
                                climb=bool(pit.climb_seen[i]&pit.platform_clear[i]&~pit.physical[i]&~pit.outside[i]),
                                stable=bool(pit.success[i]),checkpoint=selection[cases[i]['policy']]['path'],
                                violation_causes={k:bool(v[i]) for k,v in guard.violation_causes.items()})
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
    np.savez_compressed(a.output/'traces.npz',initial_q=initial_q.cpu().numpy(),initial_root=relative.cpu().numpy(),**arrays)
    manifest=dict(checkpoint=str(a.root/selection[-1]['path']),completed_updates=latest['completed_updates'],
                  history=metadata,
                  cases=cases,joint_names=robot.joint_names,body_names=robot.body_names,dt=raw.step_dt,
                  default_q=robot.data.default_joint_pos[0].cpu().tolist(),
                  request_joint_ids=raw.action_manager.get_term('joint_pos').joint_indices,
                  actuator_limits=guard.limits,initial_state_matched=True,smoke_only=a.smoke,
                  phase_telemetry='geometry/support measurements only; stage acceptance not inferred from joint pose',
                  slope_geometry=getattr(cfg.scene.terrain.terrain_generator,'slope_geometry_receipts',[]),
                  urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  controller_ids=controller_ids,observation_traces=obs_trace_names,
                  protocol='Evaluation-only late-failure diagnosis: six flat commands, 64 matched first episodes, 6s budget or first termination. Direct V6/V9/V11 students and V11 teacher; V11-student interventions to V6 student or same-model V11 teacher at 1.0/2.0/2.8s, plus forward-0.6 interventions at 2.6/3.0s. All policy observations and same-state counterfactual actions are pre-step; physical state, request and target are post-step. V11 student/teacher latents and confirmed MoE softmax weights are same-model diagnostics. Switches do not reset state. Fixed 10ms delay, no observation corruption or pushes. Matched initialization is asserted; pre-switch trajectory agreement is measured rather than assumed perfect.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',len(cases),flush=True)
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
