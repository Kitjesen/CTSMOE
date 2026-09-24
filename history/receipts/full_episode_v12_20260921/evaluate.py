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
for key in ('root','output','selection'):
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--code',type=Path,required=True)
p.add_argument('--flat25',action='store_true',help='25-second all-direction trials on enlarged flat geometry')
p.add_argument('--smoke',action='store_true',help='Three physics control steps; not a behavior evaluation')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())
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
    commands=[('forward',[.6,0.,0.]),('reverse',[-.6,0.,0.]),
              ('left',[0.,.4,0.]),('right',[0.,-.4,0.]),
              ('yaw_left',[0.,0.,.6]),('yaw_right',[0.,0.,-.6]),
              ('diagonal',[.4,.3,0.]),('diagonal_back',[-.4,-.3,0.]),
              ('forward_turn',[.4,0.,.4]),('side_turn',[0.,.3,-.4])]
    cases=[dict(policy=index,terrain='flat',level=0,mode=mode,direction=name,velocity=velocity,
                teacher=item.get('teacher',False)) for index,item in enumerate(selection)
           for mode in ('direct','settle') for name,velocity in commands]
    for index,item in enumerate(selection if not a.flat25 else []):
        cases += [dict(policy=index,terrain=terrain,level=level,mode='direct',
                       direction='forward',velocity=[speed,0.,0.],teacher=item.get('teacher',False))
                  for terrain in ('stairs_up','stairs_down','slope_up','slope_down',
                                  'wave','rough_slope','obstacles')
                  for level in (6,9) for speed in (.3,.6,1.2)]
        cases += [dict(policy=index,terrain='pit',level=level,mode='direct',
                       direction='forward',velocity=[speed,0.,0.],teacher=item.get('teacher',False))
                  for level in (0,2,4,6,9,15,21) for speed in (.3,.6,.9)]
    cases.sort(key=lambda c:c['policy'])
    assert len(cases)==(20 if a.flat25 else 83)*len(selection)
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
    assert latest['startup_repair_profile'] in ('omni_start_v9', 'omni_start_v10', 'omni_full_episode_v1')
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
    if a.flat25:
        # Keep the named terrain/command schema, but make every generated patch
        # a level 80 m square. No old obstacle protrudes into the test plane.
        from flat25_geometry import flat_patch
        from isaaclab.terrains import TerrainGenerator
        generator = cfg.scene.terrain.terrain_generator
        # The mixed generator intentionally measures real 45-degree slope meshes.
        # This separate flat-only protocol uses the standard generator instead.
        generator.class_type = TerrainGenerator
        generator.size = (80., 80.)
        generator.num_rows = 1
        generator.use_cache = False
        cfg.scene.terrain.max_init_terrain_level = 0
        for terrain_cfg in generator.sub_terrains.values():
            terrain_cfg.function = flat_patch
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
    indices=[torch.tensor([c['id'] for c in cases if c['policy']==k],device=raw.device)
             for k in range(len(policies))]
    def act(model,obs,teacher):
        if not teacher:
            return model.act_inference(obs)
        critic=model.critic_obs_normalizer(model.get_critic_obs(obs))
        single=model.single_obs_normalizer(obs['single_obs'])
        return model.actor(torch.cat((model.teacher_encoder(critic),single),dim=-1))
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
    # RslRlVecEnvWrapper.__init__ resets after the terrain and scripted command
    # overrides above. Assert that runtime contract before the first action.
    expected_initial_commands = velocities*(settle == 0.)[:,None]
    torch.testing.assert_close(command.commands, expected_initial_commands, atol=1e-6, rtol=0.)
    assert bool((raw.episode_length_buf == 0).all())
    flat_initial = torch.tensor([c['terrain']=='flat' for c in cases], device=raw.device)
    assert bool((relative[flat_initial,:2].abs() < .001).all())
    for ids in indices[1:]:
        torch.testing.assert_close(relative[ids],relative[indices[0]],atol=2e-4,rtol=0)
        torch.testing.assert_close(initial_q[ids],initial_q[indices[0]],atol=1e-6,rtol=0)
    print('MATCHED_INITIAL_STATES',len(cases),flush=True)
    active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    traces={k:[] for k in ('body_pos','body_quat','joint_pos','joint_vel','target','torque','command','velocity','root_clearance','climb','stable','active','angular_velocity','raw_action','requested','torque_peak','speed_peak','hard_excess','saturation','gravity','base_contact','ground_relief','ground_scan_valid','wheel_clearance','wheel_ground_force','pit_front_gap','platform_stop','phase_weight','phase_ready','phase_stable_s')}
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
    with torch.inference_mode():
        for step in range(3 if a.smoke else 1250):
            issued=command.commands.clone()
            actions=torch.cat([act(model,{k:v[indices[i]] for k,v in obs.items()},selection[i].get('teacher',False)) for i,model in enumerate(policies)],dim=0)
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
            values=dict(phase_weight=phase['weight'],phase_ready=phase['ready'],phase_stable_s=phase['stable_s'],ground_relief=relief,ground_scan_valid=valid,
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
            budget=flat & ((step+1)*raw.step_dt>=(25. if a.flat25 else 6.))
            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|budget|((~pit.active)&(dx>=4.)& (not a.flat25)))
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
    manifest=dict(protocol_id='flat25_80m' if a.flat25 else 'matched_nine_terrain_83',checkpoint=str(a.root/selection[-1]['path']),completed_updates=latest['completed_updates'],
                  history=metadata,
                  cases=cases,joint_names=robot.joint_names,body_names=robot.body_names,dt=raw.step_dt,
                  default_q=robot.data.default_joint_pos[0].cpu().tolist(),
                  request_joint_ids=raw.action_manager.get_term('joint_pos').joint_indices,
                  actuator_limits=guard.limits,initial_state_matched=True,smoke_only=a.smoke,
                  initial_scripted_commands_verified=True,initial_flat_origins_verified=True,
                  initial_episode_age_zero_verified=True,
                  max_episode_length_steps=raw.max_episode_length,
                  phase_telemetry='geometry/support measurements only; stage acceptance not inferred from joint pose',
                  slope_geometry=getattr(cfg.scene.terrain.terrain_generator,'slope_geometry_receipts',[]),
                  urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  protocol='V10 flat outlier repair comparison, V8 landing geometry (shorter stair flight, outer plateau); phase telemetry enabled. Nine terrains, matched default joint/root states verified across policies. No noise or pushes; fixed10ms delay. 83 cases per policy: 20 all-direction flat starts (6s budget), seven other terrain types at L6/L9 and vx0.3/0.6/1.2, pit0.15/0.25/0.35/0.45/0.6/0.9/1.2m at vx0.3/0.6/0.9 (25s budget). No heading feedback. Pit external stop retained. 50Hz trajectories plus per-step 200Hz torque/speed/hard-limit peaks. Fixed cases, not success probabilities over random trials.')
    if a.flat25:
        manifest['protocol'] = ('25-second all-direction flat evaluation on level 80 m square patches. '
            '20 fixed cases per policy, direct or 2-second settled start; no 4m early completion. '
            'No noise/pushes; fixed10ms delay. Physical termination remains active; '
            'all terrain geometry is flat in this protocol; nine-terrain validation is separate.')
        manifest['flat_patch_size_m'] = [80., 80.]
        manifest['flat_boundary_crossed'] = [c['id'] for c in cases if np.abs(
            arrays['body_pos'][:c['end_step']+1,c['id'],robot.body_names.index('base_link'),:2]).max()>38.]
        if manifest['flat_boundary_crossed']:
            raise RuntimeError('A full-duration case left the validated flat evaluation area')
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
