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
    cases=[]
    variants=[
        ("baseline", {}), ("slew3", {"slew":3.}), ("slew6", {"slew":6.}),
        ("slew24", {"slew":24.}), ("delay0", {"delay":0}),
        ("delay5ms", {"delay":1}), ("ramp0p2", {"ramp":.2}),
        ("ramp0p5", {"ramp":.5}), ("ramp1", {"ramp":1.}),
        ("settle0p2", {"wait":.2}), ("settle0p5", {"wait":.5}),
        ("settle1", {"wait":1.}), ("settle2", {"wait":2.}),
        ("clear_initial_delay", {"clear_delay":True}),
        ("prime_initial_position", {"prime_position":True}),
    ]
    for pi,item in enumerate(selection):
        for name,extra in (variants if item['label']=='V12_student' else variants[:1]):
            cases.append(dict(id=len(cases), policy=pi, terrain='flat', level=0,
                name=name, mode='direct', direction='forward', velocity=[.6,0.,0.],
                teacher=item.get('teacher',False), slew=extra.get('slew',12.),
                delay=extra.get('delay',2), wait=extra.get('wait',0.),
                ramp=extra.get('ramp',0.), clear_delay=extra.get('clear_delay',False),
                prime_position=extra.get('prime_position',False)))
    input_hashes={x['path']:hashlib.sha256((a.root/x['path']).read_bytes()).hexdigest() for x in selection}
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
    settle=torch.tensor([c['wait'] for c in cases],device=raw.device)
    ramps=torch.tensor([c['ramp'] for c in cases],device=raw.device)
    velocities=torch.tensor([c['velocity'] for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        factor=torch.where(ramps>0,((elapsed-settle)/ramps.clamp_min(.001)).clamp(0.,1.),(elapsed>=settle).float())
        command.commands[:]=velocities*factor[:,None]
        command.commands[pit.active & pit.stop_on_platform]=0.
        pit.commands[:]=command.commands
        pit.held_speed[:]=command.commands[:,0]
    def resample(ids):
        command.time_left[ids]=1e9
        update()
    command._resample,command._update_command=resample,update
    raw.pit_valid_spawn=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device)
    # Record exact actuator inputs/outputs, including the wrapper reset before policy step zero.
    physics_records={name:[] for name in raw.scene['robot'].actuators}
    probe_clock={'step':-1}
    for group,actuator in raw.scene['robot'].actuators.items():
        delayed_slot={}
        buffer_compute=actuator.positions_delay_buffer.compute
        def capture_delay(data, original=buffer_compute, slot=delayed_slot):
            value=original(data)
            slot['delayed']=value.detach().clone()
            return value
        actuator.positions_delay_buffer.compute=capture_delay
        original_compute=actuator.compute
        def capture_actuator(control_action,joint_pos,joint_vel,original=original_compute,
                             slot=delayed_slot,actuator=actuator,group=group):
            incoming=control_action.joint_positions.detach().clone()
            q=joint_pos.detach().clone()
            qd=joint_vel.detach().clone()
            output=original(control_action,joint_pos,joint_vel)
            physics_records[group].append((probe_clock['step'],incoming,slot['delayed'].clone(),
                q,qd,actuator.computed_effort.detach().clone(),actuator.applied_effort.detach().clone()))
            return output
        actuator.compute=capture_actuator
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
    print('LOADED_DIAGNOSTIC',len(cases),flush=True)
    pit.learner.fill_(1)
    command.learner.fill_(1)
    raw.retain_terminal=True
    robot=raw.scene['robot']
    q0=robot.data.default_joint_pos[0].clone()
    origins=raw.scene.env_origins.clone()
    initial_q=robot.data.joint_pos.clone()
    initial_root=robot.data.root_state_w.clone()
    relative=initial_root.clone()
    relative[:,:3]-=origins
    torch.testing.assert_close(initial_q,initial_q[0:1].expand_as(initial_q),atol=1e-6,rtol=0.)
    torch.testing.assert_close(relative,relative[0:1].expand_as(relative),atol=2e-4,rtol=0.)
    assert bool((raw.episode_length_buf==0).all())
    joint_action=raw.action_manager.get_term('joint_pos')
    joint_action._max_delta=torch.tensor([c['slew']*raw.step_dt for c in cases],device=raw.device)[:,None]
    actuator_init={}
    clear_ids=torch.tensor([c['id'] for c in cases if c['clear_delay']],device=raw.device,dtype=torch.long)
    prime_ids=torch.tensor([c['id'] for c in cases if c['prime_position']],device=raw.device,dtype=torch.long)
    for group,actuator in robot.actuators.items():
        cb=actuator.positions_delay_buffer._circular_buffer
        actuator_init[group]=dict(joint_indices=list(actuator.joint_indices),
            target=robot.data.joint_pos_target[:,actuator.joint_indices].cpu().tolist(),
            buffer=cb.buffer.cpu().tolist(),pushes=cb._num_pushes.cpu().tolist(),
            stiffness=actuator.stiffness.cpu().tolist(),damping=actuator.damping.cpu().tolist())
        for name in ('positions_delay_buffer','velocities_delay_buffer','efforts_delay_buffer'):
            buffer=getattr(actuator,name)
            buffer.set_time_lag(torch.tensor([c['delay'] for c in cases],device=raw.device,dtype=torch.int))
            if len(clear_ids):
                buffer.reset(clear_ids)
        if len(prime_ids) and group!='wheel':
            # Diagnostic intervention only: replace pre-policy cached position requests
            # with the actual reset pose. Leave buffer age/pointer and all other cases unchanged.
            cb._buffer[:,prime_ids]=initial_q[prime_ids][:,actuator.joint_indices].unsqueeze(0)
    (a.output/'initial_buffers.json').write_text(json.dumps(actuator_init))
    frozen=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in policies]
    keys=('joint_pos','joint_vel','target','requested','command','velocity','height','raw_action',
          'torque','contact','phase_weight','phase_ready','rewards','active')
    traces={k:[] for k in keys}
    obs_trace={}
    active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    obs=env.get_observations()
    horizon=3 if a.smoke else 400
    with torch.inference_mode():
        for step in range(horizon):
            probe_clock['step']=step
            for key,value in obs.items():
                obs_trace.setdefault(key,[]).append(value.detach().cpu().numpy().copy())
            issued=command.commands.clone()
            actions=torch.cat([act(model,{k:v[indices[i]] for k,v in obs.items()},selection[i].get('teacher',False)) for i,model in enumerate(policies)],dim=0)
            actions[~active]=0.
            obs,reward,done,extras=env.step(actions)
            d=robot.data
            phase=phase_measure(raw)
            contact=torch.stack([sensor.data.force_matrix_w[:,0,0].norm(dim=-1) for sensor in pit.ground_contacts],dim=1)
            values=dict(joint_pos=d.joint_pos,joint_vel=d.joint_vel,target=d.joint_pos_target,
                requested=joint_action.requested_actions,command=issued,velocity=d.root_lin_vel_b,
                height=d.root_pos_w[:,2]-origins[:,2],raw_action=actions,torque=d.applied_torque,
                contact=contact,phase_weight=phase['weight'],phase_ready=phase['ready'],
                rewards=raw.reward_manager._step_reward,active=active)
            for key,value in values.items():
                traces[key].append(value.detach().cpu().numpy().copy())
            for ci in torch.nonzero(active & raw.reset_terminated,as_tuple=False).flatten().tolist():
                cases[ci]['physical_end_step']=step
                cases[ci]['physical_terms']=[name for name in raw.termination_manager.active_terms if raw.termination_manager.get_term(name)[ci]]
                active[ci]=False
            if step%100==0:
                print('STEP',step,flush=True)
    np.savez_compressed(a.output/'traces.npz',**{k:np.stack(v) for k,v in traces.items()})
    np.savez_compressed(a.output/'observations.npz',**{k:np.stack(v) for k,v in obs_trace.items()})
    physics={}
    for group,records in physics_records.items():
        physics[group+'_step']=np.array([r[0] for r in records])
        for col,name in enumerate(('incoming','delayed','q','qd','computed','applied'),1):
            physics[group+'_'+name]=torch.stack([r[col] for r in records]).cpu().numpy()
    np.savez_compressed(a.output/'physics.npz',**physics)
    for model,before in zip(policies,frozen):
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(hashlib.sha256((a.root/path).read_bytes()).hexdigest()==sha for path,sha in input_hashes.items())
    manifest=dict(cases=cases,history=metadata,joint_names=robot.joint_names,dt=raw.step_dt,
        default_q=q0.cpu().tolist(),request_joint_ids=joint_action.joint_indices,
        reward_terms=raw.reward_manager.active_terms,actuator_init=actuator_init,
        control_steps=horizon,optimizer_updates=0,hashes_unchanged=True,input_sha256=input_hashes,
        initial_q_matched=True,initial_root_matched=True,
        diagnostic_only=True,protocol='Frozen matched flat forward 0.6m/s; single-factor slew/delay/onset/initial-buffer interventions. No training or hardware.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (a.output/'FINALIZED.json').write_text(json.dumps(dict(control_steps=horizon,optimizer_updates=0,hashes_unchanged=True)))
    print('FINALIZED_ZERO_UPDATES',flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
