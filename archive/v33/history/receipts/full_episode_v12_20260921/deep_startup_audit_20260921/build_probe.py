"""Build a bounded diagnostic from the previously exercised fixed-case evaluator.

Only this diagnostic copy is changed. Production training and checkpoints stay frozen.
"""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
source = (HERE.parent / "evaluate.py").read_text(encoding="utf-8")
prefix = source[:source.index("    commands=[")]
setup = source[source.index("    class FirstEpisode"):source.index("    print('LOADED'")]
cases = '''    cases=[]
    variants=[
        ("baseline", {}), ("settle0p02", {"wait":.02}),
        ("settle0p04", {"wait":.04}), ("settle0p06", {"wait":.06}),
        ("settle0p1", {"wait":.1}), ("ramp0p1", {"ramp":.1}),
        ("ramp0p2", {"ramp":.2}), ("reset_targets", {"reset_targets":True}),
        ("reset_targets_ramp0p2", {"reset_targets":True,"ramp":.2}),
        ("reset_targets_slew6_ramp0p2", {"reset_targets":True,"ramp":.2,"slew":6.}),
        ("zero_past_command_history", {"zero_command_history":True}),
    ]
    for pi,item in enumerate(selection):
        for name,extra in (variants if item['label']=='V12_student' else variants[:1]):
            cases.append(dict(id=len(cases), policy=pi, terrain='flat', level=0,
                name=name, mode='direct', direction='forward', velocity=[.6,0.,0.],
                teacher=item.get('teacher',False), slew=extra.get('slew',12.),
                delay=extra.get('delay',2), wait=extra.get('wait',0.),
                ramp=extra.get('ramp',0.), clear_delay=extra.get('clear_delay',False),
                prime_position=extra.get('prime_position',False),
                reset_targets=extra.get('reset_targets',False),
                zero_command_history=extra.get('zero_command_history',False)))
    input_hashes={x['path']:hashlib.sha256((a.root/x['path']).read_bytes()).hexdigest() for x in selection}
'''
old = "    settle=torch.tensor([2. if c['mode']=='settle' else 0. for c in cases],device=raw.device)"
assert setup.count(old) == 1
setup = setup.replace(old, "    settle=torch.tensor([c['wait'] for c in cases],device=raw.device)\n    ramps=torch.tensor([c['ramp'] for c in cases],device=raw.device)")
old = "        command.commands[:]=velocities*(elapsed>=settle)[:,None]"
assert setup.count(old) == 1
setup = setup.replace(old, "        factor=torch.where(ramps>0,((elapsed-settle)/ramps.clamp_min(.001)).clamp(0.,1.),(elapsed>=settle).float())\n        command.commands[:]=velocities*factor[:,None]")
capture = '''    # Match reset target buffers to the actual reset pose in selected diagnostic cases.
    corrected_reset_mask=torch.tensor([c['reset_targets'] for c in cases],device=raw.device,dtype=torch.bool)
    original_reset=raw._reset_idx
    def reset_and_sync_targets(ids):
        original_reset(ids)
        selected=ids[corrected_reset_mask[ids]]
        if len(selected):
            asset=raw.scene['robot']
            asset.set_joint_position_target(asset.data.joint_pos[selected].clone(),env_ids=selected)
            asset.set_joint_velocity_target(torch.zeros_like(asset.data.joint_vel[selected]),env_ids=selected)
            asset.set_joint_effort_target(torch.zeros_like(asset.data.joint_vel[selected]),env_ids=selected)
    raw._reset_idx=reset_and_sync_targets
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
'''
needle = "    env=RslRlVecEnvWrapper(raw,clip_actions=MoECTSRunnerCfg().clip_actions)"
assert setup.count(needle) == 1
setup = setup.replace(needle, capture + needle)
tail = '''    print('LOADED_DIAGNOSTIC',len(cases),flush=True)
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
        actuator_init[group]=dict(joint_indices=[int(i) for i in actuator.joint_indices],
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
    history_ids=torch.tensor([c['id'] for c in cases if c['zero_command_history']],device=raw.device,dtype=torch.long)
    if len(history_ids):
        history_buffer=raw.observation_manager._group_obs_term_history_buffer['policy']['velocity_commands']
        history_buffer._buffer[:,history_ids]=0.
        history_buffer._buffer[history_buffer._pointer,history_ids]=command.commands[history_ids]
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
    # Frozen replay probes: change a single observation channel, never update weights.
    replay=[]
    model=policies[2]
    def infer_components(ob):
        latent,weights=model.student_moe_encoder(model.actor_obs_normalizer(model.get_actor_obs(ob)))
        single=model.single_obs_normalizer(ob['single_obs'])
        return model.actor(torch.cat((latent,single),dim=-1)),latent,weights,single
    with torch.inference_mode():
        initial={k:torch.tensor(v[0][2:3],device=raw.device) for k,v in obs_trace.items()}
        for vx in (-.6,0.,.1,.2,.3,.4,.5,.6,.8,1.2):
            ob={k:v.clone() for k,v in initial.items()}
            ob['single_obs'][:,6]=vx
            ob['critic'][:,9]=vx
            ob['policy'][:,60:90:3]=vx
            output,latent,weights,single=infer_components(ob)
            replay.append(dict(probe='initial_vx',vx=vx,action=output[0].cpu().tolist(),weights=weights[0].cpu().tolist(),latent=latent[0].cpu().tolist()))
        baseline,latbase,wbase,singlebase=infer_components(initial)
        zero={k:v.clone() for k,v in initial.items()}
        zero['single_obs'][:,6]=0.
        zero['policy'][:,60:90:3]=0.
        azero,latzero,wzero,singlezero=infer_components(zero)
        for name,latent,single in [('direct_latent_zero_single',latbase,singlezero),('zero_latent_direct_single',latzero,singlebase)]:
            output=model.actor(torch.cat((latent,single),dim=-1))
            replay.append(dict(probe=name,action=output[0].cpu().tolist()))
        for time in (.02,.1,.2,.5,1.):
            index=round(time/raw.step_dt)
            original={k:torch.tensor(v[index][2:3],device=raw.device) for k,v in obs_trace.items()}
            for channel in ('unchanged','past_action','command_history','joint_velocity'):
                ob={k:v.clone() for k,v in original.items()}
                if channel=='past_action':
                    ob['single_obs'][:,37:53]=0.
                    ob['policy'][:,370:530]=0.
                elif channel=='command_history':
                    ob['policy'][:,60:90]=0.
                    ob['policy'][:,87:90]=original['single_obs'][:,6:9]
                elif channel=='joint_velocity':
                    ob['single_obs'][:,21:37]=0.
                    ob['policy'][:,210:370]=0.
                output,latent,weights,single=infer_components(ob)
                replay.append(dict(probe='recorded_input_ablation',time=time,channel=channel,
                    action=output[0].cpu().tolist(),weights=weights[0].cpu().tolist()))
    (a.output/'network_replay.json').write_text(json.dumps(replay,indent=2))
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
'''
result = prefix + cases + setup + tail
ast.parse(result)
(HERE / "probe.py").write_text(result, encoding="utf-8")
(HERE / "BUILD.json").write_text(json.dumps({"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "probe_sha256": hashlib.sha256(result.encode()).hexdigest()},indent=2))
print("BUILT", HERE / "probe.py")
