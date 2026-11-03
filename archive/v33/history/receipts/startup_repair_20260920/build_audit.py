"""Generate a matched omnidirectional student/teacher regression harness."""
from pathlib import Path

p=Path(__file__).parent
source=p.parent/'posture_history_20260920/probe.py'
s=source.read_text(encoding='utf-8')
s=s.replace("for key in ('root','output'):","for key in ('root','output','selection'):")
s=s.replace("selection=json.loads((a.root/'posture_history_selection.json').read_text())", "selection=json.loads(a.selection.read_text())")
s=s.replace("latest=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=False)","latest=torch.load(a.root/selection[-1]['path'],map_location='cpu',weights_only=False)")
start=s.index('    cases=[dict(')
end=s.index('    class FirstEpisode',start)
s=s[:start]+'''    commands=[('forward',[.6,0.,0.]),('reverse',[-.6,0.,0.]),
              ('left',[0.,.4,0.]),('right',[0.,-.4,0.]),
              ('yaw_left',[0.,0.,.6]),('yaw_right',[0.,0.,-.6]),
              ('diagonal',[.4,.3,0.]),('diagonal_back',[-.4,-.3,0.]),
              ('forward_turn',[.4,0.,.4]),('side_turn',[0.,.3,-.4])]
    cases=[dict(policy=index,terrain='flat',level=0,mode=mode,direction=name,velocity=velocity,
                teacher=item.get('teacher',False)) for index,item in enumerate(selection)
           for mode in ('direct','settle') for name,velocity in commands]
    for i,c in enumerate(cases):
        c['id']=i
''' +s[end:]
s=s.replace('cfg.episode_length_s=8.','cfg.episode_length_s=6.')
start=s.index('    modes=torch.tensor(')
end=s.index('    def resample(ids):',start)
s=s[:start]+'''    settle=torch.tensor([2. if c['mode']=='settle' else 0. for c in cases],device=raw.device)
    velocities=torch.tensor([c['velocity'] for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        command.commands[:]=velocities*(elapsed>=settle)[:,None]
        pit.commands[:]=command.commands
        pit.held_speed[:]=command.commands[:,0]
''' +s[end:]
s=s.replace("    print('LOADED',len(policies),flush=True)",'''    indices=[torch.tensor([c['id'] for c in cases if c['policy']==k],device=raw.device)
             for k in range(len(policies))]
    def act(model,obs,teacher):
        if not teacher:
            return model.act_inference(obs)
        critic=model.critic_obs_normalizer(model.get_critic_obs(obs))
        single=model.single_obs_normalizer(obs['single_obs'])
        return model.actor(torch.cat((model.teacher_encoder(critic),single),dim=-1))
    print('LOADED',len(policies),flush=True)''')
s=s.replace("'root_clearance','climb','stable','active')", "'root_clearance','climb','stable','active','angular_velocity','raw_action')")
s=s.replace('range(400)','range(300)')
s=s.replace("            actions=torch.cat([model.act_inference({k:v[i*4:(i+1)*4] for k,v in obs.items()}) for i,model in enumerate(policies)],dim=0)","            actions=torch.cat([act(model,{k:v[indices[i]] for k,v in obs.items()},selection[i].get('teacher',False)) for i,model in enumerate(policies)],dim=0)")
s=s.replace("torque=d.applied_torque,command=issued,velocity=d.root_lin_vel_b,", "torque=d.applied_torque,command=issued,velocity=d.root_lin_vel_b,\n                        angular_velocity=d.root_ang_vel_b,raw_action=actions,")
s=s.replace("    print('DONE',json.dumps(cases),flush=True)","    print('DONE',len(cases),flush=True)")
s=s.replace("checkpoint=receipt['checkpoint'],completed_updates=latest['completed_updates']", "checkpoint=str(a.root/selection[-1]['path']),completed_updates=latest['completed_updates']")
start=s.index("                  protocol='")
end=s.index("\n    (a.output/'manifest.json')",start)
s=s[:start]+"                  protocol='Matched nominal flat L0, default pose and zero initial velocities; fixed10ms delay, no noise/push; ten all-direction commands with immediate and2s-settled starts;6s or4m boundary; teacher actor uses critic encoder, student actor uses history encoder; all first episodes')"+s[end:]
(p/'audit.py').write_text(s,encoding='utf-8')
