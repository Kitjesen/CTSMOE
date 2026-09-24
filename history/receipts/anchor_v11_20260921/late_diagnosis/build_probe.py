"""Build the late-failure intervention probe from the verified V11 probe."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
source = (HERE.parent / "probe_current.py").read_text(encoding="utf-8")


def replace(old: str, new: str) -> None:
    """Make one deliberate transformation and fail if the baseline drifted."""
    global source
    assert source.count(old) == 1, old[:120]
    source = source.replace(old, new)


replace(
    '"""Matched stair first episodes before/after stronger posture rewards, plus 1.2 m pit."""',
    '"""Matched flat first-episode diagnosis of the V11 late posture excursion."""',
)
replace("receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())\n", "")

start = source.index("    assert len(selection)==2")
end = source.index("    for i,c in enumerate(cases):", start)
case_source = '''    labels=[item['label'] for item in selection]
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
'''
source = source[:start] + case_source + source[end:]

replace(
    "    use_always=torch.tensor([c['policy_source']=='v6' for c in cases],device=raw.device)\n"
    "    switch_times=torch.tensor([c['intervention_time'] if c['intervention_time'] is not None else float('inf') for c in cases],device=raw.device)",
    "    controller_ids={'v6_student':0,'v9_student':1,'v11_student':2,'v11_teacher':3}\n"
    "    switch_times=torch.tensor([c['intervention_time'] if c['intervention_time'] is not None else float('inf') for c in cases],device=raw.device)\n"
    "    initial_controller=torch.tensor([controller_ids[c['controller_source']] for c in cases],device=raw.device)\n"
    "    target_controller=torch.tensor([controller_ids.get(c['intervention_target'],-1) for c in cases],device=raw.device)",
)
replace(
    "    print('LOADED',len(policies),flush=True)",
    "    def counterfactual(model,obs):\n"
    "        single=model.single_obs_normalizer(obs['single_obs'])\n"
    "        actor_obs=model.actor_obs_normalizer(model.get_actor_obs(obs))\n"
    "        latent_s,weights=model.student_moe_encoder(actor_obs)\n"
    "        critic=model.critic_obs_normalizer(model.get_critic_obs(obs))\n"
    "        latent_t=model.teacher_encoder(critic)\n"
    "        student=model.actor(torch.cat((latent_s,single),dim=-1))\n"
    "        teacher=model.actor(torch.cat((latent_t,single),dim=-1))\n"
    "        torch.testing.assert_close(student,model.act_inference(obs))\n"
    "        torch.testing.assert_close(teacher,act(model,obs,True))\n"
    "        return student,teacher,latent_s,latent_t,weights\n"
    "    print('LOADED',len(policies),flush=True)",
)
replace(
    "    traces.update({k:[] for k in ('obs_policy','obs_single','obs_critic','action_v6','action_current','uses_v6')})",
    "    traces.update({k:[] for k in ('action_v6_student','action_v9_student','action_v9_teacher',\n"
    "                                      'action_v11_student','action_v11_teacher','latent_v11_student',\n"
    "                                      'latent_v11_teacher','latent_v11_delta','gate_v11',\n"
    "                                      'actual_controller_id')})",
)
replace(
    "    obs=env.get_observations()\n    with torch.inference_mode():\n        for step in range(3 if a.smoke else 200):",
    "    obs=env.get_observations()\n"
    "    obs_trace_names={key:f'obs_{key}' for key in obs.keys()}\n"
    "    assert len(set(obs_trace_names.values()))==len(obs_trace_names)\n"
    "    traces.update({name:[] for name in obs_trace_names.values()})\n"
    "    with torch.inference_mode():\n"
    "        for step in range(3 if a.smoke else int(round(6./raw.step_dt))):",
)
replace(
    "            before_obs={k:obs[k].clone() for k in ('policy','single_obs','critic')}\n"
    "            action_v6=act(policies[0],obs,False)\n"
    "            action_current=act(policies[1],obs,False)\n"
    "            uses_v6=use_always | (step*raw.step_dt+1e-8>=switch_times)\n"
    "            actions=torch.where(uses_v6[:,None],action_v6,action_current)",
    "            before_obs={k:v.clone() for k,v in obs.items()}\n"
    "            v6_student,_,_,_,_=counterfactual(policies[0],obs)\n"
    "            v9_student,v9_teacher,_,_,_=counterfactual(policies[1],obs)\n"
    "            v11_student,v11_teacher,z_s,z_t,gate=counterfactual(policies[2],obs)\n"
    "            switched=step*raw.step_dt+1e-8>=switch_times\n"
    "            actual_controller=torch.where(switched,target_controller,initial_controller)\n"
    "            actions=v11_student.clone()\n"
    "            candidates=(v6_student,v9_student,v11_student,v11_teacher)\n"
    "            for controller_id,candidate in enumerate(candidates):\n"
    "                mask=actual_controller==controller_id\n"
    "                actions[mask]=candidate[mask]",
)
replace(
    "            values=dict(obs_policy=before_obs['policy'],obs_single=before_obs['single_obs'],\n"
    "                        obs_critic=before_obs['critic'],action_v6=action_v6,action_current=action_current,uses_v6=uses_v6,\n"
    "                        phase_weight=phase['weight']",
    "            pre_obs={obs_trace_names[k]:v for k,v in before_obs.items()}\n"
    "            values=dict(**pre_obs,action_v6_student=v6_student,action_v9_student=v9_student,\n"
    "                        action_v9_teacher=v9_teacher,action_v11_student=v11_student,\n"
    "                        action_v11_teacher=v11_teacher,latent_v11_student=z_s,latent_v11_teacher=z_t,\n"
    "                        latent_v11_delta=z_s-z_t,gate_v11=gate,actual_controller_id=actual_controller,\n"
    "                        phase_weight=phase['weight']",
)
replace("budget=flat & ((step+1)*raw.step_dt>=4.)", "budget=flat & ((step+1)*raw.step_dt>=6.)")
replace(
    "                  protocol='Intervention diagnosis, not a deployable policy: 11 flat commands, current model/V6 controls and current model-to-V6 switches after 0.10/0.20/0.40/0.80 seconds. 4s or first termination. Same default initial state, no observation noise/push, 10ms actuator delay. Raw observations are before action; joint/target states are after action. Independent environments may diverge numerically; pre-intervention parity is measured, not assumed.')",
    "                  controller_ids=controller_ids,observation_traces=obs_trace_names,\n"
    "                  protocol='Evaluation-only late-failure diagnosis: six flat commands, 64 matched first episodes, 6s budget or first termination. Direct V6/V9/V11 students and V11 teacher; V11-student interventions to V6 student or same-model V11 teacher at 1.0/2.0/2.8s, plus forward-0.6 interventions at 2.6/3.0s. All policy observations and same-state counterfactual actions are pre-step; physical state, request and target are post-step. V11 student/teacher latents and confirmed MoE softmax weights are same-model diagnostics. Switches do not reset state. Fixed 10ms delay, no observation corruption or pushes. Matched initialization is asserted; pre-switch trajectory agreement is measured rather than assumed perfect.')",
)

(HERE / "probe.py").write_text(source, encoding="utf-8")
print(HERE / "probe.py")
