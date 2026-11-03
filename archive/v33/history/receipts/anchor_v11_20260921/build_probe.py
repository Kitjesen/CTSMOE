"""Build a matched intervention probe from the already verified evaluator."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE.parent / 'flat_v10_20260921/evaluate.py').read_text(encoding='utf-8')


def replace(old, new):
    global source
    assert source.count(old) == 1, old[:100]
    source = source.replace(old, new)


start = source.index('    commands=[')
end = source.index('    for i,c in enumerate(cases):', start)
source = source[:start] + '''    assert [item['label'] for item in selection] == ['V6_43202', 'V9_44711']
    commands=[(f'vx_{v}',[v,0.,0.]) for v in (.3,.45,.6,.75,.9,1.2,-.6)]
    commands += [('left',[0.,.4,0.]),('right',[0.,-.4,0.]),
                 ('yaw_left',[0.,0.,.6]),('yaw_right',[0.,0.,-.6])]
    modes=[('v9',None),('v6',None)]+[('switch_v9_to_v6',t) for t in (.1,.2,.4,.8)]
    cases=[dict(policy=1,terrain='flat',level=0,mode='direct',direction=name,
                velocity=velocity,teacher=False,policy_source=mode,intervention_time=t)
           for name,velocity in commands for mode,t in modes]
    assert len(cases)==66
''' + source[end:]
replace("    indices=[torch.tensor([c['id'] for c in cases if c['policy']==k],device=raw.device)\n             for k in range(len(policies))]", "    use_always=torch.tensor([c['policy_source']=='v6' for c in cases],device=raw.device)\n    switch_times=torch.tensor([c['intervention_time'] if c['intervention_time'] is not None else float('inf') for c in cases],device=raw.device)")
replace("    for ids in indices[1:]:\n        torch.testing.assert_close(relative[ids],relative[indices[0]],atol=2e-4,rtol=0)\n        torch.testing.assert_close(initial_q[ids],initial_q[indices[0]],atol=1e-6,rtol=0)", "    torch.testing.assert_close(relative,relative[:1].expand_as(relative),atol=2e-4,rtol=0)\n    torch.testing.assert_close(initial_q,initial_q[:1].expand_as(initial_q),atol=1e-6,rtol=0)")
replace("    scanner=raw.scene['height_scanner']", "    traces.update({k:[] for k in ('obs_policy','obs_single','obs_critic','action_v6','action_v9','uses_v6')})\n    scanner=raw.scene['height_scanner']")
replace('for step in range(3 if a.smoke else 1250):', 'for step in range(3 if a.smoke else 200):')
replace("            actions=torch.cat([act(model,{k:v[indices[i]] for k,v in obs.items()},selection[i].get('teacher',False)) for i,model in enumerate(policies)],dim=0)", "            before_obs={k:obs[k].clone() for k in ('policy','single_obs','critic')}\n            action_v6=act(policies[0],obs,False)\n            action_v9=act(policies[1],obs,False)\n            uses_v6=use_always | (step*raw.step_dt+1e-8>=switch_times)\n            actions=torch.where(uses_v6[:,None],action_v6,action_v9)")
replace("            values=dict(phase_weight=phase['weight']", "            values=dict(obs_policy=before_obs['policy'],obs_single=before_obs['single_obs'],\n                        obs_critic=before_obs['critic'],action_v6=action_v6,action_v9=action_v9,uses_v6=uses_v6,\n                        phase_weight=phase['weight']")
replace('budget=flat & ((step+1)*raw.step_dt>=6.)', 'budget=flat & ((step+1)*raw.step_dt>=4.)')
replace("|budget|((~pit.active)&(dx>=4.)))", "|budget)")
old = source[source.index("                  protocol='V10"):source.index("    (a.output/'manifest.json')")]
source = source.replace(old, "                  protocol='Intervention diagnosis, not a deployable policy: 11 flat commands, V9/V6 controls and V9-to-V6 switches after 0.10/0.20/0.40/0.80 seconds. 4s or first termination. Same default initial state, no observation noise/push, 10ms actuator delay. Raw observations are before action; joint/target states are after action. Independent environments may diverge numerically; pre-intervention parity is measured, not assumed.')\n")
(HERE / 'probe_evaluate.py').write_text(source, encoding='utf-8')
