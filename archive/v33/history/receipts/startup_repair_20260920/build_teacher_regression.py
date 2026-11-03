from pathlib import Path

p=Path(__file__).parent
s=(p/'regression.py').read_text()
old='actions=policy.act_inference(obs)'
new='''critic=policy.critic_obs_normalizer(policy.get_critic_obs(obs))
            single=policy.single_obs_normalizer(obs['single_obs'])
            actions=policy.actor(torch.cat((policy.teacher_encoder(critic),single),dim=-1))'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('All first episodes; student mean action;',
    'All first episodes; baseline student and current TEACHER mean action;')
(p/'teacher_regression.py').write_text(s)
