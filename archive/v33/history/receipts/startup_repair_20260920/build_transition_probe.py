"""Build a read-only same-state teacher/student start-stop diagnostic."""
from pathlib import Path
import json

p = Path(__file__).parent
s = (p/'audit.py').read_text()
old = "actions=torch.cat([act(model,{k:v[indices[i]] for k,v in obs.items()},selection[i].get('teacher',False)) for i,model in enumerate(policies)],dim=0)"
new = """students, teachers, gates, latents_s, latents_t = [], [], [], [], []
            for i,model in enumerate(policies):
                local = {k:v[indices[i]] for k,v in obs.items()}
                single = model.single_obs_normalizer(local['single_obs'])
                latent_s, weights = model.student_moe_encoder(model.actor_obs_normalizer(model.get_actor_obs(local)))
                latent_t = model.teacher_encoder(model.critic_obs_normalizer(model.get_critic_obs(local)))
                student = model.actor(torch.cat((latent_s,single),dim=-1))
                teacher = model.actor(torch.cat((latent_t,single),dim=-1))
                torch.testing.assert_close(student, model.act_inference(local))
                torch.testing.assert_close(teacher, act(model,local,True))
                students.append(student)
                teachers.append(teacher)
                gates.append(weights)
                latents_s.append(latent_s)
                latents_t.append(latent_t)
            student_actions,teacher_actions = torch.cat(students),torch.cat(teachers)
            actions=torch.cat([teachers[i] if item.get('teacher',False) else students[i]
                               for i,item in enumerate(selection)])
            extra = dict(student_action=student_actions.clone(),teacher_action=teacher_actions.clone(),
                         expert_weights=torch.cat(gates),latent_student=torch.cat(latents_s),
                         latent_teacher=torch.cat(latents_t),single_observation=obs['single_obs'].clone(),
                         policy_observation=obs['policy'].clone(),pre_joint_pos=robot.data.joint_pos.clone())"""
changes = {
    old:new,
    "velocities*(elapsed>=settle)[:,None]":"velocities*((elapsed>=settle)&(elapsed<settle+2.))[:,None]",
    "'angular_velocity','raw_action')}":"'angular_velocity','raw_action','student_action','teacher_action','expert_weights','latent_student','latent_teacher','single_observation','policy_observation','pre_joint_pos')}",
    "for k,v in values.items():":"values.update(extra)\n            for k,v in values.items():",
    "|((~pit.active)&(dx>=4.))":"",
    "protocol='Matched nominal flat":"protocol='Transition probe: move for 2s then stop; record same-state teacher/student before each step. Matched nominal flat",
    ";6s or4m boundary;":";6s budget;",
}
for old,new in changes.items():
    assert s.count(old)==1, old
    s=s.replace(old,new)
(p/'transition_probe.py').write_text(s,encoding='utf-8')
selection=[]
for version,run in [('v3','request'),('v4','protected')]:
    for teacher in (False,True):
        selection.append(dict(path=f'runs/startup_repair_{run}96_20260920/model_43201.pt',
                              label=version+('_teacher' if teacher else '_student'),teacher=teacher))
(p/'transition_selection.json').write_text(json.dumps(selection,indent=2))
