"""Build a matched historical-policy evaluation from the verified stair harness."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
s = (root/'recordings/stairs_strong_20260920/record.py').read_text(encoding='utf-8')
s = s.replace("receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())", "receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())\nselection=json.loads((a.root/'posture_history_selection.json').read_text())")
start = s.index("    before_path=")
end = s.index("    class FirstEpisode", start)
s = s[:start] + '''    cases=[dict(policy=index,terrain='flat',level=0,mode=mode,speed=.6,settle_s=0.,id=index*4+j)
           for index in range(len(selection)) for j,mode in enumerate(('stand','direct','settle','ramp'))]
''' + s[end:]
s = s.replace('configure_mixed, MixedTerrainGenerator, mixed_reset_root', 'configure_mixed, mixed_reset_root')
start = s.index('    for sub in cfg.scene.terrain.terrain_generator.sub_terrains:')
end = s.index('    for name in vars(cfg.curriculum):',start)
s = s[:start] + s[end:]
s = s.replace('cfg.episode_length_s=25.', 'cfg.episode_length_s=8.')
start = s.index('    speed=torch.tensor(')
end = s.index('    def resample(ids):',start)
s = s[:start] + '''    modes=torch.tensor([('stand','direct','settle','ramp').index(c['mode']) for c in cases],device=raw.device)
    def update():
        elapsed=raw.episode_length_buf*raw.step_dt
        vx=torch.where(modes==0,0.,torch.where(modes==2,.6*(elapsed>=2),
                      torch.where(modes==3,.6*elapsed.clamp(0.,1.),.6)))
        command.commands.zero_()
        command.commands[:,0]=vx
        pit.commands.zero_()
        pit.commands[:,0]=vx
        pit.held_speed[:]=vx
''' + s[end:]
start=s.index('    old=copy.deepcopy(policy)')
end=s.index('    pit.learner.fill_(1)',start)
s=s[:start]+'''    policies=[]
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
''' + s[end:]
s=s.replace('range(1250)', 'range(400)')
s=s.replace("            actions=policy.act_inference(obs)\n            actions[old_ids]=old.act_inference({k:v[old_ids] for k,v in obs.items()})", "            actions=torch.cat([model.act_inference({k:v[i*4:(i+1)*4] for k,v in obs.items()}) for i,model in enumerate(policies)],dim=0)")
s=s.replace("**geometry[(cases[i]['terrain'],cases[i]['level'])]", "checkpoint=selection[cases[i]['policy']]['path']")
s=s.replace("    assert all('end_step' in c for c in cases)", "    for c in cases:\n        if 'end_step' not in c:\n            c.update(end_step=step,seconds=(step+1)*raw.step_dt,physical=False,terms=['budget'],passed_4m=False)")
s=s.replace("before_checkpoint=str(before_path),before_completed_updates=before['completed_updates'],", "history=metadata,")
start=s.index("                  protocol='")
end=s.index("\n    (a.output/'manifest.json')",start)
s=s[:start]+"                  protocol='Historical weights evaluated in identical current nominal physics; flat L0, zero initial velocities/default joints, yaw0, no noise/push, fixed10ms actuator delay; four first episodes per checkpoint: stand, immediate0.6, stand2s then0.6, ramp0to0.6 over1s; current guards;8s or4m boundary; not historical deployment reproduction')"+s[end:]
(Path(__file__).parent/'probe.py').write_text(s,encoding='utf-8')
