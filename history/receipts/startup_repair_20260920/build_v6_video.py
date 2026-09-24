"""Generate a bounded V6 recording from the validated first-episode probe."""
from pathlib import Path

b = Path(__file__).parent
s = (b / 'all_scene_audit.py').read_text(encoding='utf-8')


def replace(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


replace('from mixed_task import configure_mixed, mixed_reset_root',
        'from mixed_task import configure_mixed, mixed_reset_root, MixedTerrainGenerator')
replace("    assert len(cases)==71*len(selection)", """    assert len(selection)==1 and not selection[0].get('teacher',False)
    assert latest['completed_updates']==43202 and latest['startup_repair_profile']=='omni_start_v6'
    cases=[c for c in cases if c['mode']=='direct' and (
        (c['terrain']=='flat' and c['direction'] in ('forward','reverse','left','forward_turn')) or
        (c['terrain'] in ('stairs_up','stairs_down') and c['velocity'][0]==.6) or
        (c['terrain']=='pit' and c['level']==21 and c['velocity'][0]==.6))]
    assert len(cases)==9
    exports={(c['terrain'],c['level']) for c in cases}
    geometry={}
    class RecorderTerrain(MixedTerrainGenerator):
        def _add_sub_terrain(self,mesh,origin,row,col,sub_terrain_cfg):
            super()._add_sub_terrain(mesh,origin,row,col,sub_terrain_cfg)
            name=sub_terrain_cfg.recording_name
            if (name,row) in exports and (name,row) not in geometry:
                local=mesh.copy()
                local.apply_translation(-self.terrain_origins[row,col])
                local.export(a.output/f'terrain_{name}_L{row}.obj')
                heights=np.unique(np.round(local.vertices[:,2],5))
                increments=np.round(np.diff(heights),5)
                values,counts=np.unique(increments[increments>1e-4],return_counts=True)
                riser=float(values[counts.argmax()]) if len(values) and name.startswith('stairs') else None
                geometry[(name,row)]=dict(riser_m=riser)
""")
replace('    raw=FirstEpisode(cfg=cfg)', """    for name,sub in cfg.scene.terrain.terrain_generator.sub_terrains.items():
        sub.recording_name=name
    cfg.scene.terrain.terrain_generator.class_type=RecorderTerrain
    raw=FirstEpisode(cfg=cfg)""")
replace("    arrays={k:torch.stack(v).cpu().numpy() for k,v in traces.items()}", """    for c in cases:
        c.update(geometry[(c['terrain'],c['level'])])
    arrays={k:torch.stack(v).cpu().numpy() for k,v in traces.items()}""")
start = s.index("                  protocol='Nine terrains,")
end = s.index("\n    (a.output/'manifest.json')", start)
s = s[:start] + "                  protocol='V6 student checkpoint43202; nine first episodes from default pose: four flat commands, stairs up/down L6/L9 at0.6m/s, pit1.2m at0.6m/s. No noise/push; fixed10ms delay. Retain all failures. Export actual terrain meshes for visual-only pose replay.')" + s[end:]
(b / 'record_v6_video.py').write_text(s, encoding='utf-8', newline='\n')
