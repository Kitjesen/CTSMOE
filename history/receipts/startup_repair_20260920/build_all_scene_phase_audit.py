"""Add geometry/support context for stage review without changing policy actions."""
from pathlib import Path

b=Path(__file__).parent
source=(b/'all_scene_audit.py').read_text(encoding='utf-8')


def replace(old,new):
    global source
    assert source.count(old)==1,old[:80]
    source=source.replace(old,new)


replace("a=p.parse_args()", "p.add_argument('--smoke',action='store_true',help='Three physics control steps; not a behavior evaluation')\na=p.parse_args()")
replace("        for step in range(1250):", "        for step in range(3 if a.smoke else 1250):")
replace("'gravity','base_contact')}",
        "'gravity','base_contact','ground_relief','ground_scan_valid','wheel_clearance','wheel_ground_force','pit_front_gap','platform_stop')}")
replace("            start=d.root_pos_w[:,None,:].clone()",
        "            start=torch.cat((d.root_pos_w[:,None,:],d.body_pos_w[:,pit.wheel_ids]),dim=1).clone()")
replace("            values=dict(body_pos=", """            ray_z=scanner.data.ray_hits_w[...,2]
            valid=torch.isfinite(ray_z).all(-1)
            relief=torch.where(valid,ray_z.amax(-1)-ray_z.amin(-1),torch.full_like(ray_z[:,0],float('nan')))
            wheel_z=d.body_pos_w[:,pit.wheel_ids,2]
            ground_force=torch.stack([sensor.data.force_matrix_w[:,0,0].norm(dim=-1)
                                      for sensor in pit.ground_contacts],dim=1)
            front_gap=origins[:,0]+1.5-(d.body_pos_w[:,pit.wheel_ids,0].amax(-1)+.093)
            values=dict(ground_relief=relief,ground_scan_valid=valid,
                        wheel_clearance=wheel_z-hits[:,1:,2]-.093,
                        wheel_ground_force=ground_force,
                        pit_front_gap=torch.where(pit.active,front_gap,torch.full_like(front_gap,float('nan'))),
                        platform_stop=pit.stop_on_platform,body_pos=""")
replace("                  actuator_limits=guard.limits,initial_state_matched=True,",
        "                  actuator_limits=guard.limits,initial_state_matched=True,smoke_only=a.smoke,\n                  phase_telemetry='geometry/support measurements only; stage acceptance not inferred from joint pose',")
(b/'all_scene_phase_audit.py').write_text(source,encoding='utf-8',newline='\n')
print('Wrote all_scene_phase_audit.py')
