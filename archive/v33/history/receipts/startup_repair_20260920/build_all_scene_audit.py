"""Derive a wider, deterministic first-episode audit from the validated probe."""
from pathlib import Path

base = Path(__file__).parent
source = (base / 'audit.py').read_text(encoding='utf-8')


def replace(old, new):
    global source
    assert source.count(old) == 1, old[:100]
    source = source.replace(old, new)


replace("a=p.parse_args()", "p.add_argument('--code',type=Path,required=True)\na=p.parse_args()")
replace("sys.path[:0]=[receipt['code'],", "sys.path[:0]=[str(a.code),")
replace("from mixed_task import configure_mixed, mixed_reset_root", "from mixed_task import configure_mixed, mixed_reset_root\n    from isaaclab.envs.mdp.events import reset_joints_by_scale")
replace("    for i,c in enumerate(cases):", """    for index,item in enumerate(selection):
        cases += [dict(policy=index,terrain=terrain,level=level,mode='direct',
                       direction='forward',velocity=[speed,0.,0.],teacher=item.get('teacher',False))
                  for terrain in ('stairs_up','stairs_down','slope_up','slope_down',
                                  'wave','rough_slope','obstacles')
                  for level in (6,9) for speed in (.3,.6,1.2)]
        cases += [dict(policy=index,terrain='pit',level=level,mode='direct',
                       direction='forward',velocity=[speed,0.,0.],teacher=item.get('teacher',False))
                  for level in (9,15,21) for speed in (.3,.6,.9)]
    cases.sort(key=lambda c:c['policy'])
    assert len(cases)==71*len(selection)
    for i,c in enumerate(cases):""")
replace("        mixed_reset_root(env,env_ids,pose_range,velocity_range,asset_cfg)", """        robot=env.scene['robot']
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
            env.pit_valid_spawn[pit_ids]=True""")
replace("    cfg.episode_length_s=6.", "    cfg.episode_length_s=25.")
replace("    cfg.events.reset_robot_joints.params.update(position_range=(1.,1.),velocity_range=(0.,0.))", "    cfg.events.reset_robot_joints.func=reset_joints_by_scale\n    cfg.events.reset_robot_joints.params.update(position_range=(1.,1.),velocity_range=(0.,0.))")
replace("        pit.commands[:]=command.commands", "        command.commands[pit.active & pit.stop_on_platform]=0.\n        pit.commands[:]=command.commands")
replace("    initial=robot.data.root_pos_w.clone()", """    initial=robot.data.root_pos_w.clone()
    initial_q=robot.data.joint_pos.clone()
    initial_root=robot.data.root_state_w.clone()
    relative=initial_root.clone()
    relative[:,:3]-=origins
    for ids in indices[1:]:
        torch.testing.assert_close(relative[ids],relative[indices[0]],atol=2e-4,rtol=0)
        torch.testing.assert_close(initial_q[ids],initial_q[indices[0]],atol=1e-6,rtol=0)
    print('MATCHED_INITIAL_STATES',len(cases),flush=True)""")
replace("'angular_velocity','raw_action')}", "'angular_velocity','raw_action','requested','torque_peak','speed_peak','hard_excess','saturation','gravity','base_contact')}")
replace("    obs=env.get_observations()", """    step_stats={k:torch.zeros_like(robot.data.joint_pos) for k in
                ('torque_peak','speed_peak','hard_excess','saturation')}
    original_measure=guard.measure
    def measure_substep():
        original_measure()
        d=robot.data
        step_stats['torque_peak'].copy_(torch.maximum(step_stats['torque_peak'],d.applied_torque.abs()))
        step_stats['speed_peak'].copy_(torch.maximum(step_stats['speed_peak'],d.joint_vel.abs()))
        excess=torch.maximum(guard.hard[...,0]-d.joint_pos,d.joint_pos-guard.hard[...,1]).clamp_min(0)*guard.leg
        step_stats['hard_excess'].copy_(torch.maximum(step_stats['hard_excess'],excess))
        step_stats['saturation'].add_((d.computed_torque-d.applied_torque).abs().gt(.001)/raw.cfg.decimation)
    guard.measure=measure_substep
    flat=torch.tensor([c['terrain']=='flat' for c in cases],device=raw.device)
    obs=env.get_observations()""")
replace("        for step in range(300):", "        for step in range(1250):")
replace("            obs,_,_,_=env.step(actions)", "            for value in step_stats.values():\n                value.zero_()\n            obs,_,_,_=env.step(actions)")
replace("                        angular_velocity=d.root_ang_vel_b,raw_action=actions,", """                        angular_velocity=d.root_ang_vel_b,raw_action=actions,
                        requested=raw.action_manager.get_term('joint_pos').requested_actions,
                        gravity=d.projected_gravity_b,
                        base_contact=pit.contact.data.net_forces_w_history[:,:,pit.base_contact_ids].norm(dim=-1).amax(dim=(1,2)),
                        **step_stats,""")
replace("            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|((~pit.active)&(dx>=4.)))", """            budget=flat & ((step+1)*raw.step_dt>=6.)
            ended=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs|budget|((~pit.active)&(dx>=4.)))""")
replace("                                stable=bool(pit.success[i]),checkpoint=selection[cases[i]['policy']]['path'])", """                                stable=bool(pit.success[i]),checkpoint=selection[cases[i]['policy']]['path'],
                                violation_causes={k:bool(v[i]) for k,v in guard.violation_causes.items()})""")
replace("    np.savez_compressed(a.output/'traces.npz',**arrays)", "    np.savez_compressed(a.output/'traces.npz',initial_q=initial_q.cpu().numpy(),initial_root=relative.cpu().numpy(),**arrays)")
replace("                  default_q=robot.data.default_joint_pos[0].cpu().tolist(),", """                  default_q=robot.data.default_joint_pos[0].cpu().tolist(),
                  request_joint_ids=raw.action_manager.get_term('joint_pos').joint_indices,
                  actuator_limits=guard.limits,initial_state_matched=True,
                  slope_geometry=getattr(cfg.scene.terrain.terrain_generator,'slope_geometry_receipts',[]),""")
start="                  protocol='Matched nominal flat L0, default pose and zero initial velocities; fixed10ms delay, no noise/push; ten all-direction commands with immediate and2s-settled starts;6s or4m boundary; teacher actor uses critic encoder, student actor uses history encoder; all first episodes')"
replace(start,"                  protocol='Nine terrains, matched default joint/root states verified across policies. No noise or pushes; fixed10ms delay. 71 cases per policy: 20 all-direction flat starts (6s budget), seven other terrain types at L6/L9 and vx0.3/0.6/1.2, pit0.6/0.9/1.2m at vx0.3/0.6/0.9 (25s budget). No heading feedback. Pit external stop retained. 50Hz trajectories plus per-step 200Hz torque/speed/hard-limit peaks. Fixed cases, not success probabilities over random trials.')")
(base/'all_scene_audit.py').write_text(source,encoding='utf-8',newline='\n')
print('Wrote all_scene_audit.py')
