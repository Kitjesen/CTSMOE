"""Frozen student diagnosis of every platform success predicate; no training edits."""
import argparse
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
for key in ("root","code","checkpoint","output"):
    p.add_argument("--"+key,type=Path,required=True)
p.add_argument("--stochastic",action="store_true")
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.code),str(a.root/"upstream/source/rsl_rl"),str(a.root/"upstream/source/robot_lab"),str(a.root/"thunder/source/doso_train")]
os.environ["OMNI_KIT_ACCEPT_EULA"]="YES"
from isaaclab.app import AppLauncher  # noqa: E402 -- simulator must start before Isaac scene imports
app=AppLauncher(headless=True,device="cuda:0").app
try:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from actuator_guard import ActuatorGuard
    from policy_contract import checkpoint_profile
    from train import apply_command_curriculum_start
    from training_profiles import configure_base_height_penalty
    from stair_speed import configure_stair_speed
    from wave_reset import configure_wave_reset
    from mixed_task import configure_mixed
    from pit_geometry import EDGE_OFFSET

    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    profile=checkpoint_profile(saved)
    cases=[dict(speed=speed,repeat=i,level=2) for speed in (.3,.5,.7,.9) for i in range(8)]
    class FirstEpisodeEnv(Go2Env):
        retain_terminal=False
        def _reset_idx(self,ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    cfg=make_config(len(cases),a.root/"cache/usd",actuator_profile=profile)
    cfg.seed,cfg.log_dir=123,str(a.output)
    apply_command_curriculum_start(cfg.commands.base_velocity,saved["command_curriculum_start"])
    configure_base_height_penalty(cfg,saved["base_height_penalty"])
    configure_stair_speed(cfg,saved["stair_command_profile"])
    configure_wave_reset(cfg,saved["wave_reset_profile"])
    configure_mixed(cfg,slope_profile=saved["slope_profile"])
    for key in vars(cfg.curriculum):
        setattr(cfg.curriculum,key,None)
    raw=FirstEpisodeEnv(cfg=cfg)
    guard=ActuatorGuard(raw,profile)
    terrain=raw.scene.terrain
    terrain.terrain_levels.fill_(2)
    terrain.terrain_types.fill_(20)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    raw.scene.env_origins[:]=terrain.env_origins
    command=raw.command_manager.get_term("base_velocity")
    command.refresh_membership()
    c=command.pit
    speeds=torch.tensor([x["speed"] for x in cases],device=raw.device)
    def resample(ids):
        c.held_speed[ids]=speeds[ids]
        c.speed_bins[ids]=torch.arange(len(cases),device=raw.device)[ids]//8
        c.commands[ids]=0
        c.commands[ids,0]=speeds[ids]
        c.time_left[ids]=1e9
    c._resample=resample
    env=RslRlVecEnvWrapper(raw,clip_actions=MoECTSRunnerCfg().clip_actions)
    agent=MoECTSRunnerCfg().to_dict()
    agent["robogauge"]={"enabled":False}
    runner=OnPolicyRunnerCTS(env,agent,log_dir=str(a.output/"loader"),device=raw.device)
    runner.alg.policy.load_state_dict(saved["model_state_dict"],strict=True)
    runner.eval_mode()
    policy=runner.alg.policy
    assert not policy.state_dependent_std and policy.noise_std_type=="scalar"
    noise_rng=torch.Generator(device=raw.device).manual_seed(987)
    c.learner.fill_(1)
    assert bool(c.active.all()) and bool((terrain.terrain_levels==2).all())
    raw.retain_terminal=True
    active=torch.ones(len(cases),device=raw.device,dtype=torch.bool)
    keys=("all_support","body_clear","upright","linear_quiet","angular_quiet","all_valid","stop_command")
    ever={k:torch.zeros_like(active) for k in keys}
    counts={k:torch.zeros(len(cases),device=raw.device) for k in keys}
    after_stop_counts={k:torch.zeros(len(cases),device=raw.device) for k in keys}
    max_hold=torch.zeros(len(cases),device=raw.device)
    last={}
    terminal=[None]*len(cases)
    old_refresh=c.refresh
    last_step=-1
    def refresh():
        global last_step,last
        old_refresh()
        if last_step==raw.common_step_counter:
            return
        last_step=raw.common_step_counter
        data=c.robot.data
        origins=raw.scene.env_origins
        rel=data.root_pos_w-origins
        edge=origins[:,0]+EDGE_OFFSET
        points=c.geometry.world_points(data.joint_pos,data.root_state_w[:,:7])
        min_x=points[...,0].amin(1)
        geometry_clear=(min_x>edge+.06)&(points[...,1].amin(1)>origins[:,1]-1.5)&(points[...,1].amax(1)<origins[:,1]+1.5)
        all_support=c.supported.all(-1)
        body_clear=all_support&(rel[:,0]>EDGE_OFFSET+.45)&geometry_clear
        linear=torch.linalg.vector_norm(data.root_lin_vel_w[:,:2],dim=-1)
        angular=torch.linalg.vector_norm(data.root_ang_vel_w,dim=-1)
        tilt=torch.rad2deg(torch.acos((-data.projected_gravity_b[:,2]).clamp(-1,1)))
        conditions=dict(all_support=all_support,body_clear=body_clear,upright=tilt<15.,
                        linear_quiet=linear<.15,angular_quiet=angular<.3,
                        stop_command=c.stop_on_platform)
        conditions["all_valid"]=body_clear&(tilt<15)&(linear<.15)&(angular<.3)&~c.physical&~c.outside&c.started_in_pit
        for key,value in conditions.items():
            ever[key]|=value&active
            counts[key]+=(value&active).float()
            after_stop_counts[key]+=(value&active&c.stop_on_platform).float()
        max_hold[:]=torch.maximum(max_hold,c.hold*active)
        last=dict(forward_m=rel[:,0],rear_clearance_m=min_x-edge,linear_speed=linear,angular_speed=angular,
                  tilt_deg=tilt,hold_seconds=c.hold*raw.step_dt,command_vx=command.commands[:,0],
                  all_support=all_support,body_clear=body_clear,stop_command=c.stop_on_platform)
    c.refresh=refresh
    obs=env.get_observations()
    with torch.inference_mode():
        for step in range(1250):
            actions=policy.act_inference(obs)
            if a.stochastic:
                # Same diagonal Normal(mean, std) as upstream policy.act, using
                # a separate RNG so action draws do not shift observation noise.
                actions=actions+torch.randn(actions.shape,device=raw.device,generator=noise_rng)*policy.std
            actions[~active]=0
            obs,_,_,_=env.step(actions)
            finish=active&(raw.termination_manager.terminated|raw.termination_manager.time_outs)
            for i in finish.nonzero().flatten().tolist():
                terminal[i]=dict(seconds=(step+1)*raw.step_dt,success=bool(c.success[i]),
                    physical=bool(c.physical[i]),outside=bool(c.outside[i]),
                    timeout=bool(raw.termination_manager.time_outs[i]),
                    last={k:float(v[i]) for k,v in last.items()})
            active[finish]=False
            if step%200==0:
                print("PIT_DIAG",step,"active",int(active.sum()),"ever",{k:int(v.sum()) for k,v in ever.items()},"maxhold",float(max_hold.max()*.02),flush=True)
            if not bool(active.any()):
                break
    for i,case in enumerate(cases):
        if terminal[i] is None:
            terminal[i]=dict(success=False,observation_budget_exhausted=True,last={k:float(v[i]) for k,v in last.items()})
        case.update(terminal[i],ever={k:bool(v[i]) for k,v in ever.items()},
                    condition_seconds={k:float(v[i]*raw.step_dt) for k,v in counts.items()},
                    after_stop_seconds={k:float(v[i]*raw.step_dt) for k,v in after_stop_counts.items()},
                    maximum_valid_hold_seconds=float(max_hold[i]*raw.step_dt))
    result=dict(checkpoint=str(a.checkpoint),completed_updates=saved["completed_updates"],
                stochastic=a.stochastic,action_std=policy.std.cpu().tolist(),
                scope="32 frozen student episodes at 25cm; four speeds; training randomization; no optimizer updates",
                cases=cases)
    (a.output/"results.json").write_text(json.dumps(result,indent=2))
    print("PIT_DIAG_COMPLETE",flush=True)
    env.close()
finally:
    app.close()
