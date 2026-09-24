"""Read-only frozen student first-episode diagnostic; production stays running."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
for key in ('root','code','checkpoint','output'):
    p.add_argument('--'+key, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False, parents=True)
sys.path[:0] = [str(a.code),str(a.root/'upstream/source/rsl_rl'),str(a.root/'upstream/source/robot_lab'),str(a.root/'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS
    from config import make_config
    from actuator_guard import ActuatorGuard
    from train import apply_command_curriculum_start
    from training_profiles import configure_base_height_penalty
    from stair_speed import configure_stair_speed
    from wave_reset import configure_wave_reset
    from mixed_task import configure_mixed

    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cases = [dict(terrain=t, level=l, noise=noise, repeat=i)
             for t in ('wave','slope_up','slope_down')
             for l in ((3,6,9) if t == 'wave' else (5,7,9))
             for noise in (False,True) for i in range(8)]
    class FirstEpisode(Go2Env):
        retain_terminal = False
        def _reset_idx(self, ids):
            if not self.retain_terminal:
                super()._reset_idx(ids)
    cfg = make_config(len(cases), a.root/'cache/usd', actuator_profile=saved['actuator_profile'])
    cfg.seed, cfg.log_dir = 2718, str(a.output)
    apply_command_curriculum_start(cfg.commands.base_velocity, saved['command_curriculum_start'])
    configure_base_height_penalty(cfg, saved['base_height_penalty'])
    configure_stair_speed(cfg, saved['stair_command_profile'])
    configure_wave_reset(cfg, saved['wave_reset_profile'])
    configure_mixed(cfg, slope_profile=saved['slope_profile'], pit_training_profile=saved['pit_training_profile'])
    for key in vars(cfg.curriculum):
        setattr(cfg.curriculum, key, None)
    cfg.rewards.lin_vel_z_l2.weight = 0
    cfg.events.reset_base.params['pose_range']['yaw'] = (0.,0.)
    raw = FirstEpisode(cfg=cfg)
    guard = ActuatorGuard(raw)
    raw.common_step_counter = saved['common_step_counter']
    terrain = raw.scene.terrain
    names = saved['terrain_column_names']
    terrain.terrain_types[:] = torch.tensor([names.index(c['terrain']) for c in cases], device=raw.device)
    terrain.terrain_levels[:] = torch.tensor([c['level'] for c in cases], device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    raw.scene.env_origins[:] = terrain.env_origins
    command = raw.command_manager.get_term('base_velocity')
    command.refresh_membership()
    def resample(ids):
        command.commands[ids] = 0
        command.commands[ids,0] = .6
        command.time_left[ids] = 1e9
    def update():
        command.commands[:] = 0
        command.commands[:,0] = .6
    command._resample, command._update_command = resample, update
    env = RslRlVecEnvWrapper(raw, clip_actions=MoECTSRunnerCfg().clip_actions)
    agent = MoECTSRunnerCfg().to_dict()
    agent['robogauge'] = {'enabled':False}
    runner = OnPolicyRunnerCTS(env, agent, log_dir=str(a.output/'loader'), device=raw.device)
    policy = runner.alg.policy
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    runner.eval_mode()
    robot = raw.scene['robot']
    materials = robot.root_physx_view.get_material_properties().cpu()
    assert materials.shape[0] == len(cases)
    origins = raw.scene.env_origins.clone()
    active = torch.ones(len(cases), device=raw.device, dtype=torch.bool)
    peaks = torch.zeros_like(robot.data.joint_vel)
    clips = torch.zeros_like(peaks)
    counts = torch.zeros(len(cases), device=raw.device)
    error = torch.zeros_like(counts)
    steps = torch.zeros_like(counts)
    max_y = torch.zeros_like(counts)
    measure = guard.measure
    def measured():
        measure()
        peaks[:] = torch.maximum(peaks, robot.data.joint_vel.abs()*active[:,None])
        clips[:] += ((robot.data.computed_torque-robot.data.applied_torque).abs() > .1)*active[:,None]
        counts[:] += active
    guard.measure = measured
    raw.retain_terminal = True
    obs = env.get_observations()
    noisy = torch.tensor([c['noise'] for c in cases], device=raw.device)
    rng = torch.Generator(device=raw.device).manual_seed(1729)
    results = [None]*len(cases)
    with torch.inference_mode():
        for step in range(1250):
            actions = policy.act_inference(obs)
            actions += noisy[:,None]*torch.randn(actions.shape, device=raw.device, generator=rng)*policy.std
            actions[~active] = 0
            obs,_,_,_ = env.step(actions)
            delta = robot.data.root_pos_w-origins
            error += (robot.data.root_lin_vel_b[:,0]-.6).abs()*active
            steps += active
            max_y[:] = torch.maximum(max_y, delta[:,1].abs()*active)
            physical = raw.termination_manager.terminated
            passed = (delta[:,0] >= 4.) & ~physical
            ended = active & (physical | passed | raw.termination_manager.time_outs)
            for i in ended.nonzero().flatten().tolist():
                terms = [n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i])]
                j = int(peaks[i].argmax())
                results[i] = dict(**cases[i], seconds=(step+1)*raw.step_dt,
                    passed=bool(passed[i]), physical=bool(physical[i]), terms=terms,
                    causes={k:bool(v[i]) for k,v in guard.violation_causes.items()},
                    displacement=delta[i].cpu().tolist(), vx_mae=float(error[i]/steps[i]),
                    lateral_max=float(max_y[i]), peak_speed=float(peaks[i,j]), peak_joint=robot.joint_names[j],
                    max_torque_clipped_fraction=float((clips[i]/counts[i].clamp_min(1)).max()),
                    material_mean=materials[i].mean(0).tolist())
            active[ended] = False
            if step % 250 == 0:
                print('PROGRESS',step,int(active.sum()),flush=True)
            if not bool(active.any()):
                break
    assert all(r is not None for r in results)
    (a.output/'results.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),
        completed_updates=saved['completed_updates'], protocol='first episode, fixed +0.6m/s, yaw command zero; training randomization and observation noise retained; action noise split; distinct randomized cases, not paired causal ablation', cases=results),indent=2))
    print('DONE',str(a.output/'results.json'),flush=True)
finally:
    app.close()
