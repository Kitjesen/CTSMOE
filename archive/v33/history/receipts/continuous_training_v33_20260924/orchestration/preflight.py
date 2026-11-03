"""V33 zero-update physical qualification; explicit all-level route coverage."""
import json
from pathlib import Path
import sys
from state_guard import MutationGuard, state_sha256

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/yaw_stairs_v32_20260924')
sys.path.insert(0,str(OLD/'orchestration'))
from v31_common import args, digest  # noqa: E402
STEPS = 1512  # 30.24s, full25s reset plus subsequent control; divisible by24.


def factory(source, output):
    def qualify(self, updates, init_at_random_ep_len=False):
        import torch
        from continuous_runtime import snapshot, FAMILIES
        raw=self.env.unwrapped
        command=raw.command_manager.get_term('base_velocity')
        terrain=raw.scene.terrain
        if updates or self.env.num_envs!=8192 or init_at_random_ep_len:
            raise RuntimeError('Qualification must be8192 and zero updates')
        before={k:state_sha256(v,torch) for k,v in (
            ('model',self.alg.policy.state_dict()),('ppo',self.alg.optimizer.state_dict()),('student',self.alg.optimizer_stu_enc.state_dict()))}
        # Compare the existing440 cells by hash, not just shape or rendered appearance.
        reference=json.loads((OLD/'training_39162_to_46194/COVERAGE_GEOMETRY.json').read_text())['mesh_receipts']
        current=raw.cfg.scene.terrain.terrain_generator.stair_transition_coverage_receipts
        oldmap={(r['row'],r['column']):r for r in reference if r['column']<20}
        for r in current:
            if r['column']<20:
                prior=oldmap[(r['row'],r['column'])]
                if r['after_sha256']!=prior['after_sha256'] or r['origin_after']!=prior['origin_after']:
                    raise RuntimeError('Original terrain geometry changed')
        all_ids=[]
        for col in range(20,26):
            ids=(terrain.terrain_types==col).nonzero().flatten()
            if len(ids)<20:
                raise RuntimeError('Missing route representation')
            terrain.terrain_levels[ids]=torch.arange(len(ids),device=raw.device)%10
            all_ids.append(ids)
        ids=torch.cat(all_ids)
        terrain.env_origins[ids]=terrain.terrain_origins[terrain.terrain_levels[ids],terrain.terrain_types[ids]]
        raw.scene.env_origins[ids]=terrain.env_origins[ids]
        # Skip promotion only during this explicit initialization, as normal resume does.
        compute=raw.curriculum_manager.compute
        raw.curriculum_manager.compute=lambda env_ids: None
        try:
            raw.reset()
        finally:
            raw.curriculum_manager.compute=compute
        command.refresh_membership()
        scanner=raw.scene['height_scanner'].data.ray_hits_w
        if not bool(torch.isfinite(scanner).all()):
            raise RuntimeError('Invalid collision scanner at route spawn')
        seen=torch.zeros((6,10),dtype=torch.long,device=raw.device)
        resets=torch.zeros(6,dtype=torch.long,device=raw.device)
        obs=self.env.get_observations().to(self.device)
        self.train_mode()
        guard=MutationGuard(self)
        with guard,torch.inference_mode():
            for step in range(STEPS):
                actions=self.alg.act(obs)
                obs,rewards,done,extras=self.env.step(actions.to(raw.device))
                obs,rewards,done=obs.to(self.device),rewards.to(self.device),done.to(self.device)
                self.alg.process_env_step(obs,rewards,done,extras)
                if not all(bool(torch.isfinite(v).all()) for v in (actions,rewards,raw.scene['robot'].data.joint_pos)):
                    raise FloatingPointError('Nonfinite qualification trajectory')
                if bool(command.pit.active.any()):
                    raise RuntimeError('Pit assignment active')
                route_reset = done.bool() & (terrain.terrain_types >= 20) & (terrain.terrain_types < 26)
                if (bool(command.continuous_success[route_reset].any())
                        or bool(command.continuous_boundary[route_reset].any())
                        or bool((command.continuous_tracking_count[route_reset] != 0).any())):
                    raise RuntimeError('Continuous episode state leaked across a real reset')
                for f in range(6):
                    selected=terrain.terrain_types==20+f
                    seen[f]+=torch.bincount(terrain.terrain_levels[selected],minlength=10)
                    resets[f]+=done[selected].long().sum()
                if (step+1)%24==0:
                    self.alg.storage.clear()
                if (step+1)%252==0:
                    print('V33_ZERO_UPDATE',step+1,flush=True)
        after={k:state_sha256(v,torch) for k,v in (
            ('model',self.alg.policy.state_dict()),('ppo',self.alg.optimizer.state_dict()),('student',self.alg.optimizer_stu_enc.state_dict()))}
        if before!=after or guard.calls or not guard.restored or self.alg.storage.step!=0 or bool((seen==0).any()) or bool((resets==0).any()):
            raise RuntimeError('Qualification state or reset coverage failed')
        result=dict(optimizer_updates=0,steps=STEPS,before=before,after=after,old_geometry_cells_verified=len(oldmap),
                    family_names=list(FAMILIES),level_control_samples=seen.cpu().tolist(),resets=resets.cpu().tolist(),
                    source_sha256=digest(source),continuous=snapshot(raw),hardware_release=False)
        (output/'FINALIZED.json').write_text(json.dumps(result,indent=2))
    return qualify


def main():
    source=ROOT/'source_migrated.pt'
    output=ROOT/'qualification'
    sys.path.insert(0,str(ROOT/'code'))
    import early_restart
    original=early_restart.configure
    def configure(cfg,profile):
        original(cfg,profile)
        from rsl_rl.runners import OnPolicyRunnerCTS
        OnPolicyRunnerCTS.learn=factory(source,output)
    early_restart.configure=configure
    argv=args(source,output,0)
    argv[argv.index('--num-envs')+1]='8192'
    sys.argv=[str(ROOT/'code/train.py'),*argv]
    from train import main as train_main
    train_main()
    if not (output/'FINALIZED.json').is_file():
        raise RuntimeError('Qualification did not finalize')


if __name__=='__main__':
    main()
