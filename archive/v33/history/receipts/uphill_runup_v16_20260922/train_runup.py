"""Explicit opt-in wrapper around frozen V15 training; no edits to the baseline runtime."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN = Path("/root/cts-early-ablation-v15-20260921/code_train")
PREFLIGHT = "--runup-preflight" in sys.argv
if PREFLIGHT:
    sys.argv.remove("--runup-preflight")
sys.path.insert(0, str(FROZEN))

import early_restart  # noqa: E402
from slope_runup import configure, snapshot  # noqa: E402

original_configure, original_snapshot = early_restart.configure, early_restart.snapshot


def configure_with_approach(cfg, profile):
    original_configure(cfg, profile)
    configure(cfg)
    if PREFLIGHT:
        from rsl_rl.runners import OnPolicyRunnerCTS
        OnPolicyRunnerCTS.learn = preflight


def snapshot_with_approach(env):
    receipt = original_snapshot(env)
    receipt["uphill_approach"] = snapshot(env)
    return receipt


def preflight(self, updates, init_at_random_ep_len=False):
    import torch
    from preflight_support import MutationGuard, state_sha256
    if updates != 0 or init_at_random_ep_len or self.env.num_envs != 4096:
        raise ValueError("Preflight requires 4096 environments, zero updates, real episode ages")
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    source = Path(sys.argv[sys.argv.index("--resume") + 1])
    def fingerprint():
        return [state_sha256(value, torch) for value in (
            self.alg.policy.state_dict(), self.alg.optimizer.state_dict(),
            self.alg.optimizer_stu_enc.state_dict())]
    before = fingerprint()
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    env = self.env.unwrapped
    obs = self.env.get_observations().to(self.device)
    self.train_mode()
    guard = MutationGuard(self)
    physical = 0
    with guard, torch.inference_mode():
        for step in range(3000):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            self.alg.process_env_step(obs, rewards, dones, extras)
            tensors = [actions, rewards, env.scene["robot"].data.root_state_w,
                       env.scene["robot"].data.joint_pos, env.scene["robot"].data.joint_vel]
            if not all(bool(torch.isfinite(t).all()) for t in tensors):
                raise FloatingPointError("Nonfinite preflight motion state")
            physical += int(env.actuator_guard.violation.sum())
            if bool(env.actuator_guard.violation_causes["torque"].any()):
                raise RuntimeError("Preflight exceeded applied torque budget")
            if (step + 1) % 24 == 0:
                self.alg.storage.clear()
                snapshot(env)
            if step % 300 == 0:
                print("RUNUP_PREFLIGHT", step, flush=True)
        self.alg.storage.clear()
    if before != fingerprint() or guard.calls or not guard.restored:
        raise RuntimeError("Preflight modified model or optimizers")
    if source_before != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Preflight modified source file")
    receipt = snapshot(env)
    if receipt["placements"] <= 0:
        raise RuntimeError("No uphill approach resets exercised")
    (out / "FINALIZED.json").write_text(json.dumps(dict(
        state="complete", optimizer_updates=0, control_steps=3000,
        completed_updates=self.current_learning_iteration,
        num_envs=4096, state_hashes_before=before, state_hashes_after=fingerprint(),
        checkpoint_sha256=source_before, approach=receipt, actuator_failure_samples=physical,
        qualification="runtime/geometry preflight; not a learned performance acceptance"), indent=2))


early_restart.configure = configure_with_approach
early_restart.snapshot = snapshot_with_approach

if __name__ == "__main__":
    from train import main
    main()
    if PREFLIGHT:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        receipt = json.loads((output / "FINALIZED.json").read_text())
        # Correct the frozen trainer's +1 convention only in this zero-update wrapper.
        result = dict(state="preflight_complete", completed_updates=receipt["completed_updates"], optimizer_updates=0)
        for name in ("SUCCESS.json", "status.json"):
            (output / name).write_text(json.dumps(result, indent=2))
