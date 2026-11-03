"""Small boundary adapter around the unmodified upstream MoeCtsRunner."""

import copy
import os
import re
from pathlib import Path

import torch
from rsl_rl.moe_cts.runner import MoeCtsRunner

from . import HISTORY_LAYOUT, UPSTREAM_COMMIT
from .checkpoint import FORMAT, load_checkpoint, validate_for_algorithm
from .exporter import export_jit, export_onnx
from .history import validate_observations
from .timeout import terminal_timeout_reward
from .environment_state import describe_state, require_formal_continuation


class ThunderMoeCtsRunner(MoeCtsRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        train_cfg = copy.deepcopy(train_cfg)  # upstream consumes class_name keys
        contract = train_cfg.get("newpose_compat", {})
        self.critic_dim = int(contract.get("critic_dim", 309))
        self.cumulative_target = int(contract.get("cumulative_target", 20000))
        if self.cumulative_target != 20000 or contract.get("history_layout") != HISTORY_LAYOUT:
            raise ValueError("Explicit NEWPOSE history layout and cumulative target 20000 are required")
        if contract.get("symmetry_enabled", False):
            raise ValueError("Thunder dynamics are not proven mirror-equivalent")
        if (train_cfg.get("algorithm", {}).get("symmetry_cfg") or {}).get("use_symmetric_augmentation", False):
            raise ValueError("Go2 mirrored PPO samples are not a verified Thunder augmentation")
        if train_cfg.get("robogauge", {}).get("enabled", False):
            raise ValueError("Go2 RoboGauge export/task must be disabled for Thunder")
        declared_layout = getattr(env.cfg, "newpose_history_layout", None)
        if declared_layout != HISTORY_LAYOUT:
            raise ValueError("Environment must explicitly declare frame-major Thunder history")
        if getattr(env.cfg, "newpose_timeout_bootstrap", None) != "terminal_observation":
            raise ValueError("Environment must declare terminal-observation timeout bootstrapping")
        if not getattr(env.cfg, "compute_final_obs", False):
            raise ValueError("compute_final_obs=True is required for the old NEWPOSE timeout semantics")
        if getattr(env.cfg, "is_finite_horizon", False):
            raise ValueError("Old NEWPOSE uses time-limit bootstrapping, not a finite-horizon terminal value")
        validate_observations(env.get_observations(), self.critic_dim)
        self.completed_updates = 0
        self._checkpoint_loaded = False
        self._compat_signature = None
        self._compat_provenance = None
        self._environment_state = None
        super().__init__(env, train_cfg, log_dir, device)

    def load(self, path, load_optimizer=True, map_location=None):
        saved = load_checkpoint(path, self.critic_dim, map_location or self.device)
        validate_for_algorithm(saved, self.alg)
        infos = super()._load_checkpoint_state(saved, load_optimizer)
        # The upstream loader restores Adam lr but not alg.learning_rate.
        self.alg.learning_rate = float(saved["learning_rate"])
        self.completed_updates = int(saved["completed_updates"])
        self.current_learning_iteration = self.completed_updates
        self._checkpoint_loaded = True
        self._optimizer_loaded = bool(load_optimizer)
        self._compat_signature = saved["signature"]
        self._compat_provenance = saved["compat_provenance"]
        self._environment_state = saved.get("environment_state")
        self.environment_resume_status = describe_state(self._environment_state, optimizer_loaded=load_optimizer)
        print("NEWPOSE_ENVIRONMENT_RESUME", self.environment_resume_status, flush=True)
        return infos

    def _validate_formal_continuation(self):
        require_formal_continuation(self)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self._checkpoint_loaded or not self._optimizer_loaded:
            raise ValueError("Continuation requires a verified checkpoint and both optimizers")
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int):
            raise ValueError("Update budget must be an integer")
        if not 0 < num_learning_iterations <= self.cumulative_target - self.completed_updates:
            raise ValueError("Requested updates exceed remaining cumulative budget or are nonpositive")
        self._validate_formal_continuation()
        self.current_learning_iteration = self.completed_updates
        upstream_update = self.alg.update
        upstream_process = self.alg.process_env_step

        def counted_update():
            result = upstream_update()
            self.completed_updates += 1
            return result

        def terminal_process(obs, rewards, dones, extras):
            raw_env = getattr(self.env, "unwrapped", self.env)
            terminated = getattr(raw_env, "reset_terminated", None)
            corrected, clean_extras, count = terminal_timeout_reward(
                self.alg, rewards, dones, extras, terminated)
            self.terminal_bootstraps += count
            return upstream_process(obs, corrected, dones, clean_extras)

        self.terminal_bootstraps = 0
        self.alg.update = counted_update
        self.alg.process_env_step = terminal_process
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.update = upstream_update
            self.alg.process_env_step = upstream_process
            self.current_learning_iteration = self.completed_updates

    def _checkpoint_state(self, infos):
        if not self._checkpoint_loaded or not self._optimizer_loaded:
            raise ValueError("A continuation checkpoint requires the verified model and both restored optimizers")
        result = super()._checkpoint_state(infos)
        result.update(compat_format=FORMAT, history_layout=HISTORY_LAYOUT,
                      signature=self._compat_signature, compat_provenance=self._compat_provenance,
                      completed_updates=self.completed_updates, iter=self.completed_updates,
                      timeout_bootstrap="terminal_observation",
                      environment_state=self._environment_state,
                      environment_resume_status=self.environment_resume_status,
                      learning_rate=float(self.alg.learning_rate), upstream_commit=UPSTREAM_COMMIT)
        return result

    def save(self, path, infos=None, *, iteration=None, final=False):
        # Upstream names files by zero-based it; label by successful updates here.
        destination = Path(path)
        if re.fullmatch(r"model_\d+\.pt", destination.name):
            destination = destination.with_name(f"model_{self.completed_updates}.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint_state(infos), temporary)
        os.replace(temporary, destination)
        self._after_save(self.completed_updates, final)

    def export_policy_as_jit(self, path, normalizer=None, filename="policy.pt", **kwargs):
        if normalizer is not None and normalizer is not self.alg.policy.actor_obs_normalizer:
            raise ValueError("A substituted normalizer changes the frozen deployment contract")
        return export_jit(self.alg.policy, path, filename)

    def export_policy_as_onnx(self, path, normalizer=None, filename="policy.onnx", verbose=False):
        if normalizer is not None and normalizer is not self.alg.policy.actor_obs_normalizer:
            raise ValueError("A substituted normalizer changes the frozen deployment contract")
        return export_onnx(self.alg.policy, path, filename, verbose)
