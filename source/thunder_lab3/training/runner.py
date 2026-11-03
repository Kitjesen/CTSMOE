"""Small boundary adapter around the unmodified upstream MoeCtsRunner."""

import copy
import os
import re
import uuid
from pathlib import Path

import torch
from rsl_rl.moe_cts.runner import MoeCtsRunner

from . import HISTORY_LAYOUT, UPSTREAM_COMMIT
from .checkpoint import FORMAT, FRESH_FORMAT, fresh_signature, load_checkpoint, validate_for_algorithm
from .exporter import export_jit, export_onnx
from .history import validate_observations
from .timeout import terminal_timeout_reward
from thunder_lab3.legacy.environment_state import describe_state, require_formal_continuation
from .fresh_state import capture_course, restore_course, training_contract, validate_course


class ThunderMoeCtsRunner(MoeCtsRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        train_cfg = copy.deepcopy(train_cfg)  # upstream consumes class_name keys
        contract = train_cfg.get("newpose_compat", {})
        self.critic_dim = int(contract.get("critic_dim", 309))
        self.cumulative_target = int(contract.get("cumulative_target", 20000))
        if self.cumulative_target not in (20000, 100000) or contract.get("history_layout") != HISTORY_LAYOUT:
            raise ValueError("Explicit Thunder history layout and supported cumulative target are required")
        symmetry = train_cfg.get("algorithm", {}).get("symmetry_cfg") or {}
        if symmetry.get("use_symmetric_augmentation", False) and symmetry.get("symmetry_class") != (
                "thunder_lab3.tasks.thunder.mdp.symmetry:ThunderSymmetry"):
            raise ValueError("Thunder requires its own 16-action symmetry mapping")
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
        self._native_run = getattr(env.cfg, "newpose_physics_schema", None) == "newton_mjwarp_triangle_v1"
        self._optimizer_loaded = self._native_run
        self._resume_course_pending = False
        self._compat_signature = None
        self._compat_provenance = None
        self._environment_state = None
        raw_env = getattr(env, "unwrapped", env)
        self._training_contract = training_contract(raw_env) if self._native_run else None
        if self._native_run:
            self._training_contract.update(network=copy.deepcopy(train_cfg["policy"]),
                                           algorithm=copy.deepcopy(train_cfg["algorithm"]),
                                           rollout_steps=train_cfg["num_steps_per_env"],
                                           clip_rewards=train_cfg.get("clip_rewards"),
                                           clip_actions=train_cfg.get("clip_actions"))
            self._training_contract['cumulative_target'] = self.cumulative_target
            self._compat_signature = fresh_signature(self.critic_dim)
            self._compat_provenance = {"initialization": "random", "run_id": uuid.uuid4().hex,
                                       "upstream_commit": UPSTREAM_COMMIT}
            self.environment_resume_status = {"mode": "fresh_random_initialization", "start_update": 0}
        super().__init__(env, train_cfg, log_dir, device)

    def load(self, path, load_optimizer=True, map_location=None):
        # The upstream play entry loads with its default True; playback never resumes a course.
        if os.environ.get("NEWPOSE_RUN_MODE") == "play":
            load_optimizer = False
        saved = load_checkpoint(path, self.critic_dim, map_location or self.device)
        validate_for_algorithm(saved, self.alg)
        native = saved["compat_format"] == FRESH_FORMAT
        if native != self._native_run:
            raise ValueError("Checkpoint lineage and task differ: select the fresh task, or --legacy for 19538 playback")
        raw_env = getattr(self.env, "unwrapped", self.env)
        if native and load_optimizer:
            if self._training_contract != saved["training_contract"]:
                raise ValueError("Native resume requires the saved task/asset/control/reward configuration")
            validate_course(raw_env, saved["environment_state"], saved["completed_updates"])
        infos = super()._load_checkpoint_state(saved, load_optimizer)
        # The upstream loader restores Adam lr but not alg.learning_rate.
        self.alg.learning_rate = float(saved["learning_rate"])
        self.completed_updates = int(saved["completed_updates"])
        self.current_learning_iteration = self.completed_updates
        self._checkpoint_loaded = True
        self._optimizer_loaded = bool(load_optimizer)
        self._native_run = native
        self._compat_signature = saved["signature"]
        self._compat_provenance = saved["compat_provenance"]
        self._environment_state = saved.get("environment_state")
        if native:
            self._training_contract = saved["training_contract"]
            self._resume_course_pending = bool(load_optimizer)
            self.environment_resume_status = {"mode": "native_resume_pending" if load_optimizer else "diagnostic_fresh_course",
                                              "course_restored": False, "physics_state_restored": False}
        else:
            self.environment_resume_status = describe_state(self._environment_state, optimizer_loaded=load_optimizer)
        print("NEWPOSE_ENVIRONMENT_RESUME", self.environment_resume_status, flush=True)
        return infos

    def _validate_formal_continuation(self):
        require_formal_continuation(self)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self._optimizer_loaded or (not self._native_run and not self._checkpoint_loaded):
            raise ValueError("Learning requires fresh native initialization or a verified resumable checkpoint")
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int):
            raise ValueError("Update budget must be an integer")
        if not 0 < num_learning_iterations <= self.cumulative_target - self.completed_updates:
            raise ValueError("Requested updates exceed remaining cumulative budget or are nonpositive")
        if self._native_run:
            if self._resume_course_pending:
                # Upstream train.py rewrites counters with iter+1 after load; restore at this boundary.
                self.environment_resume_status = restore_course(
                    getattr(self.env, "unwrapped", self.env), self._environment_state, self.completed_updates)
                self._resume_course_pending = False
                init_at_random_ep_len = False
        else:
            self._validate_formal_continuation()
        self.current_learning_iteration = self.completed_updates
        if self._native_run and self.completed_updates == 0 and self.logger.log_dir is not None:
            self.save(Path(self.logger.log_dir) / "model_0.pt")
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
        if not self._optimizer_loaded or (not self._native_run and not self._checkpoint_loaded):
            raise ValueError("A training checkpoint requires initialized or restored model and both optimizers")
        result = super()._checkpoint_state(infos)
        if self._native_run:
            self._environment_state = capture_course(getattr(self.env, "unwrapped", self.env), self.completed_updates)
        result.update(compat_format=FRESH_FORMAT if self._native_run else FORMAT, history_layout=HISTORY_LAYOUT,
                      signature=self._compat_signature, compat_provenance=self._compat_provenance,
                      completed_updates=self.completed_updates, iter=self.completed_updates,
                      timeout_bootstrap="terminal_observation",
                      environment_state=self._environment_state,
                      environment_resume_status=self.environment_resume_status,
                      learning_rate=float(self.alg.learning_rate), upstream_commit=UPSTREAM_COMMIT)
        if self._native_run:
            result["training_contract"] = self._training_contract
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
