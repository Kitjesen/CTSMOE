"""Boundary adapter for Lope. It refuses Thunder checkpoints and 16-action symmetry."""
import copy
import os
import re
import uuid
from pathlib import Path

import torch
from rsl_rl.moe_cts.runner import MoeCtsRunner

from lope_lab3.spec import CRITIC_DIM, CUMULATIVE_UPDATES
from . import FRESH_FORMAT, HISTORY_LAYOUT, SYMMETRY_CLASS, UPSTREAM_COMMIT
from .checkpoint import fresh_signature, load_checkpoint, validate_for_algorithm
from .exporter import export_jit, export_onnx
from .fresh_state import capture_course, restore_course, training_contract, validate_course
from .history import validate_observations
from .timeout import terminal_timeout_reward


class LopeMoeCtsRunner(MoeCtsRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        train_cfg = copy.deepcopy(train_cfg)
        contract = train_cfg.get("lope_compat", {})
        self.critic_dim = int(contract.get("critic_dim", 0))
        self.cumulative_target = int(contract.get("cumulative_target", 0))
        if self.critic_dim != CRITIC_DIM or self.cumulative_target != CUMULATIVE_UPDATES:
            raise ValueError(f"Lope runner requires its {CRITIC_DIM}-value critic and the rough-task update cap")
        if contract.get("history_layout") != HISTORY_LAYOUT or contract.get("robot") != "lope_rs02_12dof":
            raise ValueError("Lope runner requires the 12-DoF history contract")
        symmetry = train_cfg.get("algorithm", {}).get("symmetry_cfg") or {}
        if symmetry.get("symmetry_class") != SYMMETRY_CLASS:
            raise ValueError("Lope requires its 12-action symmetry mapping")
        if "thunder" in str(symmetry.get("symmetry_class", "")).lower():
            raise ValueError("Thunder symmetry cannot train Lope")
        if train_cfg.get("robogauge", {}).get("enabled", False):
            raise ValueError("Go2 RoboGauge must stay disabled for Lope")
        if getattr(env.cfg, "lope_history_layout", None) != HISTORY_LAYOUT:
            raise ValueError("Environment must declare frame-major Lope history")
        if getattr(env.cfg, "lope_robot", None) != "lope_rs02_12dof":
            raise ValueError("Environment is not the Lope RS02 task")
        if getattr(env.cfg, "lope_action_dim", None) != 12:
            raise ValueError("Lope action width must be 12")
        if not getattr(env.cfg, "compute_final_obs", False):
            raise ValueError("compute_final_obs=True is required")
        validate_observations(env.get_observations(), self.critic_dim)
        self.completed_updates = 0
        self._checkpoint_loaded = False
        self._resume_course_pending = False
        raw_env = getattr(env, "unwrapped", env)
        self._training_contract = training_contract(raw_env)
        self._training_contract.update(
            network=copy.deepcopy(train_cfg["policy"]),
            algorithm=copy.deepcopy(train_cfg["algorithm"]),
            rollout_steps=train_cfg["num_steps_per_env"],
            clip_rewards=train_cfg.get("clip_rewards"),
        )
        self._compat_signature = fresh_signature(self.critic_dim)
        self._compat_provenance = {
            "initialization": "random",
            "robot": "lope_rs02_12dof",
            "run_id": uuid.uuid4().hex,
            "upstream_commit": UPSTREAM_COMMIT,
            "thunder_checkpoint": False,
        }
        super().__init__(env, train_cfg, log_dir, device)

    def load(self, path, load_optimizer=True, map_location=None):
        if os.environ.get("LOPE_RUN_MODE") == "play":
            load_optimizer = False
        saved = load_checkpoint(path, self.critic_dim, map_location or self.device)
        validate_for_algorithm(saved, self.alg)
        raw_env = getattr(self.env, "unwrapped", self.env)
        if load_optimizer:
            if self._training_contract != saved["training_contract"]:
                raise ValueError("Lope resume requires the saved task/asset/control/reward configuration")
            validate_course(raw_env, saved["environment_state"], saved["completed_updates"])
        infos = super()._load_checkpoint_state(saved, load_optimizer)
        self.alg.learning_rate = float(saved["learning_rate"])
        self.completed_updates = int(saved["completed_updates"])
        self.current_learning_iteration = self.completed_updates
        self._checkpoint_loaded = True
        self._resume_course_pending = bool(load_optimizer)
        self._environment_state = saved.get("environment_state")
        self._training_contract = saved["training_contract"]
        self._compat_signature = saved["signature"]
        self._compat_provenance = saved["compat_provenance"]
        return infos

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int):
            raise ValueError("Update budget must be an integer")
        if not 0 < num_learning_iterations <= self.cumulative_target - self.completed_updates:
            raise ValueError("Requested updates exceed the Lope rough-task budget or are nonpositive")
        if self._resume_course_pending:
            restore_course(getattr(self.env, "unwrapped", self.env), self._environment_state, self.completed_updates)
            self._resume_course_pending = False
            init_at_random_ep_len = False
        self.current_learning_iteration = self.completed_updates
        if self.completed_updates == 0 and self.logger.log_dir is not None:
            self.save(Path(self.logger.log_dir) / "model_0.pt")
        upstream_update = self.alg.update
        upstream_process = self.alg.process_env_step

        def counted_update():
            result = upstream_update()
            self.completed_updates += 1
            return result

        def terminal_process(obs, rewards, dones, extras):
            raw_env = getattr(self.env, "unwrapped", self.env)
            corrected, clean_extras, count = terminal_timeout_reward(
                self.alg, rewards, dones, extras, getattr(raw_env, "reset_terminated", None)
            )
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
        result = super()._checkpoint_state(infos)
        raw_env = getattr(self.env, "unwrapped", self.env)
        self._environment_state = capture_course(raw_env, self.completed_updates)
        result.update(
            compat_format=FRESH_FORMAT,
            history_layout=HISTORY_LAYOUT,
            signature=self._compat_signature,
            compat_provenance=self._compat_provenance,
            completed_updates=self.completed_updates,
            iter=self.completed_updates,
            timeout_bootstrap="terminal_observation",
            environment_state=self._environment_state,
            learning_rate=float(self.alg.learning_rate),
            upstream_commit=UPSTREAM_COMMIT,
            training_contract=self._training_contract,
        )
        return result

    def save(self, path, infos=None, *, iteration=None, final=False):
        destination = Path(path)
        if re.fullmatch(r"model_\d+\.pt", destination.name):
            destination = destination.with_name(f"model_{self.completed_updates}.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint_state(infos), temporary)
        os.replace(temporary, destination)

    def export_policy_as_jit(self, path, normalizer=None, filename="policy.pt", **kwargs):
        return export_jit(self.alg.policy, path, filename)

    def export_policy_as_onnx(self, path, normalizer=None, filename="policy.onnx", verbose=False):
        return export_onnx(self.alg.policy, path, filename, verbose)
