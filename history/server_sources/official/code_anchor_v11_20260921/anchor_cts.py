# CTS update derived from ETH Zurich/NVIDIA rsl_rl, BSD-3-Clause.
# Upstream sha256: 136a1c1082243ae1c6654307ffcf32de2a99cf6c4d6581419ada59f288a058f6
"""Training-only normal-start action anchoring; deployment remains one student."""
import copy
import hashlib
import itertools
import math

import torch
from torch import nn
from rsl_rl.algorithms import MoECTS

from anchor_objective import (
    deterministic_mean, gradient_ratio_coefficient, masked_action_loss, normal_start_mask,
)

REFERENCE_SHA256 = 'bdde0ca02d7e0d9db9ecdfd8c9b352cc1f511db49a9af936cc14c8046f6a77f8'


class AnchoredMoECTS(MoECTS):
    def attach_anchor(self, env, reference_path, saved_anchor=None):
        reference_path = reference_path.resolve(strict=True)
        if self.is_multi_gpu or self.rnd is not None:
            raise ValueError('This bounded anchor experiment requires single GPU and no RND')
        if self.policy.state_dependent_std:
            raise ValueError('Anchor action layout requires state-independent standard deviation')
        self.anchor_env = env
        self.anchor_reference_path = str(reference_path)
        self.anchor_reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if self.anchor_reference_hash != REFERENCE_SHA256:
            raise ValueError('Reference is not the exact V6 model used in the recovery probe')
        self.anchor_reference = copy.deepcopy(self.policy)
        reference = torch.load(reference_path, map_location=self.device, weights_only=False)
        if reference['completed_updates'] != 43202:
            raise ValueError('Reference must be the audited V6 43202 checkpoint')
        self.anchor_reference.load_state_dict(reference['model_state_dict'], strict=True)
        self.anchor_reference.eval().requires_grad_(False)
        if self.policy.actor_obs_normalization or self.policy.critic_obs_normalization:
            raise ValueError('This experiment has only validated disabled observation normalization')
        self.anchor_coefficients = dict(main=None, student=None)
        self.anchor_calibration = dict(main={}, student={})
        if saved_anchor is not None:
            if saved_anchor['profile'] != 'v6_normal_start_action_v1':
                raise ValueError('Unreviewed startup anchor profile on resume')
            if saved_anchor['reference_sha256'] != self.anchor_reference_hash:
                raise ValueError('Frozen reference changed on resume')
            self.anchor_coefficients = dict(saved_anchor['coefficients'])
            if set(self.anchor_coefficients) != {'main', 'student'}:
                raise ValueError('Invalid anchor coefficient schema')
            for value in self.anchor_coefficients.values():
                if value is not None and (type(value) not in (int, float) or
                        not math.isfinite(value) or not 0 < value <= 1):
                    raise ValueError('Anchor coefficients must be None or finite values in (0,1]')
            self.anchor_calibration = copy.deepcopy(saved_anchor['calibration'])
        action_scales = []
        physical_scales = []
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            scale = torch.as_tensor(term._scale, device=self.device).expand(env.num_envs, term.action_dim)[0]
            action_scales.append(scale)
            if name == 'joint_pos':
                physical_scales.append(torch.full_like(scale, .25))
            elif name == 'joint_vel':
                physical_scales.append(torch.full_like(scale, 5.))
            else:
                raise ValueError(f'Unreviewed action term: {name}')
        self.anchor_action_scales = torch.cat(action_scales)
        self.anchor_physical_scales = torch.cat(physical_scales)
        shape = (self.storage.num_transitions_per_env, env.num_envs)
        # These fields travel with TensorDict observations through the existing
        # teacher/student permutation. They are excluded from network obs_groups.
        self.storage.observations['anchor_mask'] = torch.zeros(*shape, 1, device=self.device, dtype=torch.bool)
        self.storage.observations['anchor_reference_action'] = torch.zeros(*shape, 16, device=self.device)
        self.anchor_sample_counts = dict(teacher=0, student=0)
        self.anchor_metrics = dict(teacher=0., student=0.)
        from startup_repair import learner_cohort_mask
        learner = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        learner[self.student_env_idxs] = 1
        self.anchor_cohort = learner_cohort_mask(learner)

    def anchor_state(self):
        return dict(profile='v6_normal_start_action_v1', reference=self.anchor_reference_path,
                    reference_sha256=self.anchor_reference_hash, coefficients=self.anchor_coefficients,
                    calibration=self.anchor_calibration, sample_counter_scope='current process only',
                    startup_seconds=1., command_abs_limits=[1.2, .5, .8],
                    mask='default-reset startup cohort on flat, first1s; shared weights affect all terrains',
                    samples=self.anchor_sample_counts)

    def act(self, obs):
        env = self.anchor_env
        command = env.command_manager.get_term('base_velocity')
        mask = normal_start_mask(command.terrain_idxs == command.terrain_type2idx['flat'],
                                 env.episode_length_buf * env.step_dt, command.command)
        mask &= self.anchor_cohort
        target = torch.zeros(env.num_envs, 16, device=self.device)
        if mask.any():
            with torch.no_grad():
                target[mask] = deterministic_mean(self.anchor_reference, obs[mask], is_teacher=False)
        augmented = obs.clone(recurse=False)
        augmented['anchor_mask'] = mask[:, None]
        augmented['anchor_reference_action'] = target
        for name, ids in (('teacher', self.teacher_env_idxs), ('student', self.student_env_idxs)):
            self.anchor_sample_counts[name] += int(mask[ids].sum())
        return super().act(augmented)

    def main_anchor_loss(self, obs, teacher_samples):
        teacher = deterministic_mean(self.policy, obs[:teacher_samples], is_teacher=True)
        student = deterministic_mean(self.policy, obs[teacher_samples:], is_teacher=False, detach_student=True)
        losses = []
        for current, part in ((teacher, obs[:teacher_samples]), (student, obs[teacher_samples:])):
            losses.append(masked_action_loss(current, part['anchor_reference_action'],
                part['anchor_mask'].squeeze(-1), self.anchor_action_scales, self.anchor_physical_scales))
        for key, loss in zip(('teacher', 'student'), losses):
            self.anchor_metrics[key] += loss.item()
        return .5 * (losses[0] + losses[1])

    def student_anchor_loss(self, obs):
        current = deterministic_mean(self.policy, obs, is_teacher=False, freeze_actor=True)
        return masked_action_loss(current, obs['anchor_reference_action'], obs['anchor_mask'].squeeze(-1),
                                  self.anchor_action_scales, self.anchor_physical_scales)

    def update(self) -> dict[str, float]:
        self.anchor_metrics = dict(teacher=0., student=0.)
        mean_anchor_main = 0.0
        mean_anchor_student = 0.0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_loss = 0
        mean_load_balance_loss = 0
        mean_rnd_loss = 0 if self.rnd else None

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        data = list(generator)

        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in data:
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
 
            def _get_results(start, end, is_teacher):
                self.policy.act(obs_batch[start:end], is_teacher)
                actions_log_prob = self.policy.get_actions_log_prob(actions_batch[start:end])
                value = self.policy.evaluate(obs_batch[start:end], is_teacher)
                mu = self.policy.action_mean
                sigma = self.policy.action_std
                entropy = self.policy.entropy
                return actions_log_prob, value, mu, sigma, entropy
            teacher_results = _get_results(0, teacher_samples, is_teacher=True)
            student_results = _get_results(teacher_samples, teacher_samples + student_samples, is_teacher=False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))
            actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch = results

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_losses[:teacher_samples].mean()
            student_surrogate_loss = surrogate_losses[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            anchor_loss = self.main_anchor_loss(obs_batch, teacher_samples)
            if self.anchor_coefficients['main'] is None and obs_batch['anchor_mask'].any():
                candidate = gradient_ratio_coefficient(
                    loss, anchor_loss, tuple(self.policy.actor.parameters()) +
                    tuple(self.policy.teacher_encoder.parameters()), ratio=.1,
                    statistics=self.anchor_calibration['main'])
                if candidate > 0:
                    self.anchor_coefficients['main'] = candidate
            loss = loss + (self.anchor_coefficients['main'] or 0.) * anchor_loss
            mean_anchor_main += anchor_loss.item()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            params_to_clip = itertools.chain.from_iterable(g['params'] for g in self.optimizer.param_groups)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in data:
            obs_a_batch = self.policy.get_actor_obs(obs_batch)
            obs_a_batch = self.policy.actor_obs_normalizer(obs_a_batch)
            student_latent, gating_weights = self.policy.student_moe_encoder(obs_a_batch[teacher_samples:])
            with torch.no_grad():
                obs_c_batch = self.policy.get_critic_obs(obs_batch)
                obs_c_batch = self.policy.critic_obs_normalizer(obs_c_batch)
                teacher_latent = self.policy.teacher_encoder(obs_c_batch[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()

            mean_usage = torch.mean(gating_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))
            student_loss = latent_loss + self.load_balance_coef * load_balance_loss
            
            student_anchor = self.student_anchor_loss(obs_batch[teacher_samples:])
            if self.anchor_coefficients['student'] is None and obs_batch['anchor_mask'][teacher_samples:].any():
                candidate = gradient_ratio_coefficient(
                    student_loss, student_anchor, tuple(self.policy.student_moe_encoder.parameters()), ratio=.1,
                    statistics=self.anchor_calibration['student'])
                if candidate > 0:
                    self.anchor_coefficients['student'] = candidate
            student_loss = student_loss + (self.anchor_coefficients['student'] or 0.) * student_anchor
            mean_anchor_student += student_anchor.item()
            self.optimizer_stu_enc.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer_stu_enc.step()

            mean_latent_loss += latent_loss.item()
            mean_load_balance_loss += load_balance_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_loss /= num_updates
        mean_load_balance_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "mean_latent_loss": mean_latent_loss,
            "mean_load_balance_loss": mean_load_balance_loss
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        loss_dict.update(anchor_main=mean_anchor_main/num_updates,
                         anchor_main_teacher=self.anchor_metrics['teacher']/num_updates,
                         anchor_main_student=self.anchor_metrics['student']/num_updates,
                         anchor_student=mean_anchor_student/num_updates,
                         anchor_main_coefficient=self.anchor_coefficients['main'] or 0.,
                         anchor_student_coefficient=self.anchor_coefficients['student'] or 0.)
        return loss_dict
