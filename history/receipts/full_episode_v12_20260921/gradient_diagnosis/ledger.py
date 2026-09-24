"""Static on-policy gradient ledger for the V12 zero-update diagnosis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


SAMPLES_PER_HORIZON = 1024
TEACHER_SAMPLES = 768
STUDENT_SAMPLES = 256
WINDOWS = {'startup_0_1': (0., 1.), 'sustained_1_5': (1., 5.)}
MIN_CONFLICT_ROWS = 32
MIN_STUDENT_CONFLICT_ROWS = 8
EXPECTED_COEFFICIENTS = {
    'main': 0.21093362834788681,
    'student': 0.0022221143221353456,
}


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    vector = torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ])
    if not torch.isfinite(vector).all():
        raise ValueError('non-finite static gradient')
    return vector


def norm(vector):
    return float(torch.linalg.vector_norm(vector.double()))


def pair_metrics(primary, anchor, coefficient):
    p, a = primary.double(), anchor.double()
    pn, an = torch.linalg.vector_norm(p), torch.linalg.vector_norm(a)
    dot = torch.dot(p, a)
    cosine = None if pn == 0 or an == 0 else float(dot/(pn*an))
    return {
        'primary_gradient_norm': float(pn),
        'anchor_gradient_norm': float(an),
        'anchor_coefficient': float(coefficient),
        'weighted_anchor_gradient_norm': float(an*coefficient),
        'weighted_anchor_to_primary_norm_ratio': (None if pn == 0 else float(an*coefficient/pn)),
        'dot_product': float(dot),
        'cosine': cosine,
        'primary_nonzero': bool(pn > 0),
        'anchor_nonzero': bool(an > 0),
    }


def group_mean(values, mask):
    return values[mask].mean() if bool(mask.any()) else values.sum()*0.


def action_loss(current, reference, mask, action_scales, physical_scales):
    error = (current-reference.detach())*action_scales/physical_scales
    return error[mask].square().mean() if bool(mask.any()) else current.sum()*0.


def policy_branch_results(policy, obs, actions, part, teacher):
    branch_obs = obs[part]
    policy.act(branch_obs, teacher)
    return (policy.get_actions_log_prob(actions[part]), policy.entropy,
            policy.action_mean)


def deterministic_subsample(batch, alg):
    obs = batch[0]
    native_teacher = alg.teacher_num_envs*alg.storage.num_transitions_per_env//alg.num_mini_batches
    native_student = alg.student_num_envs*alg.storage.num_transitions_per_env//alg.num_mini_batches
    if len(obs) != native_teacher+native_student:
        raise ValueError('unexpected native minibatch size')
    teacher = torch.linspace(0, native_teacher-1, TEACHER_SAMPLES, device=obs.device).long()
    student = native_teacher + torch.linspace(
        0, native_student-1, STUDENT_SAMPLES, device=obs.device).long()
    indices = torch.cat((teacher, student))
    selected = []
    for position, item in enumerate(batch):
        selected.append(item if item is None or position >= 8 else item[indices])
    return tuple(selected), native_teacher, native_student


class Accumulator:
    def __init__(self):
        self.data = {}

    def add(self, name, primary, anchor, coefficient, weight, evidence_count):
        if weight <= 0:
            return
        item = self.data.setdefault(name, {
            'primary': torch.zeros_like(primary, device='cpu', dtype=torch.float64),
            'anchor': torch.zeros_like(anchor, device='cpu', dtype=torch.float64),
            'weight': 0., 'evidence_count': 0, 'coefficient': coefficient,
        })
        item['primary'] += primary.detach().cpu().double()*weight
        item['anchor'] += anchor.detach().cpu().double()*weight
        item['weight'] += weight
        item['evidence_count'] += evidence_count

    def finish(self):
        result = {}
        for name, item in self.data.items():
            primary = item['primary']/item['weight']
            anchor = item['anchor']/item['weight']
            metrics = pair_metrics(primary, anchor, item['coefficient'])
            minimum = MIN_STUDENT_CONFLICT_ROWS if name.endswith('student_encoder') else MIN_CONFLICT_ROWS
            metrics['sample_scarcity'] = item['evidence_count'] < minimum
            metrics['conflict_evaluable'] = (item['evidence_count'] >= minimum
                                              and metrics['primary_nonzero']
                                              and metrics['anchor_nonzero'])
            metrics['negative_cosine_conflict'] = bool(
                metrics['conflict_evaluable'] and metrics['cosine'] < 0.)
            result[name] = {**metrics, 'aggregation_weight': item['weight'],
                            'conflict_evidence_rows': item['evidence_count']}
        return result


class GradientLedger:
    def __init__(self, alg, raw, output: Path):
        self.alg, self.raw, self.output = alg, raw, output
        self.rows = []
        self.accumulator = Accumulator()
        self.main_coefficient = float(alg.anchor_coefficients['main'])
        self.student_coefficient = float(alg.anchor_coefficients['student'])
        if not (math.isfinite(self.main_coefficient) and math.isfinite(self.student_coefficient)):
            raise ValueError('invalid saved V12 anchor coefficients')
        if alg.anchor_coefficients != EXPECTED_COEFFICIENTS:
            raise ValueError('diagnosis requires the fixed coefficients saved by V12')
        ppo_ids = {id(parameter) for group in alg.optimizer.param_groups for parameter in group['params']}
        student_ids = {id(parameter) for group in alg.optimizer_stu_enc.param_groups
                       for parameter in group['params']}
        main_ids = {id(parameter) for parameter in alg.policy.actor.parameters()}
        main_ids |= {id(parameter) for parameter in alg.policy.teacher_encoder.parameters()}
        encoder_ids = {id(parameter) for parameter in alg.policy.student_moe_encoder.parameters()}
        if not main_ids <= ppo_ids or encoder_ids != student_ids or ppo_ids & student_ids:
            raise ValueError('production optimizer parameter ownership changed')
        if tuple(raw.action_manager.active_terms) != ('joint_pos', 'joint_vel'):
            raise ValueError('unreviewed action channel order')
        position = raw.action_manager.get_term('joint_pos').joint_indices
        # This is the deployed IsaacLab JointAction contract: native
        # JointVelocityAction stores and consumes resolved joints as _joint_ids.
        wheels = raw.action_manager.get_term('joint_vel')._joint_ids
        if len(position) != 12 or len(wheels) != 4:
            raise ValueError('expected 12 position and 4 wheel-velocity channels')
        self.action_channels = {
            'position_joint_ids': list(position), 'wheel_joint_ids': list(wheels),
            'position_joint_names': [raw.scene['robot'].joint_names[index] for index in position],
            'wheel_joint_names': [raw.scene['robot'].joint_names[index] for index in wheels],
        }

    def analyze(self, batch, horizon):
        # These counts use every native rollout state, before the deterministic
        # gradient subset. They measure command support, not gradient weighting.
        stored = self.alg.storage.observations
        commands = stored['audit_command'].reshape(-1, 3)
        ages = stored['audit_age_s'].reshape(-1)
        groups = stored['audit_group'].reshape(-1)
        cohort_all = stored['audit_flat_cohort'].reshape(-1)
        anchor_all = stored['anchor_mask'].reshape(-1)
        forward_region = ((commands[:, 0] >= .45) & (commands[:, 0] <= .75)
                          & (commands[:, 1].abs() < .05) & (commands[:, 2].abs() < .05))
        native_support = {}
        for group_id, group_name in enumerate(('teacher', 'student')):
            group = groups == group_id
            startup = group & cohort_all & (ages >= 0.) & (ages < 1.)
            native_support[group_name] = {
                'all_rollout_rows': int(group.sum()),
                'flat_cohort_startup_rows': int(startup.sum()),
                'real_anchor_rows': int((group & anchor_all).sum()),
                'flat_cohort_startup_forward_region_rows': int((startup & forward_region).sum()),
                'flat_cohort_forward_region_rows': int((group & cohort_all & forward_region).sum()),
            }
        full_advantages = batch[3]
        if self.alg.normalize_advantage_per_mini_batch:
            full_advantages = ((full_advantages-full_advantages.mean())
                               /(full_advantages.std()+1e-8))
            batch = (*batch[:3], full_advantages, *batch[4:])
        batch, native_teacher, native_student = deterministic_subsample(batch, self.alg)
        (obs, actions, _target_values, advantages, _returns, old_log_prob,
         _old_mu, _old_sigma, _hidden, _masks) = batch
        policy = self.alg.policy
        teacher_slice = slice(0, TEACHER_SAMPLES)
        student_slice = slice(TEACHER_SAMPLES, SAMPLES_PER_HORIZON)

        teacher_log, teacher_entropy, teacher_mean = policy_branch_results(
            policy, obs, actions, teacher_slice, True)
        student_log, student_entropy, student_mean = policy_branch_results(
            policy, obs, actions, student_slice, False)
        log_prob = torch.cat((teacher_log, student_log))
        entropy = torch.cat((teacher_entropy, student_entropy))
        current_mean = torch.cat((teacher_mean, student_mean))
        ratio = torch.exp(log_prob-old_log_prob.squeeze(-1))
        adv = advantages.squeeze(-1)
        surrogate = torch.maximum(-adv*ratio, -adv*ratio.clamp(
            1.-self.alg.clip_param, 1.+self.alg.clip_param))

        from anchor_objective import deterministic_mean
        reference = deterministic_mean(self.alg.anchor_reference, obs, is_teacher=False)
        student_frozen = deterministic_mean(policy, obs[student_slice], is_teacher=False,
                                            freeze_actor=True)
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs[student_slice]))
        student_latent, gates = policy.student_moe_encoder(actor_obs)
        with torch.no_grad():
            critic_obs = policy.critic_obs_normalizer(policy.get_critic_obs(obs[student_slice]))
            teacher_latent = policy.teacher_encoder(critic_obs)
        latent_rows = (teacher_latent-student_latent).square().mean(-1)
        usage = gates.mean(0)
        load_balance = ((usage-torch.full_like(usage, 1./gates.shape[1])).square()).mean()

        main_parameters = tuple(policy.actor.parameters())+tuple(policy.teacher_encoder.parameters())
        student_parameters = tuple(policy.student_moe_encoder.parameters())
        all_mask = torch.ones(SAMPLES_PER_HORIZON, dtype=torch.bool, device=obs.device)
        group_teacher = torch.arange(SAMPLES_PER_HORIZON, device=obs.device) < TEACHER_SAMPLES
        group_student = ~group_teacher
        if (obs['audit_group'][:TEACHER_SAMPLES] != 0).any() or (
                obs['audit_group'][TEACHER_SAMPLES:] != 1).any():
            raise ValueError('native CTS teacher/student ordering changed')
        age = obs['audit_age_s'].squeeze(-1)
        cohort = obs['audit_flat_cohort'].squeeze(-1)
        moving = obs['audit_command'].norm(dim=-1) > .05
        real_anchor = obs['anchor_mask'].squeeze(-1)
        masks = {
            'global': all_mask,
            'startup_0_1': real_anchor & (age >= 0.) & (age < 1.),
            'sustained_1_5': cohort & moving & (age >= 1.) & (age < 5.),
        }
        row = {
            'horizon': horizon,
            'all_native_rollout_support': native_support,
            'native_minibatch_rows': native_teacher+native_student,
            'native_teacher_rows': native_teacher,
            'native_student_rows': native_student,
            'selected_rows': SAMPLES_PER_HORIZON,
            'selected_teacher_rows': TEACHER_SAMPLES,
            'selected_student_rows': STUDENT_SAMPLES,
            'subsample': 'systematic linspace within deterministic first native minibatch; 3:1 rows, production equal branch means',
            'windows': {},
        }
        for name, selected in masks.items():
            teacher_mask, student_mask = selected & group_teacher, selected & group_student
            anchor_selected = real_anchor if name == 'global' else selected
            anchor_teacher = anchor_selected & group_teacher
            anchor_student = anchor_selected & group_student
            ppo_loss = (group_mean(surrogate, teacher_mask)+group_mean(surrogate, student_mask)
                        - self.alg.entropy_coef*group_mean(entropy, selected))
            anchor_reference = (obs['anchor_reference_action'] if name != 'sustained_1_5'
                                else reference)
            main_anchor = .5*(
                action_loss(current_mean[:TEACHER_SAMPLES], anchor_reference[:TEACHER_SAMPLES],
                            anchor_teacher[:TEACHER_SAMPLES], self.alg.anchor_action_scales,
                            self.alg.anchor_physical_scales)
                + action_loss(current_mean[TEACHER_SAMPLES:], anchor_reference[TEACHER_SAMPLES:],
                              anchor_student[TEACHER_SAMPLES:], self.alg.anchor_action_scales,
                              self.alg.anchor_physical_scales))
            student_selected = student_mask[TEACHER_SAMPLES:]
            if bool(student_selected.any()):
                selected_usage = gates[student_selected].mean(0)
                selected_balance = ((selected_usage-torch.full_like(
                    selected_usage, 1./gates.shape[1])).square()).mean()
            else:
                selected_balance = load_balance*0.
            student_primary = (group_mean(latent_rows, student_selected)
                               + self.alg.load_balance_coef*selected_balance)
            student_anchor = action_loss(
                student_frozen, anchor_reference[TEACHER_SAMPLES:],
                anchor_student[TEACHER_SAMPLES:],
                self.alg.anchor_action_scales, self.alg.anchor_physical_scales)
            main_gradient = gradient_vector(ppo_loss, main_parameters)
            main_anchor_gradient = gradient_vector(main_anchor, main_parameters)
            student_gradient = gradient_vector(student_primary, student_parameters)
            student_anchor_gradient = gradient_vector(student_anchor, student_parameters)
            main_pair = pair_metrics(main_gradient, main_anchor_gradient, self.main_coefficient)
            student_pair = pair_metrics(student_gradient, student_anchor_gradient,
                                        self.student_coefficient)
            count = int(selected.sum())
            student_count = int(student_selected.sum())
            anchor_count = int(anchor_selected.sum())
            anchor_student_count = int(anchor_student.sum())
            main_evidence_count = min(count, anchor_count)
            student_evidence_count = min(student_count, anchor_student_count)
            main_pair['sample_scarcity'] = main_evidence_count < MIN_CONFLICT_ROWS
            main_pair['conflict_evaluable'] = (main_evidence_count >= MIN_CONFLICT_ROWS
                                                and main_pair['primary_nonzero']
                                                and main_pair['anchor_nonzero'])
            main_pair['negative_cosine_conflict'] = bool(
                main_pair['conflict_evaluable'] and main_pair['cosine'] < 0.)
            student_pair['sample_scarcity'] = student_evidence_count < MIN_STUDENT_CONFLICT_ROWS
            student_pair['conflict_evaluable'] = (
                student_evidence_count >= MIN_STUDENT_CONFLICT_ROWS
                and student_pair['primary_nonzero'] and student_pair['anchor_nonzero'])
            student_pair['negative_cosine_conflict'] = bool(
                student_pair['conflict_evaluable'] and student_pair['cosine'] < 0.)
            self.accumulator.add(f'{name}/main_actor', main_gradient, main_anchor_gradient,
                                 self.main_coefficient, SAMPLES_PER_HORIZON if name == 'global' else count,
                                 main_evidence_count)
            self.accumulator.add(f'{name}/student_encoder', student_gradient,
                                 student_anchor_gradient, self.student_coefficient,
                                 STUDENT_SAMPLES if name == 'global' else student_count,
                                 student_evidence_count)
            action_error = ((current_mean-reference)*self.alg.anchor_action_scales
                            /self.alg.anchor_physical_scales).square()
            window = {
                'anchor_semantics': ('counterfactual_not_applied_in_training' if name == 'sustained_1_5'
                                     else 'real_applied_anchor_mask'),
                'rows': count,
                'teacher_rows': int(teacher_mask.sum()),
                'student_rows': int(student_mask.sum()),
                'conflict_evidence_rows': main_evidence_count,
                'student_conflict_evidence_rows': student_evidence_count,
                'real_or_counterfactual_anchor_rows': int(anchor_selected.sum()),
                'anchor_teacher_rows': int(anchor_teacher.sum()),
                'anchor_student_rows': int(anchor_student.sum()),
                'unique_envs': int(torch.unique(obs['audit_env_id'][selected]).numel()),
                'support_valid_rows': int((obs['audit_support_count'].squeeze(-1)[selected] >= 3).sum()),
                'nonzero_advantage_rows': int((adv[selected].abs() > 1e-8).sum()),
                'command_abs_mean': (obs['audit_command'][selected].abs().mean(0).detach().cpu().tolist()
                                     if count else None),
                'command_nonzero_axis_counts': ((obs['audit_command'][selected].abs() > .05).sum(0)
                                                .detach().cpu().tolist() if count else None),
                'main_actor': main_pair,
                'student_encoder': student_pair,
                'per_action_v6_normalized_mse': (action_error[selected].mean(0).detach().cpu().tolist()
                                                  if count else None),
                'per_position_joint_abs_target_minus_position': (
                    (obs['audit_position_target'][selected]-obs['audit_position'][selected])
                    .abs().mean(0).detach().cpu().tolist() if count else None),
                'per_position_joint_abs_request_minus_position': (
                    (obs['audit_position_request'][selected]-obs['audit_position'][selected])
                    .abs().mean(0).detach().cpu().tolist() if count else None),
                'per_wheel_abs_target_minus_velocity': (
                    (obs['audit_wheel_velocity_target'][selected]
                     -obs['audit_wheel_velocity'][selected]).abs().mean(0).detach().cpu().tolist()
                    if count else None),
                'mean_supporting_wheels': (float(obs['audit_support_count'][selected].float().mean())
                                           if count else None),
                'base_contact_fraction': (float(obs['audit_base_contact'][selected].float().mean())
                                          if count else None),
            }
            row['windows'][name] = window
        self.rows.append(row)
        with (self.output/'gradient_horizons.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')

    def finish(self):
        result = {
            'protocol': 'V12 native on-policy static gradient ledger v1',
            'optimizer_steps': 0,
            'horizons': len(self.rows),
            'control_steps': len(self.rows)*24,
            'selected_rows_per_horizon': SAMPLES_PER_HORIZON,
            'teacher_student_branch_weighting': 'teacher mean + student mean, matching production PPO surrogate',
            'aggregate': self.accumulator.finish(),
            'sample_counts_are_evidence_not_loss_magnitude': True,
            'post_1s_anchor_is_counterfactual': True,
            'v11_historical_calibration_not_reused_as_v12_measurement': True,
            'action_channels': self.action_channels,
        }
        (self.output/'gradient_ledger.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
