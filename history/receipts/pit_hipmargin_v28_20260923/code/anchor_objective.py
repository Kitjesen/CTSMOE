"""Frozen-policy anchors for retaining normal flat starts during multi-task RL.

The caller owns the frozen reference lifecycle: load V6, switch it to evaluation
mode, and disable parameter gradients.  Action scales must be supplied in the
policy output order from the environment's active action terms.  Gradient-ratio
calibration is intended for the first eligible batch, not every optimizer step.
"""

import torch
from torch.func import functional_call


def normal_start_mask(terrain_is_flat, age_s, command):
    """Select finite, validated flat starts during the first episode second."""
    if terrain_is_flat.ndim != 1 or terrain_is_flat.dtype != torch.bool:
        raise ValueError("terrain_is_flat must be a boolean [N] tensor")
    if age_s.shape != terrain_is_flat.shape:
        raise ValueError("age_s must have shape [N]")
    if command.shape != (terrain_is_flat.shape[0], 3):
        raise ValueError("command must have shape [N, 3]")
    if terrain_is_flat.device != age_s.device or terrain_is_flat.device != command.device:
        raise ValueError("mask inputs must be on the same device")
    finite = torch.isfinite(age_s) & torch.isfinite(command).all(dim=-1)
    bounded = (
        (command[:, 0].abs() <= 1.2)
        & (command[:, 1].abs() <= 0.5)
        & (command[:, 2].abs() <= 0.8)
    )
    return terrain_is_flat & finite & (age_s >= 0.0) & (age_s < 1.0) & bounded


def masked_action_loss(current, reference, mask, action_scales, physical_scales):
    """Return masked MSE after converting actor means to physical units.

    ``action_scales`` converts raw actor output to physical units, while
    ``physical_scales`` defines the per-channel normalization.  Keeping these
    explicit avoids assuming where leg and wheel actions occur in the vector.
    """
    if current.ndim != 2 or current.shape[1] != 16:
        raise ValueError("current must have shape [N, 16]")
    if reference.shape != current.shape:
        raise ValueError("reference must have the same shape as current")
    if mask.shape != (current.shape[0],) or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean [N] tensor")
    if action_scales.shape != (current.shape[1],):
        raise ValueError("action_scales must have shape [16]")
    if physical_scales.shape != (current.shape[1],):
        raise ValueError("physical_scales must have shape [16]")
    tensors = (current, reference, mask, action_scales, physical_scales)
    if any(tensor.device != current.device for tensor in tensors):
        raise ValueError("loss inputs must be on the same device")
    floating = (current, reference, action_scales, physical_scales)
    if any(not tensor.is_floating_point() for tensor in floating):
        raise ValueError("action tensors and scales must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in floating):
        raise ValueError("loss inputs must contain only finite values")
    if (physical_scales <= 0.0).any():
        raise ValueError("physical_scales must be positive")
    if not mask.any():
        return current.sum() * 0.0

    normalized_error = (
        (current - reference.detach()) * action_scales / physical_scales
    )
    return normalized_error[mask].square().mean()


def deterministic_mean(
    policy,
    obs,
    is_teacher,
    detach_student=False,
    freeze_actor=False,
):
    """Compute the deterministic actor mean through the policy's native encoders."""
    single_obs = policy.single_obs_normalizer(obs["single_obs"])
    if is_teacher:
        critic_obs = policy.critic_obs_normalizer(policy.get_critic_obs(obs))
        latent = policy.teacher_encoder(critic_obs)
    else:
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
        latent, _ = policy.student_moe_encoder(actor_obs)
        if detach_student:
            latent = latent.detach()

    actor_input = torch.cat((latent, single_obs), dim=-1)
    if freeze_actor:
        detached_state = {
            name: value.detach()
            for name, value in (
                *policy.actor.named_parameters(),
                *policy.actor.named_buffers(),
            )
        }
        actor_output = functional_call(policy.actor, detached_state, (actor_input,))
    else:
        actor_output = policy.actor(actor_input)

    if policy.state_dependent_std:
        return actor_output[..., 0, :]
    return actor_output


def gradient_ratio_coefficient(
    primary_loss,
    anchor_loss,
    parameters,
    *,
    ratio=0.1,
    max_coefficient=1.0,
    statistics=None,
):
    """Calibrate an anchor coefficient from gradient norms without writing grads."""
    if primary_loss.numel() != 1 or anchor_loss.numel() != 1:
        raise ValueError("losses must be scalar tensors")
    if not torch.isfinite(primary_loss.detach()).all() or not torch.isfinite(
        anchor_loss.detach()
    ).all():
        raise ValueError("losses must be finite")
    if not torch.isfinite(torch.tensor((ratio, max_coefficient))).all():
        raise ValueError("ratio and max_coefficient must be finite")
    if ratio < 0.0 or max_coefficient < 0.0:
        raise ValueError("ratio and max_coefficient must be non-negative")

    trainable_parameters = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    if (
        not trainable_parameters
        or not primary_loss.requires_grad
        or not anchor_loss.requires_grad
    ):
        return 0.0

    primary_gradients = torch.autograd.grad(
        primary_loss,
        trainable_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    anchor_gradients = torch.autograd.grad(
        anchor_loss,
        trainable_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def gradient_norm(gradients):
        terms = [
            gradient.detach().double().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not terms:
            return None
        norm = torch.stack(terms).sum().sqrt()
        if not torch.isfinite(norm):
            raise ValueError("gradient norms must be finite")
        return norm.item()

    primary_norm = gradient_norm(primary_gradients)
    anchor_norm = gradient_norm(anchor_gradients)
    if primary_norm is None or anchor_norm is None or anchor_norm == 0.0:
        return 0.0
    coefficient = min(max_coefficient, ratio * primary_norm / anchor_norm)
    if statistics is not None:
        statistics.update(primary_gradient_norm=primary_norm, anchor_gradient_norm=anchor_norm,
                          requested_ratio=ratio, coefficient=coefficient,
                          achieved_ratio=coefficient*anchor_norm/primary_norm if primary_norm > 0 else 0.)
    return coefficient
