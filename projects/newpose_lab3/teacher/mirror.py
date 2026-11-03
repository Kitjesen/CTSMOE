"""Sagittal reflection and optional student-only consistency; no mirrored PPO transitions."""
import torch

from .privileged import JOINTS

PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14)


class ThunderMirror:
    def __init__(self, default_leg_pos, action_scale, joint_names=JOINTS):
        if tuple(joint_names) != JOINTS:
            raise ValueError('Expected canonical FR FL RR RL leg joints, then four wheels')
        self.default = torch.as_tensor(default_leg_pos).detach().clone()
        self.scale = torch.as_tensor(action_scale).detach().clone()
        if self.default.shape != (12,) or self.scale.shape != (16,):
            raise ValueError('Expected 12 default leg positions and 16 action scales')
        if not torch.isfinite(self.default).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('Finite default positions and positive finite action scales required')

    def positions(self, relative):
        if relative.shape[-1] != 12:
            raise ValueError('Expected 12 relative leg positions')
        default = self.default.to(relative)
        absolute = relative + default
        return -absolute[..., list(PERM[:12])] - default

    def joint_vectors(self, value):
        if value.shape[-1] != 16:
            raise ValueError('Expected 16 joint velocities, accelerations or torques')
        return -value[..., list(PERM)]

    def actions(self, value):
        if value.shape[-1] != 16:
            raise ValueError('Expected 12 leg target residuals and four wheel-speed actions')
        scale = self.scale.to(value)
        target = value * scale
        positions = self.positions(target[..., :12])
        wheel_speeds = -target[..., [13, 12, 15, 14]]
        return torch.cat((positions, wheel_speeds), dim=-1) / scale

    def frame(self, value):
        if value.shape[-1] != 53:
            raise ValueError('Expected 53-dimensional Thunder student frame')
        return torch.cat((value[..., :3] * value.new_tensor((-1, 1, -1)),
                          value[..., 3:6] * value.new_tensor((1, -1, 1)),
                          value[..., 6:9] * value.new_tensor((1, -1, -1)),
                          self.positions(value[..., 9:21]), self.joint_vectors(value[..., 21:37]),
                          self.actions(value[..., 37:53])), dim=-1)

    def history(self, value):
        if value.shape[-1] != 530:
            raise ValueError('Expected frame-major, oldest-first history of ten 53-D frames')
        return self.frame(value.reshape(*value.shape[:-1], 10, 53)).reshape_as(value)


def student_mirror_loss(student_encoder, actor, history, mapper, weight=0.0):
    """Soft prior on student outputs; actor parameters are constants and original action is detached.

    Add to the student distillation loss only. This does not assume mirrored physical
    transitions have the same returns. Asymmetric robot dynamics can make this prior harmful.
    """
    if not 0 <= weight < float('inf'):
        raise ValueError('Mirror consistency weight must be finite and nonnegative')
    if weight == 0:
        return history.new_zeros(())
    state = {name: value.detach() for name, value in actor.named_parameters()}
    state.update({name: value.detach() for name, value in actor.named_buffers()})
    def actions(obs):
        encoded = student_encoder(obs)
        latent = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
        return torch.func.functional_call(actor, state, (torch.cat((latent, obs[..., -53:]), -1),))
    with torch.no_grad():
        target = mapper.actions(actions(history))
    prediction = actions(mapper.history(history))
    return weight * (prediction - target).square().mean()


def reflected_height_indices(ray_starts, tolerance=1e-6):
    """Derive scan ordering from actual ray coordinates; never guess reshape ordering."""
    if ray_starts.ndim != 2 or ray_starts.shape[-1] != 3 or not torch.isfinite(ray_starts).all():
        raise ValueError('Expected finite local-frame ray starts (N,3)')
    reflected = ray_starts * ray_starts.new_tensor((1, -1, 1))
    distances = torch.cdist(reflected, ray_starts, compute_mode='donot_use_mm_for_euclid_dist')
    error, indices = distances.min(dim=1)
    if (error > tolerance).any() or len(indices.unique()) != len(indices):
        raise ValueError('Ray grid is not bijectively symmetric under sagittal reflection')
    return indices
