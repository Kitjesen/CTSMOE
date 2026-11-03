"""Append force/actuator features to the teacher and critic, never to the student."""
from dataclasses import dataclass

import torch

LEGS = ('fr', 'fl', 'rr', 'rl')
LEG_JOINTS = tuple(f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf'))
JOINTS = LEG_JOINTS + tuple(f'{leg}_foot_joint' for leg in LEGS)
WHEEL_BODIES = tuple(f'{leg}_foot_Link' for leg in LEGS)
FEATURE_NAMES = ('joint_acc_16', 'joint_applied_torque_or_proxy_16', 'wheel_normal_contact_force_magnitude_4')


@dataclass(frozen=True)
class FeatureScales:
    acceleration: float = 1e-4
    torque: float = 0.01
    contact_force: float = 0.001
    clip: float = 10.0

    def __post_init__(self):
        if not all(0 < value < float('inf') for value in vars(self).values()):
            raise ValueError('Feature scales and clip must be finite and positive')


def tensor(value):
    """Isaac Lab 2 Tensor or Isaac Lab 3 ProxyArray; no CPU copy."""
    result = value if isinstance(value, torch.Tensor) else getattr(value, 'torch', None)
    if not isinstance(result, torch.Tensor):
        raise TypeError('Expected torch Tensor or ProxyArray.torch')
    return result


def normalized_features(acceleration, torque, force_w, reset_mask, scales=FeatureScales()):
    """Force is the norm of each wheel's net normal-contact vector, not a sum of point magnitudes."""
    acceleration, torque, force_w = map(tensor, (acceleration, torque, force_w))
    n = acceleration.shape[0]
    if acceleration.shape != (n, 16) or torque.shape != (n, 16) or force_w.shape != (n, 4, 3):
        raise ValueError('Expected canonical (N,16), (N,16), (N,4,3) features')
    reset_mask = tensor(reset_mask).bool()
    if reset_mask.shape != (n,):
        raise ValueError('Reset mask must have one entry per environment')
    if not all(torch.isfinite(value).all() for value in (acceleration, torque, force_w)):
        raise ValueError('Nonfinite privileged physics signal')
    # Reset velocities are discontinuous and must not become acceleration evidence.
    acceleration = torch.where(reset_mask[:, None], 0., acceleration)
    features = torch.cat((acceleration * scales.acceleration, torque * scales.torque,
                          torch.linalg.vector_norm(force_w, dim=-1) * scales.contact_force), dim=-1)
    return features.clamp(-scales.clip, scales.clip)


def support_privileged(env, asset_cfg, sensor_cfg, scales=FeatureScales(), contact_force_field='net_normal_forces_w'):
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    ids, bodies = asset_cfg.joint_ids, sensor_cfg.body_ids
    if isinstance(ids, slice) or tuple(asset.joint_names[i] for i in ids) != JOINTS:
        raise ValueError('Teacher joint feature order must match the explicit 16-joint contract')
    if isinstance(bodies, slice) or tuple(sensor.body_names[i] for i in bodies) != WHEEL_BODIES:
        raise ValueError('Teacher contact feature order must be FR FL RR RL wheels')
    if contact_force_field not in ('net_normal_forces_w', 'net_forces_w'):
        raise ValueError('Unknown contact force field')
    return normalized_features(tensor(asset.data.joint_acc)[:, ids],
        tensor(asset.data.applied_torque)[:, ids], tensor(getattr(sensor.data, contact_force_field))[:, bodies],
        tensor(env.episode_length_buf) == 0, scales)


def attach_privileged_observations(cfg, scales=FeatureScales(), contact_force_field='net_normal_forces_w'):
    """Call before environment construction. Existing 274 critic features stay first."""
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    group = cfg.observations.critic
    if getattr(group, 'support_privileged', None) is not None:
        raise ValueError('Privileged extension is already configured')
    group.support_privileged = ObservationTermCfg(
        func=support_privileged,
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(JOINTS), preserve_order=True),
                'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(WHEEL_BODIES), preserve_order=True),
                'scales': scales, 'contact_force_field': contact_force_field},
    )
    return cfg


def require_observation_contract(obs, newton=False):
    expected = 309 if newton else 310
    if obs['policy'].shape[-1] != 53 or obs['critic'].shape[-1] != expected:
        raise ValueError(f'Expected unchanged student frame 53 and extended teacher/critic {expected}')
