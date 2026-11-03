"""NumPy observation/target reference for this ONNX artifact; no motor I/O.

Read runtime_contract.json for the measured source configuration. This module
does not provide a realtime hardware control or emergency-stop implementation.
"""
import numpy as np

TERM_DIMS = (3, 3, 3, 12, 16, 16)
TERM_SCALES = (.25, 1., 1., 1., .05, 1.)
LEG_NAMES = tuple(f'{side}_{joint}_joint' for side in ('fr', 'fl', 'rr', 'rl')
                  for joint in ('hip', 'thigh', 'calf'))
WHEEL_NAMES = tuple(f'{side}_foot_joint' for side in ('fr', 'fl', 'rr', 'rl'))


def vector(values, count):
    out = np.asarray(values, dtype=np.float32)
    if out.shape != (count,) or not np.isfinite(out).all():
        raise ValueError(f'Expected finite vector of shape ({count},)')
    return out


def build_single(angular_velocity_body, projected_gravity_body, velocity_command,
                 leg_q, all_joint_dq, previous_action, default_leg_q):
    """All joint inputs are URDF-signed SI values in LEG_NAMES + WHEEL_NAMES.

    projected_gravity_body is unit gravity in body axes, e.g. [0,0,-1] upright.
    previous_action is the preceding normalized policy output after the wrapper
    action clip, before position scaling/soft limits/slew constraints.
    """
    terms = (vector(angular_velocity_body, 3), vector(projected_gravity_body, 3),
             vector(velocity_command, 3), vector(leg_q, 12)-vector(default_leg_q, 12),
             vector(all_joint_dq, 16), vector(previous_action, 16))
    return np.concatenate([np.clip(x, -100., 100.)*scale
                           for x, scale in zip(terms, TERM_SCALES)]).astype(np.float32)


class ObservationHistory:
    """Ten entries per term, oldest first. Reset pads with the first observation."""
    def __init__(self):
        self.frames = None

    def reset(self):
        self.frames = None

    def push(self, single):
        single = vector(single, 53)
        if self.frames is None:
            self.frames = np.repeat(single[None], 10, axis=0)
        else:
            self.frames[:-1] = self.frames[1:].copy()
            self.frames[-1] = single
        start = 0
        blocks = []
        for size in TERM_DIMS:
            blocks.append(self.frames[:, start:start+size].reshape(-1))
            start += size
        return np.concatenate(blocks)[None].astype(np.float32), single[None].copy()


class TargetReference:
    """Postprocessing reference only; target state is reset to measured leg q."""
    def __init__(self, contract, measured_leg_q):
        if contract['policy_leg_order'] != list(LEG_NAMES):
            raise ValueError('Unexpected policy joint ordering')
        physics_order = contract['physics_order']
        ids = [physics_order.index(name) for name in LEG_NAMES]
        self.default = np.asarray(contract['default_q'], dtype=np.float32)[ids]
        bounds = np.asarray(contract['soft_joint_pos_limits'], dtype=np.float32)[ids]
        self.lower, self.upper = bounds[:, 0], bounds[:, 1]
        scale = np.asarray(contract['position_scale'], dtype=np.float32)
        if scale.ndim == 2:
            if not np.array_equal(scale, np.broadcast_to(scale[:1], scale.shape)):
                raise ValueError('Per-environment position scales are not identical')
            scale = scale[0]
        self.scale = np.broadcast_to(scale, (12,))
        if self.scale.shape != (12,):
            raise ValueError('Expected twelve action position scales')
        self.clip_actions = contract['clip_actions']
        self.delta = contract['actuator_limits']['leg_target_rate_rad_s']*contract['dt']
        self.wheel_limit = contract['actuator_limits']['wheel_speed_rad_s']
        self.previous = np.clip(vector(measured_leg_q, 12), self.lower, self.upper)

    def step(self, raw):
        action = vector(raw, 16)
        if self.clip_actions is not None:
            action = np.clip(action, -self.clip_actions, self.clip_actions)
        requested = self.default + action[:12]*self.scale
        bounded = np.clip(requested, self.lower, self.upper)
        target = np.clip(self.previous + np.clip(bounded-self.previous, -self.delta, self.delta),
                         self.lower, self.upper)
        self.previous = target.copy()
        wheels = np.clip(action[12:]*5., -self.wheel_limit, self.wheel_limit)
        return dict(previous_action=action, requested_leg_q=requested, leg_q_target=target,
                    wheel_velocity_target=wheels)
