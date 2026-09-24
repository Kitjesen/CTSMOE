"""Reuse Isaac Lab command delay with Thunder torque-speed limits."""
from isaaclab.actuators import DCMotor, DCMotorCfg, DelayedPDActuator
from isaaclab.utils import configclass


class ThunderDelayedDCMotor(DelayedPDActuator, DCMotor):
    """Delay setpoints, then run DCMotor PD and its speed-dependent torque clipping."""


@configclass
class ThunderDelayedDCMotorCfg(DCMotorCfg):
    class_type: type = ThunderDelayedDCMotor
    min_delay: int = 0
    max_delay: int = 4


class FastDelayMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from fast_delay import FastDelayBuffer
        for name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
            setattr(self, name, FastDelayBuffer(getattr(self, name)))


class FastThunderDelayedDCMotor(FastDelayMixin, ThunderDelayedDCMotor):
    """Same actuator computations and resets; only the buffer compute path differs."""


class FastWheelDelayedPDActuator(FastDelayMixin, DelayedPDActuator):
    """Wheel counterpart; preserves the independent per-actuator random delays."""
