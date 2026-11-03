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
