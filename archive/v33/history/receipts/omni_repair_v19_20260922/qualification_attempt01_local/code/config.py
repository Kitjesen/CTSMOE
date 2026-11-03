"""Minimal Thunder morphology adapter over the pinned upstream Go2 task."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp as lab_mdp
from doso_train.assets import DOSO_THUNDER_V4_CFG
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from actuators import ThunderDelayedDCMotorCfg
from wheel_rewards import wheel_feet_regulation

LEGS = [f"{side}_{joint}_joint" for side in ("fr", "fl", "rr", "rl") for joint in ("hip", "thigh", "calf")]
WHEELS = [f"{side}_foot_joint" for side in ("fr", "fl", "rr", "rl")]
ALL_JOINTS = LEGS + WHEELS


def adapt_entities(value, seen=None):
    """Remap upstream joint and rigid-body selectors, preserving term functions."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, SceneEntityCfg):
        for attr in ("joint_names", "body_names"):
            names = getattr(value, attr)
            if names is None:
                continue
            scalar = isinstance(names, str)
            mapped = []
            for name in ([names] if scalar else names):
                for before, after in (("FL_", "fl_"), ("FR_", "fr_"), ("RL_", "rl_"), ("RR_", "rr_")):
                    name = name.replace(before, after)
                if attr == "body_names":
                    name = name.replace("_foot", "_foot_Link").replace("_thigh", "_thigh_Link").replace("_calf", "_calf_Link")
                    if name == "base":
                        name = "base_link"
                mapped.append(name)
            setattr(value, attr, mapped[0] if scalar else mapped)
        return
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        children = vars(value).values()
    else:
        return
    for child in children:
        adapt_entities(child, seen)


def make_config(num_envs, usd_dir, fast_delay=False, actuator_profile="original"):
    cfg = Go2EnvCfg()
    adapt_entities(cfg)
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = DOSO_THUNDER_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.spawn.usd_dir = str(usd_dir)
    # Preserve Thunder asset nominal pose and PD gains, with explicit sim limits.
    for name in ("hip", "thigh", "calf"):
        cfg.scene.robot.actuators[name] = ThunderDelayedDCMotorCfg(
            joint_names_expr=[f".*_{name}_joint"], effort_limit=120.0,
            effort_limit_sim=120.0, saturation_effort=120.0,
            velocity_limit=17.48, velocity_limit_sim=17.48,
            stiffness=120.0, damping=8.0, friction=0.0,
        )
    cfg.scene.robot.actuators["wheel"] = DelayedPDActuatorCfg(
        joint_names_expr=[".*_foot_joint"], effort_limit=17.0,
        effort_limit_sim=17.0, velocity_limit=44.0, velocity_limit_sim=44.0,
        stiffness=0.0, damping=1.0, friction=0.0, min_delay=0, max_delay=4,
    )
    if fast_delay:
        from actuators import FastThunderDelayedDCMotor, FastWheelDelayedPDActuator
        for name in ("hip", "thigh", "calf"):
            cfg.scene.robot.actuators[name].class_type = FastThunderDelayedDCMotor
        cfg.scene.robot.actuators["wheel"].class_type = FastWheelDelayedPDActuator
    for sensor in (cfg.scene.height_scanner, cfg.scene.height_scanner_small):
        sensor.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.actions.joint_pos.joint_names = LEGS
    cfg.actions.joint_vel = lab_mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEELS, scale=5.0,
        use_default_offset=True, clip={".*": (-44.0, 44.0)}, preserve_order=True,
    )
    # Isaac Lab clips processed targets AFTER scaling, hence +/-44 rad/s.
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        group.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEGS, preserve_order=True)
        group.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)
    for name in ("joint_acc", "joint_torque"):
        getattr(cfg.observations.critic, name).params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)
    # Energy/acceleration penalties also include wheels; position posture terms remain leg-only.
    for name in ("joint_acc_l2", "joint_power", "joint_torques_l2"):
        getattr(cfg.rewards, name).params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)
    cfg.rewards.base_height_l2.params["target_height"] = 0.46
    cfg.rewards.feet_regulation.params["base_height_target"] = 0.46
    cfg.rewards.feet_regulation.func = wheel_feet_regulation
    cfg.rewards.feet_regulation.params["wheel_radius"] = 0.093
    if actuator_profile != "original":
        from actuator_guard import configure_guard
        configure_guard(cfg, actuator_profile)
    return cfg
