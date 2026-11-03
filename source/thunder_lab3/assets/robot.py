"""Thunder URDF import and actuator configuration for Newton."""
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from .contract import POSE, validate_urdf


def robot_cfg():
    asset = validate_urdf(os.environ["NEWPOSE_URDF"])
    cache = str(Path(os.environ["NEWPOSE_USD_CACHE"]).resolve())
    return ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(asset_path=asset["urdf"], usd_dir=cache,
            force_usd_conversion=True, fix_base=False, merge_fixed_joints=True,
            collision_from_visuals=False, collision_type="Convex Hull", self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
            activate_contact_sensors=True, articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)]),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,0.46),rot=(0.,0.,0.,1.),joint_pos=POSE,joint_vel={".*":0.}),
        soft_joint_pos_limit_factor=0.9,
        actuators={"legs": DCMotorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            actuator_effort_limit=120., actuator_velocity_limit=17.48, joint_effort_limit=120., joint_velocity_limit=17.48,
            saturation_effort=120., stiffness=90., damping=6.93, friction=0.),
            "wheels": ImplicitActuatorCfg(joint_names_expr=[".*_foot_joint"],joint_effort_limit=17.,
                joint_velocity_limit=44.,stiffness=0.,damping=1.,friction=0.)})
