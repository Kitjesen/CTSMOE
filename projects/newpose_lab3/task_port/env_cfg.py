"""Minimal native manager task; rewards/randomization/terrain are not the 19538 training task."""
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from runtime.thunder_contract import POSE, validate_urdf
from teacher.privileged import JOINTS, LEG_JOINTS
from .observations import policy_frame, critic_frame


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


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = None
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0., history_length=3)
    height_scanner = RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/.*base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,20.)),
        pattern_cfg=patterns.GridPatternCfg(resolution=.1,size=(1.6,1.0)),ray_alignment="yaw",
        mesh_prim_paths=["/World/ground"], global_world_only=True, max_distance=100., update_period=0.02)


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        frame = ObservationTermCfg(func=policy_frame)
        enable_corruption = False
        concatenate_terms = True
    @configclass
    class Critic(ObservationGroupCfg):
        frame = ObservationTermCfg(func=critic_frame)
        enable_corruption = False
        concatenate_terms = True
    policy = Policy()
    critic = Critic()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",joint_names=list(LEG_JOINTS),preserve_order=True,
        scale={".*_hip_joint":.125,"^(?!.*_hip_joint).*":.25},use_default_offset=True,clip={".*":(-100.,100.)})
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot",joint_names=list(JOINTS[12:]),preserve_order=True,
        scale=5.,use_default_offset=True,clip={".*":(-44.,44.)})


def fresh_actions_cfg():
    """Bound fresh-policy leg targets by URDF limits after scale and default-pose offset."""
    actions = ActionsCfg()
    actions.joint_pos.clip = validate_urdf(os.environ["NEWPOSE_URDF"])["joint_position_limits"]
    return actions


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(asset_name="robot",resampling_time_range=(20.,20.),
        rel_standing_envs=0.,heading_command=False,debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.,0.),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default,mode="reset")


@configclass
class RewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive,weight=1.)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out,time_out=True)


@configclass
class ThunderCompatibilityCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=4,env_spacing=3.,lazy_sensor_update=False)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = None
    def __post_init__(self):
        self.scene.robot = robot_cfg()
        self.decimation = 8
        self.episode_length_s = 20.
        self.sim = sim_utils.SimulationCfg(dt=.0025,render_interval=8,use_newton_actuators=False,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),num_substeps=1))
