"""Sensors and the 12-DoF position action. There is no wheel velocity term."""
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from lope_lab3.assets.contract import validate_urdf
from lope_lab3.assets.robot import urdf_path
from lope_lab3.spec import ACTION_SCALE_RAD, JOINT_NAMES, action_clips


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=3, track_air_time=True
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        max_distance=100.0,
        update_period=0.02,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(JOINT_NAMES),
        preserve_order=True,
        scale={
            ".*_hip_joint": ACTION_SCALE_RAD["hip"],
            ".*_thigh_joint": ACTION_SCALE_RAD["thigh"],
            ".*_calf_joint": ACTION_SCALE_RAD["calf"],
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


def actions_cfg():
    """Clip position targets to Lope's 0.9 soft window around each URDF interval."""
    limits = validate_urdf(urdf_path())["joint_position_limits"]
    actions = ActionsCfg()
    actions.joint_pos.clip = action_clips(limits)
    return actions
