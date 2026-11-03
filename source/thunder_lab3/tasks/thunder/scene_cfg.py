"""Scene sensors and wheel-leg actions shared by fresh and legacy playback tasks."""
import os
from isaaclab.envs import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from thunder_lab3.assets.contract import validate_urdf
from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS, LEG_JOINTS


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
