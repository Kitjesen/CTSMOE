"""From-scratch pure locomotion Lope task using the official mixed Go2 CTS recipe.

Hardware, 14S RS02 delayed actuators, adducted stance, height target, action
scales, stairs, and slopes stay Lope's. Commands use the official unscaled RLGym
ranges. Fall recovery is deferred to an independent policy/task.
"""
import os

import isaaclab.sim as sim_utils
from go2_rl_lab.tasks.mixed.mdp import rewards as go2_rewards
from go2_rl_lab.tasks.mixed.mdp.events import reset_root_state_uniform
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from lope_lab3.assets.robot import robot_cfg
from go2_rl_lab.tasks.mixed.mdp.curriculums import TerrainLevelsVelGym, gradual_reward_weight_modification
from lope_lab3.spec import (
    ACTION_DIM,
    BAD_ORIENTATION_LIMIT_RAD,
    BASE_HEIGHT_FINAL_UPDATE,
    BASE_HEIGHT_FINAL_WEIGHT,
    BASE_HEIGHT_TARGET_M,
    BASE_HEIGHT_WEIGHT,
    DECIMATION,
    EPISODE_LENGTH_S,
    HEIGHT_SCAN_X,
    HEIGHT_SCAN_Y,
    HIP_STAND_STILL_SCALE,
    ILLEGAL_CONTACT_THRESHOLD_N,
    JOINT_ACC_WEIGHT,
    JOINT_NAMES,
    JOINT_POWER_WEIGHT,
    JOINT_TORQUE_WEIGHT,
    LIN_VEL_Z_FINAL_UPDATE,
    LIN_VEL_Z_FINAL_WEIGHT,
    LIN_VEL_Z_WEIGHT,
    LOPE_PUSH_LINEAR,
    LOPE_RESET_LINEAR_VELOCITY,
    NUM_ENVS,
    OFFICIAL_PUSH_ANGULAR,
    OFFICIAL_RESET_ANGULAR_VELOCITY,
    PHYSICS_DT_S,
    SPAWN_HEIGHT_M,
    TERRAIN,
    TRACK_LIN_STD,
    TRACK_YAW_STD,
    UNDESIRED_CONTACT_THRESHOLD_N,
)
from lope_lab3.tasks.lope.commands import CommandsCfg
from lope_lab3.tasks.lope.mdp.observations import NoisyPolicyFrame, critic_frame, policy_frame
from lope_lab3.tasks.lope.mdp.randomization import configure_randomization
from lope_lab3.tasks.lope.scene_cfg import ActionsCfg, SceneCfg, actions_cfg
from lope_lab3.tasks.lope.terrain import LopeTerrainImporterCfg, terrain_cfg

LEGS = SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)
HIPS = SceneEntityCfg("robot", joint_names=[".*_hip_joint"])
THIGH_CALF = SceneEntityCfg("robot", joint_names=[".*_(thigh|calf)_joint"])
FEET = SceneEntityCfg("robot", body_names=[".*_foot_Link"], preserve_order=True)
FOOT_CONTACTS = SceneEntityCfg("contact_forces", body_names=[".*_foot_Link"], preserve_order=True)
LEG_CONTACTS = SceneEntityCfg("contact_forces", body_names=[".*_(thigh|calf)_Link"])
HEIGHT_SENSOR = SceneEntityCfg("height_scanner")


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        frame = ObservationTermCfg(func=policy_frame, history_length=10, flatten_history_dim=True)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class SingleObs(ObservationGroupCfg):
        frame = ObservationTermCfg(func=policy_frame)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Critic(ObservationGroupCfg):
        frame = ObservationTermCfg(func=critic_frame)
        enable_corruption = False
        concatenate_terms = True

    policy = Policy()
    single_obs = SingleObs()
    critic = Critic()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=lab_mdp.reset_scene_to_default, mode="reset")
    reset_root = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": LOPE_RESET_LINEAR_VELOCITY["x"],
                "y": LOPE_RESET_LINEAR_VELOCITY["y"],
                "z": LOPE_RESET_LINEAR_VELOCITY["z"],
                "roll": OFFICIAL_RESET_ANGULAR_VELOCITY["roll"],
                "pitch": OFFICIAL_RESET_ANGULAR_VELOCITY["pitch"],
                "yaw": OFFICIAL_RESET_ANGULAR_VELOCITY["yaw"],
            },
        },
    )


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=go2_rewards.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": TRACK_LIN_STD},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=go2_rewards.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": TRACK_YAW_STD},
    )
    lin_vel_z_l2 = RewardTermCfg(func=go2_rewards.lin_vel_z_l2, weight=LIN_VEL_Z_WEIGHT)
    ang_vel_xy_l2 = RewardTermCfg(func=go2_rewards.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewardTermCfg(
        func=lab_mdp.joint_torques_l2, weight=JOINT_TORQUE_WEIGHT, params={"asset_cfg": LEGS}
    )
    joint_acc_l2 = RewardTermCfg(
        func=go2_rewards.clipped_joint_acc_l2,
        weight=JOINT_ACC_WEIGHT,
        params={"asset_cfg": LEGS, "max_penalty": 1.0e8},
    )
    joint_power = RewardTermCfg(func=go2_rewards.joint_power, weight=JOINT_POWER_WEIGHT, params={"asset_cfg": LEGS})
    joint_pos_limits = RewardTermCfg(func=lab_mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": LEGS})
    action_rate_l2 = RewardTermCfg(
        func=go2_rewards.clipped_action_rate_l2, weight=-0.01, params={"max_penalty": 20.0}
    )
    action_smoothness_l2 = RewardTermCfg(
        func=go2_rewards.clipped_action_smoothness_l2, weight=-0.01, params={"max_penalty": 50.0}
    )
    base_height_l2 = RewardTermCfg(
        func=go2_rewards.base_height_l2,
        weight=BASE_HEIGHT_WEIGHT,
        params={
            "target_height": BASE_HEIGHT_TARGET_M,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": HEIGHT_SENSOR,
            "x_range": HEIGHT_SCAN_X,
            "y_range": HEIGHT_SCAN_Y,
        },
    )
    feet_regulation = RewardTermCfg(
        func=go2_rewards.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET_M,
            "asset_cfg": FEET,
            "sensor_cfg": HEIGHT_SENSOR,
            "x_range": HEIGHT_SCAN_X,
            "y_range": HEIGHT_SCAN_Y,
        },
    )
    undesired_contacts = RewardTermCfg(
        func=go2_rewards.undesired_contacts,
        weight=-1.0,
        params={"threshold": UNDESIRED_CONTACT_THRESHOLD_N, "sensor_cfg": LEG_CONTACTS},
    )
    hip_pos_penalty_l1 = RewardTermCfg(
        func=go2_rewards.joint_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": HIPS,
            "stand_still_scale": HIP_STAND_STILL_SCALE,
            "stand_cmd_idxs": [1, 2],
        },
    )
    joint_pos_penalty_l1 = RewardTermCfg(
        func=go2_rewards.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": THIGH_CALF,
            "stand_still_scale": HIP_STAND_STILL_SCALE,
        },
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
    illegal_contact = TerminationTermCfg(
        func=lab_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": ILLEGAL_CONTACT_THRESHOLD_N,
        },
    )
    bad_orientation = TerminationTermCfg(
        func=lab_mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": BAD_ORIENTATION_LIMIT_RAD,
        },
    )


@configclass
class CurriculumCfg:
    """Official terrain-level rule and the two gradual reward-weight schedules."""

    terrain_levels = CurriculumTermCfg(func=TerrainLevelsVelGym)
    base_linear_velocity_z = CurriculumTermCfg(
        func=gradual_reward_weight_modification,
        params={
            "term_name": "lin_vel_z_l2",
            "initial_weight": LIN_VEL_Z_WEIGHT,
            "final_weight": LIN_VEL_Z_FINAL_WEIGHT,
            "start_it": 0,
            "end_it": LIN_VEL_Z_FINAL_UPDATE,
        },
    )
    base_height = CurriculumTermCfg(
        func=gradual_reward_weight_modification,
        params={
            "term_name": "base_height_l2",
            "initial_weight": BASE_HEIGHT_WEIGHT,
            "final_weight": BASE_HEIGHT_FINAL_WEIGHT,
            "start_it": 0,
            "end_it": BASE_HEIGHT_FINAL_UPDATE,
        },
    )


@configclass
class LopeFreshEnvCfg(ManagerBasedRLEnvCfg):
    """12-action Lope locomotion. Do not resume a Thunder checkpoint into this task."""

    scene = SceneCfg(num_envs=NUM_ENVS, env_spacing=0.5, lazy_sensor_update=False)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    lope_history_layout: str = "frame_major_current_last"
    lope_timeout_bootstrap: str = "terminal_observation"
    lope_action_dim: int = ACTION_DIM
    lope_robot: str = "lope_rs02_12dof"
    compute_final_obs: bool = True
    lope_terrain_schema: str = "lope_mixed_go2_scaled_stairs_5_25cm_slope_35deg"
    lope_physics_schema: str = "newton_mjwarp_triangle_clean_self_col_rs02_14s_v4"

    def __post_init__(self):
        self.scene.robot = robot_cfg()
        self.scene.terrain = LopeTerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_cfg(),
            max_init_terrain_level=TERRAIN["max_init_terrain_level"],
            collision_group=-1,
            visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain/mesh"]
        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim = sim_utils.SimulationCfg(
            dt=PHYSICS_DT_S,
            render_interval=DECIMATION,
            use_newton_actuators=False,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), num_substeps=1),
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        self.sim.physics.solver_cfg.integrator = "implicitfast"
        self.sim.physics.solver_cfg.nconmax = 128
        self.sim.physics.default_shape_cfg = NewtonShapeCfg(margin=0.01)
        self.actions = actions_cfg()
        self.observations.policy.frame.func = NoisyPolicyFrame
        self.observations.single_obs.frame.func = NoisyPolicyFrame
        self.observations.policy.enable_corruption = True
        self.observations.single_obs.enable_corruption = True
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        configure_randomization(self)
        if self.rewards.base_height_l2.params["target_height"] != BASE_HEIGHT_TARGET_M:
            raise ValueError("Lope height reward must use base_height_target_m")
        if self.scene.robot.init_state.pos[2] != SPAWN_HEIGHT_M:
            raise ValueError("Lope must spawn at initial_root_height_m")


def configure_playback(cfg):
    cfg.scene.num_envs = 32
    cfg.curriculum = None
    choice = os.environ.get("LOPE_EVAL_TERRAIN", "mixed")
    if choice != "mixed":
        if choice not in cfg.scene.terrain.terrain_generator.sub_terrains:
            raise ValueError("unknown LOPE_EVAL_TERRAIN")
        selected = cfg.scene.terrain.terrain_generator.sub_terrains[choice]
        selected.proportion = 1.0
        cfg.scene.terrain.terrain_generator.sub_terrains = {choice: selected}
    level = os.environ.get("LOPE_EVAL_LEVEL")
    if level is not None:
        level = int(level)
        rows = cfg.scene.terrain.terrain_generator.num_rows
        if not 0 <= level < rows:
            raise ValueError(f"LOPE_EVAL_LEVEL must be 0..{rows - 1}")
        cfg.scene.terrain.fixed_level = level


@configclass
class LopeFreshPlayEnvCfg(LopeFreshEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_playback(self)
        self.observations.policy.enable_corruption = False
        self.observations.single_obs.enable_corruption = False
        for name in vars(self.events):
            if name.startswith("randomize_"):
                setattr(self.events, name, None)
        self.sim.physics.solver_cfg.body_com_buckets = None
