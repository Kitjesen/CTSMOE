"""Official mixed-task randomization on Newton's CoM path.

Push linear velocity is the official +-0.4 m/s. The CoM pattern matches Lope's
``base_link`` name. There is no wheel gain event.
"""
from dataclasses import replace

from go2_rl_lab.physics import BodyComBucketRandomizationCfg, RandomizedMJWarpSolverCfg
from go2_rl_lab.tasks.mixed.mdp.events import randomize_action_joint_pos_offset
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from newton import ShapeFlags

from lope_rl_lab.assets.lope_cfg import (
    BODY_COM_OFFSET_X,
    BODY_COM_OFFSET_Y,
    BODY_COM_OFFSET_Z,
    JOINT_NAMES,
    SOLVER_NJMAX,
)
from lope_rl_lab.tasks.locomotion.config import LOPE_PUSH_LINEAR, OFFICIAL_PUSH_ANGULAR


def configure_randomization(cfg):
    cfg.events.randomize_rigid_body_mass_base = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    cfg.events.randomize_rigid_body_mass_others = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base_link$).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    cfg.events.randomize_rigid_body_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    cfg.events.randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    cfg.events.reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    cfg.events.randomize_motor_zero_offset = EventTermCfg(
        func=randomize_action_joint_pos_offset,
        mode="reset",
        params={"action_term_name": "joint_pos", "offset_range": (-0.035, 0.035)},
    )
    cfg.events.randomize_push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": {
            "x": LOPE_PUSH_LINEAR["x"],
            "y": LOPE_PUSH_LINEAR["y"],
            "roll": OFFICIAL_PUSH_ANGULAR["roll"],
            "pitch": OFFICIAL_PUSH_ANGULAR["pitch"],
            "yaw": OFFICIAL_PUSH_ANGULAR["yaw"],
        }},
    )
    # Official Newton CoM buckets. The body pattern is Lope's base_link.
    solver = RandomizedMJWarpSolverCfg(
        class_type="go2_rl_lab.physics.resettable_mjwarp_manager:NewtonMJWarpResettableManager",
        njmax=SOLVER_NJMAX,
        nconmax=128,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
        update_data_interval=1,
        use_mujoco_contacts=False,
        sensor_bvh_shape_flags=int(ShapeFlags.COLLIDE_SHAPES),
        body_com_buckets=BodyComBucketRandomizationCfg(
            body_name_pattern=r"(?:^|/)Robot/Geometry/base_link$",
            x_offset_range=BODY_COM_OFFSET_X,
            y_offset_range=BODY_COM_OFFSET_Y,
            z_offset_range=BODY_COM_OFFSET_Z,
            num_buckets=10,
            seed=42,
        ),
    )
    cfg.sim.physics = replace(cfg.sim.physics, class_type=None, solver_cfg=solver)
