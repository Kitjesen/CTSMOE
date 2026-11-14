"""Startup randomization from Lope's flat/rough config, on Newton's CoM path.

Push disturbances and motor zero-offset are absent: SmallDogEnvCfg sets both
events to None. Wheel gain randomization is not created.
"""
from dataclasses import replace

from newton import ShapeFlags

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from go2_rl_lab.physics import BodyComBucketRandomizationCfg, RandomizedMJWarpSolverCfg

from lope_lab3.spec import JOINT_NAMES


def configure_randomization(cfg):
    cfg.events.randomize_rigid_body_mass_base = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.3),
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
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.05),
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
        params={"position_range": (0.99, 1.01), "velocity_range": (0.0, 0.0)},
    )
    # Lope's CoM box is ±1 cm in x/y and ±0.5 cm in z. Newton has to bake that
    # into the model, so the stack's bucket hook carries those ranges.
    solver = RandomizedMJWarpSolverCfg(
        class_type="go2_rl_lab.physics.resettable_mjwarp_manager:NewtonMJWarpResettableManager",
        njmax=300,
        nconmax=128,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
        update_data_interval=1,
        use_mujoco_contacts=False,
        sensor_bvh_shape_flags=int(ShapeFlags.COLLIDE_SHAPES),
        body_com_buckets=BodyComBucketRandomizationCfg(
            body_name_pattern=r"(?:^|/)Robot/Geometry/base_link$",
            x_offset_range=(-0.01, 0.01),
            y_offset_range=(-0.01, 0.01),
            z_offset_range=(-0.005, 0.005),
            num_buckets=10,
            seed=42,
        ),
    )
    cfg.sim.physics = replace(cfg.sim.physics, class_type=None, solver_cfg=solver)
