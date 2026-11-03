"""Fresh-run Go2 randomization, adapted to Thunder's mixed actuator ownership."""

from dataclasses import replace

from newton import ShapeFlags

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from go2_rl_lab.physics import BodyComBucketRandomizationCfg, RandomizedMJWarpSolverCfg
from go2_rl_lab.tasks.mixed.mdp.events import randomize_action_joint_pos_offset

from .privileged import LEG_JOINTS, JOINTS


def configure_randomization(cfg):
    """Add startup/reset/interval events without replacing the task's reset events."""
    cfg.events.randomize_rigid_body_mass_base = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # Relative uncertainty scales with Thunder's 16.868 kg base, not Go2's base mass.
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINTS), preserve_order=True),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Lab recommends initialization-only writes for implicit actuator gains.
    cfg.events.randomize_wheel_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINTS[12:]), preserve_order=True),
            "stiffness_distribution_params": None,
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
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
            "x": (-0.4, 0.4), "y": (-0.4, 0.4),
            "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6),
        }},
    )
    # CoM changes must precede model finalization so MJWarp builds the correct mass matrix.
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
            x_offset_range=(0.0, 0.1),
            y_offset_range=(-0.05, 0.05),
            z_offset_range=(0.0, 0.05),
            num_buckets=10,
            seed=42,
        ),
    )
    # NewtonCfg derives manager dispatch only in __post_init__; replacing solver_cfg alone skips CoM hooks.
    cfg.sim.physics = replace(cfg.sim.physics, class_type=None, solver_cfg=solver)

