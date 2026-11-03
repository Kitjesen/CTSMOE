"""Full eight-terrain Thunder task using pinned Go2/Lab3 utilities.

Nominal Newton dynamics are the comparison baseline, not a claim that the entire
19538 PhysX randomization/solver distribution has been migrated unchanged.
"""
import math
import os
from collections.abc import Mapping
from types import SimpleNamespace
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg,mdp as lab_mdp
from isaaclab.managers import (ObservationGroupCfg,ObservationTermCfg,EventTermCfg,
    RewardTermCfg,CurriculumTermCfg,TerminationTermCfg,SceneEntityCfg)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg
from task_port.env_cfg import SceneCfg,ActionsCfg,robot_cfg
from task_port.observations import policy_frame,critic_frame
from teacher.privileged import LEG_JOINTS,JOINTS,WHEEL_BODIES
from height_curriculum import configure as configure_candidates
from height_curriculum.height import HeightOptions
from height_curriculum.boxes import gradual_random_grid_terrain
from .terrain import terrain_cfg,ThunderTerrainImporterCfg
from . import mdp


LEGS=SceneEntityCfg('robot',joint_names=list(LEG_JOINTS),preserve_order=True)
WHEELS=SceneEntityCfg('robot',joint_names=list(JOINTS[12:]),preserve_order=True)
WHEEL_CONTACTS=SceneEntityCfg('contact_forces',body_names=list(WHEEL_BODIES),preserve_order=True)
LEG_CONTACTS=SceneEntityCfg('contact_forces',body_names=['.*_(thigh|calf)_Link'])
NONWHEEL_CONTACTS=SceneEntityCfg('contact_forces',body_names=['^(?!.*_foot_Link).*'])


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        frame=ObservationTermCfg(func=policy_frame,history_length=10,flatten_history_dim=True)
        enable_corruption=False
        concatenate_terms=True
    @configclass
    class SingleObs(ObservationGroupCfg):
        frame=ObservationTermCfg(func=policy_frame)
        enable_corruption=False
        concatenate_terms=True
    @configclass
    class Critic(ObservationGroupCfg):
        frame=ObservationTermCfg(func=critic_frame)
        enable_corruption=False
        concatenate_terms=True
    policy=Policy()
    single_obs=SingleObs()
    critic=Critic()


@configclass
class CommandsCfg:
    base_velocity=mdp.ThunderVelocityCommandCfg(asset_name='robot',resampling_time_range=(10.,10.),
        rel_standing_envs=.1,rel_pure_yaw_envs=.1,rel_pure_lateral_envs=.1,heading_command=False,
        rel_heading_envs=0.,debug_vis=False,
        ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.,1.),lin_vel_y=(-1.,1.),ang_vel_z=(-1.5,1.5)))


@configclass
class EventsCfg:
    reset_scene=EventTermCfg(func=lab_mdp.reset_scene_to_default,mode='reset')
    reset_root=EventTermCfg(func=mdp.reset_stair_aligned,mode='reset',params={
        'pose_range':{'x':(-.5,.5),'y':(-.5,.5),'z':(0.,.05),'roll':(-.1,.1),'pitch':(-.1,.1),'yaw':(-3.14,3.14)},
        'velocity_range':{key:(0.,0.) for key in ('x','y','z','roll','pitch','yaw')}})


@configclass
class RewardsCfg:
    lin_vel_z_l2=RewardTermCfg(func=mdp.vertical,weight=-2.)
    ang_vel_xy_l2=RewardTermCfg(func=mdp.angular_xy,weight=-.05)
    joint_torques_l2=RewardTermCfg(func=lab_mdp.joint_torques_l2,weight=-3.e-5,params={'asset_cfg':LEGS})
    joint_acc_l2=RewardTermCfg(func=lab_mdp.joint_acc_l2,weight=-1.e-7,params={'asset_cfg':LEGS})
    joint_acc_wheel_l2=RewardTermCfg(func=lab_mdp.joint_acc_l2,weight=-2.5e-10,params={'asset_cfg':WHEELS})
    joint_pos_limits=RewardTermCfg(func=lab_mdp.joint_pos_limits,weight=-5.,params={'asset_cfg':LEGS})
    joint_power=RewardTermCfg(func=mdp.power,weight=-1.e-5,params={'asset_cfg':LEGS})
    wheel_torques_l2=RewardTermCfg(func=lab_mdp.joint_torques_l2,weight=-.0015,params={'asset_cfg':WHEELS})
    wheel_power=RewardTermCfg(func=mdp.power,weight=-3.e-5,params={'asset_cfg':WHEELS})
    stand_still=RewardTermCfg(func=mdp.stand_still,weight=-2.,params={'asset_cfg':LEGS})
    joint_pos_penalty=RewardTermCfg(func=mdp.posture,weight=-1.,params={'asset_cfg':LEGS})
    joint_mirror=RewardTermCfg(func=mdp.diagonal_mirror,weight=-.05)
    action_rate_l2=RewardTermCfg(func=lab_mdp.action_rate_l2,weight=-.01)
    undesired_contacts=RewardTermCfg(func=mdp.nonwheel_contact_cost,weight=-1.,params={'threshold':1.,'sensor_cfg':NONWHEEL_CONTACTS})
    contact_forces=RewardTermCfg(func=lab_mdp.contact_forces,weight=-.01,params={'threshold':450.,'sensor_cfg':WHEEL_CONTACTS})
    touchdown_velocity_l2=RewardTermCfg(func=mdp.TouchdownVelocity,weight=-.2,params={'velocity_threshold':.15,
        'sensor_cfg':WHEEL_CONTACTS,'asset_cfg':SceneEntityCfg('robot',body_names=list(WHEEL_BODIES),preserve_order=True)})
    track_lin_vel_xy_exp=RewardTermCfg(func=mdp.track_xy,weight=3.75)
    track_ang_vel_z_exp=RewardTermCfg(func=mdp.track_yaw,weight=1.875)
    upward=RewardTermCfg(func=mdp.upward,weight=3.)
    feet_contact_without_cmd=RewardTermCfg(func=mdp.feet_contact_without_cmd,weight=.1,params={'sensor_cfg':WHEEL_CONTACTS})
    flat_lateral_ang_vel_l2=RewardTermCfg(func=mdp.lateral_rate,weight=-.02)


@configclass
class TerminationsCfg:
    time_out=TerminationTermCfg(func=lab_mdp.time_out,time_out=True)
    terrain_out_of_bounds=TerminationTermCfg(func=mdp.terrain_out_of_bounds,time_out=True)
    stair_base_contact=TerminationTermCfg(func=mdp.base_contact,params={'threshold':1.,'sensor_cfg':SceneEntityCfg('contact_forces',body_names='base_link')})
    leg_contact=TerminationTermCfg(func=mdp.leg_contact_failure,params={'sensor_cfg':LEG_CONTACTS})


@configclass
class CurriculumCfg:
    terrain_levels=CurriculumTermCfg(func=mdp.terrain_levels)


@configclass
class ThunderNewposeEnvCfg(ManagerBasedRLEnvCfg):
    scene=SceneCfg(num_envs=4096,env_spacing=2.5,lazy_sensor_update=False)
    observations=ObservationsCfg()
    actions=ActionsCfg()
    commands=CommandsCfg()
    events=EventsCfg()
    rewards=RewardsCfg()
    terminations=TerminationsCfg()
    curriculum=CurriculumCfg()
    newpose_history_layout: str='frame_major_current_last'
    newpose_timeout_bootstrap: str='terminal_observation'
    compute_final_obs: bool=True
    gradual_boxes: bool=False
    height_weight: float=0.
    height_options: object=None
    hip_extra_weight: float=0.
    def __post_init__(self):
        self.scene.robot=robot_cfg()
        self.scene.terrain=ThunderTerrainImporterCfg(prim_path='/World/ground',
            terrain_type='generator',terrain_generator=terrain_cfg(),max_init_terrain_level=5,
            collision_group=-1,visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
        self.scene.height_scanner.mesh_prim_paths=['/World/ground/terrain/mesh']
        self.scene.contact_forces.track_air_time=True
        self.scene.contact_forces.update_period=.005
        self.scene.contact_forces.history_length=3
        self.decimation=8;self.episode_length_s=20.
        # Match Go2's triangle-mesh collision path; keep other contact settings unchanged.
        self.sim=sim_utils.SimulationCfg(dt=.0025,render_interval=8,use_newton_actuators=False,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),num_substeps=1))
        self.apply_candidate_features()

    def apply_candidate_features(self):
        """Idempotently resolve explicit candidates after composition, before gym.make."""
        if self.hip_extra_weight>0 or not math.isfinite(self.hip_extra_weight):
            raise ValueError('hip_extra_weight must be finite/nonpositive')
        if not isinstance(self.gradual_boxes,bool):raise TypeError('gradual_boxes must be boolean')
        options=HeightOptions(**self.height_options) if isinstance(self.height_options,Mapping) else self.height_options
        generator=self.scene.terrain.terrain_generator
        boxes=generator.sub_terrains.get('boxes')
        if self.gradual_boxes and boxes is None:raise ValueError('gradual_boxes requires a boxes terrain')
        # Validate and build in isolation so an invalid override cannot erase prior terms.
        candidate=SimpleNamespace(scene=self.scene,rewards=SimpleNamespace())
        configure_candidates(candidate,height_weight=self.height_weight,height_options=options)
        self.height_options=options
        if self.height_weight:self.rewards.cropped_ground_height=candidate.rewards.cropped_ground_height
        elif hasattr(self.rewards,'cropped_ground_height'):delattr(self.rewards,'cropped_ground_height')
        if self.gradual_boxes:
            if boxes.function is not gradual_random_grid_terrain:self._baseline_boxes_function=boxes.function
            boxes.function=gradual_random_grid_terrain
            generator.use_cache=False
        elif boxes is not None and boxes.function is gradual_random_grid_terrain:
            boxes.function=self._baseline_boxes_function
        if self.hip_extra_weight:
            self.rewards.hip_extra=RewardTermCfg(func=lab_mdp.joint_deviation_l1,weight=self.hip_extra_weight,
                params={'asset_cfg':SceneEntityCfg('robot',joint_names=['.*_hip_joint'])})
        elif hasattr(self.rewards,'hip_extra'):delattr(self.rewards,'hip_extra')
        return self


@configclass
class ThunderNewposePlayEnvCfg(ThunderNewposeEnvCfg):
    def __post_init__(self):
        super().__post_init__();self.scene.num_envs=64
        self.curriculum=None
        self.events.play_camera_reset=EventTermCfg(func=mdp.follow_play_camera,mode='reset')
        self.events.play_camera_interval=EventTermCfg(func=mdp.follow_play_camera,mode='interval',
            interval_range_s=(.02,.02),is_global_time=True,resample_interval_on_reset=False)
        choice=os.environ.get('NEWPOSE_EVAL_TERRAIN','mixed')
        names={'flat':'flat','stairs_up':'pyramid_stairs_inv','stairs_down':'pyramid_stairs'}
        if choice!='mixed':
            if choice not in names:raise ValueError('unknown NEWPOSE_EVAL_TERRAIN')
            key=names[choice];selected=self.scene.terrain.terrain_generator.sub_terrains[key]
            selected.proportion=1.;self.scene.terrain.terrain_generator.sub_terrains={key:selected}
        level=os.environ.get('NEWPOSE_EVAL_LEVEL')
        if level is not None:
            level=int(level)
            if not 0<=level<=9:raise ValueError('NEWPOSE_EVAL_LEVEL must be 0..9')
            self.scene.terrain.fixed_level=level
            # Playback never advances the training curriculum.
