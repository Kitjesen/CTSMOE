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
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg,NewtonCollisionPipelineCfg,NewtonShapeCfg
from thunder_lab3.assets.robot import robot_cfg
from thunder_lab3.assets.contract import A_POSE,A_POSE_CONTACT_HEIGHT,A_POSE_SPAWN_HEIGHT
from go2_rl_lab.tasks.mixed.mdp.curriculums import gradual_reward_weight_modification
from go2_rl_lab.tasks.mixed.mdp import rewards as go2_rewards
from .commands import ThunderRLGymVelocityCommandCfg,ThunderTerrainLevels
from .scene_cfg import SceneCfg,ActionsCfg,fresh_actions_cfg
from thunder_lab3.tasks.thunder.mdp.observations import policy_frame,critic_frame,NoisyPolicyFrame
from .mdp.randomization import configure_randomization
from thunder_lab3.tasks.thunder.mdp.privileged import LEG_JOINTS,JOINTS,WHEEL_BODIES
from .mdp.height_config import configure as configure_candidates
from thunder_lab3.tasks.thunder.mdp.height import HeightOptions
from thunder_lab3.tasks.thunder.boxes import gradual_random_grid_terrain
from .terrain import terrain_cfg,ThunderTerrainImporterCfg
from .mdp import terms as mdp


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
class FreshCommandsCfg:
    base_velocity=ThunderRLGymVelocityCommandCfg(debug_vis=False)


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
class FreshRewardsCfg(RewardsCfg):
    # Contact onsets reward bouncing, and diagonal joint signs are not a verified symmetry.
    feet_contact_without_cmd=None
    joint_mirror=None
    # Velocity rewards already gate on uprightness; avoid a large stand-still survival bonus.
    upward=None
    track_lin_vel_xy_exp=RewardTermCfg(func=go2_rewards.track_lin_vel_xy_exp,weight=2.,
        params={'command_name':'base_velocity','std':.5})
    track_ang_vel_z_exp=RewardTermCfg(func=go2_rewards.track_ang_vel_z_exp,weight=1.,
        params={'command_name':'base_velocity','std':.5})
    lin_vel_z_l2=RewardTermCfg(func=go2_rewards.lin_vel_z_l2,weight=-2.)
    joint_torques_l2=RewardTermCfg(func=lab_mdp.joint_torques_l2,weight=-6.e-5,params={'asset_cfg':LEGS})
    joint_acc_l2=RewardTermCfg(func=go2_rewards.clipped_joint_acc_l2,weight=-3.e-7,
        params={'asset_cfg':LEGS,'max_penalty':1.e8})
    joint_power=RewardTermCfg(func=go2_rewards.joint_power,weight=-1.5e-5,params={'asset_cfg':LEGS})
    joint_pos_limits=RewardTermCfg(func=lab_mdp.joint_pos_limits,weight=-2.,params={'asset_cfg':LEGS})
    action_rate_l2=RewardTermCfg(func=go2_rewards.clipped_action_rate_l2,weight=-.01,
        params={'max_penalty':20.})
    action_smoothness_l2=RewardTermCfg(func=go2_rewards.clipped_action_smoothness_l2,weight=-.01,
        params={'max_penalty':50.})
    # Replace the old whole-leg posture costs instead of stacking both objectives.
    stand_still=None
    joint_pos_penalty=None
    hip_pos_penalty_l1=RewardTermCfg(func=go2_rewards.joint_pos_penalty_l1,weight=-.05,
        params={'command_name':'base_velocity','asset_cfg':SceneEntityCfg('robot',joint_names=['.*_hip_joint']),
                'stand_still_scale':10.,'stand_cmd_idxs':[1,2]})
    joint_pos_penalty_l1=RewardTermCfg(func=go2_rewards.joint_pos_penalty_l1,weight=-.01,
        params={'command_name':'base_velocity','asset_cfg':SceneEntityCfg('robot',joint_names=['.*_(thigh|calf)_joint']),
                'stand_still_scale':10.})


@configclass
class TerminationsCfg:
    time_out=TerminationTermCfg(func=lab_mdp.time_out,time_out=True)
    terrain_out_of_bounds=TerminationTermCfg(func=mdp.terrain_out_of_bounds,time_out=True)
    stair_base_contact=TerminationTermCfg(func=mdp.base_contact,params={'threshold':1.,'sensor_cfg':SceneEntityCfg('contact_forces',body_names='base_link')})
    leg_contact=TerminationTermCfg(func=mdp.leg_contact_failure,params={'sensor_cfg':LEG_CONTACTS})


@configclass
class FreshTerminationsCfg(TerminationsCfg):
    leg_contact=TerminationTermCfg(func=mdp.fresh_leg_contact_failure,params={'sensor_cfg':LEG_CONTACTS})
    terrain_tile_exit=TerminationTermCfg(func=mdp.terrain_tile_exit,time_out=True)
    body_inverted=TerminationTermCfg(func=mdp.body_inverted)


@configclass
class CurriculumCfg:
    terrain_levels=CurriculumTermCfg(func=mdp.terrain_levels)


@configclass
class FreshCurriculumCfg(CurriculumCfg):
    terrain_levels=CurriculumTermCfg(func=ThunderTerrainLevels)
    base_linear_velocity_z=CurriculumTermCfg(func=gradual_reward_weight_modification,
        params={'term_name':'lin_vel_z_l2','initial_weight':-2.,'final_weight':0.,
                'start_it':0,'end_it':1500})
    base_height=CurriculumTermCfg(func=gradual_reward_weight_modification,
        params={'term_name':'cropped_ground_height','initial_weight':-1.,'final_weight':-10.,
                'start_it':0,'end_it':5000})


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
        if isinstance(self.curriculum,FreshCurriculumCfg):
            self.curriculum.base_height=(CurriculumTermCfg(func=gradual_reward_weight_modification,
                params={'term_name':'cropped_ground_height','initial_weight':self.height_weight,
                        'final_weight':10*self.height_weight,'start_it':0,'end_it':5000})
                if self.height_weight else None)
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


def configure_playback(cfg):
    cfg.scene.num_envs=64
    cfg.curriculum=None
    cfg.events.play_camera_reset=EventTermCfg(func=mdp.follow_play_camera,mode='reset')
    cfg.events.play_camera_interval=EventTermCfg(func=mdp.follow_play_camera,mode='interval',
        interval_range_s=(.02,.02),is_global_time=True,resample_interval_on_reset=False)
    choice=os.environ.get('NEWPOSE_EVAL_TERRAIN','mixed')
    names={'flat':'flat','stairs_up':'pyramid_stairs_inv','stairs_down':'pyramid_stairs'}
    if choice!='mixed':
        if choice not in names:raise ValueError('unknown NEWPOSE_EVAL_TERRAIN')
        key=names[choice];selected=cfg.scene.terrain.terrain_generator.sub_terrains[key]
        selected.proportion=1.;cfg.scene.terrain.terrain_generator.sub_terrains={key:selected}
        cfg.gradual_boxes=False
    level=os.environ.get('NEWPOSE_EVAL_LEVEL')
    if level is not None:
        level=int(level)
        if not 0<=level<=9:raise ValueError('NEWPOSE_EVAL_LEVEL must be 0..9')
        cfg.scene.terrain.fixed_level=level


@configclass
class ThunderNewposePlayEnvCfg(ThunderNewposeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_playback(self)


@configclass
class ThunderFreshEnvCfg(ThunderNewposeEnvCfg):
    """Independent from-scratch baseline; old policy playback retains its original contract."""
    scene=SceneCfg(num_envs=16384,env_spacing=2.5,lazy_sensor_update=False)
    commands=FreshCommandsCfg()
    curriculum=FreshCurriculumCfg()
    rewards=FreshRewardsCfg()
    terminations=FreshTerminationsCfg()
    gradual_boxes: bool=True
    height_weight: float=-1.
    height_options: object=HeightOptions(target_height=A_POSE_CONTACT_HEIGHT,tolerance=.02,
        max_reference_speed=.5,max_reference_lag=.1,min_valid_fraction=.8)
    newpose_terrain_schema: str='thunder_fresh_10level_v4_go2_aligned_a_2dp'
    newpose_physics_schema: str='newton_mjwarp_triangle_v1'

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot=robot_cfg(pose=A_POSE,spawn_height=A_POSE_SPAWN_HEIGHT)
        self.decimation=4
        self.episode_length_s=25.
        self.sim.dt=.005
        self.sim.render_interval=self.decimation
        self.observations.policy.frame.func=NoisyPolicyFrame
        self.observations.single_obs.frame.func=NoisyPolicyFrame
        self.observations.policy.enable_corruption=True
        self.observations.single_obs.enable_corruption=True
        # Match Go2's allocation: the 1M default overflows contact hashing at 16384 environments.
        self.sim.physics.collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        self.sim.physics.solver_cfg.integrator='implicitfast'
        self.sim.physics.solver_cfg.nconmax=128
        self.sim.physics.default_shape_cfg=NewtonShapeCfg(margin=.01)
        self.actions=fresh_actions_cfg()
        self.scene.terrain.max_init_terrain_level=0
        self.scene.terrain.terrain_generator.sub_terrains['boxes'].grid_height_range=(.01,.20)
        self.scene.contact_forces.update_period=self.sim.dt
        self.scene.contact_forces.history_length=self.decimation
        configure_randomization(self)


@configclass
class ThunderFreshPlayEnvCfg(ThunderFreshEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_playback(self)
        self.observations.policy.enable_corruption=False
        self.observations.single_obs.enable_corruption=False
        for name in vars(self.events):
            if name.startswith('randomize_'):
                setattr(self.events,name,None)
        self.sim.physics.solver_cfg.body_com_buckets=None
