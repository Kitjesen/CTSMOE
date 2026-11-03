"""Thunder rewards, restricted stair commands and command-aware terrain grading.

Ported from the frozen NEWPOSE mixed branch, using Lab 3 XYZW/ProxyArray APIs.
Newton contact terms explicitly use net normal forces; backend equivalence is not claimed.
"""
import math
import torch
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import SceneEntityCfg,ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat,quat_from_euler_xyz,wrap_to_pi
from go2_rl_lab.utils.terrains import get_terrain_masks,get_assigned_terrain_masks
from go2_rl_lab.tasks.mixed.mdp.events import reset_root_state_uniform as go2_reset_root_state_uniform


def follow_play_camera(env, env_ids, eye_offset=(2.,-2.,1.2)):
    """Point official visualizers at environment zero; never write robot or command state."""
    target=tuple(env.scene['robot'].data.root_pos_w.torch[0].detach().cpu().tolist())
    eye=tuple(position+offset for position,offset in zip(target,eye_offset))
    env.sim.set_camera_view(eye=eye,target=target)


def env_indices(env_ids,num_envs,device):
    """Normalize Lab 3 whole/partial reset selectors to explicit backend indices."""
    if env_ids is None:return torch.arange(num_envs,device=device,dtype=torch.long)
    if isinstance(env_ids,slice):
        return torch.tensor(list(range(*env_ids.indices(num_envs))),device=device,dtype=torch.long)
    ids=torch.as_tensor(env_ids,device=device)
    if ids.ndim!=1:raise ValueError('environment indices must be one-dimensional')
    if ids.dtype==torch.bool:
        if len(ids)!=num_envs:raise ValueError('environment mask length must match num_envs')
        return ids.nonzero().flatten()
    if ids.numel() and (ids.is_floating_point() or ids.is_complex()):
        raise TypeError('environment indices must be integral')
    ids=ids.long()
    if torch.any((ids<0)|(ids>=num_envs)):raise IndexError('environment index out of range')
    return ids


def stairs(env,assigned=False):
    masks=(get_assigned_terrain_masks if assigned else get_terrain_masks)(env)
    zero=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    return masks.get('pyramid_stairs',zero)|masks.get('pyramid_stairs_inv',zero)


def upright_scale(env):
    return (-env.scene['robot'].data.projected_gravity_b.torch[:,2]).clamp(0,.7)/.7


def moving_stairs(env):
    return stairs(env)&(env.command_manager.get_command('base_velocity')[:,0]>.1)


def track_xy(env,std=.5,command_name='base_velocity'):
    err=(env.command_manager.get_command(command_name)[:,:2]-env.scene['robot'].data.root_lin_vel_b.torch[:,:2]).square().sum(1)
    variance=torch.where(moving_stairs(env),.25**2,std**2)
    return torch.exp(-err/variance)*upright_scale(env)


def track_yaw(env,std=.5,command_name='base_velocity'):
    return lab_mdp.track_ang_vel_z_exp(env,std,command_name)*upright_scale(env)


def vertical(env):
    cost=env.scene['robot'].data.root_lin_vel_b.torch[:,2].square()*upright_scale(env)
    return torch.where(moving_stairs(env),0.,cost)


def angular_xy(env):
    return env.scene['robot'].data.root_ang_vel_b.torch[:,:2].square().sum(1)*upright_scale(env)


def lateral_rate(env):
    cmd=env.command_manager.get_command('base_velocity')
    cost=env.scene['robot'].data.root_ang_vel_b.torch[:,:2].square().sum(1)
    return torch.where(cmd[:,1].abs()>=.2,cost,0.)


def upward(env):
    value=(1-env.scene['robot'].data.projected_gravity_b.torch[:,2]).square()
    return torch.where(moving_stairs(env),.25*value,value)


def power(env,asset_cfg):
    data=env.scene[asset_cfg.name].data;ids=asset_cfg.joint_ids
    return (data.joint_vel.torch[:,ids]*data.applied_torque.torch[:,ids]).abs().sum(1)


def stand_still(env,asset_cfg,command_name='base_velocity',command_threshold=.1):
    value=lab_mdp.joint_deviation_l1(env,asset_cfg)
    return value*(env.command_manager.get_command(command_name).norm(dim=1)<command_threshold)*upright_scale(env)


def posture(env,asset_cfg,stand_still_scale=5.,velocity_threshold=.5,command_threshold=.1,command_name='base_velocity'):
    data=env.scene[asset_cfg.name].data
    deviation=(data.joint_pos.torch[:,asset_cfg.joint_ids]-data.default_joint_pos.torch[:,asset_cfg.joint_ids]).norm(dim=1)
    moving=(env.command_manager.get_command(command_name).norm(dim=1)>command_threshold)|(data.root_lin_vel_b.torch[:,:2].norm(dim=1)>velocity_threshold)
    cost=torch.where(moving,deviation,stand_still_scale*deviation)*upright_scale(env)
    return torch.where(moving_stairs(env),.25*cost,cost)


def diagonal_mirror(env):
    robot=env.scene['robot'];data=robot.data
    q=data.joint_pos.torch-data.default_joint_pos.torch
    costs=[]
    for a,b in [('fr','rl'),('fl','rr')]:
        ai=[robot.joint_names.index(f'{a}_{kind}_joint') for kind in ('hip','thigh','calf')]
        bi=[robot.joint_names.index(f'{b}_{kind}_joint') for kind in ('hip','thigh','calf')]
        costs.append((q[:,ai]-q[:,bi]).square().sum(1))
    return .5*(costs[0]+costs[1])*upright_scale(env)


def nonwheel_contact_cost(env,threshold,sensor_cfg):
    forces=env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch
    peak=forces[:,:,sensor_cfg.body_ids].norm(dim=-1).amax(1)
    return 4.*(peak>threshold).float().sum(1)+(peak/100.).clamp(max=10.).sum(1)


def leg_contact_failure(env,sensor_cfg):
    if not hasattr(env,'_leg_contact'):
        env._leg_contact={'duration':torch.zeros(env.num_envs,device=env.device),
            'touched':torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),
            'failed':torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),'step':-1}
    state=env._leg_contact
    if state['step']!=env.common_step_counter:
        force=env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch
        peak=force[:,:,sensor_cfg.body_ids].norm(dim=-1).amax((1,2))
        state['duration']=torch.where(peak>20.,state['duration']+env.step_dt,0.)
        state['touched']|=peak>5.;state['failed']|=(peak>=500.)|(state['duration']>=.1)
        state['step']=env.common_step_counter
    return state['failed']


def base_contact(env,sensor_cfg,threshold=1.):
    return stairs(env)&lab_mdp.illegal_contact(env,threshold,sensor_cfg)


def feet_contact_without_cmd(env,sensor_cfg,command_name='base_velocity'):
    first=env.scene.sensors[sensor_cfg.name].compute_first_contact(env.step_dt)
    if not isinstance(first,torch.Tensor):first=first.torch
    return first[:,sensor_cfg.body_ids].sum(1).float()*(env.command_manager.get_command(command_name).norm(dim=1)<.1)*upright_scale(env)


class TouchdownVelocity(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        asset=cfg.params['asset_cfg'];sensor=cfg.params['sensor_cfg']
        asset_names=[env.scene[asset.name].body_names[i] for i in asset.body_ids]
        sensor_names=[env.scene.sensors[sensor.name].body_names[i] for i in sensor.body_ids]
        if asset_names!=sensor_names:raise ValueError('Touchdown robot/sensor body order differs')
        self.previous=torch.zeros((env.num_envs,len(asset.body_ids)),device=env.device)

    def reset(self,env_ids=None):
        self.previous[env_indices(env_ids,self.previous.shape[0],self.previous.device)]=0.

    def __call__(self,env,asset_cfg,sensor_cfg,velocity_threshold):
        first=env.scene.sensors[sensor_cfg.name].compute_first_contact(env.step_dt)
        if not isinstance(first,torch.Tensor):first=first.torch
        cost=(first[:,sensor_cfg.body_ids].float()*(-self.previous-velocity_threshold).clamp_min(0).square()).sum(1)
        self.previous.copy_(env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:,asset_cfg.body_ids,2])
        return cost


def tilt_flags(gravity):
    roll=torch.atan2(-gravity[:,1],-gravity[:,2]).abs()
    pitch=torch.atan2(gravity[:,0],gravity[:,1:].norm(dim=1)).abs()
    return (roll>math.radians(50))|(pitch>math.radians(60)),gravity[:,2]>=0


def reset_stair_aligned(env,env_ids,pose_range,velocity_range,asset_cfg=SceneEntityCfg('robot')):
    ids=env_indices(env_ids,env.num_envs,env.device)
    # Lab 3's namesake is a manager class; Go2 supplies the reusable function.
    go2_reset_root_state_uniform(env,ids,pose_range,velocity_range,asset_cfg)
    ids=ids[stairs(env,assigned=True)[ids]]
    if not len(ids):return
    robot=env.scene[asset_cfg.name];pose=robot.data.root_pose_w.torch[ids].clone()
    roll,pitch,yaw=euler_xyz_from_quat(pose[:,3:7]);yaw=torch.round(yaw/(math.pi/2))*(math.pi/2)
    pose[:,3:7]=quat_from_euler_xyz(roll,pitch,yaw)
    robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)


class ThunderVelocityCommand(lab_mdp.UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if cfg.rel_standing_envs+cfg.rel_pure_yaw_envs+cfg.rel_pure_lateral_envs>1:
            raise ValueError('command mode proportions exceed one')
        self.is_pure_yaw_env=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.is_pure_lateral_env=torch.zeros_like(self.is_pure_yaw_env)
        self.was_on_stairs=torch.zeros_like(self.is_pure_yaw_env)
        self.stair_heading=torch.zeros(self.num_envs,device=self.device)
        self.travel_expected=torch.zeros_like(self.stair_heading);self.travel_progress=torch.zeros_like(self.stair_heading)
        self.travel_peak=torch.zeros_like(self.stair_heading);self.travel_radius=torch.zeros_like(self.stair_heading)
        self.travel_failed=torch.zeros_like(self.is_pure_yaw_env);self.tilt_duration=torch.zeros_like(self.stair_heading)
        self.travel_previous=self.robot.data.root_pos_w.torch[:,:2].clone()
        self.travel_yaw=euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2].clone()
        self.travel_axis=torch.zeros((self.num_envs,2),device=self.device)
        self.travel_step=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)

    def _set_stair_targets(self,ids):
        if not len(ids):return
        yaw=euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2]
        self.stair_heading[ids]=torch.round(yaw/(math.pi/2))*(math.pi/2)+torch.empty(len(ids),device=self.device).uniform_(-math.pi/18,math.pi/18)
        self.vel_command_b[ids,0]=torch.empty(len(ids),device=self.device).uniform_(.25,.5)
        self.vel_command_b[ids,1]=0.;self.is_pure_yaw_env[ids]=False;self.is_pure_lateral_env[ids]=False;self.is_heading_env[ids]=False

    def _constrain_stairs(self,mask):
        ids=mask.nonzero().flatten()
        if not len(ids):return
        yaw=euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2]
        self.vel_command_b[ids,0]=self.vel_command_b[ids,0].clamp(.25,.5)
        self.vel_command_b[ids,1]=0.
        self.vel_command_b[ids,2]=(.5*wrap_to_pi(self.stair_heading[ids]-yaw)).clamp(-.2,.2)
        self.vel_command_b[ids[self.is_standing_env[ids]]]=0.

    def _resample_command(self,env_ids):
        ids=env_indices(env_ids,self.num_envs,self.device)
        probability=self.cfg.rel_standing_envs;self.cfg.rel_standing_envs=0.
        try:super()._resample_command(ids)
        finally:self.cfg.rel_standing_envs=probability
        self.vel_command_b[ids,:2]*=(self.vel_command_b[ids,:2].norm(dim=1)>.2)[:,None]
        mode=torch.rand(len(ids),device=self.device);yaw_end=probability+self.cfg.rel_pure_yaw_envs
        standing=mode<probability;pure_yaw=(mode>=probability)&(mode<yaw_end)
        lateral=(mode>=yaw_end)&(mode<yaw_end+self.cfg.rel_pure_lateral_envs)
        self.is_standing_env[ids]=standing;self.is_pure_yaw_env[ids]=pure_yaw;self.is_pure_lateral_env[ids]=lateral
        self.is_heading_env[ids[standing|pure_yaw|lateral]]=False;self.vel_command_b[ids[standing]]=0.
        for mask,axis,limits in ((pure_yaw,2,self.cfg.ranges.ang_vel_z),(lateral,1,self.cfg.ranges.lin_vel_y)):
            selected=ids[mask]
            if len(selected):
                value=torch.empty(len(selected),device=self.device).uniform_(*limits)
                value=torch.where(value>=0,value.clamp(min=.2),value.clamp(max=-.2))
                self.vel_command_b[selected]=0.;self.vel_command_b[selected,axis]=value
        mask=stairs(self._env);self._set_stair_targets(ids[mask[ids]]);self._constrain_stairs(mask)

    def _update_command(self):
        super()._update_command();mask=stairs(self._env)
        leaving=self.was_on_stairs&~mask
        if leaving.any():self._resample_command(leaving.nonzero().flatten())
        self._set_stair_targets((mask&~self.was_on_stairs).nonzero().flatten())
        self._constrain_stairs(mask);self.was_on_stairs=mask

    def reset(self,env_ids=None):
        ids=env_indices(env_ids,self.num_envs,self.device)
        result=super().reset(ids)
        state=getattr(self._env,'_leg_contact',None)
        if state is not None:
            for key in ('duration','touched','failed'):state[key][ids]=0
        for name in ('travel_expected','travel_progress','travel_peak','travel_radius','tilt_duration'):getattr(self,name)[ids]=0
        self.travel_failed[ids]=False;self.travel_previous[ids]=self.robot.data.root_pos_w.torch[ids,:2];self.travel_step[ids]=-1
        yaw=euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2];self.travel_yaw[ids]=yaw
        yaw=torch.round(yaw/(math.pi/2))*(math.pi/2);self.travel_axis[ids]=torch.stack((yaw.cos(),yaw.sin()),1)
        return result

    def _update_metrics(self):
        super()._update_metrics();self.accumulate(torch.arange(self.num_envs,device=self.device))

    def accumulate(self,ids):
        ids=env_indices(ids,self.num_envs,self.device)
        step=self._env.common_step_counter
        ids=ids[(self._env.episode_length_buf[ids]>0)&(self.travel_step[ids]!=step)]
        if not len(ids):return
        position=self.robot.data.root_pos_w.torch[ids,:2];delta=position-self.travel_previous[ids]
        command=self.command[ids,:2];speed=command.norm(dim=1);yaw=self.travel_yaw[ids]
        direction=torch.stack((yaw.cos()*command[:,0]-yaw.sin()*command[:,1],yaw.sin()*command[:,0]+yaw.cos()*command[:,1]),1)/speed.clamp_min(1e-6)[:,None]
        budget=speed*self._env.step_dt;self.travel_expected[ids]+=budget
        self.travel_progress[ids]+=torch.minimum((delta*direction).sum(1).clamp_min(0),budget)
        offset=position-self._env.scene.env_origins[ids,:2]
        self.travel_peak[ids]=torch.maximum(self.travel_peak[ids],(offset*self.travel_axis[ids]).sum(1))
        self.travel_radius[ids]=torch.maximum(self.travel_radius[ids],offset.abs().amax(1))
        bad,inverted=tilt_flags(self.robot.data.projected_gravity_b.torch[ids])
        self.tilt_duration[ids]=torch.where(bad,self.tilt_duration[ids]+self._env.step_dt,0.)
        self.travel_failed[ids]|=inverted|(self.tilt_duration[ids]>=.2)
        self.travel_previous[ids]=position;self.travel_yaw[ids]=euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2]
        self.travel_step[ids]=step


@configclass
class ThunderVelocityCommandCfg(lab_mdp.UniformVelocityCommandCfg):
    class_type: type=ThunderVelocityCommand
    rel_pure_yaw_envs: float=.1
    rel_pure_lateral_envs: float=.1


def terrain_levels(env,env_ids):
    ids=env_indices(env_ids,env.num_envs,env.device)
    terrain=env.scene.terrain;robot=env.scene['robot'];term=env.command_manager.get_term('base_velocity')
    term.accumulate(ids);expected=term.travel_expected[ids]
    delta=robot.data.root_pos_w.torch[ids,:2]-env.scene.env_origins[ids,:2]
    forward=(delta*term.travel_axis[ids]).sum(1);lateral=delta[:,0]*term.travel_axis[ids,1]-delta[:,1]*term.travel_axis[ids,0]
    failed=term.travel_failed[ids]|env.termination_manager.terminated[ids]
    bad,inverted=tilt_flags(robot.data.projected_gravity_b.torch[ids]);stable=~bad&~inverted
    contacts=getattr(env,'_leg_contact',None)
    if contacts is not None:stable&=~contacts['touched'][ids]
    stair=stairs(env,assigned=True)[ids]
    crossed=torch.where(stair,torch.minimum(torch.maximum(forward,term.travel_peak[ids]),forward),term.travel_radius[ids])
    stable&=~stair|(lateral.abs()<1.5)
    moving=expected>.5;tracked=term.travel_progress[ids]>=expected*.5
    up=moving&~failed&stable&tracked&(crossed>terrain.cfg.terrain_generator.size[0]/2)
    down=~up&(failed|(moving&~tracked))
    terrain.update_env_origins(ids,up,down)
    return terrain.terrain_levels.float().mean()


def terrain_out_of_bounds(env,distance_buffer=3.):
    gen=env.scene.terrain.cfg.terrain_generator;xy=env.scene['robot'].data.root_pos_w.torch[:,:2]
    extent=torch.tensor([gen.num_rows*gen.size[0],gen.num_cols*gen.size[1]],device=env.device)/2+gen.border_width-distance_buffer
    return (xy.abs()>extent).any(1)
