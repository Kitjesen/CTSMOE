"""CPU checks of commands, curriculum and the full terrain/reward contract."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml

from thunder_lab3.tasks.thunder.env_cfg import (RewardsCfg,FreshRewardsCfg,CommandsCfg,ObservationsCfg,
    ThunderNewposeEnvCfg,ThunderNewposePlayEnvCfg,ThunderFreshEnvCfg,ThunderFreshPlayEnvCfg)
from thunder_lab3.tasks.thunder.mdp.height import HeightOptions
from thunder_lab3.tasks.thunder.boxes import gradual_random_grid_terrain
from thunder_lab3.tasks.thunder.terrain import terrain_cfg,pyramid_slope
from thunder_lab3.tasks.thunder.mdp import terms as mdp
from thunder_lab3.tasks.thunder import make_env
from thunder_lab3.assets.contract import A_POSE,A_POSE_CONTACT_HEIGHT,A_POSE_SPAWN_HEIGHT,POSE


class FullContractTest(unittest.TestCase):
    @staticmethod
    def contact_fixture(count=1):
        history=torch.zeros(count,8,1,3)
        sensor=NS(cfg=NS(update_period=.0025,history_length=8),
            data=NS(net_normal_forces_w_history=NS(torch=history)))
        env=NS(num_envs=count,device='cpu',physics_dt=.0025,step_dt=.02,common_step_counter=0,
            cfg=NS(scene=NS(lazy_sensor_update=False)),scene=NS(sensors={'contact_forces':sensor}),
            _physics_handles_decimation=False)
        return env,NS(name='contact_forces',body_ids=[0]),history

    def test_fresh_contact_does_not_turn_sparse_impacts_into_continuous_duration(self):
        env,cfg,history=self.contact_fixture()
        for step in range(1,6):
            env.common_step_counter=step;history.zero_();history[:,3,0,2]=30.
            self.assertFalse(bool(mdp.fresh_leg_contact_failure(env,cfg).any()))
            self.assertEqual(env._leg_contact['samples'].item(),0)
            self.assertEqual(env._leg_contact['duration'].item(),0.)
        self.assertTrue(env._leg_contact['touched'].item())

    def test_fresh_contact_exactly_forty_consecutive_samples_fail_not_thirty_nine(self):
        env,cfg,history=self.contact_fixture(2)
        history[:,:,:,2]=30.
        for step in range(1,5):
            env.common_step_counter=step
            self.assertFalse(bool(mdp.fresh_leg_contact_failure(env,cfg).any()))
        env.common_step_counter=5
        history[1,0,0,2]=0.
        self.assertEqual(mdp.fresh_leg_contact_failure(env,cfg).tolist(),[True,False])
        self.assertEqual(env._leg_contact['samples'].tolist(),[40,0])
        self.assertAlmostEqual(env._leg_contact['duration'][0].item(),.1,places=7)

    def test_fresh_contact_retains_peak_and_touch_thresholds(self):
        env,cfg,history=self.contact_fixture(3)
        history[:,4,0,2]=torch.tensor([500.,499.,5.])
        self.assertEqual(mdp.fresh_leg_contact_failure(env,cfg).tolist(),[True,False,False])
        self.assertEqual(env._leg_contact['touched'].tolist(),[True,True,False])
        self.assertEqual(env._leg_contact['duration'].tolist(),[0.,0.,0.])

    def test_fresh_contact_rejects_overlapping_or_lazy_history(self):
        for key,value in (('period',.005),('length',7),('physics',.005),('lazy',True),('decimation',True)):
            env,cfg,_=self.contact_fixture()
            sensor=env.scene.sensors['contact_forces']
            if key=='period':sensor.cfg.update_period=value
            elif key=='length':sensor.cfg.history_length=value
            elif key=='physics':env.physics_dt=value
            elif key=='lazy':env.cfg.scene.lazy_sensor_update=value
            else:env._physics_handles_decimation=value
            with self.assertRaises(ValueError):mdp.fresh_leg_contact_failure(env,cfg)

    def test_fresh_contact_partial_reset_and_duplicate_calls_do_not_recount_samples(self):
        env,cfg,history=self.contact_fixture(2)
        history[:,:,:,2]=30.
        env.common_step_counter=1
        mdp.fresh_leg_contact_failure(env,cfg)
        mdp.fresh_leg_contact_failure(env,cfg)
        self.assertEqual(env._leg_contact['samples'].tolist(),[8,8])
        term=object.__new__(mdp.ThunderVelocityCommand);term._env=env
        term.robot=NS(data=NS(root_pos_w=NS(torch=torch.zeros(2,3)),
            root_quat_w=NS(torch=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1))))
        for name in ('travel_expected','travel_progress','travel_peak','travel_radius','tilt_duration'):
            setattr(term,name,torch.ones(2))
        term.travel_failed=torch.ones(2,dtype=torch.bool)
        term.travel_previous=torch.ones(2,2);term.travel_step=torch.ones(2,dtype=torch.long)
        term.travel_yaw=torch.ones(2);term.travel_axis=torch.ones(2,2)
        with patch.object(mdp.lab_mdp.UniformVelocityCommand,'reset',return_value={}):term.reset([0])
        self.assertEqual(env._leg_contact['samples'].tolist(),[0,8])
        mdp.fresh_leg_contact_failure(env,cfg)
        self.assertEqual(env._leg_contact['samples'].tolist(),[0,8])
        env.common_step_counter=2
        mdp.fresh_leg_contact_failure(env,cfg)
        self.assertEqual(env._leg_contact['samples'].tolist(),[8,16])

    def test_explicit_seed_reaches_environment_before_construction(self):
        cfg=NS(seed=0,apply_candidate_features=lambda:None)
        with patch.dict('os.environ',{'NEWPOSE_SEED':'17','NEWPOSE_CANDIDATE_OVERRIDES':'{}'}), \
                patch('isaaclab.envs.ManagerBasedRLEnv',return_value='created') as constructor:
            self.assertEqual(make_env(cfg),'created')
            self.assertEqual(cfg.seed,17)
            constructor.assert_called_once_with(cfg=cfg)
        with patch.dict('os.environ',{'NEWPOSE_SEED':'-1','NEWPOSE_CANDIDATE_OVERRIDES':'{}'}):
            with self.assertRaises(ValueError):make_env(cfg)

    def test_fresh_defaults_are_independent_of_legacy_and_start_on_level_zero(self):
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',return_value=None), \
                patch('thunder_lab3.tasks.thunder.env_cfg.fresh_actions_cfg',return_value=NS(fresh_actions=True)):
            fresh=ThunderFreshEnvCfg();legacy=ThunderNewposeEnvCfg()
        self.assertEqual(fresh.newpose_terrain_schema,'thunder_fresh_10level_v3_go2_commands_a_pose')
        self.assertEqual(fresh.newpose_physics_schema,'newton_mjwarp_triangle_v1')
        self.assertEqual(fresh.scene.terrain.max_init_terrain_level,0)
        self.assertEqual(legacy.scene.terrain.max_init_terrain_level,5)
        self.assertTrue(fresh.actions.fresh_actions)
        self.assertEqual(fresh.episode_length_s,20.)
        self.assertEqual(fresh.scene.contact_forces.update_period,.0025)
        self.assertEqual(fresh.scene.contact_forces.history_length,8)
        self.assertAlmostEqual(fresh.scene.contact_forces.update_period*fresh.scene.contact_forces.history_length,
            fresh.sim.dt*fresh.decimation)
        self.assertEqual(legacy.scene.contact_forces.history_length,3)
        self.assertIs(fresh.scene.terrain.terrain_generator.sub_terrains['boxes'].function,gradual_random_grid_terrain)
        self.assertEqual(fresh.scene.terrain.terrain_generator.sub_terrains['boxes'].grid_height_range,(.01,.20))
        self.assertIsNot(legacy.scene.terrain.terrain_generator.sub_terrains['boxes'].function,gradual_random_grid_terrain)
        self.assertEqual((fresh.height_weight,fresh.hip_extra_weight),(-1.,0.))
        self.assertEqual((legacy.height_weight,legacy.hip_extra_weight),(0.,0.))
        self.assertIs(fresh.terminations.terrain_tile_exit.func,mdp.terrain_tile_exit)
        self.assertTrue(fresh.terminations.terrain_tile_exit.time_out)
        self.assertFalse(fresh.terminations.body_inverted.time_out)
        self.assertIs(fresh.terminations.leg_contact.func,mdp.fresh_leg_contact_failure)
        self.assertIs(legacy.terminations.leg_contact.func,mdp.leg_contact_failure)

    def test_fresh_pose_height_and_reward_course_do_not_mutate_legacy(self):
        from go2_rl_lab.tasks.mixed.env_cfg import PhysicsCfg as Go2PhysicsCfg
        from isaaclab_newton.physics import NewtonCollisionPipelineCfg

        def asset(**kwargs):
            return NS(init_state=NS(joint_pos=dict(kwargs.get('pose',POSE)),
                pos=(0.,0.,kwargs.get('spawn_height',.46))))
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',side_effect=asset), \
                patch('thunder_lab3.tasks.thunder.env_cfg.fresh_actions_cfg',return_value=None):
            fresh=ThunderFreshEnvCfg();legacy=ThunderNewposeEnvCfg();other=ThunderFreshEnvCfg()
        self.assertEqual(fresh.scene.robot.init_state.joint_pos,A_POSE)
        self.assertEqual(fresh.scene.robot.init_state.pos[2],A_POSE_SPAWN_HEIGHT)
        self.assertEqual(legacy.scene.robot.init_state.joint_pos,POSE)
        self.assertEqual(legacy.scene.robot.init_state.pos[2],.46)
        self.assertEqual(fresh.scene.num_envs,16384)
        self.assertEqual(legacy.scene.num_envs,4096)
        self.assertEqual(fresh.sim.physics.collision_cfg.max_triangle_pairs,2_500_000)
        self.assertEqual(fresh.sim.physics.collision_cfg.max_triangle_pairs,
            Go2PhysicsCfg().newton_mjwarp.collision_cfg.max_triangle_pairs)
        self.assertIsNone(legacy.sim.physics.collision_cfg)
        self.assertEqual(NewtonCollisionPipelineCfg().max_triangle_pairs,1_000_000)
        official=Go2PhysicsCfg().newton_mjwarp
        for name in ('integrator','njmax','nconmax','cone','impratio','update_data_interval','use_mujoco_contacts'):
            self.assertEqual(getattr(fresh.sim.physics.solver_cfg,name),getattr(official.solver_cfg,name))
        self.assertEqual(fresh.sim.physics.default_shape_cfg.margin,official.default_shape_cfg.margin)
        self.assertEqual((legacy.sim.physics.solver_cfg.integrator,legacy.sim.physics.solver_cfg.nconmax,
            legacy.sim.physics.default_shape_cfg.margin),('euler',None,0.))
        self.assertEqual((fresh.sim.dt,fresh.decimation,fresh.episode_length_s),(.0025,8,20.))
        self.assertEqual(fresh.height_options.target_height,A_POSE_CONTACT_HEIGHT)
        self.assertEqual(fresh.height_options.tolerance,.02)
        self.assertEqual(fresh.height_options.max_reference_speed,.5)
        self.assertEqual(fresh.height_options.max_reference_lag,.1)
        self.assertEqual(fresh.height_options.min_valid_fraction,.8)
        self.assertEqual(fresh.rewards.cropped_ground_height.weight,-1.)
        self.assertFalse(hasattr(legacy.rewards,'cropped_ground_height'))
        self.assertEqual(fresh.curriculum.base_linear_velocity_z.params,
            dict(term_name='lin_vel_z_l2',initial_weight=-2.,final_weight=0.,start_it=0,end_it=1500))
        self.assertEqual(fresh.curriculum.base_height.params,
            dict(term_name='cropped_ground_height',initial_weight=-1.,final_weight=-10.,start_it=0,end_it=5000))
        self.assertFalse(hasattr(legacy.curriculum,'base_height'))
        fresh.scene.robot.init_state.joint_pos['fr_thigh_joint']=0.
        fresh.height_options.target_height=.5
        self.assertEqual(other.scene.robot.init_state.joint_pos,A_POSE)
        self.assertEqual(other.height_options.target_height,A_POSE_CONTACT_HEIGHT)
        self.assertEqual(legacy.scene.robot.init_state.joint_pos,POSE)

    def test_fresh_height_ablation_removes_and_restores_its_schedule(self):
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',return_value=None), \
                patch('thunder_lab3.tasks.thunder.env_cfg.fresh_actions_cfg',return_value=None):
            cfg=ThunderFreshEnvCfg()
        cfg.height_weight=0.;cfg.apply_candidate_features()
        self.assertFalse(hasattr(cfg.rewards,'cropped_ground_height'))
        self.assertIsNone(cfg.curriculum.base_height)
        cfg.height_weight=-.5;cfg.apply_candidate_features()
        self.assertEqual(cfg.rewards.cropped_ground_height.weight,-.5)
        self.assertEqual(cfg.curriculum.base_height.params['initial_weight'],-.5)
        self.assertEqual(cfg.curriculum.base_height.params['final_weight'],-5.)

    def test_fresh_reward_changes_remove_contact_onset_bonus_and_unverified_mirror(self):
        old=RewardsCfg();fresh=FreshRewardsCfg()
        def weights(cfg):return {k:v.weight for k,v in vars(cfg).items() if hasattr(v,'weight') and v.weight}
        expected=weights(old)
        for name in ('feet_contact_without_cmd','joint_mirror','upward'):expected.pop(name)
        self.assertEqual(weights(fresh),expected)
        self.assertEqual(fresh.track_lin_vel_xy_exp.params,{'std':.5,'stair_std':.5})
        self.assertIsNone(fresh.upward)
        self.assertFalse(any('air_time' in name or 'lift' in name for name in weights(fresh)))

    def test_fresh_stair_tracking_signal_and_normal_slope_upright_gate(self):
        data=NS(root_lin_vel_b=NS(torch=torch.zeros(2,3)),
            projected_gravity_b=NS(torch=torch.tensor([[0.,0.,-1.],[math.sin(math.radians(40)),0.,-math.cos(math.radians(40))]])))
        env=NS(scene={'robot':NS(data=data)},command_manager=NS(get_command=lambda _:torch.tensor([[.5,0.,0.],[.5,0.,0.]])))
        with patch.object(mdp,'moving_stairs',return_value=torch.ones(2,dtype=torch.bool)):
            legacy=mdp.track_xy(env);fresh=mdp.track_xy(env,stair_std=.5)
        torch.testing.assert_close(legacy,torch.full((2,),math.exp(-4)))
        torch.testing.assert_close(fresh,torch.full((2,),math.exp(-1)))
        self.assertFalse(bool(mdp.body_inverted(env).any()))
        data.projected_gravity_b.torch[1,2]=.1
        self.assertEqual(mdp.body_inverted(env).tolist(),[False,True])

    def test_tile_exit_and_promotion_use_the_same_boundary(self):
        n=3;called=[]
        positions=torch.tensor([[4.,0.,.5],[4.001,0.,.5],[-4.001,0.,.5]])
        term=NS(accumulate=lambda ids:None,travel_expected=torch.full((n,),5.),
            travel_progress=torch.full((n,),4.5),travel_axis=torch.tensor([[1.,0.]]).repeat(n,1),
            travel_failed=torch.zeros(n,dtype=torch.bool),travel_peak=positions[:,0].clone(),travel_radius=positions[:,0].abs())
        robot=NS(data=NS(root_pos_w=NS(torch=positions),projected_gravity_b=NS(torch=torch.tensor([[0.,0.,-1.]]).repeat(n,1))))
        terrain=NS(cfg=NS(terrain_generator=NS(size=(8.,8.))),terrain_levels=torch.zeros(n,dtype=torch.long),
            update_env_origins=lambda ids,up,down:called.append((up.clone(),down.clone())))
        class Scene(dict):pass
        scene=Scene(robot=robot);scene.terrain=terrain;scene.env_origins=torch.zeros(n,3)
        env=NS(scene=scene,num_envs=n,device='cpu',command_manager=NS(get_term=lambda _:term),
            termination_manager=NS(terminated=torch.zeros(n,dtype=torch.bool)))
        self.assertEqual(mdp.terrain_tile_exit(env).tolist(),[False,True,True])
        with patch.object(mdp,'stairs',return_value=torch.ones(n,dtype=torch.bool)):
            mdp.terrain_levels(env,None)
        self.assertEqual(called[-1][0].tolist(),[False,True,False])
        self.assertEqual(called[-1][1].tolist(),[False,False,False])

    def test_fresh_playback_holds_exact_stair_level_and_disables_curriculum(self):
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',return_value=None), \
                patch('thunder_lab3.tasks.thunder.env_cfg.fresh_actions_cfg',return_value=None), \
                patch.dict('os.environ',{'NEWPOSE_EVAL_TERRAIN':'stairs_up','NEWPOSE_EVAL_LEVEL':'9'}):
            cfg=ThunderFreshPlayEnvCfg()
        self.assertIsNone(cfg.curriculum)
        self.assertEqual(list(cfg.scene.terrain.terrain_generator.sub_terrains),['pyramid_stairs_inv'])
        self.assertEqual(cfg.scene.terrain.fixed_level,9)
        self.assertFalse(cfg.gradual_boxes)
        cfg.apply_candidate_features()

    def test_play_camera_tracks_far_environment_without_changing_robot_or_training(self):
        positions=torch.tensor([[80.,-24.,.5],[0.,0.,1.]])
        before=positions.clone();calls=[]
        env=NS(scene={'robot':NS(data=NS(root_pos_w=NS(torch=positions)))},
            sim=NS(set_camera_view=lambda **kwargs:calls.append(kwargs)))
        mdp.follow_play_camera(env,None)
        self.assertEqual(calls,[{'eye':(82.,-26.,1.7),'target':(80.,-24.,.5)}])
        torch.testing.assert_close(positions,before)
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',return_value=None):
            training=ThunderNewposeEnvCfg();play=ThunderNewposePlayEnvCfg()
        self.assertIsNotNone(training.curriculum)
        self.assertFalse(hasattr(training.events,'play_camera_interval'))
        self.assertIsNone(play.curriculum)
        self.assertEqual(play.events.play_camera_interval.interval_range_s,(.02,.02))
        self.assertTrue(play.events.play_camera_interval.is_global_time)
        self.assertIs(play.events.play_camera_reset.func,mdp.follow_play_camera)

    def test_flat_mesh_is_closed_volume_with_unchanged_zero_height_surface(self):
        generator=terrain_cfg();cfg=generator.sub_terrains['flat'];cfg.size=generator.size
        for difficulty in (0.,1.):
            meshes,origin=cfg.function(difficulty,cfg)
            self.assertEqual(len(meshes),1)
            mesh=meshes[0]
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_volume)
            np.testing.assert_allclose(mesh.bounds,[[0.,0.,-.1],[8.,8.,0.]],atol=1e-12)
            np.testing.assert_allclose(origin,[4.,4.,0.],atol=1e-12)
            self.assertAlmostEqual(mesh.volume,6.4)

    def test_terrain_shape_and_nominal_limits(self):
        cfg=terrain_cfg()
        self.assertEqual((cfg.num_rows,cfg.num_cols,cfg.size),(10,20,(8.,8.)))
        self.assertEqual(list(cfg.sub_terrains),['flat','pyramid_stairs','pyramid_stairs_inv','boxes','random_rough','hf_pyramid_slope','hf_pyramid_slope_inv','wave'])
        self.assertAlmostEqual(sum(x.proportion for x in cfg.sub_terrains.values()),1.)
        self.assertEqual(cfg.sub_terrains['pyramid_stairs_inv'].step_height_range,(.05,.25))
        self.assertAlmostEqual(math.atan(cfg.sub_terrains['hf_pyramid_slope'].slope_range[1]),math.radians(40))
        for key in ('hf_pyramid_slope','hf_pyramid_slope_inv'):
            sub=cfg.sub_terrains[key];sub.size=cfg.size
            meshes,origin=pyramid_slope(1.,sub)
            self.assertTrue(meshes[0].is_watertight)
            self.assertAlmostEqual(origin[2],(-1 if sub.inverted else 1)*3.*math.tan(math.radians(40)))

    def test_nonzero_reward_weights_match_frozen_formal_yaml(self):
        frozen=Path(__file__).resolve().parent/'fixtures/legacy_19538_env.yaml'
        with frozen.open(encoding='utf-8') as file:old=yaml.load(file,Loader=yaml.BaseLoader)['rewards']
        expected={k:float(v['weight']) for k,v in old.items() if isinstance(v,dict) and float(v['weight'])!=0}
        cfg=RewardsCfg();actual={k:v.weight for k,v in vars(cfg).items() if hasattr(v,'weight') and v.weight!=0}
        self.assertEqual(actual,expected)

    def test_observation_history_and_global_commands(self):
        obs=ObservationsCfg();cmd=CommandsCfg().base_velocity
        self.assertEqual(obs.policy.frame.history_length,10)
        self.assertTrue(obs.policy.frame.flatten_history_dim)
        self.assertIs(obs.policy.frame.func,obs.single_obs.frame.func)
        self.assertFalse(obs.policy.enable_corruption)
        self.assertEqual(cmd.ranges.lin_vel_x,(-1.,1.))
        self.assertEqual(cmd.ranges.lin_vel_y,(-1.,1.))
        self.assertEqual(cmd.ranges.ang_vel_z,(-1.5,1.5))
        self.assertEqual(cmd.resampling_time_range,(10.,10.))

    def test_full_terrain_uses_newton_triangle_collision_path(self):
        with patch('thunder_lab3.tasks.thunder.env_cfg.robot_cfg',return_value=None):
            cfg=ThunderNewposeEnvCfg()
        self.assertFalse(cfg.sim.physics.solver_cfg.use_mujoco_contacts)
        self.assertIsNone(cfg.sim.physics.collision_cfg)
        self.assertEqual(cfg.sim.physics.default_shape_cfg.margin,0.)
        self.assertEqual(cfg.sim.dt,.0025)
        self.assertEqual(cfg.decimation,8)

    def test_stairs_remove_lateral_and_bound_yaw_keep_standing(self):
        term=NS(device='cpu')
        term.robot=NS(data=NS(root_quat_w=NS(torch=torch.tensor([[0.,0.,0.,1.]]).repeat(3,1))))
        term.vel_command_b=torch.tensor([[-1.,1.,1.5],[1.,-1.,-1.5],[.4,1.,1.]])
        term.stair_heading=torch.tensor([1.,-1.,1.])
        term.is_standing_env=torch.tensor([False,False,True])
        mdp.ThunderVelocityCommand._constrain_stairs(term,torch.ones(3,dtype=torch.bool))
        torch.testing.assert_close(term.vel_command_b,torch.tensor([[.25,0.,.2],[.5,0.,-.2],[0.,0.,0.]]))

    def test_curriculum_rejects_leg_touch_and_untracked_progress(self):
        n=3;called=[]
        term=NS(accumulate=lambda ids:None,travel_expected=torch.tensor([5.,5.,5.]),
            travel_progress=torch.tensor([4.,4.,1.]),travel_axis=torch.tensor([[1.,0.]]).repeat(n,1),
            travel_failed=torch.zeros(n,dtype=torch.bool),travel_peak=torch.full((n,),5.),travel_radius=torch.full((n,),5.))
        robot=NS(data=NS(root_pos_w=NS(torch=torch.tensor([[5.,0.,0.]]).repeat(n,1)),
                        projected_gravity_b=NS(torch=torch.tensor([[0.,0.,-1.]]).repeat(n,1))))
        terrain=NS(cfg=NS(terrain_generator=NS(size=(8.,8.))),terrain_levels=torch.zeros(n,dtype=torch.long),
                   update_env_origins=lambda ids,up,down:called.append((up.clone(),down.clone())))
        class Scene(dict):pass
        scene=Scene(robot=robot);scene.terrain=terrain;scene.env_origins=torch.zeros(n,3)
        env=NS(scene=scene,device='cpu',num_envs=n,command_manager=NS(get_term=lambda _:term),
            termination_manager=NS(terminated=torch.zeros(n,dtype=torch.bool)),
            _leg_contact={'touched':torch.tensor([False,True,False])})
        for ids,want_up,want_down in ((None,[True,False,False],[False,False,True]),
                (slice(None),[True,False,False],[False,False,True]),
                (slice(1,3),[False,False],[False,True]),
                ([0,2],[True,False],[False,True])):
            with patch.object(mdp,'stairs',return_value=torch.ones(n,dtype=torch.bool)):
                mdp.terrain_levels(env,ids)
            self.assertEqual(called[-1][0].tolist(),want_up)
            self.assertEqual(called[-1][1].tolist(),want_down)

    def test_reset_selector_normalization_and_invalid_indices(self):
        for selector,expected in ((None,[0,1,2,3]),(slice(None),[0,1,2,3]),
                (slice(1,None,2),[1,3]),(slice(None,None,-1),[3,2,1,0]),
                ([2,0],[2,0]),(torch.tensor([1,3]),[1,3]),([],[]),
                (torch.tensor([False,True,False,True]),[1,3])):
            ids=mdp.env_indices(selector,4,'cpu')
            self.assertEqual(ids.tolist(),expected);self.assertEqual(ids.dtype,torch.long)
        for invalid in ([.5],[-1],[4],[[1]],torch.tensor([True,False])):
            with self.assertRaises((ValueError,TypeError,IndexError)):
                mdp.env_indices(invalid,4,'cpu')

    def test_command_reset_only_clears_selected_environments(self):
        term=object.__new__(mdp.ThunderVelocityCommand);n=3
        term._env=NS(num_envs=n,device='cpu',_leg_contact={
            'duration':torch.ones(n),'touched':torch.ones(n,dtype=torch.bool),'failed':torch.ones(n,dtype=torch.bool)})
        term.robot=NS(data=NS(root_pos_w=NS(torch=torch.zeros(n,3)),
            root_quat_w=NS(torch=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1))))
        def fill():
            for name in ('travel_expected','travel_progress','travel_peak','travel_radius','tilt_duration'):
                setattr(term,name,torch.ones(n))
            term.travel_failed=torch.ones(n,dtype=torch.bool)
            term.travel_previous=torch.ones(n,2);term.travel_step=torch.ones(n,dtype=torch.long)
            term.travel_yaw=torch.ones(n);term.travel_axis=torch.ones(n,2)
        fill()
        with patch.object(mdp.lab_mdp.UniformVelocityCommand,'reset',return_value={}):
            term.reset(slice(1,2))
            self.assertEqual(term.travel_expected.tolist(),[1.,0.,1.])
            self.assertEqual(term.travel_step.tolist(),[1,-1,1])
            self.assertEqual(term._env._leg_contact['touched'].tolist(),[True,False,True])
            self.assertEqual(term.travel_axis[1].tolist(),[1.,0.])
            term.reset(None)
            self.assertEqual(term.travel_expected.tolist(),[0.,0.,0.])
            self.assertFalse(bool(term.travel_failed.any()))
            self.assertFalse(bool(term._env._leg_contact['touched'].any()))

    def test_root_reset_uses_actual_go2_api_and_aligns_only_selected_stairs(self):
        n=4
        defaults=torch.zeros(n,13);defaults[:,2]=.46;defaults[:,6]=1.;defaults[:,7]=1.
        pose=torch.full((n,7),99.);velocity=torch.full((n,6),99.)
        def write_pose(root_pose,env_ids):pose[env_ids]=root_pose
        def write_velocity(root_velocity,env_ids):velocity[env_ids]=root_velocity
        robot=NS(device='cpu',data=NS(default_root_state=NS(torch=defaults),root_pose_w=NS(torch=pose)),
            write_root_pose_to_sim_index=write_pose,write_root_velocity_to_sim_index=write_velocity)
        class Scene(dict):pass
        scene=Scene(robot=robot);scene.num_envs=n
        scene.env_origins=torch.tensor([[10.,0.,0.],[20.,0.,0.],[30.,0.,0.],[40.,0.,0.]])
        env=NS(scene=scene,num_envs=n,device='cpu')
        for selected in (slice(1,3),None,slice(None)):
            pose.fill_(99.);velocity.fill_(99.)
            ids=mdp.env_indices(selected,n,'cpu')
            with patch.object(mdp,'stairs',return_value=torch.tensor([False,True,True,False])):
                mdp.reset_stair_aligned(env,selected,{'x':(.1,.1),'yaw':(.6,.6)},{'x':(.2,.2)})
            torch.testing.assert_close(pose[ids,0],scene.env_origins[ids,0]+.1)
            torch.testing.assert_close(pose[ids,2],torch.full((len(ids),),.46))
            torch.testing.assert_close(velocity[ids,0],torch.full((len(ids),),1.2))
            want_yaw=torch.tensor([.6,0.,0.,.6])[ids]
            torch.testing.assert_close(mdp.euler_xyz_from_quat(pose[ids,3:7])[2],want_yaw)
            if len(ids)==2:
                self.assertTrue(bool((pose[[0,3]]==99.).all()))
                self.assertTrue(bool((velocity[[0,3]]==99.).all()))

    def test_candidate_composition_is_explicit_idempotent_and_reversible(self):
        cfg=NS(scene=NS(terrain=NS(terrain_generator=terrain_cfg())),rewards=RewardsCfg(),
            gradual_boxes=False,height_weight=0.,height_options=None,hip_extra_weight=0.,curriculum=None)
        apply=lambda:ThunderNewposeEnvCfg.apply_candidate_features(cfg)
        boxes=cfg.scene.terrain.terrain_generator.sub_terrains['boxes'];baseline=boxes.function
        apply();apply()
        self.assertIs(boxes.function,baseline)
        self.assertFalse(hasattr(cfg.rewards,'cropped_ground_height'))
        cfg.gradual_boxes=True;cfg.height_weight=-.5;cfg.hip_extra_weight=-.1
        cfg.height_options=dict(target_height=.46,tolerance=.03,max_reference_speed=.5,max_reference_lag=.04)
        apply();apply()
        self.assertIs(boxes.function,gradual_random_grid_terrain)
        self.assertIsInstance(cfg.height_options,HeightOptions)
        self.assertEqual(cfg.rewards.cropped_ground_height.weight,-.5)
        self.assertEqual(cfg.rewards.hip_extra.weight,-.1)
        previous=cfg.rewards.cropped_ground_height
        cfg.height_weight=.5
        with self.assertRaises(ValueError):apply()
        self.assertIs(cfg.rewards.cropped_ground_height,previous)
        cfg.gradual_boxes=False;cfg.height_weight=0.;cfg.hip_extra_weight=0.
        apply();apply()
        self.assertIs(boxes.function,baseline)
        self.assertFalse(hasattr(cfg.rewards,'cropped_ground_height'))
        self.assertFalse(hasattr(cfg.rewards,'hip_extra'))


if __name__=='__main__':unittest.main()
