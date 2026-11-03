"""Diagnostic validation exercises real Lab action processing with actual Go2 zero-offset DR."""
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction,JointVelocityAction
from go2_rl_lab.tasks.mixed.mdp.events import randomize_action_joint_pos_offset

from thunder_lab3.assets.contract import A_POSE,validate_urdf
from thunder_lab3.tasks.thunder.mdp.observations import verify_student_contract
from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS


URDF=Path(__file__).resolve().parents[1]/'source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf'


def fixture(randomized=True):
    n=8
    defaults=torch.tensor([[A_POSE[name] for name in JOINTS]]).repeat(n,1)
    robot=NS(joint_names=list(JOINTS),data=NS(default_joint_pos=NS(torch=defaults)))
    limits=validate_urdf(URDF,pose=A_POSE)['joint_position_limits']
    # Allocate only the tensors used by the unmodified real process_actions;
    # constructing a physics articulation is unnecessary for this boundary check.
    position=object.__new__(JointPositionAction)
    position.cfg=NS(clip={'enabled':True});position._joint_ids=list(range(12))
    position._raw_actions=torch.zeros(n,12);position._processed_actions=torch.zeros(n,12)
    position._offset=defaults[:,:12].clone()
    position._scale=torch.tensor([.125,.25,.25]*4)[None]
    position._clip=torch.tensor([limits[name] for name in JOINTS[:12]])[None].repeat(n,1,1)
    velocity=object.__new__(JointVelocityAction)
    velocity.cfg=NS(clip={'enabled':True});velocity._joint_ids=list(range(12,16))
    velocity._raw_actions=torch.zeros(n,4);velocity._processed_actions=torch.zeros(n,4)
    velocity._offset=torch.zeros(n,4);velocity._scale=5.
    velocity._clip=torch.tensor([-44.,44.])[None,None].repeat(n,4,1)
    terms={'joint_pos':position,'joint_vel':velocity}
    def process(value):
        position.process_actions(value[:,:12])
        velocity.process_actions(value[:,12:])
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.num_envs=n
    event=NS(params={'action_term_name':'joint_pos','offset_range':(-.035,.035)}) if randomized else None
    env=NS(num_envs=n,device='cpu',scene=scene,
        cfg=NS(scene=NS(robot=NS(init_state=NS(joint_pos=A_POSE))),
            events=NS(randomize_motor_zero_offset=event),observations=NS(policy=NS(enable_corruption=True))),
        action_manager=NS(get_term=lambda name:terms[name],process_action=process))
    if randomized:
        torch.manual_seed(42)
        randomize_action_joint_pos_offset(env,None,**event.params)
    return env,position


class ActionOffsetContractTests(unittest.TestCase):
    def verify(self,env):
        with patch.dict(os.environ,{'NEWPOSE_URDF':str(URDF)}):
            return verify_student_contract(env,fresh=True)

    def test_real_randomization_and_actual_action_processing_pass(self):
        env,position=fixture()
        result=self.verify(env)
        measured=result['motor_zero_offset']['measured_range_rad']
        self.assertTrue(result['motor_zero_offset']['enabled'])
        self.assertGreaterEqual(measured[0],-.035001)
        self.assertLessEqual(measured[1],.035001)
        self.assertLess(measured[0],-.02)
        self.assertGreater(measured[1],.02)
        self.assertGreater(float(position.processed_actions.abs().max()),0.)

    def test_out_of_range_offset_fails_before_exercising_actions(self):
        env,position=fixture()
        position._offset[0,0]=A_POSE['fr_hip_joint']+.04
        with self.assertRaisesRegex(ValueError,'exceeds'):self.verify(env)
        self.assertEqual(torch.count_nonzero(position.raw_actions).item(),0)

    def test_no_event_requires_exact_nominal_offset(self):
        env,position=fixture(randomized=False)
        self.assertEqual(self.verify(env)['motor_zero_offset'],
            {'enabled':False,'configured_range_rad':None,'measured_range_rad':[0.,0.]})
        position._offset[0,0]+=1.e-6
        with self.assertRaises(AssertionError):self.verify(env)

    def test_valid_offset_does_not_hide_action_scale_regression(self):
        env,position=fixture()
        position._scale[:]=.9
        with self.assertRaises(AssertionError):self.verify(env)


if __name__=='__main__':
    unittest.main()
