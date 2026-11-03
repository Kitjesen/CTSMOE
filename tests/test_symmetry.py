"""16-action Thunder mirror semantics and the real upstream CTS batch interface."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensordict import TensorDict

from thunder_lab3.assets.contract import A_POSE,validate_urdf
from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS
from thunder_lab3.tasks.thunder.mdp.symmetry import (ThunderSymmetry,ThunderSymmetryMapper,
    JOINT_PERM,FOOT_PERM,GAIN_PERM,reflected_ray_indices)
from tools.audits.geometry_audit import leg_fk,S


URDF=Path(__file__).resolve().parents[1]/'source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf'


def make_env(ordering='xy'):
    x=torch.linspace(-.8,.8,17);y=torch.linspace(-.5,.5,11)
    if ordering=='xy':
        yy,xx=torch.meshgrid(y,x,indexing='ij')
    else:
        xx,yy=torch.meshgrid(x,y,indexing='ij')
    starts=torch.stack((xx.flatten(),yy.flatten(),torch.full((187,),20.)),dim=-1)
    directions=torch.tensor([0.,0.,-1.]).expand_as(starts).clone()
    robot=NS(joint_names=list(JOINTS),data=NS(default_joint_pos=NS(torch=torch.tensor([[A_POSE[n] for n in JOINTS]]))))
    return NS(scene={'robot':robot,'height_scanner':NS(ray_starts=NS(torch=starts[None]),
        ray_directions=NS(torch=directions[None]))}),starts


def observations(count=5):
    frame=torch.randn(count,10,53)
    return TensorDict({'policy':frame.flatten(-2),'single_obs':frame[:,-1].clone(),
        'critic':torch.randn(count,309)},batch_size=[count])


class SymmetryTests(unittest.TestCase):
    def test_all_observation_columns_and_actions_are_involutions(self):
        for ordering in ('xy','yx'):
            env,starts=make_env(ordering);mapper=ThunderSymmetryMapper(env)
            original=observations();twice=mapper.reverse_obs(mapper.reverse_obs(original))
            for key in original.keys():self.assertTrue(torch.equal(original[key],twice[key]),key)
            mirrored=mapper.reverse_obs(original)
            self.assertTrue(torch.equal(mirrored['policy'].reshape(5,10,53)[:,-1],mirrored['single_obs']))
            action=torch.randn(5,16)
            self.assertTrue(torch.equal(action,mapper.reverse_joints(mapper.reverse_joints(action))))
            signal=starts[:,1].square()+starts[:,0]*3
            odd=starts[:,1]
            self.assertTrue(torch.equal(signal[mapper.height_perm],signal))
            torch.testing.assert_close(odd[mapper.height_perm],-odd,atol=1e-7,rtol=0.)

    def test_critic_semantics_include_force_norm_and_gain_group_order(self):
        env,_=make_env();mapper=ThunderSymmetryMapper(env)
        original=torch.arange(309,dtype=torch.float32)[None]
        result=mapper.reverse_critic(original)
        self.assertTrue(torch.equal(result[:,243:255],original[:,243:255][:,list(GAIN_PERM[:12])]))
        self.assertTrue(torch.equal(result[:,255:271],original[:,255:271][:,list(GAIN_PERM)]))
        self.assertTrue(torch.equal(result[:,271:273],original[:,271:273]))
        for start in (24,40,273,289):
            self.assertTrue(torch.equal(result[:,start:start+16],-original[:,start:start+16][:,list(JOINT_PERM)]))
        self.assertTrue(torch.equal(result[:,305:309],original[:,305:309][:,list(FOOT_PERM)]))

    def test_target_clipping_pd_torque_and_joint_power_commute_with_mirror(self):
        env,_=make_env();mapper=ThunderSymmetryMapper(env)
        limits=validate_urdf(URDF,pose=A_POSE)['joint_position_limits']
        bounds=torch.tensor([limits[name] for name in JOINTS[:12]])
        default=torch.tensor([A_POSE[name] for name in JOINTS[:12]])
        scale=torch.tensor([.125,.25,.25]*4+[5.]*4)
        action=torch.randn(20,16)*20
        def targets(a):
            return torch.cat(((a[:,:12]*scale[:12]+default).clamp(bounds[:,0],bounds[:,1]),
                              (a[:,12:]*scale[12:]).clamp(-44.,44.)),dim=-1)
        torch.testing.assert_close(targets(mapper.reverse_joints(action)),mapper.reverse_joints(targets(action)))
        q,dq=torch.randn(20,16),torch.randn(20,16)
        kp,kd=torch.rand(20,16)*100,torch.rand(20,16)*10
        target=targets(action)
        torque=kp*(target-q)-kd*dq
        mirrored_torque=mapper.permute_joints(kp)*(mapper.reverse_joints(target)-mapper.reverse_joints(q))-mapper.permute_joints(kd)*mapper.reverse_joints(dq)
        torch.testing.assert_close(mirrored_torque,mapper.reverse_joints(torque))
        torch.testing.assert_close((torque*dq).sum(1),(mirrored_torque*mapper.reverse_joints(dq)).sum(1))

    def test_actual_urdf_axes_require_negative_wheel_speed_and_torque(self):
        root=ET.parse(URDF).getroot()
        rng=np.random.default_rng(42)
        for right,left in (('fr','fl'),('rr','rl')):
            for _ in range(16):
                q=rng.uniform([-.2,-1.2,-1.,-2.],[.2,1.2,1.,2.])
                _,axis=leg_fk(root,right,q)
                _,mirrored_axis=leg_fk(root,left,-q)
                speed=2.3
                # Angular velocity is axial; reflection is det(S)*S, not S.
                np.testing.assert_allclose(-speed*mirrored_axis,-S@(speed*axis),atol=1e-6)
                velocity=np.cross(speed*axis,[0.,0.,-.093])
                mirrored_velocity=np.cross(-speed*mirrored_axis,[0.,0.,-.093])
                np.testing.assert_allclose(mirrored_velocity,S@velocity,atol=1e-6)
        # Every canonical joint axis, not only the wheels, needs a sign flip.
        for right,left in zip((0,1,2,6,7,8,12,14),(3,4,5,9,10,11,13,15)):
            a=np.fromstring(root.find(f"joint[@name='{JOINTS[right]}']/axis").get('xyz'),sep=' ')
            b=np.fromstring(root.find(f"joint[@name='{JOINTS[left]}']/axis").get('xyz'),sep=' ')
            np.testing.assert_allclose(-b,-S@a,atol=1e-12)

    def test_actual_upstream_segmented_cts_interface_preserves_teacher_student_order(self):
        env,_=make_env();symmetry=ThunderSymmetry(env)
        obs=observations(3);actions=torch.randn(3,16);mu=torch.randn(3,16);sigma=torch.rand(3,16)+.1
        ids=torch.arange(3,dtype=torch.float32)[:,None]
        out=symmetry.augment_moe_cts_batch(obs,actions,ids,ids,ids,ids,mu,sigma,teacher_samples=2)
        self.assertEqual(out[0].batch_size,torch.Size([6]))
        self.assertEqual(out[2][:,0].tolist(),[0.,1.,0.,1.,2.,2.])
        torch.testing.assert_close(out[1][2:4],symmetry.mapper.reverse_joints(actions[:2]))
        torch.testing.assert_close(out[6][2:4],symmetry.mapper.reverse_joints(mu[:2]))
        torch.testing.assert_close(out[7][2:4],symmetry.mapper.permute_joints(sigma[:2]))
        self.assertTrue(bool((out[7]>0).all()))
        self.assertEqual(symmetry.get_original_mask(2,1,'cpu').tolist(),[True,True,False,False,True,False])

    def test_rejects_incompatible_scan_pose_and_observation_width(self):
        env,starts=make_env();mapper=ThunderSymmetryMapper(env)
        invalid=starts.clone();invalid[0,1]=.333
        with self.assertRaises(ValueError):reflected_ray_indices(invalid)
        env.scene['robot'].data.default_joint_pos.torch[0,0]+=.01
        with self.assertRaises(ValueError):ThunderSymmetryMapper(env)
        with self.assertRaises(ValueError):mapper.reverse_joints(torch.zeros(1,12))
        with self.assertRaises(ValueError):mapper.reverse_critic(torch.zeros(1,274))


if __name__=='__main__':
    unittest.main()
