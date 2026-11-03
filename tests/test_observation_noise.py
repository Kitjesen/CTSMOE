"""Fresh actor noise must be the same sensor sample in current frame and history."""
from types import SimpleNamespace as NS
import unittest

import torch
from isaaclab.managers import ObservationTermCfg

from thunder_lab3.tasks.thunder.mdp.observations import NoisyPolicyFrame, policy_frame, _SharedFrameNoise
from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS


def make_env(n=8):
    proxy=lambda value:NS(torch=value)
    data=NS(root_ang_vel_b=proxy(torch.zeros(n,3)),projected_gravity_b=proxy(torch.zeros(n,3)),
        joint_pos=proxy(torch.zeros(n,16)),default_joint_pos=proxy(torch.zeros(n,16)),
        joint_vel=proxy(torch.zeros(n,16)),default_joint_vel=proxy(torch.zeros(n,16)))
    return NS(num_envs=n,device='cpu',common_step_counter=0,
        cfg=NS(observations=NS(policy=NS(enable_corruption=True),single_obs=NS(enable_corruption=True))),
        scene={'robot':NS(joint_names=list(JOINTS),data=data)},
        command_manager=NS(get_command=lambda name:torch.zeros(n,3)),
        action_manager=NS(action=torch.zeros(n,16)))


class ObservationNoiseTests(unittest.TestCase):
    def test_raw_noise_amplitudes_and_unmodified_commands_actions(self):
        torch.manual_seed(42)
        env=make_env(2048)
        term=NoisyPolicyFrame(ObservationTermCfg(func=NoisyPolicyFrame),env)
        sample=term(env)
        maximum=torch.tensor([.05]*3+[.05]*3+[0.]*3+[.03]*12+[.1]*16+[0.]*16)
        self.assertTrue(bool((sample.abs()<=maximum).all()))
        self.assertTrue(bool((sample[:,maximum>0].abs().amax(0)>.9*maximum[maximum>0]).all()))
        self.assertTrue(torch.equal(sample[:,6:9],torch.zeros(2048,3)))
        self.assertTrue(torch.equal(sample[:,37:],torch.zeros(2048,16)))
        # The clean source used by the critic is neither mutated nor made noisy.
        self.assertTrue(torch.equal(policy_frame(env),torch.zeros(2048,53)))

    def test_two_manager_terms_share_sample_across_reads_steps_and_partial_reset(self):
        torch.manual_seed(9)
        env=make_env()
        a=NoisyPolicyFrame(ObservationTermCfg(func=NoisyPolicyFrame),env)
        b=NoisyPolicyFrame(ObservationTermCfg(func=NoisyPolicyFrame),env)
        first=a(env)
        self.assertTrue(torch.equal(first,b(env)))
        self.assertTrue(torch.equal(first,a(env)))
        env.common_step_counter+=1
        second=b(env)
        self.assertFalse(torch.equal(first,second))
        self.assertTrue(torch.equal(second,a(env)))
        # Both observation groups reset; this must invalidate only these rows.
        a.reset(torch.tensor([1,4]));b.reset(torch.tensor([1,4]))
        reset=a(env)
        self.assertTrue(torch.equal(reset,b(env)))
        self.assertTrue(torch.equal(reset[[0,2,3,5,6,7]],second[[0,2,3,5,6,7]]))
        self.assertFalse(torch.equal(reset[[1,4]],second[[1,4]]))

    def test_corruption_flags_disable_without_touching_clean_frame(self):
        env=make_env()
        term=NoisyPolicyFrame(ObservationTermCfg(func=NoisyPolicyFrame),env)
        env.cfg.observations.policy.enable_corruption=False
        with self.assertRaises(ValueError):term(env)
        env.cfg.observations.single_obs.enable_corruption=False
        self.assertTrue(torch.equal(term(env),policy_frame(env)))

    def test_noise_precedes_clip_and_scale(self):
        state=_SharedFrameNoise()
        raw=torch.full((2,53),100.)
        state.apply(raw,0)
        # Install a deterministic valid sensor sample to expose processing order.
        state.sample[:]=0.
        state.sample[:,0]=.2
        state.sample[:,21]=2.
        result=state.apply(raw,0)
        self.assertTrue(torch.equal(result[:,0],torch.full((2,),25.)))
        self.assertTrue(torch.equal(result[:,21],torch.full((2,),5.)))
        state.sample[:,0]=-.2
        result=state.apply(raw,0)
        torch.testing.assert_close(result[:,0],torch.full((2,),24.95))


if __name__=='__main__':
    unittest.main()
