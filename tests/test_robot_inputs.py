"""CPU tensor contract tests: python -m unittest tests.test_robot_inputs -v."""
from types import SimpleNamespace
import unittest

import torch

from thunder_lab3.assets.contract import POSE
from thunder_lab3.tasks.thunder.mdp.observations import critic_frame, policy_frame, policy_tensor_frame
from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS, WHEEL_BODIES, normalized_features


def proxy(value):
    return SimpleNamespace(torch=value)


def fake_env():
    """Deliberately scramble backend order to exercise canonical observation indexing."""
    names = list(reversed(JOINTS))
    defaults = torch.tensor([[POSE[name] for name in names]]).repeat(2, 1)
    values = torch.arange(16, dtype=torch.float32)[None].repeat(2, 1)
    data = SimpleNamespace(
        joint_pos=proxy(defaults + values / 100), default_joint_pos=proxy(defaults),
        joint_vel=proxy(values), default_joint_vel=proxy(torch.zeros_like(values)),
        joint_acc=proxy(values * 100), root_lin_vel_b=proxy(torch.tensor([[1., 2., 3.]]).repeat(2, 1)),
        root_ang_vel_b=proxy(torch.tensor([[4., 5., 6.]]).repeat(2, 1)),
        projected_gravity_b=proxy(torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
    gains = SimpleNamespace(joint_indices=slice(None),
        stiffness=torch.tensor([[0. if "foot" in name else 90. for name in names]]).repeat(2, 1),
        damping=torch.tensor([[1. if "foot" in name else 6.93 for name in names]]).repeat(2, 1))
    class Actuators(dict):
        pass
    actuators = Actuators(all=gains)
    actuators.applied_effort = proxy(values * 2)
    robot = SimpleNamespace(data=data, joint_names=names, actuators=actuators)
    contacts = SimpleNamespace(body_names=list(reversed(WHEEL_BODIES)), data=SimpleNamespace(
        net_normal_forces_w=proxy(torch.tensor([[[0., 0., v] for v in (400., 300., 200., 100.)]]).repeat(2, 1, 1))))
    hit = torch.zeros(2, 187, 3)
    hit[:, :, 2] = torch.linspace(0., .2, 187)
    scanner = SimpleNamespace(data=SimpleNamespace(
        pos_w=proxy(torch.tensor([[0., 0., .46]]).repeat(2, 1)), ray_hits_w=proxy(hit)))
    env = SimpleNamespace(scene={"robot": robot, "contact_forces": contacts, "height_scanner": scanner},
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.5, -.5, 1.5]]).repeat(2, 1)),
        action_manager=SimpleNamespace(action=values / 10), episode_length_buf=torch.tensor([0, 1]), num_envs=2,
        _newpose_material_reader=SimpleNamespace(read=lambda: torch.tensor([[1., 0.]]).repeat(2, 1)),
        _newpose_wheel_order=[3, 2, 1, 0])
    return env


class RobotInputTests(unittest.TestCase):
    def test_student_53_has_body_signals_and_canonical_joint_order(self):
        env = fake_env()
        frame = policy_frame(env)
        self.assertEqual(frame.shape, (2, 53))
        torch.testing.assert_close(frame[:, :3], env.scene["robot"].data.root_ang_vel_b.torch * .25)
        torch.testing.assert_close(frame[:, 6:9], env.command_manager.get_command("base_velocity"))
        ids = [env.scene["robot"].joint_names.index(name) for name in JOINTS]
        relative = env.scene["robot"].data.joint_pos.torch-env.scene["robot"].data.default_joint_pos.torch
        torch.testing.assert_close(frame[:, 9:21], relative[:, ids[:12]])
        torch.testing.assert_close(frame[:, 21:37], env.scene["robot"].data.joint_vel.torch[:, ids] * .05)
        torch.testing.assert_close(frame[:, 37:], env.action_manager.action)

    def test_legacy_observation_clip_precedes_normalization(self):
        frame = policy_tensor_frame(torch.full((1, 3), 200.), torch.zeros(1, 3), torch.zeros(1, 3),
                                    torch.zeros(1, 12), torch.full((1, 16), -200.), torch.zeros(1, 16))
        torch.testing.assert_close(frame[:, :3], torch.full((1, 3), 25.))
        torch.testing.assert_close(frame[:, 21:37], torch.full((1, 16), -5.))

    def test_teacher_309_keeps_height_and_normal_contact_inputs(self):
        env = fake_env()
        frame = critic_frame(env)
        self.assertEqual(frame.shape, (2, 309))
        torch.testing.assert_close(frame[:, :3], env.scene["robot"].data.root_lin_vel_b.torch)
        expected_height = .46-env.scene["height_scanner"].data.ray_hits_w.torch[:, :, 2]-.5
        torch.testing.assert_close(frame[:, 56:243], expected_height)
        self.assertGreater(float(frame[0, 56:243].std()), .01)
        torch.testing.assert_close(frame[:, 243:271], torch.ones(2, 28))
        torch.testing.assert_close(frame[:, 271:273], torch.tensor([[1., 0.]]).repeat(2, 1))
        torch.testing.assert_close(frame[:, -4:], torch.tensor([[.1, .2, .3, .4]]).repeat(2, 1))
        self.assertEqual(torch.count_nonzero(frame[0, 273:289]).item(), 0)
        self.assertGreater(torch.count_nonzero(frame[1, 273:289]).item(), 0)

    def test_privileged_force_is_vector_norm_and_reset_acceleration_is_masked(self):
        acceleration, torque = torch.ones(2, 16)*100, torch.ones(2, 16)*2
        normal = torch.tensor([[[3., 4., 0.]]*4]*2)
        features = normalized_features(acceleration, torque, normal, torch.tensor([True, False]))
        torch.testing.assert_close(features[:, -4:], torch.full((2, 4), .005))
        torch.testing.assert_close(features[0, :16], torch.zeros(16))
        torch.testing.assert_close(features[1, :16], torch.full((16,), .01))
        normal[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            normalized_features(acceleration, torque, normal, torch.tensor([False, False]))


if __name__ == "__main__":
    unittest.main()
