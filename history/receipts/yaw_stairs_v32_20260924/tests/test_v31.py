import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from omni_stairs_v31 import body_command, world_xy  # noqa: E402
from landing_v31 import penalties  # noqa: E402


def test_nonroute_reset_draws_no_command_groups():
    from omni_stairs_v31 import sample_groups
    result = sample_groups(0, 'cpu')
    assert result.shape == (0,) and result.dtype == torch.long


def test_lateral_body_direction_crosses_riser_for_all_four_exits():
    for axis in range(4):
        for sign in (-1., 1.):
            group = torch.tensor([1])
            command = body_command(group, torch.tensor([sign]), torch.ones(1), torch.tensor([.3]), torch.zeros(1), torch.zeros(1, dtype=torch.bool))
            velocity = world_xy(command, torch.tensor([axis * math.pi / 2 - sign * math.pi / 2]))
            expected = torch.tensor([[.3 * math.cos(axis * math.pi / 2), .3 * math.sin(axis * math.pi / 2)]])
            torch.testing.assert_close(velocity, expected)
            assert command[0, 0] == command[0, 2] == 0


def test_mixed_command_activates_at_geometry_trigger_not_clock_alone():
    groups = torch.tensor([2, 3, 4])
    command = body_command(groups, torch.ones(3), torch.ones(3), torch.full((3,), .5), torch.ones(3), torch.zeros(3, dtype=torch.bool))
    assert not command[:, 1:].any()
    command = body_command(groups, torch.ones(3), torch.ones(3), torch.full((3,), .5), torch.ones(3), torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(command[:, 1:], torch.tensor([[0., .25], [.15, 0.], [.15, .25]]))
    after = body_command(groups, torch.ones(3), torch.ones(3), torch.full((3,), .5), torch.full((3,), 3.1), torch.ones(3, dtype=torch.bool))
    assert not after[:, 1:].any()


def test_landing_penalties_distinguish_support_fast_impact_and_no_event():
    force = torch.tensor([[100., 100., 100., 100.], [600., 0., 0., 0.], [600., 0., 0., 0.]])
    downward = torch.tensor([[.1] * 4, [1., 0., 0., 0.], [1., 0., 0., 0.]])
    event = torch.tensor([[True] * 4, [True, False, False, False], [False] * 4])
    f, v = penalties(force, downward, event, event, torch.full((3,), 400.))
    torch.testing.assert_close(f, torch.tensor([0., 2.25, 0.]))
    torch.testing.assert_close(v, torch.tensor([0., .5625, 0.]))


def test_landing_sensor_preserves_original_critic_history(monkeypatch):
    import landing_v31
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', SimpleNamespace(RewardTermCfg=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'isaaclab.sensors', SimpleNamespace(ContactSensorCfg=lambda **kw: SimpleNamespace(**kw)))
    cfg = SimpleNamespace(scene=SimpleNamespace(contact_forces=SimpleNamespace(history_length=3)),
                          sim=SimpleNamespace(dt=.005), rewards=SimpleNamespace())
    landing_v31.configure(cfg)
    assert cfg.scene.contact_forces.history_length == 3
    assert cfg.scene.landing_contacts.history_length == 4
    assert cfg.scene.landing_contacts.update_period == .005
    assert cfg.rewards.landing_velocity_v31.weight == -1.


def test_groups_disable_side_and_keep_mixed():
    from omni_stairs_v31 import sample_groups
    torch.manual_seed(42)
    groups = sample_groups(10000, 'cpu')
    assert (groups == 1).sum() == 0
    assert set(groups.tolist()) == {0, 2, 3, 4}


def test_yaw_coverage_and_quaternion_preserves_tilt():
    from omni_stairs_v31 import sample_yaw_offsets, rotate_world_yaw
    torch.manual_seed(42)
    angles = sample_yaw_offsets(10000, 'cpu')
    assert angles.abs().max() <= math.pi/4
    assert angles.min() < -.77 and angles.max() > .77
    assert .23 < (angles == 0).float().mean() < .27
    q = torch.tensor([[.9, .1, .2, .3]])
    q = q/q.norm(dim=1, keepdim=True)
    result = rotate_world_yaw(q, torch.tensor([.7]))
    torch.testing.assert_close(result.norm(dim=1), torch.ones(1))
    # A world-Z rotation preserves the world-Z component of body axes.
    def body_z_world_z(t):
        return 1-2*(t[:,1]**2+t[:,2]**2)
    torch.testing.assert_close(body_z_world_z(result), body_z_world_z(q))


def test_reset_applies_yaw_to_all_stairs_after_parent(monkeypatch):
    import omni_stairs_v31 as routes
    calls = []
    state = torch.zeros(3, 13)
    state[:,3] = 1
    robot = SimpleNamespace(data=SimpleNamespace(root_state_w=state))
    robot.write_root_pose_to_sim = lambda pose, env_ids: calls.append((pose.clone(), env_ids.clone()))
    scene = {'robot': robot}
    class Scene(dict):
        terrain = SimpleNamespace(terrain_types=torch.tensor([6, 8, 0]))
    scene = Scene(scene)
    cmd = SimpleNamespace(stair_mask=lambda: torch.tensor([True, True, False]))
    env = SimpleNamespace(num_envs=3, device='cpu', scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _:cmd))
    def original(e, ids, pose, velocity, asset):
        e.stair_axis = torch.zeros(3, dtype=torch.long)
        e.stair_motion_sign = torch.ones(3)
    monkeypatch.setattr(routes, 'sample_groups', lambda n,d:torch.zeros(n, dtype=torch.long))
    monkeypatch.setattr(routes, 'sample_yaw_offsets', lambda n,d:torch.tensor([.5,-.6]))
    monkeypatch.setitem(sys.modules,'robot_lab.tasks.go2.mdp.events', SimpleNamespace(reset_root_state_uniform=lambda *a,**k:None))
    routes.reset_root(env, torch.arange(3), {}, {}, SimpleNamespace(name='robot'), original)
    assert calls[0][1].tolist() == [0,1]
    torch.testing.assert_close(2*torch.atan2(calls[0][0][:,6],calls[0][0][:,3]), torch.tensor([.5,-.6]))
    assert env.stair_initial_yaw_offset[2] == 0
