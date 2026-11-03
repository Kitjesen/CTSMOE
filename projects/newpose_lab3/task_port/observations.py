"""Real Newton 53/309 observations with explicit material-schema change."""
import torch

from teacher.privileged import JOINTS, LEG_JOINTS, WHEEL_BODIES, normalized_features


def _ids(env):
    if not hasattr(env, "_newpose_joint_order"):
        env._newpose_joint_order = [env.scene["robot"].joint_names.index(n) for n in JOINTS]
    return env._newpose_joint_order


def policy_frame(env):
    robot = env.scene["robot"]
    ids = _ids(env)
    return policy_tensor_frame(robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
        env.command_manager.get_command("base_velocity"),
        (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids[:12]],
        (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids], env.action_manager.action)


def policy_tensor_frame(angular_velocity, gravity, commands, relative_legs, relative_velocities, actions):
    """53 values: body omega 3, gravity 3, command 3, leg q 12, joint dq 16, previous action 16."""
    # Preserve the old reference contract: observation terms are clipped before scaling.
    return torch.cat(tuple(value.clamp(-100,100) * scale for value, scale in (
        (angular_velocity,.25),(gravity,1.),(commands,1.),(relative_legs,1.),
        (relative_velocities,.05),(actions,1.))), dim=-1)


def actuator_ratios(robot, canonical):
    # Explicit PD stores gains in its actuator, while implicit gains belong to the articulation.
    kp = torch.zeros_like(robot.data.joint_pos.torch)
    kd = torch.zeros_like(kp)
    for actuator in robot.actuators.values():
        kp[:, actuator.joint_indices] = actuator.stiffness
        kd[:, actuator.joint_indices] = actuator.damping
    # Historical critic slots concatenate hip, thigh, calf groups, unlike the leg-major actor.
    gain_order = [robot.joint_names.index(f"{leg}_{kind}_joint")
                  for kind in ("hip", "thigh", "calf", "foot") for leg in ("fr", "fl", "rr", "rl")]
    normalizer = torch.tensor([6.93] * 12 + [1.0] * 4, device=kp.device, dtype=kp.dtype)
    return (kp[:, gain_order[:12]] / 90.).clamp(0,3), (kd[:, gain_order] / normalizer).clamp(0,3)


class NewtonWheelMaterial:
    """Read actual live wheel-shape mu/restitution; no PhysX dynamic-friction placeholder."""
    def __init__(self, env):
        robot = env.scene["robot"]
        model = env.sim.physics_manager.get_model()
        view = robot._root_view
        self.mu = view.get_attribute("shape_material_mu", model)[:, 0]
        self.restitution = view.get_attribute("shape_material_restitution", model)[:, 0]
        body_ids = [robot.body_names.index(name) for name in WHEEL_BODIES]
        backend_ids = robot.map_body_ids_to_backend(body_ids)
        shapes = [shape for body in backend_ids for shape in view.body_shapes[body]]
        if not shapes:
            raise ValueError("Wheel material observation has no collision shapes")
        self.shapes = torch.tensor(shapes, device=env.device, dtype=torch.long)

    def read(self):
        import warp as wp
        values = torch.stack((wp.to_torch(self.mu)[:, self.shapes].mean(dim=1),
                              wp.to_torch(self.restitution)[:, self.shapes].mean(dim=1)), dim=-1)
        if not torch.isfinite(values).all():
            raise RuntimeError("Nonfinite Newton wheel materials")
        return values.clamp(0,2)


def critic_frame(env):
    robot = env.scene["robot"]
    ids = _ids(env)
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "_newpose_material_reader"):
        env._newpose_material_reader = NewtonWheelMaterial(env)
        env._newpose_wheel_order = [sensor.body_names.index(n) for n in WHEEL_BODIES]
    height = (env.scene["height_scanner"].data.pos_w.torch[:, 2:3]
        - env.scene["height_scanner"].data.ray_hits_w.torch[..., 2] - 0.5).clamp(-1, 1)
    kp, kd = actuator_ratios(robot, ids)
    privileged = normalized_features(robot.data.joint_acc.torch[:, ids],
        robot.actuators.applied_effort.torch[:, ids],
        sensor.data.net_normal_forces_w.torch[:, env._newpose_wheel_order], env.episode_length_buf == 0)
    core = torch.cat((robot.data.root_lin_vel_b.torch, robot.data.root_ang_vel_b.torch,
        robot.data.projected_gravity_b.torch, env.command_manager.get_command("base_velocity"),
        (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids[:12]],
        (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids],
        env.action_manager.action), dim=-1).clamp(-100,100)
    result = torch.cat((core,height,kp,kd,env._newpose_material_reader.read(),privileged),dim=-1)
    if result.shape != (env.num_envs, 309) or not torch.isfinite(result).all():
        raise RuntimeError(f"Expected finite real-physics teacher observations [N,309], got {result.shape}")
    return result


def verify_student_contract(env, *, fresh=False):
    """Exercise actual Lab action processing; select fresh URDF bounds explicitly."""
    from runtime.thunder_contract import POSE, validate_urdf
    robot=env.scene["robot"]
    position=env.action_manager.get_term("joint_pos")
    velocity=env.action_manager.get_term("joint_vel")
    if tuple(robot.joint_names[i] for i in position._joint_ids) != LEG_JOINTS:
        raise RuntimeError("Student leg action order does not match 19538")
    if tuple(robot.joint_names[i] for i in velocity._joint_ids) != JOINTS[12:]:
        raise RuntimeError("Student wheel action order does not match 19538")
    ids=_ids(env)
    expected=torch.tensor([POSE[n] for n in JOINTS],device=env.device)
    torch.testing.assert_close(robot.data.default_joint_pos.torch[:,ids],expected[None].expand(env.num_envs,-1))
    probe=torch.arange(16,device=env.device,dtype=torch.float32)[None].repeat(env.num_envs,1)-4
    probe[:,-1]=100.
    probe[1::2].neg_()
    env.action_manager.process_action(probe)
    scales=torch.tensor([.125,.25,.25]*4,device=env.device)
    targets=probe[:,:12]*scales+expected[:12]
    if fresh:
        import os
        limits=validate_urdf(os.environ["NEWPOSE_URDF"])["joint_position_limits"]
        bounds=torch.tensor([limits[name] for name in LEG_JOINTS],device=env.device)
        targets=targets.clamp(min=bounds[:,0],max=bounds[:,1])
    else:
        targets=targets.clamp(-100,100)
    torch.testing.assert_close(position.processed_actions,targets)
    torch.testing.assert_close(velocity.processed_actions,(probe[:,12:]*5.).clamp(-44,44))
    return {"action_order":list(JOINTS),"pose":POSE,"observation_clip_before_scale":True,
            "action_clip_after_scale":True,
            "leg_target_bounds":"URDF hard limits" if fresh else "historical +/-100 rad",
            "history":"10 frames, oldest first, frame-major","noise":"disabled in interface probe"}
