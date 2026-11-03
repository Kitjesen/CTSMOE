"""Real Newton 53/309 observations with explicit material-schema change."""
import torch
from isaaclab.managers import ManagerTermBase

from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS, LEG_JOINTS, WHEEL_BODIES, normalized_features


def _ids(env):
    if not hasattr(env, "_newpose_joint_order"):
        env._newpose_joint_order = [env.scene["robot"].joint_names.index(n) for n in JOINTS]
    return env._newpose_joint_order


def policy_frame(env):
    return policy_tensor_frame(*_policy_inputs(env))


def _policy_inputs(env):
    robot = env.scene["robot"]
    ids = _ids(env)
    return (robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
        env.command_manager.get_command("base_velocity"),
        (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids[:12]],
        (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids], env.action_manager.action)


def policy_tensor_frame(angular_velocity, gravity, commands, relative_legs, relative_velocities, actions):
    """53 values: body omega 3, gravity 3, command 3, leg q 12, joint dq 16, previous action 16."""
    # Preserve the old reference contract: observation terms are clipped before scaling.
    return torch.cat(tuple(value.clamp(-100,100) * scale for value, scale in (
        (angular_velocity,.25),(gravity,1.),(commands,1.),(relative_legs,1.),
        (relative_velocities,.05),(actions,1.))), dim=-1)


class _SharedFrameNoise:
    """One sensor-noise sample per physics observation, shared by both actor groups."""
    def __init__(self):
        self.sample = None
        self.valid = None
        self.step = None

    def reset(self, env_ids=None):
        if self.valid is not None:
            self.valid[slice(None) if env_ids is None else env_ids] = False

    def apply(self, raw, step):
        if raw.ndim != 2 or raw.shape[1] != 53:
            raise ValueError("Expected one unscaled 53-value Thunder frame per environment")
        if self.sample is None:
            self.sample = torch.zeros_like(raw)
            self.valid = torch.zeros(len(raw), dtype=torch.bool, device=raw.device)
        elif self.sample.shape != raw.shape or self.sample.device != raw.device or self.sample.dtype != raw.dtype:
            raise ValueError("Observation noise cache shape/device/dtype changed")
        if self.step != step:
            self.valid.zero_()
            self.step = step
        refresh = ~self.valid
        # Go2 raw sensor amplitudes; commands and previous actions stay exact.
        amplitude = raw.new_tensor([.2]*3 + [.05]*3 + [0.]*3 + [.03]*12 + [2.]*16 + [0.]*16)
        self.sample[refresh] = (2*torch.rand_like(raw[refresh])-1)*amplitude
        self.valid[refresh] = True
        scale = raw.new_tensor([.25]*3 + [1.]*18 + [.05]*16 + [1.]*16)
        return (raw+self.sample).clamp(-100,100)*scale


class NoisyPolicyFrame(ManagerTermBase):
    """Fresh actor term with consistent current/history noise and partial reset.

    Use this class for both policy.frame and single_obs.frame. Their corruption
    flags must agree; disabling both also disables this term's noise. Sampling
    precedes clipping/scaling and ObservationManager owns the history afterward.
    The critic continues to call critic_frame directly and remains clean.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, "_thunder_policy_frame_noise"):
            env._thunder_policy_frame_noise = _SharedFrameNoise()
        self.state = env._thunder_policy_frame_noise

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env):
        observations = env.cfg.observations
        enabled = observations.policy.enable_corruption
        if enabled != observations.single_obs.enable_corruption:
            raise ValueError("Thunder policy and single_obs must use identical corruption settings")
        if not enabled:
            return policy_frame(env)
        raw = torch.cat(_policy_inputs(env), dim=-1)
        return self.state.apply(raw, int(env.common_step_counter))


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
    from thunder_lab3.assets.contract import validate_urdf
    robot=env.scene["robot"]
    position=env.action_manager.get_term("joint_pos")
    velocity=env.action_manager.get_term("joint_vel")
    if tuple(robot.joint_names[i] for i in position._joint_ids) != LEG_JOINTS:
        raise RuntimeError("Student leg action order does not match 19538")
    if tuple(robot.joint_names[i] for i in velocity._joint_ids) != JOINTS[12:]:
        raise RuntimeError("Student wheel action order does not match 19538")
    ids=_ids(env)
    pose=env.cfg.scene.robot.init_state.joint_pos
    expected=torch.tensor([pose[n] for n in JOINTS],device=env.device)
    torch.testing.assert_close(robot.data.default_joint_pos.torch[:,ids],expected[None].expand(env.num_envs,-1))
    offset=position._offset
    if not isinstance(offset,torch.Tensor) or offset.shape!=(env.num_envs,12) or not torch.isfinite(offset).all():
        raise ValueError("Expected finite per-environment position-action offsets [N,12]")
    nominal=expected[None,:12].expand_as(offset)
    delta=offset-nominal
    event=getattr(getattr(env.cfg,"events",None),"randomize_motor_zero_offset",None)
    allowed=None
    if event is None:
        torch.testing.assert_close(offset,nominal,atol=0.,rtol=0.)
    else:
        if event.params.get("action_term_name")!="joint_pos":
            raise ValueError("Motor zero-offset event must target joint_pos")
        allowed=torch.as_tensor(event.params.get("offset_range"),device=offset.device,dtype=offset.dtype)
        if allowed.shape!=(2,) or not torch.isfinite(allowed).all() or allowed[0]>allowed[1]:
            raise ValueError("Expected finite ordered motor zero-offset range")
        # Allow only floating-point subtraction error, not an unbounded offset.
        tolerance=4*torch.finfo(offset.dtype).eps*nominal.abs().clamp_min(1.)
        if ((delta<allowed[0]-tolerance)|(delta>allowed[1]+tolerance)).any():
            raise ValueError("Actual motor zero offset exceeds the configured randomization range")
    probe=torch.arange(16,device=env.device,dtype=torch.float32)[None].repeat(env.num_envs,1)-4
    probe[:,-1]=100.
    probe[1::2].neg_()
    env.action_manager.process_action(probe)
    scales=torch.tensor([.125,.25,.25]*4,device=env.device)
    targets=probe[:,:12]*scales+offset
    if fresh:
        import os
        limits=validate_urdf(os.environ["NEWPOSE_URDF"],pose=pose)["joint_position_limits"]
        bounds=torch.tensor([limits[name] for name in LEG_JOINTS],device=env.device)
        targets=targets.clamp(min=bounds[:,0],max=bounds[:,1])
    else:
        targets=targets.clamp(-100,100)
    torch.testing.assert_close(position.processed_actions,targets)
    torch.testing.assert_close(velocity.processed_actions,(probe[:,12:]*5.).clamp(-44,44))
    return {"action_order":list(JOINTS),"pose":dict(pose),"observation_clip_before_scale":True,
            "action_clip_after_scale":True,
            "motor_zero_offset":{"enabled":event is not None,
                "configured_range_rad":None if allowed is None else allowed.detach().cpu().tolist(),
                "measured_range_rad":[float(delta.min()),float(delta.max())]},
            "leg_target_bounds":"URDF hard limits" if fresh else "historical +/-100 rad",
            "history":"10 frames, oldest first, frame-major",
            "noise":("shared raw sensor noise before clip/scale" if env.cfg.observations.policy.enable_corruption
                     else "disabled")}
