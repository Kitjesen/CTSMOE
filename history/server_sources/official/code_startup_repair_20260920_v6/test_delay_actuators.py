"""Standalone parity test for the optimized Thunder delayed actuators.

Run this file with the Isaac Lab Python interpreter.  It intentionally starts
Kit only from ``main`` so importing the module has no simulator side effects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


def main() -> None:
    from isaaclab.app import AppLauncher

    result_path = Path(__file__).with_suffix(".result.json")
    result_path.unlink(missing_ok=True)
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        import torch

        from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
        from isaaclab.utils.types import ArticulationActions

        from actuators import (
            FastThunderDelayedDCMotor,
            FastWheelDelayedPDActuator,
            ThunderDelayedDCMotor,
            ThunderDelayedDCMotorCfg,
        )

        device = "cuda:0"
        num_envs = 17
        num_joints = 4
        joint_names = [f"joint_{index}" for index in range(num_joints)]

        def leg_cfg():
            return ThunderDelayedDCMotorCfg(
                joint_names_expr=[".*"],
                effort_limit=120.0,
                effort_limit_sim=120.0,
                saturation_effort=120.0,
                velocity_limit=17.48,
                velocity_limit_sim=17.48,
                stiffness=120.0,
                damping=8.0,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            )

        def wheel_cfg():
            return DelayedPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=17.0,
                effort_limit_sim=17.0,
                velocity_limit=44.0,
                velocity_limit_sim=44.0,
                stiffness=0.0,
                damping=1.0,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            )

        def build(actuator_type, cfg):
            return actuator_type(
                cfg=cfg,
                joint_names=joint_names,
                joint_ids=slice(None),
                num_envs=num_envs,
                device=device,
                stiffness=0.0,
                damping=0.0,
                armature=0.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=torch.inf,
                velocity_limit=torch.inf,
            )

        group_specs = (
            ("hip", ThunderDelayedDCMotor, FastThunderDelayedDCMotor, leg_cfg),
            ("thigh", ThunderDelayedDCMotor, FastThunderDelayedDCMotor, leg_cfg),
            ("calf", ThunderDelayedDCMotor, FastThunderDelayedDCMotor, leg_cfg),
            ("wheel", DelayedPDActuator, FastWheelDelayedPDActuator, wheel_cfg),
        )
        reference = {name: build(ref_type, cfg_factory()) for name, ref_type, _, cfg_factory in group_specs}
        optimized = {name: build(fast_type, cfg_factory()) for name, _, fast_type, cfg_factory in group_specs}

        def rng_state():
            return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()

        def restore_rng(state):
            cpu_state, cuda_states = state
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)

        def assert_rng_equal(left, right):
            assert torch.equal(left[0], right[0]), "CPU RNG state diverged"
            assert len(left[1]) == len(right[1])
            assert all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True)), "CUDA RNG state diverged"

        def lag(actuator):
            position = actuator.positions_delay_buffer._time_lags
            assert torch.equal(position, actuator.velocities_delay_buffer._time_lags)
            assert torch.equal(position, actuator.efforts_delay_buffer._time_lags)
            return position.clone()

        def paired_reset(env_ids, *, allow_matching_error=False):
            initial_rng = rng_state()
            reference_error = None
            try:
                for actuator in reference.values():
                    actuator.reset(env_ids)
            except Exception as error:  # parity includes rejected empty reset behavior
                reference_error = error
            reference_rng = rng_state()

            restore_rng(initial_rng)
            optimized_error = None
            try:
                for actuator in optimized.values():
                    actuator.reset(env_ids)
            except Exception as error:
                optimized_error = error
            optimized_rng = rng_state()

            assert_rng_equal(reference_rng, optimized_rng)
            if reference_error is not None or optimized_error is not None:
                assert reference_error is not None and optimized_error is not None
                assert type(reference_error) is type(optimized_error)
                assert str(reference_error) == str(optimized_error)
                if allow_matching_error:
                    return
                raise reference_error
            for name in reference:
                assert torch.equal(lag(reference[name]), lag(optimized[name])), f"{name} lag mismatch"

        torch.manual_seed(20260917)
        torch.cuda.manual_seed_all(20260917)
        paired_reset(None)
        sampled_lags = [lag(reference[name]) for name in reference]
        assert any(
            not torch.equal(sampled_lags[0], other) for other in sampled_lags[1:]
        ), "the four actuator groups did not receive independent lag samples"

        def clone_action(action):
            return ArticulationActions(
                joint_positions=action.joint_positions.clone(),
                joint_velocities=action.joint_velocities.clone(),
                joint_efforts=action.joint_efforts.clone(),
            )

        def assert_optional_equal(left, right, label):
            assert (left is None) == (right is None), f"{label} None mismatch"
            if left is not None:
                assert torch.equal(left, right), f"{label} mismatch"

        base = torch.arange(num_envs * num_joints, device=device, dtype=torch.float32).reshape(num_envs, num_joints)
        for step in range(19):
            if step == 6:
                paired_reset(torch.tensor([1, 5, 9, 16], device=device))
            elif step == 10:
                paired_reset(torch.empty(0, dtype=torch.long, device=device), allow_matching_error=True)
            elif step == 13:
                paired_reset(slice(None))

            joint_pos = torch.sin(base * 0.07 + step * 0.13)
            joint_vel = torch.cos(base * 0.11 - step * 0.17) * 24.0
            source = ArticulationActions(
                joint_positions=torch.sin(base * 0.03 + step * 0.41) * 1.7,
                joint_velocities=torch.cos(base * 0.05 + step * 0.29) * 38.0,
                joint_efforts=torch.sin(base * 0.09 - step * 0.23) * 19.0,
            )
            for name in reference:
                ref_output = reference[name].compute(clone_action(source), joint_pos, joint_vel)
                fast_output = optimized[name].compute(clone_action(source), joint_pos, joint_vel)
                assert_optional_equal(ref_output.joint_positions, fast_output.joint_positions, f"{name} positions")
                assert_optional_equal(ref_output.joint_velocities, fast_output.joint_velocities, f"{name} velocities")
                assert_optional_equal(ref_output.joint_efforts, fast_output.joint_efforts, f"{name} efforts")
                assert torch.equal(reference[name].computed_effort, optimized[name].computed_effort), (
                    f"{name} computed effort mismatch"
                )
                assert torch.equal(reference[name].applied_effort, optimized[name].applied_effort), (
                    f"{name} applied effort mismatch"
                )

        # Check the physical clipping laws against independently calculated values.
        dc_motor = build(ThunderDelayedDCMotor, leg_cfg())
        dc_motor.reset(None)
        dc_velocity = torch.tensor([[0.0, 8.74, -8.74, 17.48]], device=device).repeat(num_envs, 1)
        dc_action = ArticulationActions(
            joint_positions=torch.tensor([[10.0, 10.0, -10.0, 10.0]], device=device).repeat(num_envs, 1),
            joint_velocities=torch.zeros((num_envs, num_joints), device=device),
            joint_efforts=torch.zeros((num_envs, num_joints), device=device),
        )
        dc_motor.compute(dc_action, torch.zeros_like(dc_velocity), dc_velocity)
        dc_expected = torch.tensor([120.0, 60.0, -60.0, 0.0], device=device).repeat(num_envs, 1)
        assert torch.equal(dc_motor.applied_effort, dc_expected), "DCMotor torque-speed clipping law changed"

        wheel = build(DelayedPDActuator, wheel_cfg())
        wheel.reset(None)
        wheel_action = ArticulationActions(
            joint_positions=torch.zeros((num_envs, num_joints), device=device),
            joint_velocities=torch.tensor([[100.0, -100.0, 25.0, -25.0]], device=device).repeat(num_envs, 1),
            joint_efforts=torch.zeros((num_envs, num_joints), device=device),
        )
        wheel.compute(wheel_action, torch.zeros((num_envs, num_joints), device=device), torch.zeros((num_envs, num_joints), device=device))
        wheel_expected = torch.tensor([17.0, -17.0, 17.0, -17.0], device=device).repeat(num_envs, 1)
        assert torch.equal(wheel.applied_effort, wheel_expected), "wheel PD effort clipping law changed"

        result = {
            "status": "passed",
            "device": device,
            "num_envs": num_envs,
            "groups": list(reference),
            "steps": 19,
            "checks": ["rng", "delay", "reset", "actions", "dc_clip", "wheel_clip"],
        }
        result_json = json.dumps(result, sort_keys=True)
        result_path.write_text(result_json + "\n", encoding="utf-8")
        print("ACTUATOR_PARITY_PASS " + result_json, flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()
