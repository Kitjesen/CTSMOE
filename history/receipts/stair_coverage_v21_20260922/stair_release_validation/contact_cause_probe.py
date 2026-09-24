"""Passive PhysX contact reads around three preserved stair transients.

Read contact_physx_view directly, without updating ContactSensor.data/history.
The installed Isaac Lab ContactSensor implementation uses these same getters;
the tensor API documents the six contact buffers and force=impulse/physics_dt.
Only reporter buffers and copied states are read. Full baseline trajectory parity
is required before these diagnostic observations may support a conclusion.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "stair_release_validation_attempt02_20260922"
CODE = HERE.parent / "stair_coverage_v21/code"
sys.path[:0] = [str(SOURCE), str(CODE)]


def main():
    import numpy as np
    import torch
    import actuator_guard

    spec = importlib.util.spec_from_file_location("preserved_evaluator", SOURCE / "evaluate_stairs_onnx_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args()
    selected = [33] if args.perturbed else [28, 64]
    event_time = {28: 5.605, 64: 1.660, 33: 2.620}
    original = actuator_guard.ActuatorGuard.measure
    fields = ("q", "dq", "ddq", "target", "torque", "computed", "body_state", "net_contact", "wheel_ground")
    records = {k: [] for k in fields}
    events = []
    holder = []
    steps = []
    detail_views = {}

    def measure(guard):
        original(guard)
        env, robot = guard.env, guard.robot
        pit = env.command_manager.get_term("base_velocity").pit
        if not holder:
            holder.append(guard)
        elif holder[0] is not guard:
            raise RuntimeError("Expected a single environment guard")
        steps.append(env._sim_step_counter)
        data = robot.data
        for field, value in (("q", data.joint_pos), ("dq", data.joint_vel),
                             ("ddq", data.joint_acc), ("target", data.joint_pos_target),
                             ("torque", data.applied_torque), ("computed", data.computed_torque),
                             ("body_state", data.body_state_w)):
            records[field].append(value[selected].clone())
        net = pit.contact.contact_physx_view.get_net_contact_forces(dt=env.physics_dt)
        records["net_contact"].append(net.reshape(env.num_envs, pit.contact.num_bodies, 3)[selected].clone())
        wheel_forces = []
        t = len(steps) * env.physics_dt
        active_ids = [i for i in selected if abs(t - event_time[i]) <= .1050001]
        for side, sensor in zip(("fr", "fl", "rr", "rl"), pit.ground_contacts, strict=True):
            view = sensor.contact_physx_view
            matrix = view.get_contact_force_matrix(dt=env.physics_dt)
            wheel_forces.append(matrix[selected, 0].clone())
            if active_ids:
                if side not in detail_views:
                    # The unchanged production reporter reserves only four
                    # contact points per wheel. Attempt02 measured a valid
                    # pair start=1247/count=12 against capacity288. A separate
                    # passive reporter reserves128 points per wheel; it does
                    # not alter collision geometry, material, solver or sensors.
                    # Exact original trajectory parity is still mandatory.
                    pattern = sensor.cfg.prim_path.replace(".*", "*")
                    assert "{" not in pattern
                    detail_views[side] = sensor._physics_sim_view.create_rigid_contact_view(
                        pattern,
                        filter_patterns=[p.replace(".*", "*") for p in sensor.cfg.filter_prim_paths_expr],
                        max_contact_data_count=128 * env.num_envs)
                detail = detail_views[side]
                forces, points, normals, distances, counts, starts = detail.get_contact_data(dt=env.physics_dt)
                for case_id in active_ids:
                    count, start = int(counts[case_id, 0]), int(starts[case_id, 0])
                    # PhysX documents buffer indexing only for count > 0.
                    # Empty pairs may carry an unsigned sentinel start index;
                    # they have no contacts to address, so use an empty slice.
                    if count > 0 and start + count > detail.max_contact_data_count:
                        raise RuntimeError(f"Contact buffer range exceeds capacity: {case_id=} {side=} {count=} {start=} capacity={detail.max_contact_data_count}")
                    sl = slice(start, start + count) if count > 0 else slice(0, 0)
                    events.append(dict(time_s=t, case_id=case_id, wheel=side, count=count,
                        force_N=forces[sl].cpu().tolist(), point_w=points[sl].cpu().tolist(),
                        normal_w=normals[sl].cpu().tolist(), separation_m=distances[sl].cpu().tolist(),
                        matrix_N=matrix[case_id, 0].cpu().tolist()))
        records["wheel_ground"].append(torch.stack(wheel_forces, 1))

    actuator_guard.ActuatorGuard.measure = measure
    module.run(args)
    guard = holder[0]
    with np.load(args.output / "traces.npz") as z:
        control = dict(z)
    expected = len(control["active"]) * 4
    assert len(steps) == expected
    np.testing.assert_array_equal(np.diff(steps), np.ones(expected - 1))
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
    np.testing.assert_array_equal(arrays["dq"][3::4], control["joint_vel"][:, selected])
    baseline = SOURCE / "results_verified" / ("perturbed" if args.perturbed else "nominal")
    with np.load(baseline / "traces.npz") as old:
        parity = {k: float(abs(control[k] - old[k]).max()) for k in
                  ("joint_pos", "joint_vel", "target", "torque", "raw_action", "body_pos")}
    before = json.loads((baseline / "manifest.json").read_text())
    after = json.loads((args.output / "manifest.json").read_text())
    outcomes = all((a["end_step"], a["end_reason"], a["terms"]) ==
                   (b["end_step"], b["end_reason"], b["terms"])
                   for a, b in zip(before["cases"], after["cases"], strict=True))
    np.savez_compressed(args.output / "CONTACT_CAUSE_5MS.npz", sim_steps=np.asarray(steps), **arrays)
    (args.output / "CONTACT_EVENTS.json").write_text(json.dumps(events))
    result = dict(optimizer_updates=0, case_ids=selected, samples=expected, physics_dt=.005,
                  baseline_max_abs_difference=parity, baseline_outcomes_identical=outcomes,
                  contact_names=guard.env.command_manager.get_term("base_velocity").pit.contact.body_names,
                  body_names=guard.robot.body_names, joint_names=guard.robot.joint_names,
                  wheel_order=["fr", "fl", "rr", "rl"],
                  hard_limits=guard.hard[0].cpu().tolist(),
                  env_origins=guard.env.scene.env_origins[selected].cpu().tolist(),
                  sensor_history_updated_by_probe=False)
    (args.output / "CONTACT_CAPTURE.json").write_text(json.dumps(result, indent=2))
    if any(parity.values()) or not outcomes:
        raise RuntimeError("Diagnostic reads changed the trajectory")
    print("CONTACT_CAPTURE", json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
