"""Read-only 5ms joint load capture around the preserved ONNX evaluator.

Do not query new contact sensors here: lazy sensor history updates can affect
termination/reward state. Joint acceleration is already read by ActuatorGuard
each physics step. The additional operations only clone existing robot buffers.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'stair_release_validation_attempt02_20260922'
CODE = HERE.parent/'stair_coverage_v21/code'
sys.path[:0] = [str(SOURCE), str(CODE)]


def main():
    import numpy as np
    import torch
    import actuator_guard

    original = actuator_guard.ActuatorGuard.measure
    records = {k: [] for k in ('acceleration', 'velocity', 'torque', 'computed_torque')}
    sim_steps = []
    captured_envs = []

    def measure(self):
        original(self)
        data = self.robot.data
        if not captured_envs:
            captured_envs.append(self.env)
        elif captured_envs[0] is not self.env:
            raise RuntimeError('Probe expects one environment')
        sim_steps.append(self.env._sim_step_counter)
        records['acceleration'].append(data.joint_acc.clone())
        records['velocity'].append(data.joint_vel.clone())
        records['torque'].append(data.applied_torque.clone())
        records['computed_torque'].append(data.computed_torque.clone())

    actuator_guard.ActuatorGuard.measure = measure
    spec = importlib.util.spec_from_file_location('preserved_stair_evaluator', SOURCE/'evaluate_stairs_onnx_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args()
    module.run(args)
    raw = captured_envs[0]
    control = np.load(args.output/'traces.npz')
    expected = control['active'].shape[0] * raw.cfg.decimation
    if len(sim_steps) != expected or not np.array_equal(np.diff(sim_steps), np.ones(expected-1)):
        raise RuntimeError(f'Non-contiguous or extra physics samples: {len(sim_steps)} != {expected}')
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
    np.savez_compressed(args.output/'PHYSICS_5MS.npz', sim_steps=np.asarray(sim_steps), **arrays)
    # Logged control-step joint velocity must equal every fourth physics sample.
    np.testing.assert_array_equal(arrays['velocity'][raw.cfg.decimation-1::raw.cfg.decimation], control['joint_vel'])
    baseline_dir = SOURCE/'results_verified'/('perturbed' if args.perturbed else 'nominal')
    with np.load(baseline_dir/'traces.npz') as previous:
        parity = {key: float(np.max(np.abs(control[key]-previous[key])))
                  for key in ('joint_pos', 'joint_vel', 'target', 'torque', 'raw_action', 'body_pos')}
    old_manifest = json.loads((baseline_dir/'manifest.json').read_text())
    new_manifest = json.loads((args.output/'manifest.json').read_text())
    same_outcomes = all((a['end_step'],a['end_reason'],a['terms']) == (b['end_step'],b['end_reason'],b['terms'])
                        for a,b in zip(old_manifest['cases'],new_manifest['cases'], strict=True))
    receipt = dict(optimizer_updates=0, physics_dt=raw.physics_dt, samples=len(sim_steps),
        contiguous_physics_steps=True, control_velocity_alignment_exact=True,
        baseline_control_max_abs_difference=parity, baseline_outcomes_identical=same_outcomes,
        source_evaluator=str(SOURCE/'evaluate_stairs_onnx_v3.py'), sensor_update_behavior_unchanged=True)
    (args.output/'PHYSICS_CAPTURE.json').write_text(json.dumps(receipt,indent=2))
    print('PHYSICS_CAPTURE',json.dumps(receipt),flush=True)


if __name__ == '__main__':
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
