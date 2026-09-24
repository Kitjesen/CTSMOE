"""Check deployment IO against actual simulator observations and action targets."""
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from reference_io import LEG_NAMES, TERM_DIMS, ObservationHistory, TargetReference, build_single


def main(root):
    folder = root/'results_verified/nominal'
    manifest = json.loads((folder/'manifest.json').read_text())
    contract = json.loads((folder/'RUNTIME_CONTRACT.json').read_text())
    # Materialize once: NpzFile otherwise decompresses a complete member on each
    # time-step lookup, turning trajectory parity into repeated archive decoding.
    with np.load(folder/'traces.npz') as archive:
        traces = dict(archive)
    with np.load(folder/'REAL_OBSERVATIONS.npz') as archive:
        observed = dict(archive)
    names = manifest['joint_names']
    legs = [names.index(n) for n in LEG_NAMES]
    all_ids = [names.index(n) for n in contract['policy_all_order']]
    default_leg = np.asarray(manifest['default_q'])[legs]
    case_count = len(manifest['cases'])
    target_peak = observation_peak = history_peak = 0.
    target_comparisons = observation_comparisons = 0
    for case in manifest['cases']:
        i, end = case['id'], case['end_step']+1
        reference = TargetReference(contract, traces['initial_q'][i, legs])
        history = ObservationHistory()
        for step in range(end):
            target = reference.step(traces['raw_action'][step, i])
            expected = traces['target'][step, i, legs]
            np.testing.assert_allclose(target['leg_q_target'], expected, atol=3e-5, rtol=1e-5)
            target_peak = max(target_peak, float(abs(target['leg_q_target']-expected).max()))
            target_comparisons += 1
            if step == 0:
                angular, gravity, q, dq, action = np.zeros(3), [0., 0., -1.], traces['initial_q'][i, legs], np.zeros(16), np.zeros(16)
            else:
                angular, gravity = traces['angular_velocity'][step-1, i], traces['gravity'][step-1, i]
                q, dq, action = traces['joint_pos'][step-1, i, legs], traces['joint_vel'][step-1, i, all_ids], traces['raw_action'][step-1, i]
            single = build_single(angular, gravity, traces['command'][step, i], q, dq, action, default_leg)
            packed, current = history.push(single)
            if step % 25 == 0:
                idx = (step//25)*case_count+i
                actual_single, actual_history = observed['single_obs'][idx], observed['history'][idx]
                np.testing.assert_allclose(current[0], actual_single, atol=3e-5, rtol=1e-5)
                np.testing.assert_allclose(packed[0], actual_history, atol=3e-5, rtol=1e-5)
                observation_peak = max(observation_peak, float(abs(current[0]-actual_single).max()))
                history_peak = max(history_peak, float(abs(packed[0]-actual_history).max()))
                observation_comparisons += 1
    # The exported two-input graph is also callable on one robot, not only the
    # 72-robot batch used by the simulator. Use measured observations for parity.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(root/'export_candidate/thunder_cts_moe_34823.onnx'),
                                   sess_options=opts, providers=['CPUExecutionProvider'])
    single_batch_peak = 0.
    for idx in np.linspace(0, len(observed['history'])-1, 32, dtype=int):
        got = session.run(None, dict(history=observed['history'][idx:idx+1], single_obs=observed['single_obs'][idx:idx+1]))[0][0]
        expected = observed['actions'][idx]
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
        single_batch_peak = max(single_batch_peak, float(abs(got-expected).max()))
    report = dict(passed=True, target_comparisons=target_comparisons,
        observation_comparisons=observation_comparisons, target_max_error_rad=target_peak,
        single_observation_max_error=observation_peak, history_max_error=history_peak,
        onnx_batch1_vs_batch72_max_error=single_batch_peak,
        term_dimensions=TERM_DIMS, history_layout='term-major; per-term oldest-first; first-frame padding')
    (root/'IO_VERIFICATION.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
