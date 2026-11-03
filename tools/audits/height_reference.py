"""Independent NumPy replay of the candidate ground-reference rate limit.

Uses reconstructed scan means from historical trajectories, not live sensors.
The settings are a diagnostic example, not a calibrated reward recommendation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def replay(raw, dt, rate, max_lag):
    raw = np.asarray(raw, dtype=float)
    result = np.full_like(raw, np.nan)
    active = np.zeros(len(raw), dtype=bool)
    previous = None
    for i, ground in enumerate(raw):
        if not np.isfinite(ground):
            continue
        if previous is None:
            previous = ground
        else:
            previous = max(previous - rate * dt, min(previous + rate * dt, ground))
        result[i] = previous
        active[i] = abs(previous - ground) <= max_lag
    return result, active


def selfcheck():
    signal = np.array([0., 0., -.239, np.nan, -.239])
    reference, active = replay(signal, .02, .5, .04)
    np.testing.assert_allclose(reference, [0., 0., -.01, np.nan, -.02], atol=1e-12)
    assert active.tolist() == [True, True, False, False, False]
    translated, active2 = replay(signal + 27, .02, .5, .04)
    np.testing.assert_allclose(reference + 27, translated, atol=1e-12)
    assert np.array_equal(active, active2)
    recovered, active3 = replay(np.r_[0., np.full(30, -.239)], .02, .5, .04)
    assert active3[-1] and abs(recovered[-1] + .239) < 1e-12
    assert np.nanmax(np.abs(np.diff(recovered))) <= .01 + 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Directory containing eight *_series_v2.npz traces')
    parser.add_argument('--output', type=Path, required=True, help='Destination JSON report')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if not source.is_dir():
        parser.error('--source must be a directory')
    selfcheck()
    cases = []
    for path in sorted(source.glob('*_series_v2.npz')):
        with np.load(path, allow_pickle=False) as data:
            dt = float(np.median(np.diff(data['time_s'])))
            if not np.isclose(dt, .02):
                raise ValueError(f'unexpected timestep {dt}')
            raw = data['scanner_z']
            filtered, active = replay(raw, dt, .5, .04)
            finite = np.isfinite(raw)
            lag = np.abs(filtered - raw)
            cases.append({
                'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'samples': len(raw), 'finite_scanner_frames': int(finite.sum()),
                'active_frames': int(active.sum()),
                'suppressed_fraction_of_finite': float((finite & ~active).sum() / max(finite.sum(), 1)),
                'raw_max_change_m': float(np.nanmax(np.abs(np.diff(raw)))),
                'filtered_max_change_m': float(np.nanmax(np.abs(np.diff(filtered)))),
                'lag_max_m': float(np.nanmax(lag)),
                'lag_p95_m': float(np.nanpercentile(lag, 95)),
            })
            assert cases[-1]['filtered_max_change_m'] <= .01 + 1e-10
    if len(cases) != 8:
        raise ValueError(f'expected 8 historical traces, found {len(cases)}')
    result = {'selfcheck': 'passed', 'rate_m_s': .5, 'max_lag_m': .04,
              'scope': 'CPU replay of reconstructed scan means; no simulator, policy, reward or training run',
              'cases': cases}
    output = args.output.resolve()
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'output': str(output),
                      'max_filtered_change_m': max(c['filtered_max_change_m'] for c in cases),
                      'suppressed_fraction_range': [min(c['suppressed_fraction_of_finite'] for c in cases),
                                                    max(c['suppressed_fraction_of_finite'] for c in cases)]}))


if __name__ == '__main__':
    main()
