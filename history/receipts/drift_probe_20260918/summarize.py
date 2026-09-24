"""Summarize per-episode peak drift; do not call it mean trajectory error."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(cases):
    cross = [case['max_cross_m'] for case in cases]
    out = {'trials': len(cases), 'peak_cross_mean_m': statistics.mean(cross),
           'peak_cross_median_m': statistics.median(cross), 'peak_cross_max_m': max(cross),
           'exceed_075_m': sum(value >= .75 for value in cross),
           'exceed_1_m': sum(value >= 1 for value in cross),
           'exceed_2_m': sum(value >= 2 for value in cross),
           'reference_exit_passes': int(sum(case['reference_exit_pass'] for case in cases)),
           'physical_failures': int(sum(case['physical_failure'] for case in cases)),
           'worst_case': max(cases, key=lambda case: case['max_cross_m'])}
    if 'signed_peak_cross_m' in cases[0]:
        for case in cases:
            assert abs(abs(case['signed_peak_cross_m']) - case['max_cross_m']) < 1.e-5
        out.update(peak_left_count=sum(case['signed_peak_cross_m'] > 0 for case in cases),
                   peak_right_count=sum(case['signed_peak_cross_m'] < 0 for case in cases),
                   max_yaw_median_deg=statistics.median(case['max_abs_yaw_deg'] for case in cases),
                   max_yaw_max_deg=max(case['max_abs_yaw_deg'] for case in cases))
    return out


parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
args = parser.parse_args()
result = json.loads(args.results.read_text(encoding='utf-8'))
summary = {'completed_updates': result['completed_updates'], 'groups': {}}
for learner in ('teacher', 'student'):
    for terrain in ('stairs_up', 'stairs_down'):
        cases = [case for case in result['cases'] if case['learner'] == learner and case['terrain'] == terrain]
        group = summarize(cases)
        group['by_level'] = {str(level): summarize([case for case in cases if case['level'] == level])
                             for level in (3, 5, 7)}
        group['by_speed'] = {str(speed): summarize([case for case in cases if case['speed'] == speed])
                             for speed in (.3, .5, .7, .9)}
        summary['groups'][learner + '_' + terrain] = group
destination = args.results.with_name(args.results.stem + '_summary.json')
destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({key: {k: v for k, v in value.items() if k not in ('by_level', 'by_speed')}
                  for key, value in summary['groups'].items()}, indent=2))
