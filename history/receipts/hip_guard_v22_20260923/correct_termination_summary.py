"""Separate success/task endings from physical failures in preserved evaluations.

TerminationManager.terminated includes pit_success. The new common evaluator's
raw 'physical' boolean was consequently too broad for pit cases. Original terms,
trajectories and outcomes are retained; this file derives the reporting category
from explicit registered reasons. It does not alter any training or verdict.
"""
import json
from pathlib import Path
import sys

PHYSICAL = {'illegal_contact','actuator_violation'}
TASK = {'pit_success','pit_outside','time_out','stair_route_exit','budget'}


def summarize(folder):
    folder = Path(folder)
    m = json.loads((folder/'manifest.json').read_text())
    groups, cases = {}, []
    for c in m['cases']:
        terms = set(c.get('terms', []))
        if terms - PHYSICAL - TASK:
            raise ValueError(f'Unclassified termination reasons: {terms-PHYSICAL-TASK}')
        physical = bool(terms & PHYSICAL)
        key = m['history'][c['policy']]['label']+'/'+c['terrain']
        group = groups.setdefault(key,dict(cases=0,physical_failures=0,pit_success=0,
            outside_task=0,route4m=0,straight4m=0,platform_dwell=0))
        group['cases'] += 1
        group['physical_failures'] += int(physical)
        group['outside_task'] += int('pit_outside' in terms)
        group['pit_success'] += int('pit_success' in terms and not physical)
        group['route4m'] += int(c.get('passed_4m',False) and not physical)
        group['straight4m'] += int(c.get('straight_4m',False) and not physical)
        group['platform_dwell'] += int(c.get('end_reason') == 'supported_platform_dwell' and not physical)
        cases.append(dict(id=c['id'],terms=sorted(terms),physical_failure=physical,
                          raw_physical_field=c.get('physical'),pit_success='pit_success' in terms))
    result=dict(fixed_cases_not_random_success_rate=True,groups=groups,cases=cases,
        reason='Explicit success/task termination separated from physical failure; raw traces unchanged')
    path=folder/'TERMINATION_REVIEW_CORRECTED.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Existing corrected summary differs')
    else:
        path.write_text(json.dumps(result,indent=2))
    print(json.dumps(groups))


if __name__ == '__main__':
    summarize(sys.argv[1])
