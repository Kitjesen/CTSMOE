"""Compare requested and limited targets using the verified action ordering/scale."""
import json
import argparse
from pathlib import Path
import numpy as np

p=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--directories',nargs='+',default=['baseline','candidate96','repeat96'])
args=parser.parse_args()
rows=[]
for directory in args.directories:
    m=json.loads((p/directory/'manifest.json').read_text())
    with np.load(p/directory/'traces.npz') as f:
        actions,targets=f['raw_action'],f['target']
    names=[f'{side}_{joint}_joint' for side in ('fr','fl','rr','rl') for joint in ('hip','thigh','calf')]
    ids=[m['joint_names'].index(n) for n in names]
    nominal=np.array(m['default_q'])[ids]
    for c in m['cases']:
        if c['mode']!='direct' or c['direction']!='forward':
            continue
        raw=actions[:50,c['id'],:12]
        request=nominal+.25*raw
        limited=targets[:50,c['id']][:,ids]
        gap=np.abs(request-limited)
        rows.append(dict(dataset=directory,policy=m['history'][c['policy']]['label'],
            raw_action_abs_max=float(np.abs(raw).max()),
            requested_offset_rad_max=float(np.abs(.25*raw).max()),
            request_limit_gap_mean_rad=float(gap.mean()),request_limit_gap_max_rad=float(gap.max()),
            fraction_joint_samples_gap_over_half_rad=float((gap>.5).mean())))
(p/'raw_target_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
