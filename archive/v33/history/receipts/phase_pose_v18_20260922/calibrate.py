"""Show the provisional coefficient's magnitude on preserved measured flat traces."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'posture_stop_v17_20260922/recording_final_32831'
manifest = json.loads((OLD / 'forward_manifest.json').read_text())
data = np.load(OLD / 'forward.npz')
legs = [i for i,n in enumerate(manifest['joint_names']) if '_foot_' not in n]
errors = abs(data['joint_pos'][...,legs] - np.asarray(manifest['default_q'])[legs])
rows = []
for case in manifest['cases']:
    i = case['id']
    for label, window in [('first1s', slice(0,50)), ('last5s', slice(1000,1250))]:
        cost = .20 * errors[window,i].sum(-1)
        rows.append(dict(checkpoint=case['checkpoint'], window=label,
            extra_cost_mean=float(cost.mean()), extra_cost_max=float(cost.max()),
            extra_cost_integral=float(cost.sum()*.02)))
(HERE / 'COEFFICIENT_MAGNITUDE.json').write_text(json.dumps(dict(
    chosen_extra_weight=-.20, scope='flat gate=1 actual-angle L1 sum over 12 leg joints',
    evidence='Fixed recorded trajectories only, not proof that training will improve.',
    rows=rows), indent=2))
print(json.dumps(rows, indent=2))
