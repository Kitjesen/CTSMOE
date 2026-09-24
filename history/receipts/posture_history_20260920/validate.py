"""Verify trace integrity and replay summary arithmetic for this audit."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).parent
checks=[]
for directory in (p,p/'fine'):
    m=json.loads((directory/'manifest.json').read_text())
    rows=json.loads((directory/'summary.json').read_text())
    assert len(rows)==len(m['cases'])
    with np.load(directory/'traces.npz') as a:
        arrays={k:a[k] for k in a.files}
    n=len(m['cases'])
    for key,value in arrays.items():
        assert value.shape[1]==n,(key,value.shape,n)
        assert np.isfinite(value).all(),key
    assert all(c['terrain']=='flat' and c['level']==0 for c in m['cases'])
    for c,r in zip(m['cases'],rows):
        i=c['id']
        count=c['end_step']+1
        assert arrays['active'][:count,i].all()
        command=arrays['command'][:count,i]
        assert np.abs(command[:,1:]).max()==0
        assert command[:,0].min()>=0 and command[:,0].max()<=.60001
        if c['mode']=='stand':
            assert np.abs(command).max()==0
        assert r['checkpoint']==m['history'][c['policy']]['label']
    checks.append(dict(directory=directory.name,cases=n,steps=len(arrays['active']),finite=True))
fine=json.loads((p/'fine/summary.json').read_text())
for label in {r['checkpoint'] for r in fine}:
    rr=[r for r in fine if r['checkpoint']==label and r['mode']=='direct']
    assert len(rr)==2
    assert abs(rr[0]['first_second_rms']-rr[1]['first_second_rms'])<.005
(p/'validation.json').write_text(json.dumps(dict(checks=checks,fine_duplicate_startup_tolerance_rad=.005),indent=2))
print(json.dumps(checks))
