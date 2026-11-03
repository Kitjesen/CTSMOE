from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(__file__).parent
d = json.loads((p/'controlled/results.json').read_text())
z = np.load(p/'controlled/traces.npz')['control']
fields, offset = {}, 0
for name, width in d['fields']:
    fields[name] = slice(offset, offset+width)
    offset += width
fig, axes = plt.subplots(3, 1, figsize=(9,8), sharex=True)
for i, label in ((61,'40 deg: passes'), (90,'45 deg: stalls')):
    end = d['cases'][i]['step']+1
    t = np.arange(1,end+1)*.02
    c = z[:end,i]
    axes[0].plot(t,c[:,fields['xyz']][:,0],label=label)
    axes[1].plot(t,c[:,fields['vel_b']][:,0],label=label,alpha=.8)
    support = (c[:,fields['wheel_force']] > 5).sum(1)
    smooth = np.convolve(support, np.ones(25)/25, mode='valid')
    axes[2].plot(t[24:],smooth,label=label)
axes[0].axhline(4,color='gray',linestyle='--'); axes[0].set_ylabel('Forward position (m)')
axes[1].axhline(.6,color='gray',linestyle='--'); axes[1].set_ylabel('Body vx (m/s)')
axes[2].set_ylabel('Contact wheels\n0.5 s moving mean'); axes[2].set_xlabel('Time (s)')
axes[0].legend()
for ax in axes: ax.grid(alpha=.2)
fig.suptitle('Frozen student 37283: fixed wheel friction 1.2, no pushes\n40 deg without heading feedback; 45 deg with heading feedback')
fig.tight_layout()
fig.savefig(p/'slope_traces.png',dpi=150)
