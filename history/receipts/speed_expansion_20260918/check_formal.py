"""Read-only verification of the speed-expanded process and source checkpoint."""
import datetime
import json
from pathlib import Path
import statistics

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/speed_expansion_20260918'
run = root / 'runs/official4096_30000_speed1_20260918'
receipt = json.loads((run / 'run.json').read_text())
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
saved = torch.load(receipt['resume'], map_location='cpu', weights_only=False)
levels, cols = saved['terrain_levels'].cpu(), saved['terrain_types'].cpu()
names = ['wave', 'slope_up', 'slope_up', 'slope_down', 'slope_down', 'rough_slope',
         'stairs_up', 'stairs_up', 'stairs_up', 'stairs_up', 'stairs_up',
         'stairs_down', 'stairs_down', 'obstacles', 'obstacles', 'obstacles', 'obstacles',
         'flat', 'flat', 'flat']
terrain = {}
for name in dict.fromkeys(names):
    selected = torch.zeros_like(cols, dtype=torch.bool)
    for i, candidate in enumerate(names):
        if candidate == name:
            selected |= cols == i
    subset = levels[selected]
    terrain[name] = {'n': int(subset.numel()), 'mean': float(subset.float().mean()),
                     'counts_L0_to_L9': torch.bincount(subset.long(), minlength=10).tolist()}
result = {'queried_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'run_receipt': receipt, 'latest': rows[-1], 'new_updates_observed': len(rows),
          'seconds_mean_last10': statistics.mean(row['seconds'] for row in rows[-10:]),
          'checkpoint_terrain': terrain, 'column_mapping': dict(enumerate(names))}
(audit / 'formal_verified.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
