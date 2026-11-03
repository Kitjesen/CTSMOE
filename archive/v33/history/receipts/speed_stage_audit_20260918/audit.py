"""Read-only training status and checkpoint terrain distribution."""
import datetime
import json
from pathlib import Path
import statistics

import torch
import yaml


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root / 'runs/official4096_30000'
env = yaml.load((run / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
status = json.loads((run / 'status.json').read_text())
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
checkpoint = max(run.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
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
result = {
    'queried_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': status,
    'terrain_mean_last_500_updates': statistics.mean(r['terrain'] for r in rows[-500:]),
    'seconds_mean_last_100_updates': statistics.mean(r['seconds'] for r in rows[-100:]),
    'episode_length_s': float(env['episode_length_s']),
    'control_dt': float(env['sim']['dt']) * int(env['decimation']),
    'commands': {key: env['commands']['base_velocity'][key] for key in
                 ('ranges', 'command_range_curriculum', 'dynamic_resample_commands', 'num_steps_per_iter')},
    'checkpoint': checkpoint.name, 'checkpoint_updates': saved['completed_updates'],
    'checkpoint_common_step_counter': saved['common_step_counter'],
    'checkpoint_terrain': terrain,
    'column_mapping': dict(enumerate(names)),
    'remaining_seconds_to_20000_estimate': max(0, 20000 - status['completed_updates']) *
        statistics.mean(r['seconds'] for r in rows[-100:]),
}
output = Path(__file__).with_name('result.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
