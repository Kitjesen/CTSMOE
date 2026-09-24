"""Read-only compact status; keep original and new terrain means separate."""
import json
from pathlib import Path

stage = Path(__file__).resolve().parents[1]
result = {"pipeline": json.loads((stage / 'STATUS.json').read_text())}
for name in ('continuation_after_flat_recording', 'continuation', 'integration16'):
    path = stage / name / 'metrics.jsonl'
    if not path.exists():
        continue
    lines = path.read_text().splitlines()
    if not lines:
        continue
    row = json.loads(lines[-1])
    kinds = row['terrain_by_type']
    old = [v for k, v in kinds.items() if not k.startswith('continuous_')]
    result['training'] = dict(completed=row['completed_updates'], seconds=row['seconds'],
        old_terrain_mean=sum(v['mean']*v['count'] for v in old)/sum(v['count'] for v in old),
        terrain={k:{'count':v['count'], 'mean':v['mean']} for k,v in kinds.items()})
    break
path = stage / 'qualification/FINALIZED.json'
if path.exists():
    proof = json.loads(path.read_text())
    result['qualification'] = {k:proof[k] for k in ('steps','optimizer_updates','old_geometry_cells_verified','resets')}
print(json.dumps(result, indent=2))
