"""Read saved curriculum state without changing any running job."""
from collections import Counter
import json
from pathlib import Path
import torch

p=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
selection=json.loads((p/'posture_history_fine_selection.json').read_text())[:7]
rows=[]
for item in selection:
    s=torch.load(p/item['path'],map_location='cpu',weights_only=False)
    types=s['terrain_types'].tolist()
    names=s['terrain_column_names']
    course=s['pit_course']
    # Pre-migration checkpoints predate this field; preserve absence explicitly.
    rows.append(dict(checkpoint=s['completed_updates'],profile=s.get('pit_training_profile'),
        terrain_counts=dict(Counter(names[i] for i in types)),
        frontier=course.get('frontier'),course_keys=list(course),
        decisions=course.get('decisions',[]),base_height_penalty=s['base_height_penalty'],
        actuator_profile=s['actuator_profile'],learning_rate=s['learning_rate']))
    del s
(p/'posture_history_checkpoint_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ('decisions','course_keys')} for r in rows],indent=2))
