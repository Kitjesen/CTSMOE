"""Read existing TensorBoard diagnostics; does not modify the training process."""
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_20260919')
events = EventAccumulator(str(path), size_guidance={'scalars': 200}).Reload()
result = {}
for tag in events.Tags()['scalars']:
    if any(word in tag.lower() for word in ('episode_reward', 'episode_termination', 'learning_rate')):
        rows = events.Scalars(tag)[-20:]
        if rows:
            result[tag] = {'step': rows[-1].step, 'last20_mean': sum(x.value for x in rows)/len(rows)}
print(json.dumps(result, indent=2))
