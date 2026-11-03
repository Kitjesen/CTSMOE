"""Capture actual V9 reward contributions under the existing matched protocol."""
from pathlib import Path

stage = Path(__file__).resolve().parent
source = stage.parent/'flat_v9_20260921/evaluate_v9.py'
s = source.read_text(encoding='utf-8')
instrument = '''    original_reward_compute = raw.reward_manager.compute
    reward_capture = {}
    def capture_reward(dt):
        result = original_reward_compute(dt)
        reward_capture['weighted_reward'] = raw.reward_manager._step_reward.clone()*dt
        reward_capture['reward_total'] = result.clone()
        reward_capture['reward_command'] = command.commands.clone()
        torch.testing.assert_close(reward_capture['weighted_reward'].sum(-1), result, atol=1e-5, rtol=1e-5)
        return result
    raw.reward_manager.compute = capture_reward
'''
changes = {
    "    obs=env.get_observations()": instrument+"    obs=env.get_observations()",
    "'phase_ready','phase_stable_s')}": "'phase_ready','phase_stable_s','weighted_reward','reward_total','reward_command')}",
    "            for k,v in values.items():": "            values.update(reward_capture)\n            for k,v in values.items():",
    "                  history=metadata,": "                  history=metadata,\n                  reward_names=raw.reward_manager.active_terms,\n                  reward_weights={name:raw.reward_manager.get_term_cfg(name).weight for name in raw.reward_manager.active_terms},",
    "protocol='V9 flat continuous repair,": "protocol='Actual V9 reward contributions, four matched fixed policies (no optimizer updates); V9 flat continuous repair,",
}
for old, new in changes.items():
    assert s.count(old) == 1, old
    s = s.replace(old, new)
(stage/'reward_probe.py').write_text(s, encoding='utf-8')
root='/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917'
(stage/'launch.sh').write_text(f'''#!/bin/bash
set -euo pipefail
cd {root}/code_flat_v9_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u {root}/posture_cause_stage_20260921/reward_probe.py --root {root} --code {root}/code_flat_v9_20260921 --selection {root}/flat_v9_selection_20260921.json --output {root}/recording/posture_cause_reward_20260921 >{root}/posture_cause_reward_20260921.log 2>&1 </dev/null &
echo PROBE_PID=$!
''', encoding='utf-8', newline='\n')
print('Reward probe and launch script built')
