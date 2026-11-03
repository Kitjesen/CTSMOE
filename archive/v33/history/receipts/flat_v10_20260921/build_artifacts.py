"""Create versioned evaluation and audit artifacts from the existing protocol."""
import difflib
import hashlib
import json
from pathlib import Path

stage = Path(__file__).resolve().parent
code = stage.parent.parent
v9 = stage.parent/'flat_v9_20260921'
evaluation = (v9/'evaluate_v9.py').read_text(encoding='utf-8')
evaluation = evaluation.replace("assert latest['startup_repair_profile']=='omni_start_v9'",
                                "assert latest['startup_repair_profile'] in ('omni_start_v9', 'omni_start_v10')")
evaluation = evaluation.replace("configure_startup(cfg,'omni_start_v9')",
                                "configure_startup(cfg,latest['startup_repair_profile'])")
evaluation = evaluation.replace('V9 flat continuous repair,', 'V10 flat outlier repair comparison,')
(stage/'evaluate.py').write_text(evaluation, encoding='utf-8')
analysis = (v9/'analyze_interim.py').read_text(encoding='utf-8')
analysis = analysis.replace("version='V9'", "version='V10'")
analysis = analysis.replace("max_rms=float(rms.max()),", """first_peak_rms=float(first.max()) if len(first) else None,
        first_peak_joint_rad=float(np.abs(q[start:min(n,start+50)]).max()) if len(first) else None,
        first_peak_time_s=float((np.argmax(first)+1)*m['dt']) if len(first) else None,
        max_rms=float(rms.max()),""")
analysis = analysis.replace("row['first_rms']<.25", "row['first_peak_rms']<.25 and row['first_peak_joint_rad']<.60")
analysis = analysis.replace("('label','first_rms','max_rms'", "('label','first_rms','first_peak_rms','first_peak_joint_rad','first_peak_time_s','max_rms'")
# Add per-joint evidence using exactly the same valid first-episode samples.
analysis += '''
joint_rows=[]
requests=m['request_joint_ids']
for c in m['cases']:
    if c['terrain']!='flat':
        continue
    start=100 if c['mode']=='settle' else 0
    n=min(c['end_step']+1,start+50)
    if n<=start:
        continue
    for j in legs:
        actual=d['joint_pos'][start:n,c['id'],j]-default[j]
        target=d['target'][start:n,c['id'],j]-default[j]
        requested=d['requested'][start:n,c['id'],requests.index(j)]-default[j]
        k=int(np.argmax(abs(actual)))
        target_delta=np.diff(d['target'][max(0,start-1):n,c['id'],j])
        joint_rows.append(dict(label=m['history'][c['policy']]['label'],case=c['id'],
            direction=c['direction'],mode=c['mode'],joint=m['joint_names'][j],
            actual_peak_rad=float(abs(actual).max()),target_peak_rad=float(abs(target).max()),
            request_peak_rad=float(abs(requested).max()),peak_time_s=float((k+1)*m['dt']),
            target_at_actual_peak_rad=float(target[k]),request_at_actual_peak_rad=float(requested[k]),
            target_step_peak_rad=float(abs(target_delta).max()) if len(target_delta) else None))
(a.directory/'startup_joints.json').write_text(json.dumps(joint_rows,indent=2),encoding='utf-8')
'''
(stage/'analyze.py').write_text(analysis, encoding='utf-8')
changed = ['startup_repair.py', 'stance_strength.py', 'stair_speed.py', 'train.py']
diff = []
for name in changed:
    diff.extend(difflib.unified_diff((stage/name).read_text(encoding='utf-8').splitlines(True),
                (code/name).read_text(encoding='utf-8').splitlines(True), fromfile='v9/'+name, tofile='v10/'+name))
for name in ['flat_outlier.py', 'test_flat_outlier.py']:
    diff.extend(difflib.unified_diff([], (code/name).read_text(encoding='utf-8').splitlines(True),
                                   fromfile='/dev/null', tofile='v10/'+name))
(stage/'change.diff').write_text(''.join(diff), encoding='utf-8')
names=changed+['flat_outlier.py', 'test_flat_outlier.py']
(stage/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256((code/n).read_bytes()).hexdigest()
                                                for n in names},indent=2), encoding='utf-8')
root='/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917'
(stage/'launch.sh').write_text(f'''#!/bin/bash
set -euo pipefail
cd {root}/code_flat_v10_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u {root}/flat_v10_stage_20260921/run_pipeline.py >{root}/flat_v10_1000_20260921.log 2>&1 </dev/null &
echo PIPELINE_PID=$!
''',encoding='utf-8',newline='\n')
print('Artifacts written', stage)
