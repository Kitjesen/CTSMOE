"""Run a bounded V10 continuation after the existing V9 evaluation completes."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
stage = root/'flat_v10_stage_20260921'
code = root/'code_flat_v10_20260921'
smoke = root/'runs/flat_v10_smoke_20260921'
formal = root/'runs/flat_v10_1000_20260921'
source = root/'runs/flat_v9_1000_20260921'


def state(value, **details):
    record = dict(state=value, **details)
    (stage/'pipeline_status.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def train(output, resume, updates):
    if output.exists():
        raise FileExistsError(output)
    subprocess.run([sys.executable, '-u', 'train.py', '--root', str(root),
                    '--output', str(output), '--num-envs', '4096', '--updates', str(updates),
                    '--mixed-pit', '--startup-repair-profile', 'omni_start_v10',
                    '--resume', str(resume)], cwd=code, check=True)


try:
    state('waiting_for_v9_training_and_evaluation')
    deadline = time.monotonic()+3600
    while 'FLAT_V9_TRAINING_AND_EVALUATIONS_COMPLETE' not in (root/'flat_v9_1000_20260921.log').read_text(errors='replace'):
        if time.monotonic() > deadline:
            raise TimeoutError('V9 did not finish training and evaluation within one hour')
        if 'Traceback (most recent call last)' in (root/'flat_v9_1000_20260921.log').read_text(errors='replace'):
            raise RuntimeError('V9 pipeline has a traceback; inspect before continuing')
        time.sleep(10)
    assert json.loads((source/'SUCCESS.json').read_text())['completed_updates'] == 44711
    state('v9_complete_analyzing_baseline')
    baseline = root/'recording/flat_v9_landing_20260921'
    subprocess.run([sys.executable, str(stage/'analyze.py'), str(baseline)], check=True)
    state('v10_smoke_training', target_completed_updates=44727)
    train(smoke, source/'model_44710.pt', 16)
    run = json.loads((smoke/'run.json').read_text())
    assert json.loads((smoke/'SUCCESS.json').read_text())['completed_updates'] == 44727
    for key in ('model', 'ppo_optimizer', 'student_optimizer', 'terrain_restored_exactly',
                'terrain_assignment', 'pit_individual', 'terrain_allocation'):
        assert run['resume_checks'][key], key
    assert not (smoke/'phase_foundation_migration.json').exists()
    rows = [json.loads(line) for line in (smoke/'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 16
    for row in rows:
        cost = row['flat_continuous']
        assert cost['profile'] == 'flat_outlier_v1' and cost['tail_weight'] == .5
        assert cost['teacher']['count']+cost['student']['count'] == 1639
        assert row['phase_transition'] is not None
    (stage/'deployment_receipt.json').write_text(json.dumps(dict(
        completed_updates=44727, resume_checks=run['resume_checks'],
        reward=rows[-1]['flat_continuous'], seconds=rows[-1]['seconds']), indent=2))
    state('v10_evaluation_smoke')
    smoke_selection = stage/'smoke_selection.json'
    with smoke_selection.open('x') as stream:
        json.dump([dict(path=str((smoke/'model_44726.pt').relative_to(root)),
                        label='V10_smoke', teacher=False)], stream)
    smoke_evaluation = root/'recording/flat_v10_eval_smoke_20260921'
    subprocess.run([sys.executable, '-u', str(stage/'evaluate.py'), '--root', str(root),
                    '--code', str(code), '--selection', str(smoke_selection),
                    '--output', str(smoke_evaluation), '--smoke'], cwd=code, check=True)
    assert json.loads((smoke_evaluation/'manifest.json').read_text())['smoke_only']
    state('v10_formal_training', target_completed_updates=45711)
    train(formal, smoke/'model_44726.pt', 984)
    assert json.loads((formal/'SUCCESS.json').read_text())['completed_updates'] == 45711
    selection = [
        dict(path='runs/startup_repair_flat40_96_20260920/model_43201.pt', label='V6_43202', teacher=False),
        dict(path='runs/flat_v9_1000_20260921/model_44710.pt', label='V9_44711', teacher=False),
        dict(path='runs/flat_v10_1000_20260921/model_45710.pt', label='V10_45711', teacher=False),
        dict(path='runs/flat_v10_1000_20260921/model_45710.pt', label='V10_45711_teacher', teacher=True)]
    selection_path = stage/'selection.json'
    with selection_path.open('x') as stream:
        json.dump(selection, stream, indent=2)
    destination = root/'recording/flat_v10_landing_20260921'
    state('v10_matched_evaluation', cases=332)
    subprocess.run([sys.executable, '-u', str(stage/'evaluate.py'), '--root', str(root),
                    '--code', str(code), '--selection', str(selection_path),
                    '--output', str(destination)], cwd=code, check=True)
    subprocess.run([sys.executable, str(stage/'analyze.py'), str(destination)], check=True)
    state('complete_pending_behavior_review', completed_updates=45711,
          evaluation=str(destination), automatic_further_training=False)
except BaseException as exc:
    state('failed', error=repr(exc))
    raise
