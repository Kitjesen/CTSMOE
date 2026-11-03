"""Run one evidence-gated anchor candidate after the existing V10 finishes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from gates import continuation_gate, skill_regressions

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
stage = root/'anchor_v11_stage_20260921'
code = root/'code_anchor_v11_20260921'
source = root/'runs/flat_v10_1000_20260921/model_45710.pt'
reference = root/'runs/startup_repair_flat40_96_20260920/model_43201.pt'
base_label = 'V10_45711'
base_completed = 45711


def state(value, **details):
    data = dict(state=value, updated_unix=time.time(), **details)
    temporary = stage/'pipeline_status.tmp'
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(stage/'pipeline_status.json')
    print(json.dumps(data), flush=True)


def run(script, *args):
    subprocess.run([sys.executable, '-u', str(stage/script), *map(str, args)], cwd=code, check=True)


def train(name, resume, updates):
    destination = root/'runs'/name
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run([sys.executable, '-u', 'train.py', '--root', str(root), '--output', str(destination),
                    '--num-envs', '4096', '--updates', str(updates), '--mixed-pit', '--resume', str(resume),
                    '--startup-anchor-reference', str(reference)], cwd=code, check=True)
    completed = json.loads((destination/'SUCCESS.json').read_text())['completed_updates']
    receipt = json.loads((destination/'run.json').read_text())
    for check in ('model','ppo_optimizer','student_optimizer','terrain_restored_exactly','terrain_assignment',
                  'pit_individual','terrain_allocation'):
        assert receipt['resume_checks'][check], check
    rows = [json.loads(line) for line in (destination/'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == updates
    assert rows[-1]['startup_anchor']['samples']['teacher'] > 0
    assert rows[-1]['startup_anchor']['samples']['student'] > 0
    assert rows[-1]['anchor_main_coefficient'] > 0 and rows[-1]['anchor_student_coefficient'] > 0
    assert not (destination/'phase_foundation_migration.json').exists()
    return destination/f'model_{completed-1}.pt', completed


def evaluate(checkpoint, completed, suffix):
    selection = [dict(path=str(source.relative_to(root)),label=base_label,teacher=False),
                 dict(path=str(source.relative_to(root)),label=base_label+'_teacher',teacher=True),
                 dict(path=str(checkpoint.relative_to(root)),label=f'V11_{completed}',teacher=False),
                 dict(path=str(checkpoint.relative_to(root)),label=f'V11_{completed}_teacher',teacher=True)]
    selection_path = stage/f'selection_{suffix}.json'
    with selection_path.open('x') as stream:
        json.dump(selection, stream, indent=2)
    destination = root/'recording'/f'anchor_v11_{suffix}_20260921'
    run('evaluate.py','--root',root,'--code',code,'--selection',selection_path,'--output',destination)
    run('analyze.py',destination)
    run('plot_comparison.py',destination)
    return destination, json.loads((destination/'analysis.json').read_text())


try:
    for name, expected in json.loads((stage/'source_hashes.json').read_text()).items():
        actual = hashlib.sha256((code/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Unreviewed candidate source: {name}')
    assert json.loads((stage/'integration_verification.json').read_text())['passed']
    state('waiting_for_v10_final_training_and_evaluation', v10_target=45711)
    deadline = time.monotonic()+4*3600
    while True:
        status = json.loads((root/'flat_v10_stage_20260921/pipeline_status.json').read_text())
        if status['state'] == 'complete_pending_behavior_review':
            break
        if status['state'] == 'failed' or time.monotonic() > deadline:
            raise RuntimeError(f'V10 predecessor did not complete: {status}')
        time.sleep(20)
    assert json.loads((source.parent/'SUCCESS.json').read_text())['completed_updates'] == 45711
    baseline_analysis = json.loads((root/'recording/flat_v10_landing_20260921/analysis.json').read_text())
    regressions = skill_regressions(baseline_analysis, 'V9_44711', 'V10_45711')
    # Do not inherit a newer model at the cost of already measured climbing skills.
    # Both choices remain archived; this fixed decision uses completed matched tests.
    if regressions:
        source = root/'runs/flat_v9_1000_20260921/model_44710.pt'
        base_label, base_completed = 'V9_44711', 44711
    (stage/'source_decision.json').write_text(json.dumps(dict(source=str(source), label=base_label,
        v10_skill_regressions=regressions, all_previous_checkpoints_retained=True),indent=2))
    state('checking_selected_source_recovery', source_label=base_label)
    selection_path = stage/'recovery_selection_v10.json'
    with selection_path.open('x') as stream:
        json.dump([dict(path=str(reference.relative_to(root)),label='V6_43202'),
                   dict(path=str(source.relative_to(root)),label=base_label)], stream)
    recovery = root/'recording/anchor_source_recovery_20260921'
    run('probe_current.py','--root',root,'--code',code,'--selection',selection_path,'--output',recovery)
    run('analyze_current_recovery.py',recovery)
    recovery_gate = json.loads((recovery/'recovery_analysis.json').read_text())['gate']
    if not recovery_gate['ready_for_bounded_anchor_experiment']:
        state('rejected_before_training', evidence=str(recovery), reason=recovery_gate)
        sys.exit(0)
    state('anchor_smoke_training', updates=16)
    checkpoint, completed = train('anchor_v11_smoke_20260921',source,16)
    assert completed == base_completed+16
    state('anchor_first_pilot_training', additional_updates=32)
    checkpoint, completed = train('anchor_v11_pilot_20260921',checkpoint,32)
    assert completed == base_completed+48
    state('anchor_first_matched_evaluation', completed_updates=completed, cases=332)
    destination, analysis = evaluate(checkpoint,completed,'pilot48')
    gate = continuation_gate(analysis,base_label,f'V11_{completed}')
    teacher_gate = continuation_gate(analysis,base_label+'_teacher',f'V11_{completed}_teacher')
    (stage/'gate48.json').write_text(json.dumps(dict(student=gate,teacher=teacher_gate),indent=2))
    if not gate['accepted'] or not teacher_gate['accepted']:
        state('pilot_rejected_pending_review', completed_updates=completed, evidence=str(destination),
              student=gate, teacher=teacher_gate, automatic_further_training=False)
        sys.exit(0)
    state('anchor_confirmatory_training', additional_updates=80)
    checkpoint, completed = train('anchor_v11_confirm_20260921',checkpoint,80)
    assert completed == base_completed+128
    state('anchor_final_matched_evaluation', completed_updates=completed, cases=332)
    destination, analysis = evaluate(checkpoint,completed,'confirm128')
    gate = continuation_gate(analysis,base_label,f'V11_{completed}',final=True)
    teacher_gate = continuation_gate(analysis,base_label+'_teacher',f'V11_{completed}_teacher',final=True)
    (stage/'gate128.json').write_text(json.dumps(dict(student=gate,teacher=teacher_gate),indent=2))
    outcome = 'complete_pending_independent_review' if gate['accepted'] and teacher_gate['accepted'] else 'final_rejected_pending_review'
    state(outcome, completed_updates=completed, evidence=str(destination),
          student=gate, teacher=teacher_gate, automatic_further_training=False,
          trained_updates=128, deployed_to_hardware=False)
except BaseException as exc:
    if isinstance(exc, SystemExit) and exc.code == 0:
        raise
    state('failed', error=repr(exc), automatic_further_training=False)
    raise
