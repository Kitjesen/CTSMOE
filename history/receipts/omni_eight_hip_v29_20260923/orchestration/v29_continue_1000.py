"""User-authorized bounded continuation after the pending matched assessment."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v29_common import LOCK, STAGE, args, digest, frozen

HERE = Path(__file__).resolve().parent
STATUS = HERE / 'CONTINUE_35823_TO_36823.json'
START, TARGET = 35823, 36823
OWNED = False


def review(source, candidate):
    reasons = []
    if not candidate['state_hashes_unchanged'] or candidate['optimizer_updates'] != 0:
        reasons.append('evaluation mutated state')
    if set(source['terrain']) != set(candidate['terrain']):
        return ['terrain set mismatch']
    total_source = total_candidate = 0
    for name, baseline in source['terrain'].items():
        row = candidate['terrain'][name]
        metrics = ('physical_termination_events', 'any_hip_inward_ge_039_fraction',
                   'first_second_posture_rms_rad', 'xy_tracking_mae_mps')
        if not all(math.isfinite(row[key]) for key in metrics):
            reasons.append(f'{name}: nonfinite metric')
        a, b = baseline['physical_termination_events'], row['physical_termination_events']
        total_source += a
        total_candidate += b
        if b > max(a + 3, math.ceil(1.5 * a)):
            reasons.append(f'{name}: physical failures regressed')
        if name in ('stairs_up', 'stairs_down') and row['any_hip_inward_ge_039_fraction'] > baseline['any_hip_inward_ge_039_fraction'] / 2:
            reasons.append(f'{name}: hip improvement not retained')
    if total_candidate > total_source:
        reasons.append('total physical failures regressed')
    if candidate['terrain']['flat']['first_second_posture_rms_rad'] > source['terrain']['flat']['first_second_posture_rms_rad'] + .02:
        reasons.append('flat startup posture regressed')
    speed = candidate['max_student_wheel_speed_radps']
    if not math.isfinite(speed) or speed >= 45:
        reasons.append('wheel control-step speed gate')
    return reasons


def write(state, **extra):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(state=state, **extra), indent=2))
    temp.replace(STATUS)


def main():
    global OWNED
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-known-results', action='store_true')
    opts = parser.parse_args()
    receipt = frozen()
    assessment = STAGE / 'assessment'
    source = json.loads((assessment / 'source34823_attempt03/FINALIZED.json').read_text())
    if opts.check_known_results:
        known = json.loads((assessment / 'candidate35123_attempt03/FINALIZED.json').read_text())
        assert not review(source, known), review(source, known)
        known['max_student_wheel_speed_radps'] = 45.
        assert 'wheel control-step speed gate' in review(source, known)
        print('KNOWN_EVIDENCE_AND_OVERSPEED_REJECTION_OK')
        return
    if STATUS.exists():
        raise FileExistsError('Continuation already attempted')
    OWNED = True
    write('waiting_for_35823_assessment', target=TARGET, additional_updates=1000)
    for _ in range(360):
        prior = json.loads((assessment / 'POST700_STATUS.json').read_text())
        if prior['state'] == 'complete_pending_behavior_review':
            break
        if prior['state'] in ('technical_failure', 'training_failed_no_assessment', 'timed_out_waiting_for_training'):
            write('blocked_by_prior_pipeline', prior=prior)
            return
        time.sleep(30)
    else:
        write('timed_out_waiting_for_assessment')
        return
    extension = json.loads((HERE / 'V29_EXTENSION_STATUS.json').read_text())
    candidate = json.loads((assessment / 'candidate35823/FINALIZED.json').read_text())
    checkpoint = Path(extension['checkpoint'])
    if (extension['completed_updates'] != START or TARGET > 41823
            or candidate['student_environments'] != source['student_environments']
            or candidate['steps'] != source['steps']
            or candidate['selected_sha256'] != digest(checkpoint)
            or extension['checkpoint_sha256'] != candidate['selected_sha256']
            or candidate['common_terrain_source_sha256'] != receipt['source_sha256']):
        raise RuntimeError('Source identity or budget mismatch')
    reasons = review(source, candidate)
    if reasons:
        write('behavior_review_rejected', reasons=reasons, completed_updates=START)
        return
    output = STAGE / 'training_35823_to_36823'
    assessor = assessment / 'assess_v29_v3.py'
    contract = json.loads((assessment / 'ASSESSMENT_CONTRACT_ATTEMPT03.json').read_text())
    if digest(assessor) != contract['assessor_sha256'] or output.exists():
        raise RuntimeError('Assessor changed or output exists')
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen()
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied')
        if shutil.disk_usage(STAGE).free < 10 * 1024**3:
            raise RuntimeError('Insufficient free disk')
        with (HERE / 'train_36823.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(STAGE / 'code/train.py'),
                                        *args(checkpoint, output, 1000)], stdout=stream, stderr=subprocess.STDOUT, cwd=HERE)
            write('training', pid=process.pid, source_completed=START, target=TARGET,
                  source_sha256=candidate['selected_sha256'])
            if process.wait():
                raise RuntimeError('Training failed; preserve checkpoints, no automatic retry')
        run = json.loads((output / 'run.json').read_text())
        done = json.loads((output / 'SUCCESS.json').read_text())
        if (done['completed_updates'] != TARGET or run['additional_updates'] != 1000
                or run['resume_sha256'] != candidate['selected_sha256']
                or run['terrain_allocation']['actual_counts'].get('pit', 0) != 0):
            raise RuntimeError('Training completion contract mismatch')
        frozen()
        final_checkpoint = output / f'model_{TARGET - 1}.pt'
        write('evaluating', completed_updates=TARGET, checkpoint=str(final_checkpoint))
        with (assessment / 'candidate36823.log').open('x') as stream:
            subprocess.run([sys.executable, '-u', str(assessor), '--checkpoint', str(final_checkpoint),
                            '--output', str(assessment / 'candidate36823')], stdout=stream, stderr=subprocess.STDOUT, check=True)
        final = json.loads((assessment / 'candidate36823/FINALIZED.json').read_text())
        if final['selected_sha256'] != digest(final_checkpoint):
            raise RuntimeError('Final assessment identity mismatch')
        write('complete_pending_review', completed_updates=TARGET, checkpoint=str(final_checkpoint),
              source_comparison_flags=review(source, final),
              note='No further automatic updates; hardware stair impact remains unvalidated')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OWNED:
            write('technical_failure', error=str(exc))
        raise
