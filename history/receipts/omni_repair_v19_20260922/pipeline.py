"""Serialized 10,000-update repair with preserved stages and matched assessments."""
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
SOURCE = BASE/'runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'
SOURCE_SHA = '04fcb792c8f18d3abe913d57eecea1d38b54a4e8a36a9899324bd39540b9876d'
STAGES = (1000, 2000, 3000, 4000)
OWNER_ACQUIRED = False


def status(state, **fields):
    temp = HERE/f'pipeline_status.{os.getpid()}.tmp'
    temp.write_text(json.dumps(dict(state=state, supervisor_pid=os.getpid(),
        time_utc=datetime.now(timezone.utc).isoformat(), source_completed=31831,
        target_completed=41831, authorized_new_updates=10000, **fields), indent=2))
    temp.replace(HERE/'pipeline_status.json')


def verify_code():
    for name, expected in json.loads((HERE/'PREPARED.json').read_text())['files'].items():
        if hashlib.sha256((HERE/'code'/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen code changed: {name}')


def run(stage, argv, completed):
    with (HERE/f'{stage}.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=HERE/'code', stdout=log, stderr=subprocess.STDOUT)
        status(stage, child_pid=child.pid, completed_updates=completed)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'{stage} failed ({rc}); preserve results and diagnose before resume')


def qualified_args():
    tree = ast.parse((HERE/'run_preflight.py').read_text())
    matches = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets)]
    if len(matches) != 1:
        raise ValueError('Qualified command ambiguous')
    return eval(compile(ast.Expression(matches[0]), '<qualified>', 'eval'),
                {'sys': sys, 'HERE': HERE, 'BASE': BASE, 'str': str, '__builtins__': {}})


def collapse_metrics(directory):
    """Stop only gross physical/pose collapse; moderate early variation gets its budget."""
    summary = json.loads((directory/'flat25/summary.json').read_text())
    rows = {row['label']: row for row in summary['forward']}
    terrains = {row['label']: row for row in summary['terrains'] if row['terrain'] == 'flat'}
    sources = ['candidate', 'candidate_teacher']
    severe = any(terrains[label]['physical'] >= 10 for label in sources)
    severe |= all(rows[label]['first_peak_rms'] is not None
                  and rows[label]['first_peak_rms'] > .8 for label in sources)
    terrain_summary = json.loads((directory/'terrain83/summary.json').read_text())
    terrain_rows = {(row['label'], row['terrain']): row for row in terrain_summary['terrains']}
    regressions = []
    pose_regressions = []
    for candidate, reference in [('candidate', 'source31831'), ('candidate_teacher', 'source31831_teacher')]:
        if rows[candidate]['first_peak_rms'] is not None and rows[candidate]['first_peak_rms'] > max(.35, 2*rows[reference]['first_peak_rms']):
            pose_regressions.append(candidate)
        for terrain in ('wave', 'rough_slope', 'stairs_up', 'stairs_down', 'slope_up', 'slope_down', 'obstacles'):
            now, before = terrain_rows[candidate, terrain], terrain_rows[reference, terrain]
            if before['passed'] >= 4 and now['passed'] <= before['passed']-2:
                regressions.append(f'{candidate}:{terrain}')
    lift = json.loads((directory/'flat25/OMNI_LIFT_METRICS.json').read_text())
    counts = {}
    for row in lift['cases']:
        if row['label'] in ('candidate', 'candidate_teacher') and row['direction'] in ('left', 'right', 'yaw_left', 'yaw_right'):
            counts[f"{row['label']}:{row['direction']}:{row['mode']}"] = sum(row['qualifying_lift_landings_per_wheel'])
    return dict(severe=severe, forward=rows, flat=terrains, terrain_regressions=regressions,
                pose_regressions=pose_regressions,
                lift_counts=counts, no_candidate_lift=bool(counts) and sum(counts.values()) == 0,
                rule='at least half flat cases physically fail for either policy OR both forward startup RMS >0.8 rad')


def main():
    global OWNER_ACQUIRED
    with (HERE/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OWNER_ACQUIRED = True
        if (HERE/'pipeline_status.json').exists():
            raise RuntimeError('Existing pipeline requires evidence-based resume, not duplicate launch')
        status('starting', completed_updates=31831)
        verify_code()
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Source changed')
        run('preflight_driver', [sys.executable, str(HERE/'run_preflight.py')], 31831)
        run('verify_driver', [sys.executable, str(HERE/'verify_preflight.py')], 31831)
        source, completed, previous_verdict = SOURCE, 31831, None
        for increment in STAGES:
            target = completed+increment
            output = HERE/f'training_{completed}_to_{target}'
            with (BASE/'full_episode_v12_pipeline.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                verify_code()
                if shutil.disk_usage(HERE).free < 10*1024**3:
                    raise RuntimeError('Less than 10 GiB free; preserve checkpoints and request capacity')
                if output.exists():
                    raise RuntimeError('Training output already exists')
                if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('GPU occupied by another run')
                args = qualified_args()
                args[2] = str(HERE/'code/train.py')
                for key, value in (('--resume', source), ('--output', output), ('--updates', increment)):
                    args[args.index(key)+1] = str(value)
                (HERE/f'launch_{target}.json').write_text(json.dumps(dict(args=args, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), indent=2))
                run(f'train_{target}', args, completed)
                result = json.loads((output/'SUCCESS.json').read_text())
                if result.get('completed_updates') != target:
                    raise RuntimeError('Unexpected completed update count')
            source, completed = output/f'model_{target-1}.pt', target
            evaluation = HERE/f'evaluation_{target}'
            run(f'evaluate_{target}', [sys.executable, str(HERE/'evaluate_pair.py'),
                '--output', str(evaluation), '--sources',
                f'source31831:31831:{SOURCE}', f'candidate:{completed}:{source}'], completed)
            verdict = collapse_metrics(evaluation)
            repeated = sorted(set(verdict['terrain_regressions']) & set(previous_verdict['terrain_regressions'])) if previous_verdict else []
            no_lift = bool(previous_verdict and previous_verdict['no_candidate_lift'] and verdict['no_candidate_lift'])
            repeated_pose = sorted(set(verdict['pose_regressions']) & set(previous_verdict['pose_regressions'])) if previous_verdict else []
            verdict.update(repeated_terrain_regressions=repeated,
                           repeated_no_lift=no_lift,
                           repeated_pose_regressions=repeated_pose,
                           stop_for_review=verdict['severe'] or (completed >= 34831 and bool(repeated or no_lift or repeated_pose)))
            (evaluation/'CONTINUATION.json').write_text(json.dumps(verdict, indent=2))
            if verdict['stop_for_review']:
                status('regression_or_no_lift_review_required', completed_updates=completed,
                       checkpoint=str(source), evidence=str(evaluation), automatic_continuation=False)
                return
            previous_verdict = verdict
        status('completed_10000_pending_review', completed_updates=completed,
               checkpoint=str(source), automatic_continuation=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        # A rejected duplicate never owned this run and must not write its status.
        if not OWNER_ACQUIRED:
            raise
        current = json.loads((HERE/'pipeline_status.json').read_text()) if (HERE/'pipeline_status.json').exists() else {}
        if current and current.get('supervisor_pid') != os.getpid():
            raise
        status('technical_failure', error=str(exc), last_stage=current, automatic_retry=False)
        raise
