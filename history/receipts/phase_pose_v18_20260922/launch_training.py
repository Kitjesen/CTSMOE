"""Launch the authorized 1000-update V18 run once, under the shared GPU lock."""
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
OUTPUT = HERE / 'training_31831_to_32831'
SOURCE_SHA = '04fcb792c8f18d3abe913d57eecea1d38b54a4e8a36a9899324bd39540b9876d'


def arguments():
    # Reuse the exact qualified flags, changing only entry point, output and budget.
    tree = ast.parse((HERE / 'run_preflight.py').read_text())
    matches = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets)]
    if len(matches) != 1:
        raise RuntimeError('Qualified launch arguments are ambiguous')
    args = eval(compile(ast.Expression(matches[0]), '<qualified arguments>', 'eval'),
                {'sys': sys, 'HERE': HERE, 'BASE': BASE, 'str': str, '__builtins__': {}})
    if args[2] != str(HERE / 'preflight.py') or args[args.index('--updates') + 1] != '0':
        raise RuntimeError('Unexpected preflight contract')
    args[2] = str(HERE / 'code/train.py')
    args[args.index('--output') + 1] = str(OUTPUT)
    args[args.index('--updates') + 1] = '1000'
    return args


def record(**fields):
    data = dict(time_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
                source_completed=31831, target_completed=32831, additional_updates=1000,
                output=str(OUTPUT), **fields)
    temp = HERE / 'training_pipeline_status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(HERE / 'training_pipeline_status.json')


def main():
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = json.loads((HERE / 'VERIFIED_READY.json').read_text())
        if (verified['state'] != 'verified_not_training' or verified['source_completed'] != 31831
                or verified['source_sha256'] != SOURCE_SHA):
            raise RuntimeError('Verified source contract mismatch')
        manifest = json.loads((HERE / 'PREPARED.json').read_text())
        for name, expected in manifest['files'].items():
            if hashlib.sha256((HERE / 'code' / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen code changed: {name}')
        args = arguments()
        source = Path(args[args.index('--resume') + 1])
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Source checkpoint changed')
        if OUTPUT.exists():
            raise RuntimeError('Training output already exists; refusing duplicate run')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied; refusing concurrent training')
        with (HERE / 'training_launch.json').open('x') as receipt:
            json.dump(dict(args=args, source_sha256=SOURCE_SHA,
                           stop_after_updates=1000, automatic_continuation=False), receipt, indent=2)
        with (HERE / 'training.log').open('x') as log:
            child = subprocess.Popen(args, cwd=HERE / 'code', stdout=log, stderr=subprocess.STDOUT)
            record(state='starting', child_pid=child.pid)
            result = child.wait()
        if result != 0:
            record(state='failed', returncode=result)
            raise RuntimeError(f'Training exited {result}; no automatic retry')
        final = json.loads((OUTPUT / 'SUCCESS.json').read_text())
        if final.get('completed_updates') != 32831:
            record(state='completion_mismatch', result=final)
            raise RuntimeError('Training completion differs from the authorized budget')
        record(state='training_complete_pending_evaluation', completed_updates=32831)


if __name__ == '__main__':
    main()
