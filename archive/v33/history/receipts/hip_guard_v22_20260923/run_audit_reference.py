"""One locked, zero-update paired test of the amended local-platform diagnostic."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V20 = HERE.parent
ROOT = V20.parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'recovery01/training_32831_to_34823/model_33173.pt')
    parser.add_argument('--source-completed', type=int, default=33174)
    parser.add_argument('--candidate', type=Path, default=V20/'training_33174_to_34823/model_33864.pt')
    parser.add_argument('--completed', type=int, default=33865)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied')
        args.output.mkdir(parents=True, exist_ok=False)
        prepared = json.loads((V20/'PREPARED.json').read_text())
        for name, expected in prepared['files'].items():
            if hashlib.sha256((V20/'code'/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen V20 runtime changed: {name}')
        pairs = [('source', args.source_completed, args.source), ('candidate', args.completed, args.candidate)]
        protocol = json.loads((V20/'requested_evaluation_33865/focused/protocol.json').read_text())
        protocol['sources'] = [dict(path=str(path), completed_updates=count) for _,count,path in pairs]
        selection = [dict(path=str(path), label=label+('_teacher' if teacher else ''), teacher=teacher)
                     for label,_,path in pairs for teacher in (False, True)]
        hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for _,_,path in pairs}
        for name, data in [('protocol.json',protocol),('selection.json',selection),('hashes_before.json',hashes)]:
            (args.output/name).write_text(json.dumps(data,indent=2))
        command = [sys.executable, '-u', str(HERE/'evaluate_local_platform.py'), '--root',str(BASE),
                   '--code',str(V20/'code'),'--output',str(args.output/'stairend'),
                   '--selection',str(args.output/'selection.json'),'--protocol',str(args.output/'protocol.json'),
                   '--suite','stairend']
        with (args.output/'evaluation.log').open('x') as log:
            child = subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
            (args.output/'status.json').write_text(json.dumps(dict(state='running',pid=os.getpid(),child_pid=child.pid,
                optimizer_updates=0, started_utc=datetime.now(timezone.utc).isoformat())))
            rc = child.wait()
        if rc:
            (args.output/'status.json').write_text(json.dumps(dict(state='technical_failure',return_code=rc,optimizer_updates=0)))
            raise RuntimeError(f'Diagnostic exited {rc}; preserve evidence, no automatic retry')
        subprocess.run([sys.executable,str(HERE/'analyze_local_platform.py'),str(args.output/'stairend')],check=True)
        manifest=json.loads((args.output/'stairend/manifest.json').read_text())
        if len(manifest['cases']) != 96 or manifest['optimizer_steps'] != 0:
            raise RuntimeError('Incomplete evaluation')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in hashes.items()):
            raise RuntimeError('Model changed during evaluation')
        result=dict(state='complete',cases=96,optimizer_updates=0,hashes_unchanged=True,
                    source_completed=args.source_completed,candidate_completed=args.completed)
        (args.output/'FINALIZED.json').write_text(json.dumps(result,indent=2))
        (args.output/'status.json').write_text(json.dumps(result,indent=2))
        print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
