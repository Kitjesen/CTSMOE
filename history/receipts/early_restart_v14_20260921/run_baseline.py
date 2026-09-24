"""Read-only early checkpoint comparison; owns the shared simulation lock."""
import fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
N=Path('/root/cts-early-restart-v14-20260921')
def main():
    with (R/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('GPU occupied')
        if (N/'baseline_status.json').exists():
            raise FileExistsError('Baseline already submitted')
        ps={x['path'] for x in json.loads((N/'selection_baseline.json').read_text())}
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ps}
        state=dict(state='running',pid=os.getpid(),started=time.time(),optimizer_updates=0,checkpoint_hashes=hashes)
        def save(): (N/'baseline_status.json').write_text(json.dumps(state,indent=2))
        save()
        try:
            for suite in ('flat25','terrain83'):
                state.update(suite=suite); save()
                out=N/('baseline_'+suite)
                with (N/('baseline_'+suite+'.log')).open('x') as log:
                    subprocess.run([sys.executable,'-u',str(N/'evaluate_early.py'),'--root',str(R),'--code',str(N/'code'),'--protocol',str(N/'protocol_baseline.json'),'--selection',str(N/'selection_baseline.json'),'--suite',suite,'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run([sys.executable,str(N/'analyze.py'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
                state[suite]='complete'; save()
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            state.update(state='complete',finished=time.time(),checkpoints_unchanged=True); save()
        except BaseException as exc:
            state.update(state='failed',error=repr(exc)); save(); raise
if __name__=='__main__': main()

