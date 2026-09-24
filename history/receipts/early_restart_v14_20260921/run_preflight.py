"""Exclusive 60-second simulation audit before any V14 training."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
N=Path('/root/cts-early-restart-v14-20260921')
SOURCE=R/'runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'
def main():
    with (R/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if json.loads((N/'baseline_status.json').read_text())['state']!='complete': raise RuntimeError('Baseline incomplete')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(): raise RuntimeError('GPU occupied')
        path=N/'preflight_driver_status.json'
        if path.exists(): raise FileExistsError('Preflight already submitted')
        state=dict(state='running',pid=os.getpid(),started=time.time(),source=str(SOURCE),completed_updates=31831,optimizer_updates=0)
        def save(): path.write_text(json.dumps(state,indent=2))
        save()
        try:
            output=N/'preflight_31831'
            with (N/'preflight.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(N/'preflight/launcher.py'),'--root',str(R),'--code',str(N/'code_train'),'--output',str(output),'--checkpoint',str(SOURCE),'--source-completed','31831'],stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save()
                if child.wait()!=0: raise RuntimeError('Preflight launcher failed; inspect log')
                subprocess.run([sys.executable,str(N/'preflight/finalize_audit.py'),'--output',str(output),'--checkpoint',str(SOURCE),'--source-completed','31831'],stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(state='complete',finished=time.time());save()
        except BaseException as exc:
            state.update(state='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__': main()

