"""One locked zero-update qualification; no automatic optimizer continuation."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')


def main():
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU already occupied')
        manifest = json.loads((HERE/'PREPARED.json').read_text())
        def verify():
            for name, value in manifest['files'].items():
                if hashlib.sha256((HERE/'code'/name).read_bytes()).hexdigest() != value:
                    raise RuntimeError(f'Prepared runtime changed: {name}')
        verify()
        args = [sys.executable,'-u',str(HERE/'preflight.py'),
                '--root',str(BASE),'--output',str(HERE/'preflight_3000'),
                '--num-envs','4096','--updates','0','--mixed-pit',
                '--resume','/root/autodl-tmp/cts-omni-repair-v19-20260922/recovery01/training_32831_to_34823/model_33173.pt',
                '--early-restart-profile','early_phase_v14_no_reward',
                '--startup-repair-profile','off','--pit-posture-profile','off',
                '--surface-posture-profile','off','--stance-strength-profile','off',
                '--allocation-profile','balanced_adaptive_v1','--pit-training-profile','strict_v1',
                '--pit-course-profile','global_v1','--actuator-profile','rs0402_transient_v4',
                '--slope-profile','slope45_v1','--command-curriculum-start','intermediate',
                '--base-height-penalty','original','--terrain-training-profile','stairs80_v1',
                '--stair-command-profile','straight_speed_v1','--wave-reset-profile','wave_surface_v1']
        with (HERE/'preflight.log').open('x') as log:
            result = subprocess.run(args,cwd=HERE/'code',stdout=log,stderr=subprocess.STDOUT)
        verify()
        if result.returncode:
            raise RuntimeError(f'Preflight exited {result.returncode}; no retry or training')
        final = json.loads((HERE/'preflight_3000/FINALIZED.json').read_text())
        if final['optimizer_updates'] != 0 or final['steps'] != 3000:
            raise RuntimeError('Incomplete zero-update preflight')
        (HERE/'READY_FOR_REVIEW.json').write_text(json.dumps(dict(state='verified_not_training',
            source_completed=33174, preflight=final),indent=2))


if __name__ == '__main__':
    main()
