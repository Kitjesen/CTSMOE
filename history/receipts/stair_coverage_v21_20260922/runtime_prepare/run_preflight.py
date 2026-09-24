"""Run one locked 3000-step V21 qualification, with no training continuation."""
import json
from pathlib import Path
import subprocess
import sys

from runtime_contract import BASE, COMPLETED, NUM_ENVS, STEPS, verify_prepared

HERE = Path(__file__).resolve().parent


def command(here, identity):
    return [sys.executable, '-u', str(here / 'preflight.py'),
            '--root', str(BASE), '--output', str(here / 'preflight_3000'),
            '--num-envs', str(NUM_ENVS), '--updates', '0', '--mixed-pit',
            '--resume', identity['path'],
            '--early-restart-profile', 'early_phase_v14_no_reward',
            '--startup-repair-profile', 'off', '--pit-posture-profile', 'off',
            '--surface-posture-profile', 'off', '--stance-strength-profile', 'off',
            '--allocation-profile', 'balanced_adaptive_v1', '--pit-training-profile', 'strict_v1',
            '--pit-course-profile', 'global_v1', '--actuator-profile', 'rs0402_transient_v4',
            '--slope-profile', 'slope45_v1', '--command-curriculum-start', 'intermediate',
            '--base-height-penalty', 'original', '--terrain-training-profile', 'stairs80_v1',
            '--stair-command-profile', 'straight_speed_v1', '--wave-reset-profile', 'wave_surface_v1']


def main():
    import fcntl

    if sys.flags.optimize:
        raise RuntimeError('Preflight assertions must remain enabled')
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity, _ = verify_prepared(HERE)
        if any((HERE / name).exists() for name in (
                'preflight_3000', 'preflight.log', 'READY_FOR_REVIEW.json', 'VERIFIED_READY.json')):
            raise FileExistsError('Preflight was already attempted; preserve evidence and do not retry')
        if subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already occupied')
        with (HERE / 'preflight.log').open('x') as log:
            result = subprocess.run(command(HERE, identity), cwd=HERE / 'code',
                                    stdout=log, stderr=subprocess.STDOUT)
        verify_prepared(HERE)
        if result.returncode:
            raise RuntimeError(f'Preflight exited {result.returncode}; no retry or training')
        final = json.loads((HERE / 'preflight_3000/FINALIZED.json').read_text())
        if (final['optimizer_updates'] != 0 or final['steps'] != STEPS
                or final['source_completed'] != COMPLETED):
            raise RuntimeError('Incomplete zero-update preflight')
        with (HERE / 'READY_FOR_REVIEW.json').open('x') as stream:
            json.dump(dict(state='completed_pending_independent_verification', source_completed=COMPLETED,
                           optimizer_updates=0, preflight=final), stream, indent=2)


if __name__ == '__main__':
    main()
