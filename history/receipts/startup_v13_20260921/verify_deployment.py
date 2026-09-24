"""Check the isolated V13 source and run its affected tests before simulation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
STAGE = ROOT/'startup_v13_stage_20260921'
CODE = ROOT/'code_startup_v13_20260921'

hashes = json.loads((STAGE/'source_hashes.json').read_text())
for name, expected in hashes.items():
    actual = hashlib.sha256((CODE/name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Source hash mismatch: {name}')
env = dict(os.environ)
env['PYTHONPATH'] = f'{CODE}:{ROOT}/upstream/source/rsl_rl'
tests = ['tests/test_startup_sampling.py', 'tests/test_reset_targets.py',
         'test_startup_repair.py', 'test_stair_speed_logic.py', 'test_stance_strength.py',
         'tests/test_anchor_coefficients.py', 'tests/test_anchor_cts_integration.py']
subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=CODE, env=env, check=True)
subprocess.run([sys.executable, '-m', 'py_compile', str(CODE/'train.py'),
                str(STAGE/'evaluate.py'), str(STAGE/'run_pipeline.py')], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'preflight/test_preflight.py',
                'test_pipeline.py', 'test_gates.py'], cwd=STAGE, env=env, check=True)
with (STAGE/'integration_verification.json').open('x') as stream:
    json.dump(dict(passed=True, source_hashes=hashes, local_related_tests_passed=39,
                   remote_affected_tests_passed=True, remote_workflow_tests_passed=True,
                   zero_update_simulation_preflight_required=True), stream, indent=2)
print('V13_DEPLOYMENT_VERIFIED', len(hashes), flush=True)
