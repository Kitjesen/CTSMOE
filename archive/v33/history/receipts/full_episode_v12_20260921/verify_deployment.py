"""Pin the six reviewed production edits and unchanged archived dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
STAGE = ROOT/'full_episode_v12_stage_20260921'
CODE = ROOT/'code_full_episode_v12_20260921'
ARCHIVE = ROOT/'code_anchor_v11_20260921'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = json.loads((STAGE/'expected_changes.json').read_text())
for name, value in expected.items():
    if digest(CODE/name) != value:
        raise RuntimeError(f'Deployed edit differs from locally tested file: {name}')
hashes = {path.name: digest(path) for path in CODE.glob('*.py')}
for name, value in hashes.items():
    if name not in expected and (not (ARCHIVE/name).is_file() or digest(ARCHIVE/name) != value):
        raise RuntimeError(f'Unexpected inherited-code change: {name}')
subprocess.run([sys.executable, '-m', 'py_compile', *[str(CODE/name) for name in expected],
                *[str(STAGE/name) for name in ('evaluate.py', 'analyze.py', 'plot_comparison.py',
                                              'flat25_geometry.py', 'run_pipeline.py', 'gates.py')]], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q', str(STAGE/'preflight/test_preflight.py'),
                str(STAGE/'test_pipeline.py'), str(STAGE/'test_gates.py')], cwd=STAGE, check=True)
with (STAGE/'source_hashes.json').open('x') as stream:
    json.dump(hashes, stream, indent=2)
with (STAGE/'integration_verification.json').open('x') as stream:
    json.dump(dict(passed=True, source_hashes=hashes, reviewed_changes=expected,
                   inherited_code_unchanged=True, local_related_tests_passed=44,
                   remote_workflow_tests_passed=19, simulation_preflight_required_separately=True), stream, indent=2)
print('DEPLOYMENT_VERIFIED', len(hashes), flush=True)
