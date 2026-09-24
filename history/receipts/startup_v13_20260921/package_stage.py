"""Package only the reviewed V13 workflow files for exclusive remote extraction."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
names = ['evaluate.py', 'analyze.py', 'gates.py', 'plot_comparison.py', 'flat25_geometry.py',
         'test_gates.py', 'test_pipeline.py', 'run_pipeline.py', 'run_pipeline.sh',
         'source_hashes.json', 'expected_changes.json', 'verify_deployment.py',
         'PLAN_AND_EVIDENCE.md']
names += [f'preflight/{name}' for name in
          ('launcher.py', 'audit_stats.py', 'finalize_audit.py', 'run.sh', 'test_preflight.py')]
with tarfile.open(HERE/'workflow_r0.tar.gz', 'x:gz') as bundle:
    for name in names:
        bundle.add(HERE/name, arcname=name)
(HERE/'workflow_hashes_r0.json').write_text(json.dumps({
    name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names
}, indent=2))
print('PACKAGED', len(names), (HERE/'workflow_r0.tar.gz').stat().st_size)
