#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
S="$R/anchor_v11_stage_20260921"
C="$R/code_anchor_v11_20260921"
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
for n in anchor_cts.py anchor_objective.py; do cp "$S/reviewed_$n" "$S/$n"; cp "$S/$n" "$C/$n"; done
for n in run_pipeline.py gates.py probe_current.py source_hashes.json test_anchor_cts_integration.py test_anchor_objective.py; do cp "$S/reviewed_$n" "$S/$n"; done
export PYTHONPATH="$C:$R/upstream/source/rsl_rl"
"$P" -m pytest "$S/test_anchor_cts_integration.py" "$S/test_anchor_objective.py" -q
"$P" - <<'PYCODE'
import hashlib,json,pathlib,time
r=pathlib.Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
s=r/'anchor_v11_stage_20260921'; c=r/'code_anchor_v11_20260921'
h=json.loads((s/'source_hashes.json').read_text())
assert all(hashlib.sha256((c/n).read_bytes()).hexdigest()==v for n,v in h.items())
(s/'integration_verification.json').write_text(json.dumps(dict(passed=True,tests=18,verified_unix=time.time(),source_hashes=h),indent=2))
print('VERIFIED_CANDIDATE_SOURCES')
PYCODE
