"""Freeze the reviewed orchestration and existing common evaluation dependencies."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V20 = ROOT/'speed_brake_v20'
AUDIT = V20/'landing_audit_v21'
ANALYSIS = Path('/root/cts-early-ablation-v15-20260921')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from runtime_contract import verify_prepared
    verify_prepared(HERE)
    names = ['driver.py', 'run_braking.py', 'seal_orchestration.py', 'SHOWCASE_PRESENTED.json']
    external = [ROOT/'pipeline.py', ROOT/'evaluate_pair.py', ROOT/'evaluate_measured.py',
                ROOT/'omni_metrics.py', V20/'driver_v2.py', V20/'PREPARED.json',
                V20/'evaluate_focus.py', V20/'focus_metrics.py', V20/'focus_protocol.py',
                V20/'requested_evaluation_33865/VERDICT.json',
                V20/'requested_evaluation_33865/focused/protocol.json',
                Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922/protocol.json'),
                ANALYSIS/'analyze.py', ANALYSIS/'enrich_analysis.py',
                *[AUDIT/name for name in ('run_audit.py', 'evaluate_local_platform.py',
                    'analyze_local_platform.py', 'local_platform.py', 'focus_protocol.py')]]
    # The legacy fixed-case evaluator deliberately uses this same pinned runtime
    # for all policies. V21's capped endpoint suite is separate and also common.
    external += sorted(Path('/root/autodl-tmp/cts-posture-stop-v17-20260922/code').rglob('*.py'))
    if not external or not all(path.is_file() for path in external):
        raise FileNotFoundError('Common evaluation dependency missing')
    with (HERE/'EXTERNAL_MANIFEST.json').open('x') as stream:
        json.dump(dict(files={str(path): digest(path) for path in external}), stream, indent=2)
    names.append('EXTERNAL_MANIFEST.json')
    with (HERE/'ORCHESTRATION_MANIFEST.json').open('x') as stream:
        json.dump(dict(files={name: digest(HERE/name) for name in names}), stream, indent=2)
    print('SEALED', len(names), len(external))


if __name__ == '__main__':
    main()
