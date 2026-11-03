"""Serialized, zero-update conversion and stairs validation; no training entry."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
V21 = ROOT/'stair_coverage_v21'
CHECKPOINT = V21/'training_33881_to_34823/model_34822.pt'
STUDENT = V21/'training_33881_to_34823/student_34823.pt'
EXPECTED = '521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5'


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [CHECKPOINT, STUDENT, *sorted((V21/'code').glob('*.py'))]}


def main():
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied; refuse concurrent evaluation')
        before = hashes()
        assert before[str(CHECKPOINT)] == EXPECTED
        out = HERE/'results'
        out.mkdir(exist_ok=False)
        (out/'hashes_before.json').write_text(json.dumps(before, indent=2))
        protocol = json.loads((V21/'evaluation_34823/terrain83/manifest.json').read_text())['protocol_config']
        protocol['sources'] = [dict(path=str(CHECKPOINT), completed_updates=34823)]
        selection = [dict(path=str(CHECKPOINT), label='student34823', teacher=False)]
        for name, value in [('protocol.json', protocol), ('selection.json', selection)]:
            (out/name).write_text(json.dumps(value, indent=2))
        def invoke(name, args):
            (HERE/'status.json').write_text(json.dumps(dict(stage=name, optimizer_updates=0)))
            with (HERE/(name+'.log')).open('x') as log:
                subprocess.run([sys.executable, '-u', *map(str, args)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        invoke('onnx_conversion', [HERE/'export_onnx.py', STUDENT, HERE/'export_candidate'])
        for condition in ('nominal', 'perturbed'):
            args = [HERE/'evaluate_stairs_onnx.py', '--root', BASE, '--code', V21/'code',
                    '--output', out/condition, '--selection', out/'selection.json',
                    '--protocol', out/'protocol.json', '--suite', 'stairend',
                    '--onnx', HERE/'export_candidate/thunder_cts_moe_34823.onnx']
            if condition == 'perturbed':
                args.append('--perturbed')
            invoke(condition, args)
        after = hashes()
        if before != after:
            raise RuntimeError('Frozen source changed')
        (out/'FINALIZED.json').write_text(json.dumps(dict(state='complete_pending_behavior_review',
            optimizer_updates=0, cases=144, hashes_unchanged=True), indent=2))
        (HERE/'status.json').write_text(json.dumps(dict(stage='complete_pending_behavior_review', optimizer_updates=0)))


if __name__ == '__main__':
    main()
