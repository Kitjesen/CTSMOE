"""Resume only the unstarted physics phase after protocol-loader repair."""
import fcntl
import hashlib
import json
import subprocess
import sys

from run_validation import BASE, HERE, V21, hashes


def main():
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied')
        metadata = HERE/'results'
        out = HERE/'results_verified'
        out.mkdir(exist_ok=False)
        before = json.loads((metadata/'hashes_before.json').read_text())
        if hashes() != before:
            raise RuntimeError('Source changed')
        onnx = HERE/'export_candidate/thunder_cts_moe_34823.onnx'
        conversion = json.loads((onnx.parent/'CONVERSION.json').read_text())
        if not conversion['conversion_passed'] or hashlib.sha256(onnx.read_bytes()).hexdigest() != conversion['onnx_sha256']:
            raise RuntimeError('ONNX identity or conversion failed')
        for condition in ('nominal', 'perturbed'):
            args = [sys.executable, '-u', str(HERE/'evaluate_stairs_onnx_v3.py'),
                    '--root', str(BASE), '--code', str(V21/'code'),
                    '--output', str(out/condition), '--selection', str(metadata/'selection.json'),
                    '--protocol', str(metadata/'protocol.json'), '--suite', 'stairend', '--onnx', str(onnx)]
            if condition == 'perturbed':
                args.append('--perturbed')
            (HERE/'status.json').write_text(json.dumps(dict(stage=condition, optimizer_updates=0)))
            with (HERE/(condition+'_v3.log')).open('x') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        if hashes() != before:
            raise RuntimeError('Source changed')
        receipt = dict(state='complete_pending_behavior_review', optimizer_updates=0,
                       hashes_unchanged=True, cases=144)
        (out/'FINALIZED.json').write_text(json.dumps(receipt, indent=2))
        (HERE/'status.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
