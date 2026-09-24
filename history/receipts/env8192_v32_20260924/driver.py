"""Authorized 8192 resource trial followed by the remaining existing budget."""
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import torch
import yaml
from migrate import migrate, exact

OLD = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/yaw_stairs_v32_20260924')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD / 'orchestration'))
from v31_common import LOCK, args, frozen  # noqa: E402 -- isolated existing launch contract
TARGET = 46194


class AgentConfigLoader(yaml.SafeLoader):
    """Isaac's own YAML export uses python/tuple for optimizer betas."""


AgentConfigLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def write(state, **kw):
    p = ROOT / 'STATUS.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state, **kw), indent=2))
    tmp.replace(p)


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return text.strip()


def run(source, out, updates, phase):
    argv = args(source, out, updates)
    argv[argv.index('--num-envs') + 1] = '8192'
    frozen()
    with (ROOT / (phase + '.log')).open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(OLD / 'code/train.py'), *argv], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write(phase, pid=proc.pid, source=str(source), output=str(out), environments=8192, target=TARGET)
        with (ROOT / (phase + '_gpu.csv')).open('x') as telemetry:
            while proc.poll() is None:
                sample = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                telemetry.write(str(time.time()) + ',' + sample.strip() + '\n')
                telemetry.flush()
                time.sleep(2)
        if proc.returncode:
            raise RuntimeError(f'{phase} process failed ({proc.returncode}); source preserved')
    success = json.loads((out / 'SUCCESS.json').read_text())
    expected = torch.load(source, map_location='cpu', weights_only=False)['completed_updates'] + updates
    if success['completed_updates'] != expected:
        raise RuntimeError('Update budget mismatch')
    checkpoint = out / f'model_{expected-1}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main():
    if (ROOT / 'STATUS.json').exists():
        raise FileExistsError('This trial was already attempted')
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_pids():
            raise RuntimeError('GPU still occupied; do not overlap jobs')
        frozen()
        oldout = OLD / 'training_39162_to_46194'
        source = max(oldout.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        saved = torch.load(source, map_location='cpu', weights_only=False)
        start = saved['completed_updates']
        if not (43501 <= start < TARGET - 32):
            raise ValueError('Unexpected source or insufficient remaining budget')
        config = yaml.load((oldout / 'agent.yaml').read_text(), Loader=AgentConfigLoader)
        ratio = config['algorithm']['teacher_env_ratio']
        expanded = migrate(saved, ratio)
        migrated = ROOT / 'source_8192.pt'
        torch.save(expanded, migrated)
        loaded = torch.load(migrated, map_location='cpu', weights_only=False)
        exact(loaded, expanded)
        baseline = [json.loads(line) for line in (oldout / 'metrics.jsonl').read_text().splitlines()[-32:]]
        baseline_s = statistics.median(r['seconds'] for r in baseline)
        receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), start=start,
                       target=TARGET, environments_before=4096, environments_after=8192, teacher_ratio=ratio,
                       steps_per_env=config['num_steps_per_env'], baseline_seconds=baseline_s,
                       model_dual_optimizers_unchanged=True, aggregate_counters_unchanged=True,
                       mapping='new index i copies old index i modulo4096; learner period divides4096',
                       new_terrain_integrated=False, frozen_code=frozen()['code'])
        (ROOT / 'MIGRATION.json').write_text(json.dumps(receipt, indent=2))
        del loaded, expanded, saved
        trial = ROOT / 'trial32'
        checkpoint = run(migrated, trial, 32, 'benchmark32')
        rows = [json.loads(line) for line in (trial / 'metrics.jsonl').read_text().splitlines()]
        if len(rows) != 32:
            raise RuntimeError('Incomplete benchmark evidence')
        median_s = statistics.median(r['seconds'] for r in rows[4:])
        samples = [line.split(',') for line in (ROOT / 'benchmark32_gpu.csv').read_text().splitlines()]
        peak = max(float(s[1]) for s in samples)
        total = min(float(s[2]) for s in samples)
        gain = 2 * baseline_s / median_s
        passed = peak < .92 * total and gain >= .95
        (ROOT / 'BENCHMARK.json').write_text(json.dumps(dict(passed=passed, peak_mib=peak, total_mib=total,
            baseline_seconds=baseline_s, seconds=median_s, throughput_ratio=gain, used_updates=32,
            collect_seconds=statistics.median(r['collect_seconds'] for r in rows[4:]),
            learn_seconds=statistics.median(r['learn_seconds'] for r in rows[4:])), indent=2))
        if not passed:
            write('resource_trial_rejected', checkpoint=str(checkpoint), completed=start+32, target=TARGET)
            return
        # Stay within the original iteration budget; no extra 10000 updates are added.
        checkpoint = run(checkpoint, ROOT / 'continuation', TARGET-start-32, 'training8192')
        write('complete_pending_assessment', checkpoint=str(checkpoint), completed=TARGET)
        assessor = OLD / 'assessment/assess_v31.py'
        with (ROOT / 'assessment.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(assessor), '--checkpoint', str(checkpoint), '--output', str(ROOT / 'assessment')], stdout=log, stderr=subprocess.STDOUT, check=True)
        write('complete_pending_review', checkpoint=str(checkpoint), completed=TARGET, hardware_release=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('technical_failure', error=traceback.format_exc())
        raise
