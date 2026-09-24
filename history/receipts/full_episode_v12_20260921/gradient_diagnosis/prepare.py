"""Prepare, but never execute, the bounded V12 static-gradient diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
CHECKPOINT = ROOT/'runs/full_episode_v12_pilot300_20260921/model_45010.pt'
REFERENCE = ROOT/'runs/startup_repair_flat40_96_20260920/model_43201.pt'
STAGE = ROOT/'full_episode_v12_stage_20260921'
CODE = ROOT/'code_full_episode_v12_20260921'
COMPLETED_UPDATES = 45011
NUM_ENVS = 4096
HORIZONS = 12
STEPS_PER_HORIZON = 24
CONTROL_STEPS = HORIZONS*STEPS_PER_HORIZON
WINDOWS_S = {'startup_0_1': [0., 1.], 'sustained_1_5': [1., 5.]}
COEFFICIENTS = {'main': 0.21093362834788681, 'student': 0.0022221143221353456}

REUSED_SOURCES = (
    'receipts/anchor_v11_20260921/distribution_audit/launcher.py',
    'receipts/anchor_v11_20260921/distribution_audit/audit_stats.py',
    'receipts/anchor_v11_20260921/late_diagnosis/probe.py',
    'receipts/anchor_v11_20260921/late_diagnosis/gradient_audit.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol_manifest():
    return {
        'state': 'PREPARED_NOT_EXECUTED',
        'requires_parent_execution_decision': True,
        'checkpoint': str(CHECKPOINT),
        'reference': str(REFERENCE),
        'completed_updates': COMPLETED_UPDATES,
        'num_envs': NUM_ENVS,
        'native_horizon_steps': STEPS_PER_HORIZON,
        'horizons': HORIZONS,
        'control_steps': CONTROL_STEPS,
        'maximum_control_step_budget': 320,
        'windows_s': WINDOWS_S,
        'optimizer_steps': 0,
        'checkpoint_writes': 0,
        'coefficient_changes': 0,
        'startup_profile': 'omni_full_episode_v1',
        'fast_delay': False,
        'objectives': {
            'main_actor': 'on-policy PPO clipped surrogate plus production entropy term using fresh stored actions/log-probabilities/advantages',
            'student_primary': 'production CTS latent distillation plus load-balance loss',
            'v6_anchor': 'production masked physical-scale action loss; applied only where stored anchor_mask is true',
            'sustained_v6_anchor': 'counterfactual diagnostic only; never described as a training gradient after 1 second',
        },
        'comparisons': [
            'main_actor PPO versus weighted V6 main anchor on shared actor parameters',
            'student latent+load-balance versus weighted V6 student anchor on student encoder parameters',
            'PPO uses teacher mean plus student mean; main anchor uses the production half-sum',
        ],
        'sample_filters': {
            'startup_0_1': 'stored anchor_mask, flat special cohort, alive finite observation',
            'sustained_1_5': 'same flat cohort, episode age [1,5), nonzero command, alive, finite support telemetry',
        },
        'scarcity_evidence': [
            'raw rows and unique environments by window/teacher/student',
            'survival fraction, command-direction counts, support-valid counts',
            'nonzero-advantage and nonzero-gradient counts',
        ],
        'conflict_evidence': [
            'unweighted and production-weighted gradient norms',
            'pairwise cosine and dot product on exactly shared parameter vectors',
            'per-action-dimension contribution for 12 leg-position and 4 wheel-velocity outputs',
            'joint pose/target/request error and support/contact summaries for the same selected rows',
        ],
        'truth_constraints': {
            'fresh_on_policy_advantages_required': True,
            'compute_returns_required_each_native_horizon': True,
            'old_v11_offpolicy_observations_are_not_ppo_gradients': True,
            'post_1s_anchor_is_counterfactual': True,
            'no_automatic_coefficient_change': True,
        },
        'reused_sources': list(REUSED_SOURCES),
    }


def validate_inputs(root=ROOT, code=CODE, stage=STAGE):
    checkpoint = root/CHECKPOINT.relative_to(ROOT)
    reference = root/REFERENCE.relative_to(ROOT)
    run = json.loads((checkpoint.parent/'run.json').read_text(encoding='utf-8'))
    success = json.loads((checkpoint.parent/'SUCCESS.json').read_text(encoding='utf-8'))
    gate = json.loads((stage/'gate300.json').read_text(encoding='utf-8'))
    assert checkpoint.is_file() and reference.is_file()
    assert success['completed_updates'] == COMPLETED_UPDATES
    assert run['startup_repair_profile'] == 'omni_full_episode_v1'
    assert run['startup_anchor']['coefficients'] == COEFFICIENTS
    assert gate['accepted'] is False
    # The isolated V12 code deployment contains production modules at top level.
    # Historical V11 receipt paths document provenance only; launcher.py and
    # ledger.py vendor the reviewed zero-update mechanisms into the V12 stage.
    sources = [code/name for name in ('anchor_cts.py', 'anchor_objective.py', 'train.py')]
    for path in sources:
        assert path.is_file(), path
    return {
        'checkpoint_sha256': sha256(checkpoint),
        'reference_sha256': sha256(reference),
        'source_sha256': {str(path.relative_to(code)): sha256(path) for path in sources},
        'gate300_sha256': sha256(stage/'gate300.json'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = validate_inputs()
    args.output.mkdir(parents=True)
    payload = protocol_manifest()
    payload['input_evidence'] = evidence
    (args.output/'PREPARED.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'state': payload['state'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
