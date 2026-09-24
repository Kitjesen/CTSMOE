"""CPU-only qualification of build scope, zero-update guards, and receipt rejection."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_runtime as build  # noqa: E402
import runtime_contract as contract  # noqa: E402
import run_preflight  # noqa: E402
import verify_preflight as verifier  # noqa: E402
from preflight_support import MutationGuard  # noqa: E402

RECEIPTS = HERE.parents[1]
V20 = RECEIPTS / 'speed_brake_v20_20260922'
TRAIN = RECEIPTS / 'omni_repair_v19_20260922/recovery01/code/train.py'
COVERAGE = HERE.parent / 'coverage_patch/stair_transition_coverage.py'


def geometry(run):
    columns = run['terrain_column_names']
    groups = {}
    for learner, types in run['terrain_by_learner'].items():
        by_type = {}
        for kind, old in types.items():
            selected = old['count'] // (5 if kind == 'stairs_up' else 2) if kind.startswith('stairs_') else 0
            by_type[kind] = dict(total=old['count'], coverage=selected, original=old['count'] - selected,
                                 coverage_by_level=[selected] + [0] * 21)
        groups[learner] = dict(total=sum(v['total'] for v in by_type.values()),
                              coverage=sum(v['coverage'] for v in by_type.values()), by_type=by_type)
    count = sum(v['coverage'] for v in groups.values())
    meshes = []
    for row in range(22):
        for col in range(40):
            selected = col in (8, 11)
            mesh = dict(row=row, column=col, kind=columns[col], selected=selected,
                        before_sha256='a' * 64, after_sha256=('b' if selected else 'a') * 64,
                        origin_before=[0., 0., 0.], origin_after=[0., 0., 0.], origin_unchanged=True)
            if selected:
                mesh.update(inner_vertices_unchanged=True, inner_nondegenerate_triangles_unchanged=True,
                            xy_unchanged=True, cap_original_level_count=1,
                            changed_faces_outside_inner_square=True, changed_face_count=9000,
                            changed_faces_inner_square_overlap_count=0, changed_faces_min_axis_separation_m=.3)
                for stem in ('inner_vertices', 'inner_triangle_vertex_positions', 'inner_nondegenerate_triangles'):
                    mesh[stem + '_before_sha256'] = mesh[stem + '_after_sha256'] = 'c' * 64
            meshes.append(mesh)
    return dict(profile=contract.PROFILE, generated_cell_count=880, selected_cell_count=44,
                config=dict(columns_by_kind={'stairs_up': [8], 'stairs_down': [11]},
                            generated_rows=list(range(22)), active_original_rows=list(range(10)),
                            geometry_only=True, rewards_commands_terminations_changed=False,
                            expected_column_names=columns), mesh_receipts=meshes,
                assignments=dict(total=4096, coverage_total=count, fraction=count / 4096, by_learner=groups))


def fixture():
    run = json.loads((V20 / 'TRAINING_RUN_REMOTE.json').read_text(encoding='utf-8'))
    final = json.loads((V20 / 'PREFLIGHT_FINALIZED_REMOTE.json').read_text(encoding='utf-8'))
    run.update(resume=verifier.SOURCE_PATH, resume_sha256=verifier.SOURCE_SHA,
               starting_completed_updates=33865, target_completed_updates=33865, additional_updates=0)
    receipt = geometry(run)
    run['stair_transition_coverage'] = copy.deepcopy(receipt)
    final.update(source_completed=33865, checkpoint_sha256=verifier.SOURCE_SHA,
                 common_step_before=812760, common_step_after=815760,
                 runner_iteration_before=33865, runner_iteration_after=33865,
                 state_hashes_before=dict(model='a' * 64, ppo_optimizer='b' * 64, student_optimizer='c' * 64),
                 forbidden_calls=[], mutation_guard_restored=True, checkpoint_artifacts_written=[],
                 initial_coverage=copy.deepcopy(receipt))
    final['state_hashes_after'] = final['state_hashes_before'].copy()
    final['coverage_observation'] = dict(cache_missing_control_steps=0, groups={
        learner + '/' + kind: dict(initial_env_count=values['by_type'][kind]['coverage'])
        for learner, values in receipt['assignments']['by_learner'].items()
        for kind in ('stairs_up', 'stairs_down')})
    s = final['settings']
    s['stair_transition_coverage'] = copy.deepcopy(receipt)
    s['height_course_applied_step'] = s['height_course_current_step'] = 815760
    s['height_weight'] = -1 - 9 * ((815760 // 24 - 31831) / 5000)
    s['effective_reward_audit']['base_height_l2']['weight'] = s['height_weight']
    saved = {key: value for key, value in run.items() if key.endswith('_profile')}
    saved.update(completed_updates=33865, common_step_counter=812760,
                 command_curriculum_start='intermediate', base_height_penalty='original')
    return final, run, 'RESUME_OK 33865\n', saved


def test_identity_is_copied_verbatim_and_command_is_zero_update():
    assert (HERE / 'SOURCE_IDENTITY.json').read_bytes() == (V20 / 'EVALUATION_33865_IDENTITY.json').read_bytes()
    assert (HERE / 'preflight_support.py').read_bytes() == (
        RECEIPTS / 'uphill_runup_v16_20260922/preflight_support.py').read_bytes()
    identity = contract.load_identity(HERE / 'SOURCE_IDENTITY.json')
    args = run_preflight.command(Path('/isolated'), identity)
    assert args[args.index('--updates') + 1] == '0'
    assert args[args.index('--resume') + 1] == identity['path']
    assert '--startup-repair-profile' in args and args[args.index('--startup-repair-profile') + 1] == 'off'
    assert 'driver.py' not in (HERE / 'run_preflight.py').read_text()


def test_train_patch_is_insert_only_after_configure_and_restoration():
    source = TRAIN.read_text(encoding='utf-8')
    patched = build.patch_train(source)
    restored = patched.replace(build.CONFIG_INSERT, '').replace(build.SNAPSHOT_INSERT, '').replace(build.RECEIPT_INSERT, '')
    assert restored == source
    assert patched.index('configure_mixed(cfg,') < patched.index(build.CONFIG_INSERT) < patched.index('raw = Go2Env(cfg=cfg)')
    assert patched.index('resume_checks["terrain_restored_exactly"]') < patched.index(build.SNAPSHOT_INSERT)
    assert patched.index('resume_checks["terrain_allocation"]') < patched.index(build.SNAPSHOT_INSERT)
    assert patched.index('install_early_allocation(allocation,') < patched.index(build.SNAPSHOT_INSERT)
    assert patched.index(build.SNAPSHOT_INSERT) < patched.index('del saved')
    ast.parse(patched)
    with pytest.raises(ValueError, match='already contains'):
        build.patch_train(patched)
    with pytest.raises(ValueError, match='anchor'):
        build.patch_train(source.replace(build.CONFIG_ANCHOR, ''))


def source_tree(tmp_path, monkeypatch):
    root = tmp_path / 'remote'
    code = root / 'speed_brake_v20/code'
    code.mkdir(parents=True)
    (code / 'train.py').write_text(TRAIN.read_text(encoding='utf-8'), encoding='utf-8')
    for i in range(85):
        (code / f'original_{i}.py').write_text(f'VALUE = {i}\n', encoding='utf-8')
    files = {p.name: contract.file_sha256(p) for p in code.glob('*.py')}
    (code.parent / 'PREPARED.json').write_text(json.dumps(dict(files=files)))
    source = root / contract.SOURCE_RELATIVE
    source.parent.mkdir()
    source.write_bytes(b'test-only checkpoint bytes')
    payload = tmp_path / 'payload'
    payload.mkdir()
    for name in build.PAYLOAD_FILES:
        shutil.copyfile(HERE / name, payload / name)
    identity = contract.load_identity(payload / contract.IDENTITY_NAME)
    identity.update(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)
    (payload / contract.IDENTITY_NAME).write_text(json.dumps(identity))
    # Only this synthetic local fixture can establish its own reviewed identity bytes.
    monkeypatch.setattr(contract, 'REVIEWED_IDENTITY_SHA256', contract.file_sha256(payload / contract.IDENTITY_NAME))
    return root, payload, files


@pytest.mark.parametrize('existing_empty', [False, True])
def test_builder_audits_all86_and_changes_only_train(tmp_path, monkeypatch, existing_empty):
    root, payload, files = source_tree(tmp_path, monkeypatch)
    destination = root / 'stair_coverage_v21'
    if existing_empty:
        destination.mkdir()
    prepared = build.build(root, payload, COVERAGE)
    assert prepared['source_files'] == files
    assert len(prepared['files']) == 87 and prepared['changed_source_files'] == ['train.py']
    assert prepared['added_runtime_files'] == ['stair_transition_coverage.py']
    assert prepared['optimizer_updates'] == 0 and prepared['launched'] is False
    contract.verify_prepared(destination)
    for name, sha in files.items():
        assert contract.file_sha256(root / 'speed_brake_v20/code' / name) == sha
    with pytest.raises(FileExistsError):
        build.build(root, payload, COVERAGE)
    (destination / 'code/original_0.py').write_text('MUTATED = True\n')
    with pytest.raises(ValueError, match='Frozen runtime changed'):
        contract.verify_prepared(destination)


@pytest.mark.parametrize('damage', ['source', 'extra', 'identity'])
def test_builder_rejects_before_creating_destination(tmp_path, monkeypatch, damage):
    root, payload, _ = source_tree(tmp_path, monkeypatch)
    if damage == 'source':
        (root / 'speed_brake_v20/code/original_0.py').write_text('changed\n')
    elif damage == 'extra':
        (root / 'speed_brake_v20/code/extra.py').write_text('extra\n')
    else:
        identity = json.loads((payload / contract.IDENTITY_NAME).read_text())
        identity['completed_updates'] = 33864
        (payload / contract.IDENTITY_NAME).write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        build.build(root, payload, COVERAGE)
    assert not (root / 'stair_coverage_v21').exists()


def test_builder_rejects_same_name_unreviewed_coverage(tmp_path, monkeypatch):
    root, payload, _ = source_tree(tmp_path, monkeypatch)
    unreviewed = tmp_path / 'stair_transition_coverage.py'
    unreviewed.write_bytes(COVERAGE.read_bytes() + b'\n# unreviewed change\n')
    with pytest.raises(ValueError, match='reviewed version'):
        build.build(root, payload, unreviewed)
    assert not (root / 'stair_coverage_v21').exists()


def test_identity_receipt_rejects_a_self_consistent_replacement(tmp_path):
    identity = contract.load_identity(HERE / contract.IDENTITY_NAME)
    identity['sha256'] = '0' * 64
    candidate = tmp_path / contract.IDENTITY_NAME
    candidate.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='reviewed original bytes'):
        contract.load_identity(candidate)


def test_valid_inherited_qualification_and_geometry():
    verifier.validate(*fixture())


@pytest.mark.parametrize('damage', [
    'optimizer_update', 'optimizer_hash', 'optimizer_restore', 'iteration', 'common_step',
    'wrong_source', 'global_reward', 'flat_weight', 'mesh_unselected', 'mesh_inner',
    'missing_cell', 'assignment', 'missing_learner_coverage', 'profiles', 'weak_speed', 'missing_triangle_proof',
])
def test_independent_verifier_rejects_incomplete_or_changed_evidence(damage):
    final, run, log, saved = fixture()
    if damage == 'optimizer_update':
        final['optimizer_updates'] = 1
    elif damage == 'optimizer_hash':
        final['state_hashes_after']['student_optimizer'] = 'f' * 64
    elif damage == 'optimizer_restore':
        run['resume_checks']['student_optimizer'] = False
    elif damage == 'iteration':
        log = 'RESUME_OK 33864\n'
    elif damage == 'common_step':
        final['common_step_after'] += 1
    elif damage == 'wrong_source':
        run['resume'] = 'other.pt'
    elif damage == 'global_reward':
        final['settings']['effective_reward_audit']['landing_motion'] = dict(registered=True, weight=-.5)
    elif damage == 'flat_weight':
        final['settings']['allocation']['target_weights']['flat'] = .2
    elif damage == 'mesh_unselected':
        final['settings']['stair_transition_coverage']['mesh_receipts'][0]['after_sha256'] = 'f' * 64
    elif damage == 'mesh_inner':
        final['settings']['stair_transition_coverage']['mesh_receipts'][8]['inner_nondegenerate_triangles_after_sha256'] = 'f' * 64
    elif damage == 'missing_cell':
        final['settings']['stair_transition_coverage']['mesh_receipts'].pop()
    elif damage == 'assignment':
        run['terrain_by_learner']['student']['flat']['count'] += 1
    elif damage == 'missing_learner_coverage':
        final['settings']['stair_transition_coverage']['assignments']['by_learner']['student']['by_type']['stairs_up']['coverage'] = 0
    elif damage == 'profiles':
        saved['actuator_profile'] = 'other'
    elif damage == 'weak_speed':
        final['observed_flat_command_abs_max'][0] = 1.5
    elif damage == 'missing_triangle_proof':
        del final['settings']['stair_transition_coverage']['mesh_receipts'][8]['changed_faces_outside_inner_square']
    with pytest.raises(ValueError):
        verifier.validate(final, run, log, saved)


@pytest.mark.parametrize('owner, name', [
    ('alg', 'update'), ('alg', 'compute_returns'), ('optimizer', 'step'),
    ('optimizer_stu_enc', 'step'), ('runner', 'save'),
])
def test_frozen_mutation_guard_forbids_every_training_and_save_entry(owner, name):
    def noop():
        pass
    runner = SimpleNamespace(save=noop, alg=SimpleNamespace(update=noop, compute_returns=noop,
                            optimizer=SimpleNamespace(step=noop), optimizer_stu_enc=SimpleNamespace(step=noop)))
    target = runner if owner == 'runner' else runner.alg if owner == 'alg' else getattr(runner.alg, owner)
    guard = MutationGuard(runner)
    with guard, pytest.raises(RuntimeError, match='Forbidden call'):
        getattr(target, name)()
    assert guard.restored and len(guard.calls) == 1 and getattr(target, name) is noop


def test_preflight_has_one_fixed_loop_and_never_calls_training_update():
    tree = ast.parse((HERE / 'preflight.py').read_text(encoding='utf-8'))
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    calls = [ast.unparse(n.func) for n in ast.walk(body) if isinstance(n, ast.Call)]
    assert 'MutationGuard' in calls and 'fingerprint' in calls
    assert not {'self.alg.update', 'self.alg.compute_returns', 'self.alg.optimizer.step', 'self.save'} & set(calls)
    assert sum(isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(STEPS)' for n in ast.walk(body)) == 1
    assert contract.STEPS == 3000


def test_exposure_observation_reads_cached_gates_and_excludes_terminal_frames():
    import torch
    from coverage_observation import CoverageObservation

    command = SimpleNamespace(learner=torch.tensor([0, 0, 1, 1]), pit=SimpleNamespace(wheel_ids=[0, 1, 2, 3]))
    data = SimpleNamespace(root_pos_w=torch.tensor([[3.5, 0., 0.]] * 4),
                           body_pos_w=torch.tensor([[[3.5, 0., 0.]] * 4] * 4))
    class Scene(dict):
        terrain = SimpleNamespace(terrain_types=torch.tensor([8, 11, 8, 11]))
        env_origins = torch.zeros(4, 3)
    scene = Scene(robot=SimpleNamespace(data=data))
    env = SimpleNamespace(num_envs=4, device='cpu', step_dt=.02, scene=scene,
                          command_manager=SimpleNamespace(get_term=lambda name: command),
                          common_step_counter=1, _phase_step=1, _pose_v18_step=1,
                          episode_length_buf=torch.tensor([2, 2, 2, 2]),
                          phase_state=dict(ready=torch.tensor([True] * 4), weight=torch.ones(4)),
                          _pose_v18_state=dict(weight=torch.ones(4)))
    observer = CoverageObservation(env, torch)
    observer.observe(env, torch.tensor([False, False, False, True]))
    observer.observe(env, torch.tensor([False] * 4))
    result = observer.snapshot()['groups']
    assert result['teacher/stairs_up']['measures']['outer_band_phase_ready']['longest_continuous_seconds'] == .04
    assert result['student/stairs_down']['measures']['root_outer_band']['env_steps'] == 1
    env._phase_step = -1
    observer.observe(env, torch.tensor([False] * 4))
    assert observer.snapshot()['cache_missing_control_steps'] == 1
    env._phase_step = 1
    env.episode_length_buf[:] = 1
    observer.observe(env, torch.tensor([False] * 4))
    assert observer.snapshot()['groups'] == result
