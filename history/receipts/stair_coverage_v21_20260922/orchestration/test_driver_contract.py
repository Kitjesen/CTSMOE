"""Exercise budget and restoration rejection without Linux locks or a GPU."""
import ast
import hashlib
import json
from pathlib import Path

import pytest


def load_checks():
    tree = ast.parse(Path(__file__).with_name('driver.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('budget', 'check_resume')]
    # The full 880-cell geometry contract has separate coverage/preflight tests.
    scope = {'json': json, 'hashlib': hashlib, 'TARGET': 34823, 'validate_coverage': lambda receipt: None}
    exec(compile(tree, '<driver checks>', 'exec'), scope)
    return scope


def test_updates_count_lost_history_and_stop_at_958():
    budget = load_checks()['budget']
    assert budget(33865)['remaining_new_updates'] == 7958
    assert budget(33881)['v21_new_updates'] == 16
    assert budget(34823)['remaining_new_updates'] == 7000
    assert budget(34823)['v21_new_updates'] == 958
    for count in (33864, 34824, 41823):
        with pytest.raises(ValueError):
            budget(count)


def resume_fixture(output):
    source = output/'model_33864.pt'
    source.write_bytes(b'preserved checkpoint fixture')
    identity = dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), completed=33865,
                    common_step_counter=812760)
    row = dict(resume=str(source), starting_completed_updates=33865,
               additional_updates=16, target_completed_updates=33881, resume_sha256=identity['sha256'],
               terrain_allocation=dict(pending_moves=0),
               resume_checks={key: True for key in ('model', 'ppo_optimizer', 'student_optimizer',
                   'terrain_assignment', 'terrain_restored_exactly', 'command_terrain_membership',
                   'terrain_allocation', 'stair_episode_totals')})
    (output/'run.json').write_text(json.dumps(row))
    (output/'COVERAGE_GEOMETRY.json').write_text(json.dumps(dict(generated_cell_count=880,
        selected_cell_count=44, assignments=dict(total=4096))))
    return source, row, identity


def test_accepts_exact_restore(tmp_path):
    source, _, identity = resume_fixture(tmp_path)
    load_checks()['check_resume'](tmp_path, source, 33865, 16, identity)


@pytest.mark.parametrize('key', ['ppo_optimizer', 'student_optimizer', 'terrain_restored_exactly',
                               'terrain_allocation', 'stair_episode_totals'])
def test_rejects_missing_restore(tmp_path, key):
    source, row, identity = resume_fixture(tmp_path)
    row['resume_checks'][key] = False
    (tmp_path/'run.json').write_text(json.dumps(row))
    with pytest.raises(RuntimeError):
        load_checks()['check_resume'](tmp_path, source, 33865, 16, identity)


def test_rejects_changed_source(tmp_path):
    source, _, identity = resume_fixture(tmp_path)
    source.write_bytes(b'different checkpoint')
    with pytest.raises(RuntimeError):
        load_checks()['check_resume'](tmp_path, source, 33865, 16, identity)
