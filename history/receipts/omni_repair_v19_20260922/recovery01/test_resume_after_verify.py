import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(tmp_path, monkeypatch):
    if os.name != 'posix':
        monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None))
    spec = importlib.util.spec_from_file_location('recovery_resume_test', Path(__file__).with_name('resume_after_verify.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path/'root'
    here = root/'recovery01'
    here.mkdir(parents=True)
    source = root/'source.pt'
    source.write_bytes(b'checkpoint fixture')
    mod.ROOT, mod.HERE, mod.SOURCE = root, here, source
    mod.SOURCE_SHA = hashlib.sha256(source.read_bytes()).hexdigest()
    mod.verify_code = lambda: None
    (root/'pipeline_status.json').write_text(json.dumps(dict(
        state='technical_failure', last_stage=dict(state='recovery_verify'),
        completed_updates=32831, spent_new_updates=1008)))
    (root/'pipeline.py').write_text('')
    (root/'evaluation_32831').mkdir()
    (root/'evaluation_32831/CONTINUATION.json').write_text(json.dumps(dict(stop_for_review=False)))
    return mod


def test_wrong_previous_stage_is_rejected_without_mutating_status(module):
    f = module.ROOT/'pipeline_status.json'
    f.write_text(json.dumps(dict(state='technical_failure', last_stage=dict(state='train_34831'))))
    before = f.read_bytes()
    with pytest.raises(RuntimeError, match='recovery_verify'):
        module.main()
    assert f.read_bytes() == before
    assert not (module.HERE/'STARTED_AFTER_VERIFY.json').exists()


def test_rejected_behavior_cannot_resume(module):
    (module.ROOT/'evaluation_32831/CONTINUATION.json').write_text(json.dumps(dict(stop_for_review=True)))
    calls = []
    module.run = lambda *args: calls.append(args)
    with pytest.raises(RuntimeError, match='Behavior rejection'):
        module.main()
    assert calls == []


def test_failed_independent_verification_never_starts_training(module):
    calls = []
    def failed_verify(stage, argv, completed):
        calls.append((stage, argv, completed))
        raise RuntimeError('independent verification failed')
    module.run = failed_verify
    with pytest.raises(RuntimeError, match='independent verification failed'):
        module.main()
    assert len(calls) == 1
    assert calls[0][0] == 'recovery_verify_v2'
    assert calls[0][1][-1].endswith('verify_preflight_v2.py')
    assert calls[0][2] == 32831


def test_duplicate_marker_rejects_before_any_verification(module):
    (module.HERE/'STARTED_AFTER_VERIFY.json').write_text('{}')
    calls = []
    module.run = lambda *args: calls.append(args)
    with pytest.raises(RuntimeError, match='already been attempted'):
        module.main()
    assert calls == []
