import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recovery_verify_v2', HERE/'verify_preflight_v2.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


@pytest.fixture
def evidence():
    import torch
    final = json.loads((HERE/'PREFLIGHT_FINALIZED.json').read_text(encoding='utf-8'))
    run = json.loads((HERE/'PREFLIGHT_RUN.json').read_text(encoding='utf-8'))
    log = (HERE/'PREFLIGHT_REMOTE.log').read_text(encoding='utf-8')
    saved = torch.load(HERE.parent/'checkpoints/model_32830.pt', map_location='cpu', weights_only=False)
    return final, run, log, saved


def test_completed_preflight_and_checkpoint_have_the_exact_same_source(evidence):
    assert verify.SOURCE_PATH.endswith('/training_31831_to_32831/model_32830.pt')
    verify.validate(*evidence)


def test_wrong_checkpoint_path_is_rejected(evidence):
    final, run, log, saved = evidence
    run['resume'] = run['resume'].replace('training_31831_to_32831', 'training_32831_to_32831')
    with pytest.raises(ValueError, match='Wrong source'):
        verify.validate(final, run, log, saved)


def test_changed_reward_is_rejected(evidence):
    final, run, log, saved = evidence
    final['settings']['height_weight'] = -9.
    with pytest.raises(ValueError, match='Unexpected effective'):
        verify.validate(final, run, log, saved)
