"""A rejected duplicate must leave the active owner's evidence untouched."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

fcntl = pytest.importorskip('fcntl')


def test_lock_loser_cannot_create_false_status(tmp_path):
    script = tmp_path/'pipeline.py'
    shutil.copyfile(Path(__file__).with_name('pipeline.py'), script)
    with (tmp_path/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'BlockingIOError' in result.stderr
    assert not (tmp_path/'pipeline_status.json').exists()
    assert not list(tmp_path.glob('pipeline_status.*.tmp'))


def test_existing_owner_status_is_preserved(tmp_path):
    script = tmp_path/'pipeline.py'
    shutil.copyfile(Path(__file__).with_name('pipeline.py'), script)
    status = tmp_path/'pipeline_status.json'
    before = '{"state":"train_32831","supervisor_pid":-1}'
    status.write_text(before)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Existing pipeline requires' in result.stderr
    assert status.read_text() == before
