from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS_PATH = ROOT / "scripts/reinforcement_learning/rsl_rl/cli_args.py"
POST20K_SCRIPT = ROOT / "scripts/experiments/run_thunder_post20k_payload.sh"


def _load_cli_args():
    spec = importlib.util.spec_from_file_location("doso_rsl_cli_args", CLI_ARGS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Value:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def test_actor_warm_start_selects_only_actor_and_std():
    cli_args = _load_cli_args()
    source = {
        "actor.0.weight": _Value((4, 3)),
        "actor.0.bias": _Value((4,)),
        "critic.0.weight": _Value((4, 3)),
        "std": _Value((16,)),
    }
    target = dict(source)
    assert set(cli_args.select_actor_warm_start_state(source, target)) == {
        "actor.0.weight",
        "actor.0.bias",
        "std",
    }


def test_actor_warm_start_rejects_missing_std_and_resume_conflict():
    cli_args = _load_cli_args()
    with pytest.raises(ValueError, match="actor.* and std"):
        cli_args.select_actor_warm_start_state({"actor.0.weight": _Value((1,))}, {"actor.0.weight": _Value((1,))})

    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--resume", "--actor-warm-start", "model.pt"])


def test_latency_cli_preserves_zero_default_and_validates_ranges():
    cli_args = _load_cli_args()
    parser = argparse.ArgumentParser()
    cli_args.add_latency_args(parser)
    defaults = parser.parse_args([])
    assert cli_args.latency_kwargs(defaults, seed=42) == {
        "obs_delay_min_frames": 0,
        "obs_delay_max_frames": 0,
        "obs_drop_start_prob": 0.0,
        "obs_drop_max_frames": 0,
        "action_delay_min_frames": 0,
        "action_delay_max_frames": 0,
        "seed": 42,
    }

    configured = parser.parse_args(
        [
            "--obs-delay-min-frames",
            "1",
            "--obs-delay-max-frames",
            "3",
            "--obs-drop-start-prob",
            "0.02",
            "--obs-drop-max-frames",
            "3",
            "--action-delay-min-frames",
            "0",
            "--action-delay-max-frames",
            "2",
        ]
    )
    assert cli_args.latency_kwargs(configured, seed=7)["obs_delay_max_frames"] == 3

    invalid = parser.parse_args(
        ["--obs-delay-min-frames", "3", "--obs-delay-max-frames", "1"]
    )
    with pytest.raises(ValueError, match="observation delay"):
        cli_args.latency_kwargs(invalid, seed=42)


def test_post20k_script_has_four_stages_and_8000_iterations():
    script = POST20K_SCRIPT.read_text(encoding="utf-8")
    assert "s1_nominal_repair 1000" in script
    assert "s2_payload_4kg 2000" in script
    assert "s3_payload_8kg 2500" in script
    assert "s4_full_payload 2500" in script
    assert "--num_envs 32768 --seed 42" in script
    assert "--actor-warm-start" in script
    assert "--resume" in script
    assert "for mass in 0 12" in script

    train_source = (ROOT / "scripts/reinforcement_learning/rsl_rl/train.py").read_text(encoding="utf-8")
    assert 'agent_cfg.algorithm.learning_rate = 2.0e-4' in train_source
    assert 'agent_cfg.algorithm.schedule = "fixed"' in train_source
