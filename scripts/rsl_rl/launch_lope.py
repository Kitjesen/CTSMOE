"""Plan or run the Lope 12-DoF task with the pinned Go2 CTS entry points.

Prints a plan unless ``--execute`` is set. Fresh training loads no checkpoint.
This launcher never selects a Thunder task and never reads NEWPOSE_URDF.

Example (plan only):

    python scripts/rsl_rl/launch_lope.py train \\
        --urdf source/lope_lab3/assets/lope/urdf/lope.urdf \\
        --num-envs 32 --updates 2 --seed 42 --run-name lope-smoke
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import uuid

_SOURCE = Path(__file__).resolve().parents[2] / "source"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from lope_lab3.spec import CUMULATIVE_UPDATES, TASK_ID, TERRAIN


def main():
    project = Path(__file__).resolve().parents[2]
    if str(project / "source") not in sys.path:
        sys.path.insert(0, str(project / "source"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--go2-root", type=Path, default=project / "upstream" / "go2_rl_lab")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--usd-cache", type=Path, default=project / "generated" / "lope_usd")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the task default (4096)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, help=f"Additional updates; completed + updates <= {CUMULATIVE_UPDATES}")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument(
        "--terrain",
        choices=("mixed", "flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "obstacles", "wave"),
        help="Playback scene selection; does not change the training curriculum",
    )
    parser.add_argument("--level", type=int, help="Fixed playback terrain row")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = TERRAIN["num_rows"]
    if args.level is not None and not 0 <= args.level < rows:
        parser.error(f"--level must be 0..{rows - 1}")
    if args.num_envs is not None and (args.num_envs < 4 or args.num_envs % 4):
        parser.error("num-envs must be a positive multiple of four for the 75/25 teacher/student split")
    if args.mode == "train" and (args.updates is None or not 1 <= args.updates <= CUMULATIVE_UPDATES):
        parser.error(f"Training requires --updates in [1,{CUMULATIVE_UPDATES}]")
    if args.mode == "play" and args.updates is not None:
        parser.error("--updates only applies to training")
    if args.mode == "play" and args.checkpoint is None:
        parser.error("Playback requires --checkpoint")
    if args.resume and (args.mode != "train" or args.checkpoint is None):
        parser.error("--resume requires training and an explicit --checkpoint")
    if args.mode == "train" and args.checkpoint is not None and not args.resume:
        parser.error("Fresh training loads no checkpoint; use --resume for a Lope checkpoint")
    if args.mode != "play" and (args.terrain is not None or args.level is not None):
        parser.error("--terrain and --level are playback-only")
    if args.video_length < 1:
        parser.error("--video-length must be positive")
    go2 = args.go2_root.resolve()
    entry = go2 / "scripts" / "rsl_rl" / f"{args.mode}.py"
    play_output = None
    if args.mode == "play":
        play_output = project / "results" / (
            "lope_play_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        )
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = play_output / args.checkpoint.name if play_output else args.checkpoint.resolve()
    upstream_args = [str(entry), "--task", TASK_ID, "--device", args.device]
    if args.num_envs is not None:
        upstream_args += ["--num_envs", str(args.num_envs)]
    if checkpoint is not None:
        upstream_args += ["--checkpoint", str(checkpoint)]
    if args.mode == "train":
        upstream_args += ["--max_iterations", str(args.updates), "--logger", "tensorboard", "--seed", str(args.seed)]
        if args.resume:
            upstream_args += ["--resume"]
        if args.run_name:
            upstream_args += ["--run_name", args.run_name]
    if args.video:
        upstream_args += ["--video", "--video_length", str(args.video_length)]
    else:
        upstream_args += ["--visualizer", "none"]
    plan = {
        "execute": args.execute,
        "robot": "lope_rs02_12dof",
        "task": TASK_ID,
        "upstream_entry": str(entry),
        "argv": upstream_args[1:],
        "working_directory": str(project),
        "urdf": str(args.urdf.resolve()),
        "usd_cache": str(args.usd_cache.resolve()),
        "cumulative_target": CUMULATIVE_UPDATES,
        "action_dim": 12,
        "wheel_actions": 0,
        "play_terrain": args.terrain or "mixed" if args.mode == "play" else None,
        "play_fixed_level": args.level,
        "seed": args.seed,
        "checkpoint_source": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "play_output": str(play_output) if play_output else None,
        "training_state": (
            "playback" if args.mode == "play"
            else "lope_resume_with_episode_reset" if args.resume
            else "fresh_random_initialization_from_0"
        ),
        "thunder_checkpoint": False,
        "thunder_task": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    for path in (entry, args.urdf, *([args.checkpoint] if args.checkpoint else [])):
        if not path.is_file():
            parser.error(f"Required file is missing: {path}")
    if play_output:
        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()

        play_output.mkdir(parents=True, exist_ok=False)
        source_digest = digest(args.checkpoint)
        shutil.copy2(args.checkpoint, checkpoint)
        if digest(checkpoint) != source_digest:
            raise RuntimeError("Isolated playback checkpoint copy failed SHA-256 verification")
    paths = [project / "source", go2 / "scripts" / "rsl_rl", go2 / "source" / "rsl_rl", go2 / "source" / "go2_rl_lab"]
    sys.path[:0] = [str(path) for path in paths]
    os.environ["LOPE_URDF"] = str(args.urdf.resolve())
    os.environ["LOPE_USD_CACHE"] = str(args.usd_cache.resolve())
    os.environ["LOPE_RUN_MODE"] = args.mode
    os.environ["LOPE_SEED"] = str(args.seed)
    os.environ.pop("NEWPOSE_URDF", None)
    if args.mode == "play":
        os.environ["LOPE_EVAL_TERRAIN"] = args.terrain or "mixed"
        if args.level is not None:
            os.environ["LOPE_EVAL_LEVEL"] = str(args.level)
        else:
            os.environ.pop("LOPE_EVAL_LEVEL", None)
    os.chdir(project)
    from isaacsim.asset.importer.urdf import URDFImporter  # noqa: F401
    import lope_lab3.tasks.lope  # noqa: F401
    if args.mode == "play":
        from rsl_rl import utils as rsl_utils
        import yaml

        upstream_export_config = rsl_utils.export_model_config

        def export_lope_config(**kwargs):
            destination = Path(upstream_export_config(**kwargs))
            metadata = yaml.safe_load(destination.read_text(encoding="utf-8"))
            completed = int(kwargs["checkpoint_iteration"])
            metadata["training_progress"]["completed_learning_iterations"] = completed
            metadata["checkpoint"]["completed_updates"] = completed
            metadata["deployment_model"].update({
                "robot": "lope",
                "policy_input": "450 values: ten frame-major 45-value observations, oldest first",
                "policy_output": "12 actions: leg-major FR/FL/RR/RL hip/thigh/calf position targets",
                "wheel_actions": 0,
                "history_layout": "frame_major_current_last",
                "weight_compatible_with_thunder": False,
                "weight_compatible_with_go2": False,
            })
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
            temporary.replace(destination)
            return str(destination)

        rsl_utils.export_model_config = export_lope_config
    sys.argv = upstream_args
    runpy.run_path(str(entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
