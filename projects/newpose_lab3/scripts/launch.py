"""Plan or run pinned Go2 train/play entry points with the Thunder task registered.

Example (prints a plan only): python scripts/launch.py play --urdf /abs/thunder_v4.urdf \
    --checkpoint /abs/model_19538.pt --video --video-length 600
Add --execute to run. Training additionally requires an explicit --updates budget.
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


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--go2-root", type=Path, default=project / "upstream" / "go2_rl_lab")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--usd-cache", type=Path, default=project / "generated" / "thunder_usd")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, help="Additional updates; adapter enforces completed + updates <= 20000")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument("--terrain", choices=("mixed", "flat", "stairs_up", "stairs_down"),
                        help="Playback scene selection; does not change the training curriculum")
    parser.add_argument("--level", type=int, choices=range(10), help="Fixed playback terrain row L0..L9")
    parser.add_argument("--env-override", action="append", default=[], metavar="FIELD=JSON",
                        help="Candidate override: gradual_boxes, height_weight, height_options, hip_extra_weight")
    parser.add_argument("--execute", action="store_true", help="Run the printed plan; omitted means no simulation or learning")
    args = parser.parse_args()
    if args.num_envs < 4 or args.num_envs % 4:
        parser.error("num-envs must be a positive multiple of four for the 75/25 teacher/student split")
    if args.mode == "train" and (args.updates is None or not 1 <= args.updates <= 20000):
        parser.error("Training requires an explicit --updates value in [1,20000]")
    if args.mode == "play" and args.updates is not None:
        parser.error("--updates only applies to training")
    if args.mode != "play" and (args.terrain is not None or args.level is not None):
        parser.error("--terrain and --level are playback-only")
    if args.terrain == "flat" and args.level not in (None, 0):
        parser.error("Flat playback has no L1..L9 difficulty; omit --level or use 0")
    if args.video_length < 1:
        parser.error("--video-length must be positive")
    candidate_overrides = {}
    for item in args.env_override:
        field, separator, value = item.partition("=")
        if not separator or field not in {"gradual_boxes", "height_weight", "height_options", "hip_extra_weight"}:
            parser.error("Unknown candidate override; use FIELD=JSON with one of the four documented fields")
        try:
            candidate_overrides[field] = json.loads(value)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON value for {field}: {exc}")
    go2 = args.go2_root.resolve()
    entry = go2 / "scripts" / "rsl_rl" / f"{args.mode}.py"
    play_output = (project / "results" / ("play_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                   + "_" + uuid.uuid4().hex[:8])) if args.mode == "play" else None
    checkpoint = play_output / args.checkpoint.name if play_output else args.checkpoint.resolve()
    upstream_args = [str(entry), "--task", "Thunder-Newpose-Newton3-v0", "--num_envs", str(args.num_envs),
                     "--device", args.device, "--checkpoint", str(checkpoint)]
    if args.mode == "train":
        upstream_args += ["--resume", "--max_iterations", str(args.updates), "--logger", "tensorboard"]
    if args.video:
        # Leaving the visualizer unspecified lets Go2 select its official headless Newton GL recorder.
        upstream_args += ["--video", "--video_length", str(args.video_length)]
    else:
        upstream_args += ["--visualizer", "none"]
    plan = {"execute": args.execute, "upstream_entry": str(entry), "argv": upstream_args[1:],
            "working_directory": str(project), "urdf": str(args.urdf.resolve()),
            "usd_cache": str(args.usd_cache.resolve()), "cumulative_target": 20000,
            "play_terrain": args.terrain or "mixed" if args.mode == "play" else None,
            "play_fixed_level": args.level,
            "candidate_overrides": candidate_overrides,
            "checkpoint_source": str(args.checkpoint.resolve()),
            "play_output": str(play_output) if play_output else None,
            "training_state": "formal continuation is blocked pending validated terrain/RNG restoration",
            "checkpoint_validation": "compat.runner validates schema, SHA, history and remaining updates before learn",
            "recording": "Go2 video_utils / env.video_recorders / Newton GL" if args.video else "disabled"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    for path in (entry, args.urdf, args.checkpoint):
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
        # Preserve source parameter provenance; upstream writes exports/videos only beside this copy.
        for name in ("agent.yaml", "env.yaml"):
            source = args.checkpoint.parent / "params" / name
            if not source.is_file():
                source = args.checkpoint.parent / name
            if source.is_file():
                (play_output / "params").mkdir(exist_ok=True)
                shutil.copy2(source, play_output / "params" / name)
        (play_output / "playback_input.json").write_text(json.dumps({
            "source": str(args.checkpoint.resolve()), "copy": str(checkpoint), "sha256": source_digest,
            "mode": "diagnostic_fresh_course", "formal_progress_changed": False,
            "candidate_overrides": candidate_overrides}, indent=2), encoding="utf-8")
    paths = [project, go2 / "scripts" / "rsl_rl", go2 / "source" / "rsl_rl", go2 / "source" / "go2_rl_lab"]
    sys.path[:0] = [str(path) for path in paths]
    os.environ["NEWPOSE_URDF"] = str(args.urdf.resolve())
    os.environ["NEWPOSE_USD_CACHE"] = str(args.usd_cache.resolve())
    os.environ["NEWPOSE_CANDIDATE_OVERRIDES"] = json.dumps(candidate_overrides)
    if args.mode == "play":
        os.environ["NEWPOSE_EVAL_TERRAIN"] = args.terrain or "mixed"
        if args.level is not None:
            os.environ["NEWPOSE_EVAL_LEVEL"] = str(args.level)
        else:
            os.environ.pop("NEWPOSE_EVAL_LEVEL", None)
    os.chdir(project)
    # Register the official isolated importer's USD schemas before any Lab schema lookup.
    from isaacsim.asset.importer.urdf import URDFImporter  # noqa: F401
    import thunder_lab3  # noqa: F401
    if args.mode == "play":
        from rsl_rl import utils as rsl_utils
        import yaml
        upstream_export_config = rsl_utils.export_model_config

        def export_thunder_config(**kwargs):
            # Upstream assumes a zero-based iteration; our adapter carries completed update count.
            destination = Path(upstream_export_config(**kwargs))
            metadata = yaml.safe_load(destination.read_text(encoding="utf-8"))
            completed = int(kwargs["checkpoint_iteration"])
            metadata["training_progress"]["completed_learning_iterations"] = completed
            metadata["training_progress"]["estimated_environment_steps"] = None
            metadata["training_progress"]["count_semantics"] = "completed_updates; environment count may change across migration"
            metadata["checkpoint"]["completed_updates"] = completed
            metadata["deployment_model"].update({
                "policy_input": "530 values: ten frame-major 53-value observations, oldest first",
                "policy_output": "16 actions: leg-major FR/FL/RR/RL hip/thigh/calf, then FR/FL/RR/RL wheels",
                "history_layout": "frame_major_current_last", "history_owned_by_caller": True,
                "checkpoint_transfer_equivalence_validated": False,
                "evaluation_mode": "diagnostic_fresh_course",
            })
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
            temporary.replace(destination)
            return str(destination)

        rsl_utils.export_model_config = export_thunder_config
    sys.argv = upstream_args
    runpy.run_path(str(entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
