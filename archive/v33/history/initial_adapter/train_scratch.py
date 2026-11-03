"""Fresh CTS teacher/student entry: explicit budget, no checkpoint arguments."""
import argparse
import subprocess
import sys
from pathlib import Path

from default_pose import PROFILE, contract


def build_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--terrain-course", choices=("original8",), required=True,
                        help="Nine-terrain continuation is not a validated scratch course")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.updates <= 0 or args.num_envs <= 0:
        parser.error("updates and num-envs must be positive")
    if args.output.exists():
        parser.error("output must be a new directory")
    command = [sys.executable, str(Path(__file__).with_name("train.py")),
               "--root", str(args.root), "--output", str(args.output),
               "--updates", str(args.updates), "--num-envs", str(args.num_envs),
               "--default-pose", PROFILE, "--command-curriculum-start", "original",
               "--actuator-profile", "rs0402_transient_v4"]
    return args, command


def main():
    import json
    args, command = build_command()
    if args.dry_run:
        print(json.dumps(dict(state="prepared_not_started", command=command,
                              default_pose=contract(PROFILE), resume=None,
                              starting_completed_updates=0), indent=2))
    else:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
