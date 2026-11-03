"""Run the existing regression, followed by nine-terrain posture diagnostics."""
import argparse
from pathlib import Path
import subprocess
import sys


def commands(root,output,selection):
    core=root/'startup_repair_regression_core_20260921.py'
    extended=root/'all_scene_phase_audit_v2.py'
    code=root/'code_startup_repair_20260920_v7'
    for path in (core,extended,code):
        if not path.exists():
            raise FileNotFoundError(path)
    shared=['--root',str(root),'--selection',str(selection)]
    return [
        [sys.executable,str(core),*shared,'--output',str(output)],
        [sys.executable,str(extended),*shared,'--code',str(code),
         '--output',str(output.with_name(output.name+'_all_scenes'))],
    ]


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','output','selection'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for command in commands(args.root,args.output,args.selection):
        subprocess.run(command,check=True)
    print('REGRESSION_AND_ALL_SCENES_COMPLETE',flush=True)


if __name__=='__main__':
    main()
