"""Independent CPU replay panels; no simulation or training processes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

here = Path(__file__).parent
folder = here / "recording_v2"
urdf = Path(
    "D:/inovxio/brain/train/rough/cts_moe_scratch/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf"
)


def render(group):
    with (here / f"render_group_{group}.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                str(here / "render_routes.py"),
                str(folder),
                str(urdf),
                str(group),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("FINISHED_GROUP", group, flush=True)


with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(render, [4, 8, 12, 16, 20, 24]))
