"""Write V27's immutable source manifest from reviewed local files."""

import hashlib
import json
from pathlib import Path


stage = Path(__file__).resolve().parent
base = stage.parent / "pit_traverse_v26_20260923"
manifest = json.loads((base / "orchestration/FROZEN.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest["code"] = {
    path.relative_to(stage / "code").as_posix(): digest(path)
    for path in sorted((stage / "code").rglob("*.py"))
}
manifest["orchestration"] = {
    name: digest(stage / "orchestration" / name)
    for name in ("preflight.py", "state_guard.py", "verify_preflight.py", "driver.py")
}
output = stage / "orchestration/FROZEN.json"
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(manifest, indent=2) + "\n")
print(len(manifest["code"]), "code files frozen", output)
