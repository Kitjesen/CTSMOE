from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
files = [root / n for n in ('README.md', 'TRAINING_PLAN.md', 'routes.py', 'export.py', 'asymmetric.py', 'test_routes.py', 'package_resources.py')]
files += sorted(p for p in (root / 'assets').iterdir() if p.suffix in ('.obj', '.json', '.html', '.png'))
files.append(root / 'assets' / 'continuous_maps_v3.zip')
manifest = {'training_integrated': False, 'github_uploaded': False, 'files': [{'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in files]}
index = root / 'UPLOAD_MANIFEST.json'
index.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
with zipfile.ZipFile(root / 'github_upload_ready.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files + [index]:
        z.write(p, p.relative_to(root).as_posix())
print(f'Packaged {len(files)} files; no upload performed.')
