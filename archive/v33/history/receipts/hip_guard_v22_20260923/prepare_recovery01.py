"""Resume completed zero-update traces; load NPZ arrays once during CPU analysis."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_candidate import replace_once

HERE=Path(__file__).resolve().parent
OUT=HERE/'recovery01'
OUT.mkdir(exist_ok=False)
shutil.copytree(HERE/'assessment',OUT/'assessment_v2',ignore=shutil.ignore_patterns('__pycache__'))
metrics=(OUT/'assessment_v2/hip_metrics.py').read_text()
metrics=replace_once(metrics,"    arrays = np.load(folder/'traces.npz')",
    "    # NpzFile is lazy: repeated per-joint indexing decompresses whole arrays.\n"
    "    # Read each needed array once; formulas and preserved trajectories stay identical.\n"
    "    with np.load(folder/'traces.npz') as archive:\n"
    "        arrays = {k: archive[k] for k in ('joint_pos','target','requested','torque_peak','speed_peak',\n"
    "            'acceleration_peak','velocity','angular_velocity','command','calf_contact_peak')}" )
metrics=metrics.replace("'HIP_PER_JOINT.csv'","'HIP_PER_JOINT_v2.csv'").replace("'HIP_REVIEW.json'","'HIP_REVIEW_v2.json'")
(OUT/'assessment_v2/hip_metrics.py').write_text(metrics,encoding='utf-8')
runner=(OUT/'assessment_v2/run_assessment.py').read_text()
runner=replace_once(runner,"    args = parser.parse_args()","    parser.add_argument('--resume', action='store_true')\n    args = parser.parse_args()")
runner=replace_once(runner,'        args.output.mkdir(exist_ok=False)',
    "        if args.resume:\n"
    "            if not args.output.is_dir() or (args.output/'FINALIZED.json').exists():\n"
    "                raise ValueError('Only an incomplete existing assessment may resume')\n"
    "        else:\n"
    "            args.output.mkdir(exist_ok=False)")
runner=replace_once(runner,"            (args.output/name).write_text(json.dumps(data, indent=2))",
    "            path = args.output/name\n"
    "            if args.resume:\n"
    "                if json.loads(path.read_text()) != data:\n"
    "                    raise ValueError('Resume source/protocol/hash mismatch')\n"
    "            else:\n"
    "                path.write_text(json.dumps(data, indent=2))")
runner=replace_once(runner,"            with (args.output/f'{label}.log').open('x') as log:\n                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)",
    "            destination = args.output/label\n"
    "            if destination.exists():\n"
    "                m = json.loads((destination/'manifest.json').read_text())\n"
    "                if (not args.resume or m['optimizer_steps'] != 0 or m['smoke_only']\n"
    "                        or m['protocol_config'] != protocol or m['hip_contract']['profile'] != 'hip_inward012_v22'\n"
    "                        or m['stairs15'] != stairs15 or m['training_updates'] != 0):\n"
    "                    raise ValueError('Existing trace is not this completed zero-update suite')\n"
    "                if not (destination/'traces.npz').is_file():\n"
    "                    raise ValueError('Missing completed trajectories')\n"
    "            else:\n"
    "                with (args.output/f'{label}.log').open('x') as log:\n"
    "                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)")
(OUT/'assessment_v2/run_assessment.py').write_text(runner,encoding='utf-8')
driver=(HERE/'orchestration03/driver_v22.py').read_text()
driver=driver.replace("STARTED_V22.json","STARTED_V22_RECOVERY01.json")
driver=driver.replace("'assessment/run_assessment.py'","'assessment_v2/run_assessment.py'")
driver=replace_once(driver,"execute('baseline_assessment',","execute('baseline_assessment_recovery01',")
driver=replace_once(driver,"'--output',str(HERE/'baseline_34823_new_hip'),","'--resume', '--output',str(HERE/'baseline_34823_new_hip'),")
driver=replace_once(driver,"            execute('independent_qualification',[sys.executable,str(HERE/'verify_metadata_bridge.py')],completed)",
    "            # Completed independent qualification is retained, not rerun or fabricated.\n"
    "            retained = json.loads((HERE/'VERIFIED_READY.json').read_text())\n"
    "            if (retained['source_sha256'] != identity['sha256']\n"
    "                    or not retained['metadata_only_delta_proven']\n"
    "                    or file_hash(retained['runtime_preflight_path']) != retained['runtime_preflight_sha256']):\n"
    "                raise ValueError('Retained qualification identity changed')\n"
    "            if list(HERE.glob('training_*')):\n"
    "                raise ValueError('This recovery is only for zero-update CPU-analysis interruption')")
driver=replace_once(driver,"    return identity\n",
    "    for name,digest in json.loads((HERE/'RECOVERY01_MANIFEST.json').read_text())['files'].items():\n"
    "        if file_hash(HERE/name) != digest:\n"
    "            raise RuntimeError(f'Frozen recovery changed: {name}')\n"
    "    return identity\n")
(OUT/'driver_v22_recovery01.py').write_text(driver,encoding='utf-8')
manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in OUT.rglob('*') if p.is_file()}
(OUT/'RECOVERY01_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
with zipfile.ZipFile(HERE/'recovery01_payload.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():
            z.write(p,p.relative_to(OUT).as_posix())
print(hashlib.sha256((HERE/'recovery01_payload.zip').read_bytes()).hexdigest())
