"""Verify preserved evidence hashes and the three per-event datasets."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
for rel,record in manifest['files'].items():
    path=ROOT/rel
    assert path.is_file(), f'missing: {rel}'
    assert path.stat().st_size==record['bytes'], f'size mismatch: {rel}'
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(block)
    assert hasher.hexdigest()==record['sha256'], f'hash mismatch: {rel}'

expected={
    '01_overnight_descent.csv':100,
    '02_forward_curb.csv':53,
    '03_backward_stairs.csv':44,
}
for name,count in expected.items():
    with (ROOT/'events'/name).open(encoding='utf-8-sig',newline='') as stream:
        actual=sum(1 for _ in csv.DictReader(stream))
    assert actual==count,f'{name}: {actual} rows, expected {count}'

with (ROOT/'事件索引.csv').open(encoding='utf-8-sig',newline='') as stream:
    assert sum(1 for _ in csv.DictReader(stream))==3

print(f'PASS: {len(manifest["files"])} SHA-256 files, three event datasets and three incident rows')
