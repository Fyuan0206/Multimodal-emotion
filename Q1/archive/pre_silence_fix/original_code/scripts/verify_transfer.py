"""Verify every uploaded input against the local pre-transfer SHA256 manifest."""
import argparse
import json
import hashlib
from pathlib import Path
def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--manifest', type=Path, default=Path('configs/dataset_transfer_manifest.json'))
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
rows = json.loads(args.manifest.read_text())
failures = []
for row in rows:
    p = args.data_root / row['path']
    if not p.is_file() or p.stat().st_size != row['bytes'] or file_hash(p) != row['sha256']:
        failures.append(row['path'])
report = dict(expected_files=len(rows), expected_bytes=sum(r['bytes'] for r in rows), failures=failures, passed=not failures)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(failures))
