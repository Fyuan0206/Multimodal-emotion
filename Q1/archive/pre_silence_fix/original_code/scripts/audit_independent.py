"""独立核对样本表与视频集合，记录媒体头、音视频流信息及源文件校验值。"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--out', type=Path, default=ROOT/'outputs')
args=parser.parse_args()
OUT=args.out
OUT.mkdir(parents=True, exist_ok=True)
DATA=args.data_root
rows = pd.read_excel(DATA / 'label-100.xlsx', dtype={'video_id': str, 'clip_id': str})
keys = list(zip(rows.video_id, rows.clip_id))
expected = {f'{v}/{c}.mp4' for v, c in keys}
actual = {str(p.relative_to(DATA)) for p in DATA.glob('*/*.mp4')}
result = {'label_rows': len(rows), 'unique_keys': len(set(keys)), 'video_files': len(actual),
          'video_id_count': rows.video_id.nunique(), 'missing': sorted(expected - actual),
          'extra': sorted(actual - expected), 'samples': []}
for v, c in keys:
    p = DATA / v / f'{c}.mp4'
    with av.open(str(p)) as container:
        record = {'id': f'{v}__{c}', 'relative_path': str(p.relative_to(DATA)),
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'container_duration_s': container.duration / av.time_base, 'streams': []}
        for stream in container.streams:
            record['streams'].append({'type': stream.type, 'frames': stream.frames,
                                      'time_base': str(stream.time_base),
                                      'duration_s': float(stream.duration * stream.time_base) if stream.duration else None,
                                      'start_s': float(stream.start_time * stream.time_base) if stream.start_time is not None else None})
        result['samples'].append(record)
(OUT / 'inventory_independent.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))
dur = [x['container_duration_s'] for x in result['samples']]
print('Container duration range:', min(dur), max(dur), 'seconds; total:', sum(dur))
assert len(rows) == len(set(keys)) == len(actual) == 100 and expected == actual