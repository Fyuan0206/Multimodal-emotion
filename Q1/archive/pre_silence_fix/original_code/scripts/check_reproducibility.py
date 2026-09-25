"""比较五条重复提取样本的全部数组、形状与数据类型，报告逐元素一致性。"""
import argparse
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=ROOT/'outputs')
OUT=parser.parse_args().out
rows=[]
for p in sorted((OUT/'repeat5/features').glob('*.npz')):
    with np.load(p,allow_pickle=False) as a,np.load(OUT/'features'/p.name,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a.files)
        diffs={k:float(np.max(abs(a[k].astype(float)-b[k].astype(float)))) if a[k].size else 0 for k in a.files}
    rows.append({'sample_id':p.stem,'exact_array_equality':all(v==0 for v in diffs.values()),'max_absolute_difference':max(diffs.values())})
assert len(rows)==5,'Run the five-sample repeat command first'
result={'repeat_samples':5,'all_arrays_identical':all(r['exact_array_equality'] for r in rows),'samples':rows}
(OUT/'reproducibility.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['all_arrays_identical']
