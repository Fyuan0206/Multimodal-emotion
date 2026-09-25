"""仅对已填写且合法的人工参考边界计算误差；缺少参考时报告待核验。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=ROOT/'outputs')
OUT=parser.parse_args().out
data=pd.read_csv(OUT/'human_review_10_clips.csv')
reference=data[['reference_start_s','reference_end_s']].apply(pd.to_numeric,errors='coerce')
assert not data.duplicated(['sample_id','word_index']).any(), 'duplicate_review_word'
provided=data[['reference_start_s','reference_end_s']].notna()
assert (provided.iloc[:,0]==provided.iloc[:,1]).all(), 'incomplete_reference_pair'
assert not (provided & reference.isna()).any().any(), 'invalid_reference_number'
available=reference.notna().all(axis=1)
for sid,group in reference[available].groupby(data.loc[available,'sample_id']):
    media=json.loads((OUT/'alignment'/f'{sid}.json').read_text())['media']
    assert np.isfinite(group.to_numpy()).all(), 'nonfinite_reference'
    assert (group.reference_start_s>=media['audio_start_s']).all()
    assert (group.reference_end_s<=media['audio_end_s']+1e-6).all()
aligned=data[['pred_start_s','pred_end_s']].notna().all(axis=1)
result={'reference_rows':int(available.sum()),'total_review_rows':len(data),'status':'pending'}
if available.any():
    assert (reference[available].reference_end_s>reference[available].reference_start_s).all()
    matched=available&aligned
    result.update(status='partial' if not available.all() else 'annotated',
                  annotated_clips=int(data.loc[available,'sample_id'].nunique()),
                  annotated_but_automatically_unaligned=int((available&~aligned).sum()))
    if matched.any():
        err=np.maximum(abs(data.loc[matched,'pred_start_s']-reference.loc[matched,'reference_start_s']),
                       abs(data.loc[matched,'pred_end_s']-reference.loc[matched,'reference_end_s']))
        result.update(matched_words=int(matched.sum()),p50_boundary_error_ms=float(np.quantile(err,.5)*1000),
                      p90_boundary_error_ms=float(np.quantile(err,.9)*1000),
                      meets_p90_100ms_on_annotated_subset=bool(np.quantile(err,.9)<=.1))
(OUT/'human_review_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))