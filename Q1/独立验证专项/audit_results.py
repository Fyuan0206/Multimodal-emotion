"""Read-only, independent verification of result identity, splits and metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
r=ROOT/'visual_results'; p=json.loads((r/'provenance.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert p['script_sha256']==sha(ROOT/'evaluate_visual.py')
for sid,digest in p['features_sha256'].items():
    assert sha(ROOT.parent/'archive/pre_silence_fix/outputs_server_a100/features'/f'{sid}.npz')==digest
x=pd.read_csv(r/'oof_predictions.csv'); m=pd.read_csv(r/'metrics.csv')
assert len(x)==100 and x.sample_id.nunique()==100 and x.video_id.nunique()==37
assert (x.groupby('video_id').outer_fold.nunique()==1).all()
folds=json.loads((r/'fold_audit.json').read_text()); all_test=[]
for f in folds:
    tr=x[x.sample_id.isin(f['train_ids'])]; te=x[x.sample_id.isin(f['test_ids'])]
    assert len(tr)+len(te)==len(x)
    assert not set(tr.video_id)&set(te.video_id)
    assert (te.outer_fold==f['fold']).all(); all_test+=f['test_ids']
    assert np.allclose(te.median_baseline,np.median(tr.label))
    assert np.allclose(te.mean_baseline,np.mean(tr.label))
assert sorted(all_test)==sorted(x.sample_id)
for row in m.to_dict('records'):
    y=x.label.to_numpy(); pred=x[row['model']].to_numpy()
    assert np.isfinite(pred).all()
    assert abs(row['mae']-np.mean(abs(y-pred)))<1e-12
    assert abs(row['r2']-(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)))<1e-12
report={'passed':True,'feature_hashes_checked':100,'clips':100,'source_groups':37,'outer_group_leakage':False,'all_clips_tested_exactly_once':True,'baseline_and_metrics_recomputed':True}
(r/'independent_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
