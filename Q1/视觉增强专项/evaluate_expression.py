"""Independent grouped diagnostic; never modifies extraction artifacts."""
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

PROTOCOL = dict(outer_folds=5, inner_folds=4, alphas=[0.1,1,10,100,1000],
                selection='negative MAE; inner GroupKFold', group='video_id',
                bootstrap_seed=20260925, bootstrap_repetitions=2000,
                target='provided continuous sentiment label, not facial emotion ground truth',
                missing='zero summary plus detection fraction; keep all clips',
                primary='expression7 versus vision15; exploratory follow-up on same 100 clips',
                uncertainty='descriptive source-video cluster bootstrap of fixed OOF predictions; not full retraining CI')
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(x):
    return np.r_[x.mean(0),x.std(0)] if len(x) else np.zeros(x.shape[1]*2)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--expression',type=Path,required=True); p.add_argument('--features',type=Path,required=True); p.add_argument('--labels',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    d=pd.read_excel(a.labels); d['sample_id']=d.video_id+'__'+d.clip_id.astype(str); d=d.sort_values('sample_id').reset_index(drop=True)
    assert not d.sample_id.duplicated().any(); assert d.label.notna().all()
    groups=d.video_id.to_numpy(); y=d.label.to_numpy(float)
    blocks={k:[] for k in ['quality','framing','shape','vision15','text','text_vision','expression7','geometry_expression','text_expression']}; hashes={}; expression_hashes={}; zero=[]
    for sid in d.sample_id:
        path=a.features/f'{sid}.npz'; hashes[sid]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            mask=z['raw_vision_valid'].astype(bool); v=z['raw_vision'][mask].astype(float)
            quality=np.array([mask.mean(),float(not mask.any()),len(mask)/15.])
            # YuNet: bbox x,y,w,h, score, five (x,y) landmarks.
            pts=v[:,5:].reshape(-1,5,2)
            relative=((pts-v[:,:2,None].transpose(0,2,1))/np.maximum(v[:,2:4,None].transpose(0,2,1),1e-8)).reshape(-1,10)
            vis=np.r_[quality,summary(v)]
            text=z['text'][z['content_mask'].astype(bool)].mean(0).astype(float)
            blocks['quality'].append(quality); blocks['framing'].append(np.r_[quality,summary(v[:,:5])])
            blocks['shape'].append(np.r_[quality,summary(relative)])
            blocks['vision15'].append(vis); blocks['text'].append(text); blocks['text_vision'].append(np.r_[text,vis])
            ep=a.expression/f'{sid}.npz'; expression_hashes[sid]=sha(ep)
            with np.load(ep,allow_pickle=False) as ez:
                assert np.array_equal(ez['valid'],mask)
                assert np.array_equal(ez['times'],z['vision_times'])
                emotion=summary(ez['scores'][mask].astype(float))
            blocks['expression7'].append(np.r_[quality,emotion])
            blocks['geometry_expression'].append(np.r_[vis,emotion])
            blocks['text_expression'].append(np.r_[text,quality,emotion])
            if not mask.any(): zero.append(sid)
    blocks={k:np.array(v) for k,v in blocks.items()}
    assert all(np.isfinite(v).all() for v in blocks.values())
    predictions={k:np.full(len(y),np.nan) for k in ['median_baseline','mean_baseline',*blocks]}
    folds=np.zeros(len(y),int); audit=[]
    for fold,(tr,te) in enumerate(GroupKFold(5).split(y,y,groups)):
        assert not set(groups[tr])&set(groups[te]); folds[te]=fold
        predictions['median_baseline'][te]=np.median(y[tr]); predictions['mean_baseline'][te]=y[tr].mean()
        inner=list(GroupKFold(4).split(y[tr],y[tr],groups[tr]))
        assert all(not set(groups[tr][i])&set(groups[tr][j]) for i,j in inner)
        record={'fold':fold,'train_ids':d.sample_id.iloc[tr].tolist(),'test_ids':d.sample_id.iloc[te].tolist(),'selected_alpha':{},'inner_group_disjoint':True}
        for name,x in blocks.items():
            model=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='svd')),{'ridge__alpha':PROTOCOL['alphas']},cv=inner,scoring='neg_mean_absolute_error',n_jobs=1,error_score='raise')
            model.fit(x[tr],y[tr]); predictions[name][te]=model.predict(x[te]); record['selected_alpha'][name]=float(model.best_params_['ridge__alpha'])
        audit.append(record); print('finished fold',fold,flush=True)
    result=d[['sample_id','video_id','clip_id','label']].copy(); result['outer_fold']=folds
    metrics=[]
    for name,pred in predictions.items():
        assert np.isfinite(pred).all(); result[name]=pred
        metrics.append(dict(model=name,dimensions=blocks[name].shape[1] if name in blocks else 0,mae=mean_absolute_error(y,pred),r2=r2_score(y,pred),spearman=float(spearmanr(y,pred).statistic)))
    result.to_csv(a.out/'oof_predictions.csv',index=False); pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False)
    rng=np.random.default_rng(PROTOCOL['bootstrap_seed']); unique=np.unique(groups)
    resamples=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(PROTOCOL['bootstrap_repetitions'])]
    contrasts=[]
    for candidate,reference in [('expression7','median_baseline'),('expression7','vision15'),('expression7','quality'),('geometry_expression','vision15'),('text_expression','text')]:
        diff=abs(y-predictions[candidate])-abs(y-predictions[reference]); boot=np.array([diff[ix].mean() for ix in resamples])
        contrasts.append(dict(candidate=candidate,reference=reference,mae_difference=float(diff.mean()),descriptive_cluster_ci_low=float(np.quantile(boot,.025)),descriptive_cluster_ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(contrasts).to_csv(a.out/'paired_comparisons.csv',index=False)
    (a.out/'fold_audit.json').write_text(json.dumps(audit,indent=2))
    metadata=dict(n=len(y),source_videos=len(unique),zero_detection_samples=zero,features_sha256=hashes,expression_sha256=expression_hashes,labels_sha256=sha(a.labels),script_sha256=sha(__file__),python=sys.version,platform=platform.platform(),sklearn=sklearn.__version__,numpy=np.__version__,all_group_disjoint=True,all_predictions_out_of_fold=True)
    (a.out/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__': main()
