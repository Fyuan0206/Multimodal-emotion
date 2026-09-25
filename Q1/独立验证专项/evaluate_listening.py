"""Read genuine listening records; do not fill, infer or overwrite references."""
import argparse, datetime, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def evaluate(records, alignment):
    required=['sample_id','word_index','word','reference_start_s','reference_end_s','review_status','confidence','reviewer','review_date','playback_rate','reviewer_note']
    if not set(required)<=set(records): raise ValueError('Missing required fields')
    d=records.copy().astype(object).where(records.notna(), '')
    if d.duplicated(['sample_id','word_index']).any(): raise ValueError('Duplicate word')
    rows=[]
    for r in d.to_dict('records'):
        sid=str(r['sample_id'])
        if Path(sid).name!=sid or '/' in sid or '\\' in sid: raise ValueError('Invalid sample ID')
        j=json.loads((alignment/f'{sid}.json').read_text()); wi=float(r['word_index'])
        if not wi.is_integer() or not 0<=wi<len(j['words']): raise ValueError('Invalid word index')
        w=j['words'][int(wi)]
        if r['word']!=w['text']: raise ValueError(f'Word mismatch: {sid}/{wi}')
        state=r['review_status']
        if state not in ['pending','confirmed','uncertain','not_heard','transcript_mismatch']: raise ValueError('Invalid review status')
        s,e=r['reference_start_s'],r['reference_end_s']; present=(s!='',e!='')
        if present[0]!=present[1]: raise ValueError('Incomplete reference pair')
        if all(present):
            s,e=float(s),float(e)
            if not np.isfinite([s,e]).all() or not j['media']['audio_start_s']<=s<e<=j['media']['audio_end_s']+1e-6: raise ValueError('Invalid boundary range')
        if state in ['not_heard','transcript_mismatch'] and any(present): raise ValueError('Unexpected reference for missing word')
        if j['media'].get('audio_all_zero') and (state=='confirmed' or any(present)):
            raise ValueError('Silent original audio cannot provide human word boundaries')
        if state=='confirmed':
            if not all(present) or not str(r['reviewer']).strip() or r['confidence'] not in ['high','medium','low']: raise ValueError('Incomplete confirmed record')
            datetime.date.fromisoformat(str(r['review_date']))
            if float(r['playback_rate']) not in [.5,.75,1.]: raise ValueError('Invalid playback rate')
        out={**r,'pred_start_s':w.get('start'),'pred_end_s':w.get('end'),'automatically_aligned':bool(w['aligned']),'included_in_accuracy':state=='confirmed' and bool(w['aligned'])}
        if out['included_in_accuracy']:
            out.update(start_abs_error_ms=abs(w['start']-s)*1000,end_abs_error_ms=abs(w['end']-e)*1000,max_boundary_error_ms=max(abs(w['start']-s),abs(w['end']-e))*1000)
        rows.append(out)
    detail=pd.DataFrame(rows); used=detail[detail.included_in_accuracy]
    metrics=dict(status='pending' if not len(used) else 'reviewed_subset_only',total_records=len(d),confirmed_records=int((d.review_status=='confirmed').sum()),matched_words=len(used),sampled_clips=int(d.sample_id.nunique()),scope='reviewed subset only; selection design must be documented; not full-corpus accuracy',status_counts=d.review_status.value_counts().to_dict())
    if len(used):
        err=used.max_boundary_error_ms.to_numpy(float)
        metrics.update(start_mae_ms=float(used.start_abs_error_ms.mean()),end_mae_ms=float(used.end_abs_error_ms.mean()),p50_max_boundary_ms=float(np.quantile(err,.5)),p90_max_boundary_ms=float(np.quantile(err,.9)),within_100ms_fraction=float((err<=100).mean()),max_boundary_ms=float(err.max()))
    return detail,metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--review',type=Path,required=True); p.add_argument('--alignment',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.review.suffix.lower()=='.json':
        j=json.loads(a.review.read_text())
        if j.get('schema_version') not in ['q1-targeted-human-review-1','q1-human-review-1'] or j.get('time_origin')!='original_clip_seconds': raise ValueError('Unknown reference schema/time origin')
        d=pd.DataFrame(j['records'])
    else: d=pd.read_csv(a.review,keep_default_na=False)
    detail,metrics=evaluate(d,a.alignment)
    if a.review.suffix.lower()=='.json' and j.get('review_scope')=='duration_stratified_10_clips':
        metrics['scope']='duration-stratified audible-clip subset; not full-corpus accuracy'
    metrics['reference_sha256']=hashlib.sha256(a.review.read_bytes()).hexdigest()
    metrics['alignment_sha256']={sid:hashlib.sha256((a.alignment/f'{sid}.json').read_bytes()).hexdigest() for sid in d.sample_id.unique()}
    a.out.mkdir(parents=True,exist_ok=True); detail.to_csv(a.out/'word_errors.csv',index=False)
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
