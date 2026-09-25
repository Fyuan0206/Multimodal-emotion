"""独立核验全部样本的数据结构、时间映射、有效掩码与区间池化结果，并保留已有人工标注。"""
import argparse
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from artifact_contract import file_hash
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=ROOT/'outputs')
OUT = parser.parse_args().out
manifest=pd.read_csv(OUT/'manifest.csv');expected=set(manifest.sample_id)
assert len(manifest)==len(expected)==100
assert {p.stem for p in (OUT/'features').glob('*.npz')}==expected
assert {p.stem for p in (OUT/'alignment').glob('*.json')}==expected
assert {p.stem for p in (OUT/'records').glob('*.json')}==expected
environment=json.loads((OUT/'environment.json').read_text())
length=int(environment['uniform_length'])
summary=[]; issues=[]; errors=[];totalwords=0;alignedwords=0
records=[]
for sid in sorted(expected):
    rec=json.loads((OUT/'records'/f'{sid}.json').read_text());records.append(rec)
    doc=json.loads((OUT/'alignment'/f'{sid}.json').read_text());words=doc['words'];media=doc['media']
    assert rec['sample_id']==doc['sample_id']==sid
    assert rec['status']=='extracted_unreviewed' and rec['media']==media
    source=manifest.loc[manifest.sample_id==sid].iloc[0]
    source_fp=hashlib.sha256((environment['config_fingerprint']+source.source_sha256+source.text_sha256).encode()).hexdigest()
    assert rec['source_fingerprint']==source_fp,(sid,'stale_record')
    for folder,suffix in [('features','.npz'),('alignment','.json')]:
        assert file_hash(OUT/folder/f'{sid}{suffix}')==rec['artifact_sha256'][folder],(sid,'corrupt_artifact')
    with np.load(OUT/'features'/f'{sid}.npz',allow_pickle=False) as pack:
        d={k:pack[k] for k in pack.files}
    for key,value in d.items():
        assert value.dtype!=object,(sid,key)
        assert np.isfinite(value).all(),(sid,key)
    assert d['text'].shape==(length,768) and d['audio'].shape==(length,27) and d['vision'].shape==(length,15),sid
    pad=d['padding_mask'];assert np.array_equal(pad,~d['text_mask'])
    for mod in ['text','audio','vision']:assert not d[mod][pad].any(),(sid,mod,'nonzero_padding')
    assert not d['audio'][~d['audio_valid_mask']].any()
    assert not d['vision'][~d['vision_valid_mask']].any()
    assert not d['raw_vision'][~d['raw_vision_valid']].any()
    assert not d['audio_valid_mask'][~d['content_mask']].any()
    assert not d['vision_valid_mask'][~d['content_mask']].any()
    for mod in ['audio','vision']:
        times=d[f'{mod}_times'];assert (np.diff(times)>0).all(),sid
        m='video' if mod=='vision' else 'audio'
        assert times.min()>=media[f'{m}_start_s']-1e-6 and times.max()<=media[f'{m}_end_s']+1e-6,sid
    assert int(d['text_length'])==int(d['text_mask'].sum())==rec['token_count']
    assert np.array_equal(d['text_mask'],np.arange(length)<int(d['text_length']))
    assert np.array_equal(d['content_mask'],d['token_word_ids']>=0)
    assert np.all(d['token_times'][~d['alignment_mask']]==-1)
    assert not d['alignment_mask'][~d['content_mask']].any()
    assert (d['audio_valid_mask']<=d['alignment_mask']).all() and (d['vision_valid_mask']<=d['alignment_mask']).all()
    assert len(d['video_frame_indices'])==len(d['raw_vision'])
    assert (np.diff(d['video_frame_indices'])>0).all()
    assert d['video_frame_indices'].min()>=0 and d['video_frame_indices'].max()<media['video_frame_count']
    if media.get('audio_all_zero'):
        assert media['audio_nonzero_samples']==0 and media['audio_peak']==0 and media['audio_rms']==0
        assert not d['alignment_mask'].any() and not d['audio_valid_mask'].any() and not d['vision_valid_mask'].any()
        assert not d['raw_audio_valid'].any() and not d['raw_audio'].any()
        assert all(not w['aligned'] and w.get('reason')=='audio_all_zero' and w['start'] is None and w['end'] is None for w in words)
    for wi,w in enumerate(words):
        assert doc['text'][w['char_start']:w['char_end']]==w['text']
        ti=np.flatnonzero(d['token_word_ids']==wi)
        assert ti.tolist()==w['token_indices']
        assert (d['alignment_mask'][ti]==w['aligned']).all()
    for j,(s,e) in enumerate(d['token_offsets']):
        overlap=[max(0,min(int(e),w['char_end'])-max(int(s),w['char_start'])) for w in words]
        expected_word=int(np.argmax(overlap)) if e>s and overlap and max(overlap)>0 else -1
        assert d['token_word_ids'][j]==expected_word
    prev=-1;wordduration=0
    for word_index,w in enumerate(words):
        totalwords+=1
        if not w['aligned']:
            issues.append({'sample_id':sid,'word_index':word_index,'word':w['text'],'reason':w.get('reason','unaligned'),'score':None});continue
        alignedwords+=1
        assert media['audio_start_s']<=w['start']<w['end']<=media['audio_end_s']+1e-6
        assert w['start']>=prev-1e-8;prev=w['end'];wordduration+=w['end']-w['start']
        if w['score']<.2:issues.append({'sample_id':sid,'word_index':word_index,'word':w['text'],'reason':'low_ctc_score_review','score':w['score']})
        for mod in ['audio','vision']:
            if w.get(f'{mod}_fallback'):
                issues.append({'sample_id':sid,'word_index':word_index,'word':w['text'],'reason':f'{mod}_nearest_frame_fallback','score':w['score']})
        ti=w['token_indices'];assert np.allclose(d['token_times'][ti],[w['start'],w['end']])
        for mod in ['audio','vision']:
            times=d[f'{mod}_times'];idx=np.array(w[f'{mod}_indices'],dtype=int)
            raw=d[f'raw_{mod}'];valid=d[f'raw_{mod}_valid']
            by_interval=np.flatnonzero((times>=w['start'])&(times<w['end']))
            if not len(by_interval):
                assert w[f'{mod}_fallback']
                by_interval=np.array([int(np.argmin(abs(times-(w['start']+w['end'])/2)))])
            else:assert not w[f'{mod}_fallback']
            assert np.array_equal(idx,by_interval),(sid,mod,'wrong_source_indices')
            expected_distance=float(abs(times[idx[0]]-(w['start']+w['end'])/2)) if w[f'{mod}_fallback'] else 0.0
            assert np.isclose(w[f'{mod}_fallback_distance_s'],expected_distance)
            valididx=idx[valid[idx]]
            assert (d[f'{mod}_valid_mask'][ti]==bool(len(valididx))).all()
            value=raw[valididx].mean(0) if len(valididx) else np.zeros(raw.shape[1])
            assert np.allclose(d[mod][ti],value,atol=1e-6),(sid,mod,'wrong_pooled_features')
    for mod,dim in [('text',768),('audio',27),('vision',15)]:
        if mod=='text':
            duration=wordduration;native=int(d['text_length']);validlen=int(d['text_mask'].sum())
            rawduration=media['audio_end_s']-media['audio_start_s'];rate='source transcript tokens'
        else:
            m='video' if mod=='vision' else 'audio'
            rawduration=media[f'{m}_end_s']-media[f'{m}_start_s']
            native=len(d[f'raw_{mod}']);validlen=int(d[f'{mod}_valid_mask'].sum());rate='20 Hz' if mod=='audio' else '15 Hz nearest original PTS'
            if mod=='audio':duration=0.0 if media.get('audio_all_zero') else (media['audio_samples']-media['audio_gap_samples'])/16000
            else:
                t=d['vision_times'];edges=np.r_[media['video_start_s'],(t[:-1]+t[1:])/2,media['video_end_s']]
                duration=float(np.diff(edges)[d['raw_vision_valid']].sum())
        summary.append(dict(sample_id=sid,modality=mod,source_duration_s=rawduration,
                            effective_duration_s=duration,feature_dim=dim,native_length=native,
                            aligned_length=length,aligned_valid_length=validlen,padding_count=int(pad.sum()),
                            native_granularity=rate,aligned_granularity='WordPiece with CTC word interval',
                            padding_rule='zero; padding_mask=true',status='extracted_unreviewed'))
current=pd.read_csv(OUT/'summary.csv').sort_values(['sample_id','modality']).reset_index(drop=True)
verified=pd.DataFrame(summary).sort_values(['sample_id','modality']).reset_index(drop=True)
pd.testing.assert_frame_equal(current[verified.columns],verified,check_dtype=False,atol=1e-9,rtol=1e-9)
verified.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(issues,columns=['sample_id','word_index','word','reason','score']).to_csv(OUT/'alignment_review_issues.csv',index=False)
ordered=sorted([r for r in records if not r['media'].get('audio_all_zero',False)],key=lambda r:r['media']['container_duration_s'])
selected=[ordered[i]['sample_id'] for i in np.linspace(0,len(ordered)-1,min(10,len(ordered))).round().astype(int)]
review=[]
for sid in selected:
    d=json.loads((OUT/'alignment'/f'{sid}.json').read_text())
    for j,w in enumerate(d['words']):
        review.append(dict(sample_id=sid,word_index=j,word=w['text'],pred_start_s=w['start'],pred_end_s=w['end'],
                           reference_start_s='',reference_end_s='',review_status='pending',reviewer_note=''))
review_path=OUT/'human_review_10_clips.csv'
if review_path.exists():
    previous=pd.read_csv(review_path).fillna('')
    saved={(str(r.sample_id),int(r.word_index),str(r.word)):r for r in previous.itertuples()}
    for row in review:
        old=saved.get((row['sample_id'],row['word_index'],row['word']))
        if old is not None:
            for key in ['reference_start_s','reference_end_s','review_status','reviewer_note']:
                row[key]=getattr(old,key)
pd.DataFrame(review).to_csv(review_path,index=False)
result=dict(samples_checked=100,modality_rows=len(summary),shape_finiteness_padding_and_time_checks='passed',
            pooled_features_recomputed='passed',total_words=totalwords,aligned_words=alignedwords,
            automatic_word_coverage=alignedwords/totalwords,unaligned_words=totalwords-alignedwords,
            low_ctc_score_words=sum(x['reason']=='low_ctc_score_review' for x in issues),
            digital_silent_clips=sum(r['media'].get('audio_all_zero',False) for r in records),
            digital_silent_words=sum(x['reason']=='audio_all_zero' for x in issues),
            human_boundary_accuracy='see human_review_metrics.json; automatic checks do not measure boundary accuracy',
            human_review_sample_ids=selected,face_detection_rate_mean=float(np.mean([r['face_detection_rate'] for r in records])),
            zero_face_clips=sum(r['face_detection_rate']==0 for r in records),
            audio_gap_samples=sum(r['media']['audio_gap_samples'] for r in records),
            total_extraction_seconds=sum(r['seconds'] for r in records),
            max_audio_video_end_offset_s=max(abs(r['media']['audio_end_s']-r['media']['video_end_s']) for r in records))
(OUT/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))