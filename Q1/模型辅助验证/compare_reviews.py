"""Compare supplied machine evidence; never convert machine labels into human truth."""
import hashlib,json,re
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
A=ROOT/'模型辅助验证_KimiK3/Q1_有声10条回听记录.json'
B=ROOT/'模型辅助验证_GPT6Pro/机器辅助核查/asr_check.json'
a=json.loads(A.read_text())['records']; b=json.loads(B.read_text())['clips']
tokens=lambda s: re.findall(r"[a-z]+(?:'[a-z]+)?",s.lower())
bykey={(r['sample_id'],int(r['word_index'])):r for r in a}; assert len(bykey)==len(a)==198
assert len(b)==10 and len({c['sample_id'] for c in b})==10
clips=[]; details=[]; issues=[]
for clip in b:
 sid=clip['sample_id']; doc=json.loads((ROOT/'outputs/server_a100/alignment'/f'{sid}.json').read_text())
 assert clip['reference_text']==doc['text']
 meta=next(x for x in json.loads((ROOT/'有声回听专项/素材来源.json').read_text()) if x['sample_id']==sid)
 for folder in ['有声回听专项','模型辅助验证_GPT6Pro']:
  assert hashlib.sha256((ROOT/folder/meta['file']).read_bytes()).hexdigest()==meta['sha256']
 source_tokens=[]; wi_map=[]
 for wi,w in enumerate(doc['words']):
  ts=tokens(w['text']); source_tokens+=ts; wi_map += [wi]*len(ts)
 recognized=[]; ai_map=[]
 for ai,w in enumerate(clip['asr_words']):
  ts=tokens(w['word']); recognized+=ts; ai_map += [ai]*len(ts)
  if not np.isfinite([w['start_s'],w['end_s']]).all() or not 0<=w['start_s']<w['end_s']<=doc['media']['audio_end_s']+1e-3:
   issues.append(dict(source='ASR',sample_id=sid,index=ai,reason='invalid_or_zero_duration',start=w['start_s'],end=w['end_s']))
 match=SequenceMatcher(None,source_tokens,recognized,autojunk=False); mapping={}
 for block in match.get_matching_blocks():
  for k in range(block.size): mapping.setdefault(wi_map[block.a+k],[]).append(ai_map[block.b+k])
 statuses=[]
 for wi,w in enumerate(doc['words']):
  r=bykey[(sid,wi)]; assert r['word']==w['text']; statuses.append(r['review_status'])
  row=dict(sample_id=sid,word_index=wi,word=w['text'],kimi_reported_status=r['review_status'],kimi_note=r['reviewer_note'],ctc_start=w['start'],ctc_end=w['end'],kimi_start=r['reference_start_s'],kimi_end=r['reference_end_s'],human_reference=False)
  s,e=r['reference_start_s'],r['reference_end_s']; legal=s is not None and e is not None and np.isfinite([s,e]).all() and 0<=s<e<=doc['media']['audio_end_s']+1e-6
  row['kimi_valid_time_pair']=bool(legal)
  if (s is not None or e is not None) and not legal: issues.append(dict(source='Kimi',sample_id=sid,index=wi,reason='invalid_time_pair',start=s,end=e))
  indices=sorted(set(mapping.get(wi,[])))
  # Require one source token and one ASR token to avoid collapsing split words.
  if len(tokens(w['text']))==1 and len(indices)==1:
   aw=clip['asr_words'][indices[0]]; bs,be=aw['start_s'],aw['end_s']
   if 0<=bs<be<=doc['media']['audio_end_s']+1e-3:
    row.update(asr_start=bs,asr_end=be)
    if legal: row['kimi_asr_max_difference_ms']=1000*max(abs(s-bs),abs(e-be))
    if w['aligned']: row['ctc_asr_max_difference_ms']=1000*max(abs(w['start']-bs),abs(w['end']-be))
  if legal and w['aligned']: row['ctc_kimi_max_difference_ms']=1000*max(abs(w['start']-s),abs(w['end']-e))
  details.append(row)
 similarity=SequenceMatcher(None,tokens(doc['text']),tokens(clip['asr_text'])).ratio()
 assert abs(similarity-clip['token_sequence_similarity'])<.000051
 clips.append(dict(sample_id=sid,source_words=len(doc['words']),asr_sequence_similarity=similarity,kimi_mismatch_words=statuses.count('transcript_mismatch'),kimi_uncertain_words=statuses.count('uncertain'),kimi_not_heard_words=statuses.count('not_heard'),given_text=doc['text'],asr_text=clip['asr_text'],review_priority='high' if similarity<.2 else 'normal',disposition='model_flag_only_not_confirmed_source_error'))
d=pd.DataFrame(details); d.to_csv(OUT/'逐词模型比较.csv',index=False); pd.DataFrame(clips).to_csv(OUT/'逐片段转写核查.csv',index=False); pd.DataFrame(issues,columns=['source','sample_id','index','reason','start','end']).to_csv(OUT/'时间合法性问题.csv',index=False)
summary=dict(human_reference=False,status='machine_evidence_only',source_a='Kimi K3 assisted records per user; backend runs not supplied',source_b='GPT-6 Pro supplied faster-whisper-base CPU int8 ASR artifacts; not GPT audio inference',records=len(a),clips=len(b),kimi_status_counts=pd.Series([r['review_status'] for r in a]).value_counts().to_dict(),time_issues=len(issues),high_priority_clips=[r['sample_id'] for r in clips if r['review_priority']=='high'],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A,B]},independence='not established; possible shared Whisper base backend and exposure to CTC predictions',boundary_differences={})
for field in ['ctc_kimi_max_difference_ms','ctc_asr_max_difference_ms','kimi_asr_max_difference_ms']:
 v=d[field].dropna().to_numpy(); summary['boundary_differences'][field]=dict(pairs=len(v),median_ms=float(np.median(v)) if len(v) else None,p90_ms=float(np.quantile(v,.9)) if len(v) else None,meaning='conditional model disagreement, not boundary error or accuracy')
(OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False,indent=2))
