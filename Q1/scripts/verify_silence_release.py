"""Cross-check corrected outputs against independently decoded input and frozen baseline."""
import argparse,json
from pathlib import Path
import av,numpy as np,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--previous',type=Path,required=True); a=p.parse_args()
rows=[]
for r in pd.read_csv(a.out/'manifest.csv').to_dict('records'):
 sid=r['sample_id']; count=nonzero=0
 with av.open(str(a.data_root/r['relative_path'])) as c:
  for f in c.decode(audio=0):
   v=f.to_ndarray(); count+=v.size; nonzero+=int(np.count_nonzero(v))
 assert count>0
 doc=json.loads((a.out/'alignment'/f'{sid}.json').read_text()); silent=nonzero==0
 assert doc['media']['audio_all_zero']==silent
 with np.load(a.out/'features'/f'{sid}.npz') as now,np.load(a.previous/'features'/f'{sid}.npz') as old:
  for key in ['text','content_mask','raw_vision','raw_vision_valid','vision_times','video_frame_indices']:
   assert np.array_equal(now[key],old[key]),(sid,key)
  if silent:
   assert not now['alignment_mask'].any() and not now['raw_audio_valid'].any()
   assert all(w['start'] is None and w['end'] is None for w in doc['words'])
  else:
   for key in now.files: assert np.array_equal(now[key],old[key]),(sid,key)
 rows.append(dict(sample_id=sid,audio_all_zero=silent,words=len(doc['words']),aligned=sum(w['aligned'] for w in doc['words'])))
report=dict(passed=True,samples=len(rows),silent_samples=sum(r['audio_all_zero'] for r in rows),silent_words=sum(r['words'] for r in rows if r['audio_all_zero']),aligned_words=sum(r['aligned'] for r in rows),non_silent_all_arrays_unchanged=True,all_native_vision_and_text_unchanged=True)
(a.out/'silence_release_audit.json').write_text(json.dumps(report,indent=2)); print(report)
