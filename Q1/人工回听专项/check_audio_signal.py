"""Audit decoded original audio; zero-valued samples cannot support word alignment."""
import av,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[3]
data=root/'E题/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条'
out=Path(__file__).resolve().parent
rows=[]
for p in sorted(data.glob('*/*.mp4')):
 total=nonzero=0; peak=energy=0.
 with av.open(str(p)) as c:
  codec=c.streams.audio[0].codec_context.name
  for f in c.decode(audio=0):
   a=f.to_ndarray().astype(np.float64); total+=a.size; nonzero+=int(np.count_nonzero(a)); peak=max(peak,float(np.abs(a).max())); energy+=float(np.square(a).sum())
 rows.append(dict(sample_id=p.parent.name+'__'+p.stem,decoded_scalar_samples=total,nonzero_samples=nonzero,peak=peak,rms=(energy/total)**.5 if total else None,all_zero=total>0 and nonzero==0,codec=codec,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
d=pd.DataFrame(rows); d.to_csv(out/'音频信号审计.csv',index=False)
silent=d[d.all_zero].sample_id.tolist()
report=dict(total=len(d),all_zero_count=len(silent),all_zero_sample_ids=silent,meaning='Decoded digital silence; forced alignment times are unsupported by audible speech')
for i in [1,2]:
 p=out/'素材'/f'视频{i}.mp4'; orig=data/'-mJ2ud6oKI8'/f'{i}.mp4'
 report[f'video{i}_copy_matches_original']=hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(orig.read_bytes()).digest()
(out/'音频信号审计.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False))
