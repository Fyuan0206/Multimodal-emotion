"""Build a human-review kit only from nonzero original audio; never invent references."""
import argparse, hashlib,json,re,sys,wave
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from pipeline import decode_media
p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--destination',type=Path,required=True); a=p.parse_args()
a.destination.mkdir(parents=True,exist_ok=True); (a.destination/'素材').mkdir(exist_ok=True)
review=pd.read_csv(a.out/'human_review_10_clips.csv',keep_default_na=False)
clips=[]; rows=[]
for i,sid in enumerate(review.sample_id.unique(),1):
 j=json.loads((a.out/'alignment'/f'{sid}.json').read_text()); vid,num=sid.rsplit('__',1); source=a.data_root/vid/f'{num}.mp4'
 signal,observed,t0,*_=decode_media(source)
 assert np.count_nonzero(signal)>0 and not j['media']['audio_all_zero'] and t0>=0
 # Preserve media-relative origin: leading zero padding, no normalization, trimming or time stretch.
 pad=round(t0*16000); samples=np.r_[np.zeros(pad,np.float32),signal]
 pcm=np.rint(np.clip(samples,-1,32767/32768)*32768).astype('<i2')
 assert np.count_nonzero(pcm)>0
 wav=a.destination/'素材'/f'音频{i:02d}.wav'
 with wave.open(str(wav),'wb') as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(pcm.tobytes())
 duration=len(pcm)/16000
 clips.append(dict(sample_id=sid,file=f'素材/{wav.name}',text=j['text'],duration=duration,audio_end=j['media']['audio_end_s'],focus_window=f'0–{duration:.3f}',sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),decoded_peak=float(abs(signal).max()),decoded_rms=float(np.sqrt(np.mean(signal.astype(float)**2)))))
 for wi,w in enumerate(j['words']):
  rows.append(dict(sample_id=sid,word_index=wi,word=w['text'],reference_start_s='',reference_end_s='',review_status='pending',confidence='',reviewer='',review_date='',playback_rate='',reviewer_note=''))
s=(ROOT/'人工回听专项/回听填写.html').read_text()
s=re.sub(r'<h1>.*?</h1><p class="notice"><strong>.*?</strong></p>','<h1>Q1 有声样本词边界回听</h1>',s,count=1)
s=s.replace('Q1 六词回听记录','Q1 有声样本回听记录').replace('先完整听一遍','已检查这些音源含非零音频；请先完整听一遍')
s=s.replace('<option value="confirmed" disabled>已确认（音源无声，停用）</option>','<option value="confirmed">已确认</option>')
start=s.index('const clips='); end=s.index("const main=",start)
s=s[:start]+'const clips='+json.dumps(clips,ensure_ascii=False)+', rows='+json.dumps(rows,ensure_ascii=False)+';\n'+s[end:]
s=s.replace('<video controls','<audio controls').replace('</video>','</audio>').replace("querySelector('video')","querySelector('audio')").replace('视频${ci+1}','音频${ci+1}')
s=s.replace("schema_version:'q1-targeted-human-review-1'", "schema_version:'q1-human-review-1',review_scope:'duration_stratified_10_clips'").replace('Q1_六词回听记录.json','Q1_有声10条回听记录.json')
(a.destination/'回听填写.html').write_text(s)
pd.DataFrame(rows).to_csv(a.destination/'回听记录.csv',index=False,encoding='utf-8-sig')
(a.destination/'素材来源.json').write_text(json.dumps(clips,ensure_ascii=False,indent=2))
(a.destination/'回听操作说明.md').write_text('''# 有声样本回听

请用Chrome、Edge或Safari打开回听填写.html，点击音频播放。所有WAV已验证有非零音频；是否清晰、词是否确实存在仍需实际回听。系统输出音量也需开启。

本包从排除2条全零音频后的98条中，按时长排序等间距选10条，共计'''+str(len(rows))+'''词。它是时长分层覆盖样本，不是严格随机样本，也不是全量人工真值。不要一次凭印象填全部词。

先整段回听，再以1/0.75/0.5倍速定位。起点记目标词最初可听语音成分，终点记最后可听语音成分，不含前后静音；连读无法确定时标为uncertain并备注。播放器时间已经是原片段秒数，慢放不需换算。可跳转或每次移动0.05秒，时间字段可填2—3位小数，但不代表毫秒级准确性。

填写实际回听人、日期、把握和主要倍速；确认后才选confirmed。词未出现或转写不符分别标记，勿猜测。支持分批填写：未完成词留pending；页面不会自动保存，请在关闭/刷新前下载JSON。再次打开需保留前次文件，交给助手合并，勿覆盖丢失。单次下载包含当前页面全部词。

音频来自原视频，16kHz单声道PCM，保留时间原点，不做降噪、增益、裁切或合成。原始哈希和WAV哈希见素材来源.json。旧六词包因原音轨静音已停用。
''')
print(json.dumps(dict(clips=len(clips),words=len(rows),all_wav_nonzero=True),ensure_ascii=False))
