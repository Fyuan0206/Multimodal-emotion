"""Frozen OpenCV Zoo expression model on exactly the existing Q1 native frames."""
import argparse, datetime, hashlib, json, os, sys, traceback
from pathlib import Path
import av, cv2, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from facial_fer_model import FacialExpressionRecog
CLASSES=['angry','disgust','fearful','happy','neutral','sad','surprised']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,ensure_ascii=False)); t.replace(p)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--q1',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--limit',type=int); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'features').mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    sources=json.loads((ROOT/'model_sources.json').read_text())
    for name,meta in sources['files'].items(): assert sha(ROOT/name)==meta['sha256'],name
    model_path=ROOT/'models/facial_expression_recognition_mobilefacenet_2022july.onnx'
    model=FacialExpressionRecog(str(model_path))
    # Actual inference smoke test: exercises preprocessing + ONNX, never used as sample data.
    blob=model._preprocess(np.zeros((112,112,3),np.uint8),None)
    model._model.setInput(blob,'data'); smoke=model._model.forward('label').reshape(-1)
    assert smoke.shape==(7,) and np.isfinite(smoke).all()
    manifest=pd.read_csv(a.q1/'manifest.csv').sort_values('sample_id')
    if a.limit: manifest=manifest.iloc[:a.limit]
    state=dict(stage='extracting',completed=0,total=len(manifest),job_id=os.getenv('SLURM_JOB_ID'),classes=CLASSES,model_sha256=sha(model_path),inference_backend='OpenCV DNN CPU',opencv=cv2.__version__)
    def progress(**kw):
        state.update(kw); state['updated_at']=datetime.datetime.now().astimezone().isoformat(); write_json(a.out/'progress.json',state); print(json.dumps(state,ensure_ascii=False),flush=True)
    progress()
    try:
        for ix,row in enumerate(manifest.to_dict('records')):
            sid=row['sample_id']; source=a.data_root/row['relative_path']; original=a.q1/'features'/f'{sid}.npz'; dest=a.out/'features'/f'{sid}.npz'; meta=dest.with_suffix('.json')
            fingerprint=dict(source_sha256=sha(source),original_features_sha256=sha(original),model_sha256=state['model_sha256'],extractor_sha256=sha(__file__),vendor_sha256=sha(ROOT/'vendor/facial_fer_model.py'))
            assert fingerprint['source_sha256']==row['source_sha256']
            if dest.exists() and meta.exists() and json.loads(meta.read_text()).get('fingerprint')==fingerprint:
                with np.load(dest) as cached: assert np.isfinite(cached['scores']).all()
                progress(completed=ix+1,last_sample=sid,resumed=True); continue
            with np.load(original,allow_pickle=False) as z:
                indices=z['video_frame_indices']; times=z['vision_times']; valid=z['raw_vision_valid'].astype(bool); geometry=z['raw_vision'].copy()
            align=json.loads((a.q1/'alignment'/f'{sid}.json').read_text()); origin=align['media']['media_origin_s']
            scores=np.zeros((len(times),7),np.float32); seen=np.zeros(len(times),bool); lookup={int(f):i for i,f in enumerate(indices)}
            with av.open(str(source)) as c:
                for fi,f in enumerate(c.decode(video=0)):
                    if fi not in lookup: continue
                    j=lookup[fi]; assert abs(float(f.pts*f.time_base)-origin-times[j])<1e-6; seen[j]=True
                    if not valid[j]: continue
                    frame=cv2.resize(f.to_ndarray(format='bgr24'),(320,320))
                    bbox=np.r_[geometry[j,:4]*320,geometry[j,5:]*320]
                    inp=model._preprocess(frame,bbox); model._model.setInput(inp,'data'); output=model._model.forward('label').reshape(-1)
                    assert output.shape==(7,) and np.isfinite(output).all()
                    scores[j]=output
            assert seen.all(),sid
            # Save raw model scores, without claiming they are calibrated probabilities.
            with dest.with_suffix('.tmp').open('wb') as out:
                np.savez_compressed(out,scores=scores,valid=valid,times=times,video_frame_indices=indices,classes=np.array(CLASSES))
            dest.with_suffix('.tmp').replace(dest)
            write_json(meta,dict(sample_id=sid,fingerprint=fingerprint,frames=len(times),valid_frames=int(valid.sum()),output_sha256=sha(dest),score_semantics='raw seven-class pretrained-model scores; not human emotion labels'))
            progress(completed=ix+1,last_sample=sid,resumed=False)
        progress(stage='extraction_complete')
    except BaseException as exc:
        progress(stage='failed',error=repr(exc),traceback=traceback.format_exc()); raise
if __name__=='__main__': main()
