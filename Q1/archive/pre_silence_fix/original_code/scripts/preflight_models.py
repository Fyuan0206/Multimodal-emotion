"""检查预训练文件完整性、全文分词长度及三个模型的最小前向计算。"""
import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from transformers import BertTokenizerFast, BertModel, Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--out',type=Path,default=ROOT/'outputs')
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from artifact_contract import verify_models
verify_models(ROOT / 'models', json.loads((ROOT / 'models/manifest.json').read_text()))
DATA = args.data_root
torch.set_num_threads(4)
start = time.monotonic()
rows = pd.read_excel(DATA / 'label-100.xlsx')
tok = BertTokenizerFast.from_pretrained(ROOT / 'models/bert', local_files_only=True)
lengths = [len(tok(str(t), truncation=False)['input_ids']) for t in rows.text]
bert = BertModel.from_pretrained(ROOT / 'models/bert', local_files_only=True).eval()
with torch.inference_mode():
    bshape = list(bert(**tok(str(rows.iloc[0].text), return_tensors='pt')).last_hidden_state.shape)
del bert
processor = Wav2Vec2Processor.from_pretrained(ROOT / 'models/ctc', local_files_only=True)
ctc = Wav2Vec2ForCTC.from_pretrained(ROOT / 'models/ctc', local_files_only=True).eval()
with torch.inference_mode():
    cshape = list(ctc(**processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors='pt')).logits.shape)
det = cv2.FaceDetectorYN.create(str(ROOT / 'models/yunet.onnx'), '', (320, 320))
_, faces = det.detect(np.zeros((320, 320, 3), dtype=np.uint8))
result = {'all_imports_ok': True, 'bert_shape': bshape, 'ctc_silence_shape': cshape,
          'yunet_blank_frame_has_no_faces': faces is None, 'transcript_token_min': min(lengths),
          'transcript_token_max': max(lengths), 'transcripts_above_50_tokens': sum(x > 50 for x in lengths),
          'planned_uniform_length': max(50, max(lengths)), 'seconds': time.monotonic() - start}
(args.out / 'model_preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))