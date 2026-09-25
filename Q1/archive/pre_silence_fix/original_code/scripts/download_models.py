"""下载固定版本的公开预训练模型，保存文件来源、大小与校验值；已有模型必须通过校验。"""
import sys
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from artifact_contract import file_hash

ROOT = Path(__file__).resolve().parents[1]
KNOWN_MANIFEST = ROOT / 'models/manifest.json'
KNOWN = {x['path']: x for x in json.loads(KNOWN_MANIFEST.read_text())} if KNOWN_MANIFEST.exists() else {}
MODELS = [
    ('bert', 'google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594',
     ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']),
    ('ctc', 'facebook/wav2vec2-base-960h', '22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
     ['config.json', 'model.safetensors', 'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']),
]


def download(item):
    relative, url = item
    dest = ROOT / 'models' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'Q1-Reproducible-Feature-Extraction/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        tmp.replace(dest)
    digest = file_hash(dest)
    expected = KNOWN.get(str(relative))
    if expected and (expected['sha256'] != digest or expected['bytes'] != dest.stat().st_size):
        raise ValueError(f'model_integrity_failed: {relative}')
    print(f'{relative}: {dest.stat().st_size} bytes', flush=True)
    return {'path': str(relative), 'url': url, 'bytes': dest.stat().st_size, 'sha256': digest}


if __name__ == '__main__':
    jobs = [(f'{name}/{filename}', f'https://huggingface.co/{repo}/resolve/{revision}/{filename}')
            for name, repo, revision, files in MODELS for filename in files]
    # 该revision在首次环境搭建时确定，此后保持固定不变。
    revision = '47534e27c9851bb1128ccc0102f1145e27f23f98'
    jobs.append(('yunet.onnx', f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{revision}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'))
    rows = list(ThreadPoolExecutor(max_workers=4).map(download, jobs))
    (ROOT / 'models' / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))