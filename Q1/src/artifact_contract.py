"""校验模型与输出文件的完整性，统一流水线汇总字段，防止复用缺失或损坏的旧结果。"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    """计算文件内容的校验值，避免仅按文件名判断是否可复用。"""
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def verify_models(root, manifest):
    """逐项核对本地模型文件与固定清单中的大小和校验值。"""
    for entry in manifest:
        path = Path(root) / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError(f'model_integrity_failed: {entry["path"]}')


def reusable_record(out, sid, source_fingerprint):
    """仅当来源一致、状态成功且两个输出内容完整时允许断点复用。"""
    out = Path(out)
    try:
        record = json.loads((out / 'records' / f'{sid}.json').read_text())
        if record['sample_id'] != sid or record['status'] != 'extracted_unreviewed':
            return False
        if record.get('source_fingerprint') != source_fingerprint:
            return False
        for folder, suffix in [('features', '.npz'), ('alignment', '.json')]:
            path = out / folder / f'{sid}{suffix}'
            if not path.is_file() or file_hash(path) != record['artifact_sha256'][folder]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summarize_arrays(record, document, arrays):
    """生成固定的三模态汇总字段，分别说明来源时长与有效时长。"""
    media = document['media']
    rows = []
    for modality, dim in [('text', 768), ('audio', 27), ('vision', 15)]:
        if modality == 'text':
            duration = sum(w['end'] - w['start'] for w in document['words'] if w['aligned'])
            source_duration = media['audio_end_s'] - media['audio_start_s']
            native = int(arrays['text_length'])
            valid_length = int(arrays['text_mask'].sum())
            rate = 'source transcript tokens'
        else:
            stream = 'video' if modality == 'vision' else 'audio'
            source_duration = media[f'{stream}_end_s'] - media[f'{stream}_start_s']
            native = len(arrays[f'raw_{modality}'])
            valid_length = int(arrays[f'{modality}_valid_mask'].sum())
            rate = '20 Hz' if modality == 'audio' else '15 Hz nearest original PTS'
            if modality == 'audio':
                duration = 0.0 if media.get('audio_all_zero') else (media['audio_samples'] - media['audio_gap_samples']) / 16000
            else:
                t = arrays['vision_times']
                edges = np.r_[media['video_start_s'], (t[:-1] + t[1:]) / 2, media['video_end_s']]
                duration = float(np.diff(edges)[arrays['raw_vision_valid']].sum())
        rows.append(dict(sample_id=record['sample_id'], modality=modality,
                         source_duration_s=source_duration, effective_duration_s=duration,
                         feature_dim=dim, native_length=native, aligned_length=len(arrays['text']),
                         aligned_valid_length=valid_length, padding_count=int(arrays['padding_mask'].sum()),
                         native_granularity=rate, aligned_granularity='WordPiece with CTC word interval',
                         padding_rule='zero; padding_mask=true', status=record['status']))
    return rows
