"""问题一主流水线：清点原始视频，提取三模态特征，执行词级对齐并保存可追溯结果。
输入为附件1与本地预训练模型；输出为特征、逐词映射、清单和处理日志。"""
from __future__ import annotations
import os
import argparse, hashlib, importlib.metadata, json, platform, re, sys, time, unicodedata
from pathlib import Path
import av
import cv2
import librosa
import numpy as np
import pandas as pd
import parselmouth
import torch
from transformers import BertTokenizerFast, BertModel, Wav2Vec2Processor, Wav2Vec2ForCTC
from ctc_align import ctc_viterbi
from artifact_contract import verify_models, reusable_record, summarize_arrays, file_hash

CONFIG = dict(sample_rate=16000, n_fft=400, hop_length=800, n_mels=40, n_mfcc=25,
              pitch_floor=50, pitch_ceiling=500, visual_fps=15, detector_size=320,
              detector_threshold=.75, schema_version='q1-1.2')


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    return file_hash(path)


def pool_interval(features, times, valid, start, end):
    """左闭右开区间池化；窗口为空时取时间中点最近的原生帧并显式报告回退。"""
    idx = np.flatnonzero((times >= start) & (times < end))
    fallback = len(idx) == 0
    distance = 0.0
    if fallback and len(times):
        idx = np.array([np.argmin(np.abs(times - (start + end) / 2))])
        distance = float(abs(times[idx[0]] - (start + end) / 2))
    used = idx[valid[idx]]
    value = features[used].mean(0) if len(used) else np.zeros(features.shape[1], np.float32)
    return value.astype(np.float32), len(used) > 0, idx.tolist(), fallback, distance


def token_word_map(offsets, words):
    """按字符区间重叠把每个 WordPiece 映射回原文中以空白分隔的词。"""
    result = []
    for s, e in offsets:
        if e <= s:
            result.append(-1); continue
        overlap = [max(0, min(int(e), w['char_end']) - max(int(s), w['char_start'])) for w in words]
        result.append(int(np.argmax(overlap)) if overlap and max(overlap) else -1)
    return np.asarray(result, dtype=np.int32)


def source_words(text):
    words = []
    for match in re.finditer(r'\S+', text):
        original = match.group()
        normalized = unicodedata.normalize('NFKD', original).encode('ascii', 'ignore').decode().upper()
        unsupported = ''.join(c for c in normalized if c.isdigit())
        chars = ''.join(c for c in normalized if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
        words.append(dict(text=original, char_start=match.start(), char_end=match.end(),
                          normalized=chars, unsupported=unsupported,
                          aligned=False, start=None, end=None, score=None))
    return words


def decode_media(path):
    """保留解码器真实时间戳；显式填补音频空隙，绝不压缩或拉伸时间轴。"""
    with av.open(str(path)) as c:
        starts = [float(s.start_time * s.time_base) for s in c.streams if s.start_time is not None]
        origin = min(starts) if starts else 0.0
        container_duration = c.duration / av.time_base
    vt, frames, vend = [], [], 0.0
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 0
        for f in c.decode(stream):
            if f.pts is None:
                raise ValueError('video_missing_pts')
            t = float(f.pts * f.time_base) - origin
            if vt and t < vt[-1]: raise ValueError('video_pts_not_monotonic')
            vt.append(t)
            frames.append(cv2.resize(f.to_ndarray(format='bgr24'), (320, 320)))
            frame_duration = float(f.duration * f.time_base) if f.duration else (1 / rate if rate else 0)
            vend = max(vend, t + frame_duration)
    if not vt: raise ValueError('no_video_frames')
    times = np.asarray(vt)
    targets = np.arange(times[0], times[-1] + 1e-8, 1 / CONFIG['visual_fps'])
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(abs(times[right] - targets) < abs(times[left] - targets), right, left)
    selected = np.unique(selected)
    selected_frames = [frames[x] for x in selected]
    del frames
    chunks = []
    with av.open(str(path)) as c:
        stream = c.streams.audio[0]
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        for f in c.decode(stream):
            for r in resampler.resample(f):
                if r.pts is None: raise ValueError('audio_missing_pts')
                chunks.append((float(r.pts * r.time_base) - origin, r.to_ndarray().reshape(-1)))
        for r in resampler.resample(None):
            if r.pts is None: raise ValueError('resampler_missing_pts')
            chunks.append((float(r.pts * r.time_base) - origin, r.to_ndarray().reshape(-1)))
    if not chunks: raise ValueError('no_audio_frames')
    a0 = chunks[0][0]
    n = max(round((t - a0) * 16000) + len(x) for t, x in chunks)
    wave = np.zeros(n, np.float32); observed = np.zeros(n, bool)
    for t, x in chunks:
        offset = round((t - a0) * 16000)
        if offset < 0: raise ValueError('audio_pts_before_origin')
        wave[offset:offset + len(x)] = x
        observed[offset:offset + len(x)] = True
    info = dict(media_origin_s=origin, container_duration_s=container_duration,
                video_start_s=float(times[0]), video_end_s=vend, video_frame_count=len(times),
                video_average_fps=rate, audio_start_s=a0, audio_end_s=a0 + n / 16000,
                audio_samples=n, audio_gap_samples=int((~observed).sum()))
    return wave, observed, a0, times[selected], selected, selected_frames, info


def audio_signal_status(wave):
    """Reject digital silence, without treating nonzero signal as proof of speech."""
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('empty_or_nonfinite_audio')
    nonzero = int(np.count_nonzero(wave))
    return dict(audio_all_zero=nonzero == 0, audio_nonzero_samples=nonzero,
                audio_peak=float(np.max(np.abs(wave))),
                audio_rms=float(np.sqrt(np.mean(np.asarray(wave, dtype=np.float64)**2))))


def acoustic_features(wave, observed, t0):
    mfcc = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc=25, n_fft=400, hop_length=800,
                               n_mels=40, center=True).T.astype(np.float32)
    relative = np.arange(len(mfcc)) * .05
    keep = relative < len(wave) / 16000
    relative, mfcc = relative[keep], mfcc[keep]
    pitch = parselmouth.Sound(wave, sampling_frequency=16000).to_pitch_ac(
        time_step=.05, pitch_floor=50, pitch_ceiling=500)
    pt = pitch.xs(); pf = pitch.selected_array['frequency']
    r = np.searchsorted(pt, relative).clip(0, len(pt) - 1); l = np.maximum(0, r - 1)
    nearest = np.where(abs(pt[r] - relative) < abs(pt[l] - relative), r, l)
    f0 = pf[nearest].astype(np.float32)
    features = np.column_stack([f0, (f0 > 0).astype(np.float32), mfcc]).astype(np.float32)
    valid = np.array([observed[max(0, round(t * 16000) - 200):min(len(wave), round(t * 16000) + 200)].all()
                      for t in relative], bool)
    if audio_signal_status(wave)['audio_all_zero']:
        features[:] = 0
        valid[:] = False
    return features, relative + t0, valid, pt[nearest] + t0


def visual_features(frames, detector):
    result = np.zeros((len(frames), 15), np.float32); valid = np.zeros(len(frames), bool)
    counts = np.zeros(len(frames), np.int32)
    for j, frame in enumerate(frames):
        _, faces = detector.detect(frame)
        if faces is None: continue
        counts[j] = len(faces)
        face = max(faces, key=lambda f: (f[2] * f[3], f[14]))
        result[j] = np.concatenate((face[:4] / 320, face[14:15], face[4:14] / 320))
        valid[j] = True
    return result, valid, counts


class Extractor:
    def __init__(self, models, tokenizer, length, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer, self.length = tokenizer, length
        self.bert = BertModel.from_pretrained(models / 'bert', local_files_only=True).eval().to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(models / 'ctc', local_files_only=True)
        self.ctc = Wav2Vec2ForCTC.from_pretrained(models / 'ctc', local_files_only=True).eval().to(self.device)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.detector = cv2.FaceDetectorYN.create(str(models / 'yunet.onnx'), '', (320, 320),
                                                 CONFIG['detector_threshold'], .3, 5000)
        jump, receptive = 1, 1
        for kernel, stride in zip(self.ctc.config.conv_kernel, self.ctc.config.conv_stride):
            receptive += (kernel - 1) * jump; jump *= stride
        self.stride, self.receptive = jump, receptive

    def align(self, wave, t0, text):
        words = source_words(text)
        if audio_signal_status(wave)['audio_all_zero']:
            for w in words:
                w['reason'] = 'audio_all_zero'
            return words
        target, wi_chars = [], {}
        for wi, w in enumerate(words):
            if not w['normalized'] or w['unsupported']:
                w['reason'] = 'unsupported_characters_or_no_letters'; continue
            if target: target.append(self.vocab['|'])
            start = len(target)
            target.extend(self.vocab[c] for c in w['normalized'])
            wi_chars[wi] = (start, len(target))
        if not target: return words
        inp = self.processor(wave, sampling_rate=16000, return_tensors='pt')
        with torch.inference_mode():
            logp = torch.log_softmax(self.ctc(**{k: v.to(self.device) for k, v in inp.items()}).logits[0], -1).cpu().numpy()
        path = ctc_viterbi(logp, target, blank_id=self.ctc.config.pad_token_id)
        for wi, (s, e) in wi_chars.items():
            indices = np.flatnonzero((path >= 2 * s + 1) & (path <= 2 * (e - 1) + 1))
            emission_indices = np.concatenate([np.flatnonzero(path == 2 * k + 1) for k in range(s, e)])
            values = [logp[t, target[(path[t] - 1) // 2]] for t in emission_indices]
            # 卷积感受野中心，用半个步长单元把帧索引换算成秒并夹在有效区间内。
            start_s = (indices[0] * self.stride + (self.receptive - self.stride) / 2) / 16000 + t0
            end_s = ((indices[-1] + 1) * self.stride + (self.receptive - self.stride) / 2) / 16000 + t0
            words[wi].update(aligned=True, start=float(max(t0, start_s)),
                             end=float(min(t0 + len(wave) / 16000, end_s)),
                             score=float(np.exp(np.mean(values))))
        return words

    def extract(self, path, text):
        wave, observed, a0, vt, frame_idx, frames, media = decode_media(path)
        media.update(audio_signal_status(wave))
        audio, at, avmask, pitch_times = acoustic_features(wave, observed, a0)
        vision, vmask, face_counts = visual_features(frames, self.detector)
        words = self.align(wave, a0, text)
        enc = self.tokenizer(text, padding='max_length', max_length=self.length, truncation=False,
                             return_offsets_mapping=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')[0].numpy()
        with torch.inference_mode():
            text_features = self.bert(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0].cpu().numpy().copy()
        text_mask = enc['attention_mask'][0].numpy().astype(bool)
        text_features[~text_mask] = 0
        word_ids = token_word_map(offsets, words)
        aligned_a = np.zeros((self.length, 27), np.float32)
        aligned_v = np.zeros((self.length, 15), np.float32)
        am, vm, align_mask = [np.zeros(self.length, bool) for _ in range(3)]
        token_times = np.full((self.length, 2), -1, np.float64)
        for wi, w in enumerate(words):
            tok = np.flatnonzero(word_ids == wi); w['token_indices'] = tok.tolist()
            if not w['aligned']: continue
            pa, va, ai, af, ad = pool_interval(audio, at, avmask, w['start'], w['end'])
            pv, vv, vi, vf, vd = pool_interval(vision, vt, vmask, w['start'], w['end'])
            aligned_a[tok], aligned_v[tok] = pa, pv
            am[tok], vm[tok], align_mask[tok] = va, vv, True
            token_times[tok] = [w['start'], w['end']]
            w.update(audio_indices=ai, vision_indices=vi, audio_fallback=af, vision_fallback=vf,
                     audio_fallback_distance_s=ad, vision_fallback_distance_s=vd)
        arrays = dict(text=text_features.astype(np.float32), audio=aligned_a, vision=aligned_v,
                      raw_audio=audio, raw_vision=vision, audio_times=at, vision_times=vt,
                      raw_audio_valid=avmask, raw_vision_valid=vmask, pitch_source_times=pitch_times,
                      video_frame_indices=frame_idx, face_counts=face_counts,
                      text_mask=text_mask, padding_mask=~text_mask, content_mask=word_ids >= 0,
                      audio_valid_mask=am, vision_valid_mask=vm, alignment_mask=align_mask,
                      token_ids=enc['input_ids'][0].numpy(), token_offsets=offsets, token_word_ids=word_ids,
                      token_times=token_times, text_length=np.array(text_mask.sum(), dtype=np.int32))
        for key, val in arrays.items():
            if val.dtype.kind in 'fc' and not np.isfinite(val).all(): raise ValueError(f'nonfinite_{key}')
        assert np.all(aligned_a[~am] == 0) and np.all(aligned_v[~vm] == 0)
        assert np.all(text_features[~text_mask] == 0)
        return arrays, words, media


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--pilot', action='store_true', help='选取时长最短、中位和最长3条')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = dict(device=args.device, machine=platform.machine(), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                   deterministic=True, tf32=False)
    torch.set_num_threads(4); torch.manual_seed(2026); np.random.seed(2026)
    for d in ['features', 'alignment', 'records']: (args.out / d).mkdir(parents=True, exist_ok=True)
    labels = pd.read_excel(args.data_root / 'label-100.xlsx', dtype={'video_id': str, 'clip_id': str})
    labels['sample_id'] = labels.video_id + '__' + labels.clip_id
    if len(labels) != 100 or labels.sample_id.nunique() != 100: raise ValueError('expected_100_unique_samples')
    expected = {f'{v}/{c}.mp4' for v, c in zip(labels.video_id, labels.clip_id)}
    actual = {str(p.relative_to(args.data_root)) for p in args.data_root.glob('*/*.mp4')}
    if expected != actual: raise ValueError(f'input_coverage_mismatch missing={expected-actual} extra={actual-expected}')
    if labels.text.isna().any() or labels.text.str.strip().eq('').any(): raise ValueError('empty_text')
    tokenizer = BertTokenizerFast.from_pretrained(args.models / 'bert', local_files_only=True)
    lengths = [len(tokenizer(str(t), truncation=False)['input_ids']) for t in labels.text]
    length = max(50, max(lengths))
    if length > 512: raise ValueError('BERT_512_overflow_requires_windowed_encoding')
    model_manifest = json.loads((args.models / 'manifest.json').read_text())
    verify_models(args.models, model_manifest)
    code_hashes = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    versions = {x: importlib.metadata.version(x) for x in ['numpy', 'av', 'torch', 'transformers', 'librosa', 'praat-parselmouth', 'opencv-python-headless', 'pandas']}
    fingerprint = hashlib.sha256(json.dumps([CONFIG, length, model_manifest, code_hashes, versions, runtime], sort_keys=True).encode()).hexdigest()
    rows = []
    for (_, row), nt in zip(labels.iterrows(), lengths):
        path = args.data_root / row.video_id / f'{row.clip_id}.mp4'
        with av.open(str(path)) as c:
            duration = c.duration / av.time_base
        rows.append(dict(sample_id=row.sample_id, video_id=row.video_id, clip_id=row.clip_id,
                         relative_path=f'{row.video_id}/{row.clip_id}.mp4', source_sha256=sha(path),
                         text_sha256=hashlib.sha256(str(row.text).encode()).hexdigest(),
                         container_duration_s=duration, token_count=nt, status='audited'))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / 'manifest.csv', index=False)
    env = dict(python=platform.python_version(), platform=platform.platform(), cpu_threads=4, seed=2026,
               runtime=runtime, config=CONFIG, uniform_length=length, config_fingerprint=fingerprint,
               packages={x: importlib.metadata.version(x) for x in ['numpy', 'av', 'torch', 'transformers',
                         'librosa', 'praat-parselmouth', 'opencv-python-headless', 'pandas', 'matplotlib']},
               ffmpeg_libraries={k: list(v) for k, v in av.library_versions.items()},
               model_files=model_manifest, code_hashes=code_hashes, command=sys.argv)
    write_json(args.out / 'environment.json', env)
    print(f'Audit: 100 unique samples; uniform token length={length}', flush=True)
    if args.audit_only: return
    selected = labels
    if args.pilot:
        ordered = manifest.sort_values('container_duration_s').sample_id.tolist()
        selected = selected[selected.sample_id.isin([ordered[0], ordered[len(ordered)//2], ordered[-1]])]
    if args.ids:
        unknown = set(args.ids) - set(labels.sample_id)
        if unknown: raise ValueError(f'unknown_sample_ids: {sorted(unknown)}')
        selected = selected[selected.sample_id.isin(args.ids)]
    if args.limit is not None:
        if args.limit <= 0: raise ValueError('limit_must_be_positive')
        selected = selected.head(args.limit)
    if selected.empty: raise ValueError('empty_sample_selection')
    extractor = Extractor(args.models, tokenizer, length, args.device)
    errors = []
    for number, (_, row) in enumerate(selected.iterrows(), 1):
        sid = row.sample_id; meta = rows[labels.index[labels.sample_id == sid][0]]
        record_path = args.out / 'records' / f'{sid}.json'
        source_fp = hashlib.sha256((fingerprint + meta['source_sha256'] + meta['text_sha256']).encode()).hexdigest()
        if args.resume and reusable_record(args.out, sid, source_fp):
            print(f'[{number}/{len(selected)}] {sid} resumed', flush=True); continue
        started = time.monotonic()
        try:
            arrays, words, media = extractor.extract(args.data_root / meta['relative_path'], str(row.text))
            np.savez_compressed(args.out / 'features' / f'{sid}.npz', **arrays)
            write_json(args.out / 'alignment' / f'{sid}.json', dict(sample_id=sid, text=str(row.text),
                       time_origin='media_start_seconds', score_definition='CTC emission geometric mean, not calibrated confidence',
                       words=words, media=media))
            matched = sum(w['aligned'] for w in words)
            record = dict(sample_id=sid, source_fingerprint=source_fp, status='extracted_unreviewed',
                          seconds=time.monotonic() - started, word_count=len(words), aligned_words=matched,
                          word_coverage=matched / max(1, len(words)),
                          low_score_words=sum(w['score'] is not None and w['score'] < .2 for w in words),
                          audio_fallback_words=sum(w.get('audio_fallback', False) for w in words),
                          vision_fallback_words=sum(w.get('vision_fallback', False) for w in words),
                          face_detection_rate=float(arrays['raw_vision_valid'].mean()),
                          uniform_length=length, token_count=int(arrays['text_length']), media=media,
                          npz_bytes=(args.out / 'features' / f'{sid}.npz').stat().st_size,
                          human_boundary_validation='pending',
                          artifact_sha256={folder: sha(args.out / folder / f'{sid}{suffix}')
                              for folder, suffix in [('features', '.npz'), ('alignment', '.json')]})
            write_json(record_path, record)
            with (args.out / 'run.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            print(f'[{number}/{len(selected)}] {sid}: {record["seconds"]:.1f}s, words {matched}/{len(words)}, face {record["face_detection_rate"]:.1%}', flush=True)
        except Exception as exc:
            import traceback
            failure = dict(sample_id=sid, status='failed', error=str(exc), traceback=traceback.format_exc())
            errors.append(failure)
            with (args.out / 'run.jsonl').open('a') as f: f.write(json.dumps(failure) + '\n')
            print(f'FAILED {sid}: {exc}', flush=True)
    records = []
    failed_ids = {r['sample_id'] for r in errors}
    for meta in rows:
        sid = meta['sample_id']
        source_fp = hashlib.sha256((fingerprint + meta['source_sha256'] + meta['text_sha256']).encode()).hexdigest()
        if sid not in failed_ids and reusable_record(args.out, sid, source_fp):
            records.append(json.loads((args.out / 'records' / f'{sid}.json').read_text()))
    table = []
    for record in records:
        sid = record['sample_id']
        with np.load(args.out / 'features' / f'{sid}.npz', allow_pickle=False) as d:
            document = json.loads((args.out / 'alignment' / f'{sid}.json').read_text())
            table.extend(summarize_arrays(record, document, d))
        manifest.loc[manifest.sample_id == sid, 'status'] = record['status']
    pd.DataFrame(table).to_csv(args.out / 'summary.csv', index=False)
    manifest.to_csv(args.out / 'manifest.csv', index=False)
    quality = dict(input_samples=100, extracted_samples=len(records), summary_rows=len(table), failed_samples=errors,
                   uniform_length=length, human_validation='pending',
                   total_npz_bytes=sum(r['npz_bytes'] for r in records),
                   global_word_coverage=sum(r['aligned_words'] for r in records) / max(1, sum(r['word_count'] for r in records)))
    write_json(args.out / 'quality.json', quality)
    print(json.dumps(quality, ensure_ascii=False), flush=True)
    if errors: raise SystemExit(1)

if __name__ == '__main__': main()