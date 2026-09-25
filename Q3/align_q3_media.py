"""Candidate word times for Attachment 4 via the existing Q1 CTC pipeline."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Q1" / "src"))
from pipeline import Extractor, decode_media  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aligner = Extractor.__new__(Extractor)
    aligner.device = torch.device(args.device)
    aligner.processor = Wav2Vec2Processor.from_pretrained(args.models / "ctc", local_files_only=True)
    aligner.ctc = Wav2Vec2ForCTC.from_pretrained(args.models / "ctc", local_files_only=True).eval().to(aligner.device)
    aligner.vocab = aligner.processor.tokenizer.get_vocab()
    jump, receptive = 1, 1
    for kernel, stride in zip(aligner.ctc.config.conv_kernel, aligner.ctc.config.conv_stride):
        receptive += (kernel - 1) * jump
        jump *= stride
    aligner.stride, aligner.receptive = jump, receptive
    rows = []
    for path in sorted(args.aligned.glob("*.pkl")):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        video = args.aligned / "videos" / f"{path.stem}.mp4"
        row = {"id": path.stem, "video": str(video.relative_to(args.aligned)),
               "status": "pending"}
        try:
            wave, observed, start, times, indices, frames, media = decode_media(video)
            del observed, frames
            words = aligner.align(wave, start, str(sample["raw_text"]))
            row.update({"status": "candidate_auto", "media": media, "words": words,
                        "video_frame_pts": times.tolist(),
                        "video_frame_source_indices": indices.tolist(),
                        "aligned_words": sum(w.get("aligned", False) for w in words),
                        "total_words": len(words)})
            print(path.stem, row["aligned_words"], "/", len(words), flush=True)
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            print(path.stem, row["reason"], flush=True)
        rows.append(row)
    (args.output / "candidate_alignment.json").write_text(
        json.dumps({"method": "Q1 q1-1.2 wav2vec2 CTC forced alignment",
                    "review": "automatic candidates, no human timing reference",
                    "samples": rows}, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8")
    if len(rows) != 20:
        raise ValueError(f"Expected 20 samples, found {len(rows)}")


if __name__ == "__main__":
    main()
