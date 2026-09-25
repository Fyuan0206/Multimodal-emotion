"""Generate an independent ASR aid; never label its output as human review.

Run from this directory with: python run_asr_check.py
Requires faster-whisper and a cached Systran/faster-whisper-base model.
"""

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "asr_check.json"


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def main():
    clips = json.loads((ROOT / "素材来源.json").read_text(encoding="utf-8"))
    with (ROOT / "回听记录.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "word_index", "word", "review_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("回听记录.csv is empty or missing required columns")
    if len(rows) != 198 or len(clips) != 10:
        raise ValueError("Unexpected source row or clip count")

    model = WhisperModel(
        "Systran/faster-whisper-base", device="cpu", compute_type="int8",
        local_files_only=True,
    )
    results = []
    for clip in clips:
        audio_path = ROOT / clip["file"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        segments, info = model.transcribe(
            str(audio_path), language="en", beam_size=5, word_timestamps=True,
            vad_filter=False, condition_on_previous_text=False,
        )
        segments = list(segments)
        transcript = " ".join(segment.text.strip() for segment in segments)
        reference = tokens(clip["text"])
        recognized = tokens(transcript)
        similarity = SequenceMatcher(None, reference, recognized).ratio()
        result = {
            "sample_id": clip["sample_id"],
            "audio_file": clip["file"],
            "duration_s": clip["duration"],
            "reference_text": clip["text"],
            "asr_text": transcript,
            "token_sequence_similarity": round(similarity, 4),
            "asr_language_probability": round(info.language_probability, 4),
            "asr_words": [
                {
                    "word": word.word.strip(),
                    "start_s": word.start,
                    "end_s": word.end,
                    "probability": round(word.probability, 4),
                }
                for segment in segments for word in (segment.words or [])
            ],
        }
        results.append(result)
        print(f"{clip['sample_id']}: similarity={similarity:.3f}", flush=True)

    payload = {
        "method": "faster-whisper-base CPU int8 independent machine transcription",
        "human_review": False,
        "warning": "ASR words and timestamps are candidates only; they do not confirm audible words or human boundaries.",
        "clips": results,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
