"""Read-only aligned feature inventory. Does not train or predict on Attachment 4."""
import argparse
import csv
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np


SHAPES = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35),
          "text_bert": (3, 50)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_sample(sample, sample_id):
    for key, shape in SHAPES.items():
        array = np.asarray(sample[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{sample_id}/{key}: shape or nonfinite values")
    bert = np.asarray(sample["text_bert"])
    if not np.issubdtype(bert.dtype, np.integer):
        raise ValueError(f"{sample_id}: text_bert must be integer")
    tokens, attention, segments = bert
    if not np.isin(attention, [0, 1]).all() or not np.isin(segments, [0, 1]).all():
        raise ValueError(f"{sample_id}: invalid attention/segment values")
    # UNK is retained as a content token; zero A/V rows are indicators, not proven missingness.
    content = (attention == 1) & ~np.isin(tokens, [0, 101, 102])
    if not content.any():
        raise ValueError(f"{sample_id}: no content positions")
    row = {"id": str(sample_id), "content_positions": int(content.sum()),
           "unk_content_tokens": int(((tokens == 100) & content).sum()),
           "raw_text_characters": len(str(sample["raw_text"])),
           "has_timestamp_field": any("time" in k.lower() for k in sample)}
    for mode in ("text", "audio", "vision"):
        zero = ~np.any(np.asarray(sample[mode]) != 0, axis=-1)
        row[f"{mode}_zero_content_rows"] = int((zero & content).sum())
        row[f"{mode}_all_content_zero"] = bool(zero[content].all())
        row[f"{mode}_nonzero_outside_content"] = int((~zero & ~content).sum())
    return row


def summarize(rows):
    return {"samples": len(rows),
            "content_length_min": min(r["content_positions"] for r in rows),
            "content_length_max": max(r["content_positions"] for r in rows),
            "unk_content_tokens": sum(r["unk_content_tokens"] for r in rows),
            "samples_with_timestamp_field": sum(r["has_timestamp_field"] for r in rows),
            "all_content_zero_samples": {
                m: sum(r[f"{m}_all_content_zero"] for r in rows)
                for m in ("text", "audio", "vision")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--attachment4-aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for source in (args.attachment2, args.attachment4_aligned):
        if args.output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Output must be outside source data")
    with args.attachment2.open("rb") as stream:
        data = pickle.load(stream)
    report = {"schema": "q3-initial-structural-audit-v1", "python": platform.python_version(),
              "numpy": np.__version__, "attachment2_sha256": sha256(args.attachment2),
              "scope": "train/valid and aligned Attachment 4 structure; no test metrics or predictions",
              "splits": {}, "limitations": [
                  "Video files matched by filename only; audiovisual decode and semantic matching pending.",
                  "Feature positions have not been verified against words or physical timestamps.",
                  "All-zero rows are observed facts, not definitive missing-modality labels."]}
    all_rows, ids, groups, texts, feature_hashes = [], {}, {}, {}, {}
    for split_name in ("train", "valid"):
        split = data[split_name]
        count = len(split["id"])
        for key, shape in SHAPES.items():
            if np.shape(split[key]) != (count, *shape):
                raise ValueError(f"{split_name}/{key}: unexpected shape")
        y = np.asarray(split["regression_labels"])
        c = np.asarray(split["classification_labels"])
        if y.shape != (count,) or c.shape != (count,) or not np.isfinite(y).all():
            raise ValueError(f"{split_name}: invalid labels")
        if not ((y >= -3) & (y <= 3)).all() or not np.array_equal(c, np.sign(y) + 1):
            raise ValueError(f"{split_name}: label range/sign mismatch")
        if len(split["raw_text"]) != count:
            raise ValueError(f"{split_name}: text count mismatch")
        rows = []
        ids[split_name], groups[split_name] = set(), set()
        texts[split_name], feature_hashes[split_name] = set(), set()
        for i, sample_id in enumerate(split["id"]):
            sample_id = str(sample_id)
            if sample_id in ids[split_name]:
                raise ValueError(f"Duplicate ID: {split_name}/{sample_id}")
            ids[split_name].add(sample_id)
            groups[split_name].add(sample_id.rsplit("$_$", 1)[0])
            sample = {k: split[k][i] for k in (*SHAPES, "raw_text")}
            row = inspect_sample(sample, sample_id)
            row["has_timestamp_field"] = any("time" in k.lower() for k in split)
            row["split"] = split_name
            rows.append(row)
            texts[split_name].add(" ".join(str(sample["raw_text"]).lower().split()))
            digest = hashlib.sha256()
            for key in SHAPES:
                digest.update(np.ascontiguousarray(sample[key]).tobytes())
            feature_hashes[split_name].add(digest.hexdigest())
        report["splits"][split_name] = summarize(rows)
        report["splits"][split_name]["class_counts"] = np.bincount(c.astype(int), minlength=3).tolist()
        report["splits"][split_name]["source_groups"] = len(groups[split_name])
        all_rows.extend(rows)
    report["train_valid_overlap"] = {
        "ids": len(ids["train"] & ids["valid"]),
        "source_groups": len(groups["train"] & groups["valid"]),
        "normalized_texts": len(texts["train"] & texts["valid"]),
        "exact_feature_payloads": len(feature_hashes["train"] & feature_hashes["valid"])}
    paths = sorted(args.attachment4_aligned.glob("*.pkl"))
    videos = sorted(args.attachment4_aligned.rglob("*.mp4"))
    if not paths or len({p.stem for p in videos}) != len(videos):
        raise ValueError("Empty feature set or ambiguous video stems")
    video_map = {p.stem: p for p in videos}
    a4_rows, manifest, a4_ids = [], [], set()
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        sample_id = str(sample["id"])
        if sample_id != path.stem or sample_id in a4_ids or path.stem not in video_map:
            raise ValueError(f"Attachment 4 ID/file mismatch: {path.name}")
        a4_ids.add(sample_id)
        row = inspect_sample(sample, sample_id)
        row["split"] = "attachment4"
        a4_rows.append(row)
        manifest.append({"id": sample_id, "feature_path": path.name,
                         "feature_sha256": sha256(path), "fields": sorted(sample),
                         "video_path": video_map[path.stem].relative_to(args.attachment4_aligned).as_posix(),
                         "video_sha256": sha256(video_map[path.stem]),
                         "shapes": {k: list(np.shape(sample[k])) for k in SHAPES}})
    if set(video_map) != a4_ids:
        raise ValueError("Unmatched Attachment 4 videos")
    report["splits"]["attachment4"] = summarize(a4_rows)
    report["attachment4_manifest"] = manifest
    all_rows.extend(a4_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "sample_inventory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({"splits": report["splits"], "overlap": report["train_valid_overlap"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
