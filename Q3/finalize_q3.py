"""Map Q3 feature evidence to text and automatic video times; summarize results."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import av
import cv2
import numpy as np
from tokenizers import BertWordPieceTokenizer


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_frame(video, original_index, path):
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == original_index:
                image = frame.to_ndarray(format="bgr24")
                if not cv2.imwrite(str(path), image):
                    raise OSError(path)
                return
    raise ValueError(f"Frame index {original_index} unavailable: {video}")


def map_evidence(row, sample, encoded, aligned, video, frames_dir):
    start, end = int(row["feature_start"]), int(row["feature_end_exclusive"])
    ids = np.asarray(sample["text_bert"])[0]
    attention = np.asarray(sample["text_bert"])[1]
    if any(not attention[j] or ids[j] in (0, 101, 102) for j in range(start, end)):
        raise ValueError(f"Evidence enters a special/padded position: {row['evidence_id']}")
    offsets = encoded.offsets[start:end]
    if len(offsets) != end - start or any(e <= s for s, e in offsets):
        raise ValueError(f"No source text offsets: {row['evidence_id']}")
    char_start, char_end = min(s for s, _ in offsets), max(e for _, e in offsets)
    row.update(token_start=start, token_end_exclusive=end,
               char_start=char_start, char_end_exclusive=char_end,
               text_excerpt=str(sample["raw_text"])[char_start:char_end],
               mapping_status="text_exact_time_pending", human_verified=False,
               ctc_min_word_score="", frame_path="",
               punctuation_only_words_skipped=0)
    if aligned.get("status") != "candidate_auto":
        return row
    words = [w for w in aligned["words"]
             if max(0, min(char_end, w["char_end"]) - max(char_start, w["char_start"])) > 0]
    # Orthographic separators such as '-' have no spoken interval to align.
    spoken = [w for w in words if w.get("normalized")]
    row["punctuation_only_words_skipped"] = len(words) - len(spoken)
    if not spoken or not all(w.get("aligned") for w in spoken):
        row["mapping_status"] = "text_exact_time_unresolved"
        return row
    begin = min(w["start"] for w in spoken)
    finish = max(w["end"] for w in spoken)
    duration = float(aligned["media"]["container_duration_s"])
    if not 0 <= begin < finish <= duration + .25:
        row["mapping_status"] = "text_exact_time_out_of_bounds"
        return row
    row["start_sec"], row["end_sec"] = round(begin, 6), round(finish, 6)
    row["ctc_min_word_score"] = min(float(w["score"]) for w in spoken)
    row["mapping_status"] = "text_exact_ctc_time_candidate"
    pts = np.asarray(aligned["video_frame_pts"], dtype=float)
    if len(pts):
        index = int(np.argmin(abs(pts - (begin + finish) / 2)))
        row["video_frame_pts"] = round(float(pts[index]), 6)
        if row["modality"] == "vision":
            name = row["evidence_id"].replace(":", "_") + ".png"
            draw_frame(video, int(aligned["video_frame_source_indices"][index]),
                       frames_dir / name)
            row["frame_path"] = f"frames/{name}"
    return row


def bootstrap_gap(rows, iterations=2000):
    included = [r for r in rows if r["primary_deletion_drop"] != ""
                and r["random_mean_drop"] != ""]
    grouped = {}
    for row in included:
        key = row["id"].rsplit("$_$", 1)[0]
        grouped.setdefault(key, []).append(float(row["primary_deletion_drop"])
                                        - float(row["random_mean_drop"]))
    rng = np.random.default_rng(6205)
    keys = sorted(grouped)
    replicates = []
    for _ in range(iterations):
        picked = rng.choice(keys, len(keys), replace=True)
        replicates.append(np.mean([value for key in picked for value in grouped[key]]))
    raw = [value for values in grouped.values() for value in values]
    return {"samples": len(raw), "source_groups": len(grouped),
            "mean_selected_minus_random_drop": float(np.mean(raw)),
            "median_selected_minus_random_drop": float(np.median(raw)),
            "positive_sample_fraction": float(np.mean(np.asarray(raw) > 0)),
            "group_bootstrap_iterations": iterations,
            "group_bootstrap_95pct": np.quantile(replicates, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    args = parser.parse_args()
    run, aligned_root = args.run, args.aligned
    alignment = json.loads((run / "alignment" / "candidate_alignment.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in alignment["samples"]}
    tokenizer = BertWordPieceTokenizer(vocab=str(args.vocab), lowercase=True)
    encoded, samples, manifest = {}, {}, []
    for path in sorted(aligned_root.glob("*.pkl")):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        sample_id = str(sample["id"])
        tokenization = tokenizer.encode(str(sample["raw_text"]))
        original = np.asarray(sample["text_bert"])
        active = np.flatnonzero(original[1])
        # 50-position truncation keeps [SEP] at index 49.
        expected = tokenization.ids[:min(len(tokenization.ids), 50)]
        if len(tokenization.ids) > 50:
            expected = tokenization.ids[:49] + [102]
        if len(active) != len(expected) or not np.array_equal(original[0, active], expected):
            raise ValueError(f"Token IDs do not map to raw text: {sample_id}")
        continuation = [j for j in range(1, len(active))
                        if j < len(tokenization.tokens) and tokenization.tokens[j].startswith("##")]
        av_equal = all(np.array_equal(sample["audio"][j], sample["audio"][j - 1])
                       and np.array_equal(sample["vision"][j], sample["vision"][j - 1])
                       for j in continuation)
        encoded[sample_id], samples[sample_id] = tokenization, sample
        manifest.append({"id": sample_id, "raw_token_ids_match": True,
                         "truncated": len(tokenization.ids) > 50,
                         "wordpiece_continuations": len(continuation),
                         "same_av_on_wordpiece_continuations": av_equal})
    explanation = run / "explanation" / "attachment4"
    evidence = read_csv(explanation / "evidence_feature_only.csv")
    predictions = {r["id"]: r for r in read_csv(run / "attachment4_predictions.csv")}
    contribution = {r["id"]: r for r in read_csv(explanation / "explanations.csv")}
    frames = explanation / "frames"
    frames.mkdir(exist_ok=True)
    mapped = []
    for row in evidence:
        sample_id = row["id"]
        video = aligned_root / "videos" / f"{sample_id}.mp4"
        mapped.append(map_evidence(row, samples[sample_id], encoded[sample_id],
                                   by_id[sample_id], video, frames))
    write_csv(explanation / "attachment4_evidence.csv", mapped)
    cards = explanation / "cards"
    cards.mkdir(exist_ok=True)
    merged = []
    for sample_id in sorted(samples):
        pred, contrib = predictions[sample_id], contribution[sample_id]
        own = [r for r in mapped if r["id"] == sample_id]
        merged.append({**pred, "primary_modality": contrib["primary_modality"],
                       "top_support_modality": contrib["top_support_modality"],
                       **{f"phi_{m}": contrib[f"phi_{m}"] for m in ("text", "audio", "vision")},
                       **{f"weight_{m}": contrib[f"weight_{m}"] for m in ("text", "audio", "vision")},
                       "evidence_ids": ";".join(r["evidence_id"] for r in own),
                       "evidence_status": ("all_auto_candidates" if all(r["mapping_status"] ==
                                           "text_exact_ctc_time_candidate" for r in own)
                                           else "some_unresolved"),
                       "human_verified": False})
        lines = [f"# Q3解释卡 {sample_id}", "", f"原文：{samples[sample_id]['raw_text']}", "",
                 f"预测：{pred['polarity']}，强度 {float(pred['intensity']):.4f}；未阈值化强度 {float(pred['raw_intensity']):.4f}",
                 f"主要模态：{contrib['primary_modality']}；主要支持模态：{contrib['top_support_modality']}",
                 "", "| 模态 | 有符号贡献 | 绝对贡献占比 |", "|---|---:|---:|"]
        for mode in ("text", "audio", "vision"):
            lines.append(f"| {mode} | {float(contrib['phi_'+mode]):.4f} | {float(contrib['weight_'+mode]):.3f} |")
        lines.extend(["", "| 模态 | 文本片段 | 候选时间(s) | 证据得分 | 定位状态 |",
                      "|---|---|---|---:|---|"])
        for item in own:
            span = (f"{item['start_sec']}–{item['end_sec']}" if item["start_sec"] else "未解出")
            excerpt = item["text_excerpt"].replace("|", "\\|")
            lines.append(f"| {item['modality']} | {excerpt} | {span} | "
                         f"{float(item['signed_score']):.4f} | {item['mapping_status']} |")
        lines.extend(["", "时间由Q1自动CTC强制对齐产生，尚无人工边界真值。模态贡献是模型特征替换实验结果。",
                      "原A/V特征与词元位置的语义关系仍以题目对齐说明及WordPiece重复行检查为依据。", ""])
        (cards / f"{sample_id}.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(explanation / "attachment4_predictions_explanations.csv", merged)
    diagnostic = read_csv(run / "explanation" / "diagnostic" / "explanations.csv")
    summary = {"attachment4_samples": len(merged), "evidence_rows": len(mapped),
               "token_mapping": manifest,
               "automatic_time_candidate_rows": sum(r["mapping_status"] ==
                                                    "text_exact_ctc_time_candidate" for r in mapped),
               "manual_timing_reference_rows": 0,
               "validation_deletion_vs_random": bootstrap_gap(diagnostic)}
    (run / "explanation" / "final_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "token_mapping"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
