"""Compare feature-row deletion with real WordPiece MASK re-encoding on valid."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel

from explain_q3 import margin
from run_q3 import Q3Model, tensor_view


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def encode(bert, tokens, attention, segments, batch=32):
    vectors = []
    with torch.inference_mode():
        for begin in range(0, len(tokens), batch):
            vectors.append(bert(input_ids=tokens[begin:begin + batch],
                                attention_mask=attention[begin:begin + batch],
                                token_type_ids=segments[begin:begin + batch])
                           .last_hidden_state)
    return torch.cat(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    selected = json.loads((run / "selection.json").read_text(encoding="utf-8"))["final_record"]
    checkpoint = torch.load(run / selected["checkpoint"], map_location=args.device,
                            weights_only=True)
    model = Q3Model(checkpoint["architecture"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    bert = BertModel.from_pretrained(args.models / "bert", local_files_only=True)
    bert = bert.to(args.device).eval()
    with args.attachment2.open("rb") as stream:
        valid = pickle.load(stream)["valid"]
    lookup = {str(sample_id): i for i, sample_id in enumerate(valid["id"])}
    explanations = csv_rows(run / "explanation" / "diagnostic" / "explanations.csv")
    evidence = csv_rows(run / "explanation" / "diagnostic" / "evidence_feature_only.csv")
    spans = {}
    for row in evidence:
        if row["modality"] == "text":
            spans.setdefault(row["id"], []).append((int(row["feature_start"]),
                                                       int(row["feature_end_exclusive"])))
    shapley = {}
    with (run / "explanation" / "diagnostic" / "shapley.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            shapley[item["id"]] = item
    chosen = [row for row in explanations if row["primary_modality"] == "text"
              and spans.get(row["id"])]
    indices = [lookup[row["id"]] for row in chosen]
    device = torch.device(args.device)
    view = tensor_view(valid, protocol["normalizers"], device)
    values = tuple(v[indices] for v in view[0])
    mask = view[1][indices]
    bert_input = np.asarray(valid["text_bert"])[indices]
    tokens = torch.tensor(bert_input[:, 0], dtype=torch.long, device=device)
    attention = torch.tensor(bert_input[:, 1], dtype=torch.long, device=device)
    segments = torch.tensor(bert_input[:, 2], dtype=torch.long, device=device)
    original = encode(bert, tokens, attention, segments)
    supplied = values[0]
    valid_tokens = attention.bool()
    active_difference = abs(original - supplied)[valid_tokens]
    with torch.inference_mode():
        from_supplied = model(values, mask)[0].cpu().numpy()
        from_reencoded = model((original, *values[1:]), mask)[0].cpu().numpy()
    masked = tokens.clone()
    for i, row in enumerate(chosen):
        for start, end in spans[row["id"]]:
            masked[i, start:end] = 103
    masked_embeddings = encode(bert, masked, attention, segments)
    with torch.inference_mode():
        changed = model((masked_embeddings, *values[1:]), mask)[0].cpu().numpy()
    tau = float(checkpoint["tau"])
    records = []
    for i, row in enumerate(chosen):
        sample_id = row["id"]
        cls = int(shapley[sample_id]["class"])
        original_margin = margin(float(from_supplied[i]), cls, tau)
        masked_margin = margin(float(changed[i]), cls, tau)
        records.append({"id": sample_id,
                        "feature_deletion_drop": float(row["primary_deletion_drop"]),
                        "wordpiece_mask_drop": original_margin - masked_margin,
                        "reencode_raw_difference": abs(float(from_supplied[i] - from_reencoded[i])),
                        "masked_token_count": int(sum(end - start for start, end in spans[sample_id]))})
    output = run / "explanation" / "text_mask_check"
    output.mkdir(exist_ok=True)
    with (output / "sample_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    feature = np.asarray([r["feature_deletion_drop"] for r in records])
    literal = np.asarray([r["wordpiece_mask_drop"] for r in records])
    summary = {"samples": len(records), "mask_id": 103,
               "active_text_mean_absolute_reencoding_error": float(active_difference.mean()),
               "active_text_max_absolute_reencoding_error": float(active_difference.max()),
               "prediction_max_absolute_reencoding_error": float(np.max(abs(from_supplied - from_reencoded))),
               "mean_feature_deletion_drop": float(feature.mean()),
               "mean_wordpiece_mask_drop": float(literal.mean()),
               "fraction_mask_drop_positive": float(np.mean(literal > 0)),
               "drop_pearson": float(np.corrcoef(feature, literal)[0, 1]),
               "limitations": "MASK keeps WordPiece positions and original A/V rows fixed; BERT context changes globally."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                             allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
