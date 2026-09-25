"""Map fixed error-case spans back to supplied WordPiece IDs without retuning."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    args = parser.parse_args()
    vocab = args.vocab.read_text(encoding="utf-8").splitlines()
    assert len(vocab) == 30522
    with (args.data_root / "附件2-数据集特征文件" / "aligned_50.pkl").open("rb") as stream:
        valid = pickle.load(stream)["valid"]
    lookup = {str(s): i for i, s in enumerate(valid["id"])}
    with (args.output / "error_cases.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        i = lookup[row["sample_id"]]
        bert = np.asarray(valid["text_bert"])[i]
        ids = bert[0].astype(int)
        eligible = (bert[1] > 0) & ~np.isin(ids, [0, 101, 102])
        content = np.flatnonzero(eligible)
        start, width = int(row["start_content_index"]), int(row["span_steps"])
        positions = content[start:start + width]
        words = [vocab[t] for t in ids[positions]]
        row["span_wordpieces"] = " ".join(words).replace(" ##", "")
        row["text_actually_removed"] = "text" in row["missing_mode"].split("+")
        row["span_token_ids"] = json.dumps(ids[positions].tolist())
        row["span_aligned_indices"] = json.dumps(positions.tolist())
    with (args.output / "error_cases_token_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "error_token_audit_provenance.json").write_text(json.dumps(dict(
        vocab_model="google-bert/bert-base-uncased", revision="86b5e0934494bd15c9632b12f734a8a67f723594",
        vocab_sha256=hashlib.sha256(args.vocab.read_bytes()).hexdigest(),
        method="Fixed error_cases.csv content-rank intervals mapped to original text_bert IDs; WordPiece continuation markers joined for display",
        attachment3_used=False, cases=len(rows)), indent=2), encoding="utf-8")
    for row in rows[:6]:
        print(row["sample_id"], row["span_wordpieces"])


if __name__ == "__main__":
    main()
