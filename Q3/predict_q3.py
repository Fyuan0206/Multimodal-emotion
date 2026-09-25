"""Checkpoint-only Attachment 4 inference with source and row verification."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from run_q3 import Q3Model, predict, sha256, tensor_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    record = json.loads((run / "selection.json").read_text(encoding="utf-8"))["final_record"]
    checkpoint_path = run / record["checkpoint"]
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Q3Model(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    paths = sorted(args.aligned.glob("*.pkl"))
    if len(paths) != 20:
        raise ValueError("Expected 20 aligned Attachment 4 samples")
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        if str(sample["id"]) != path.stem:
            raise ValueError(f"ID mismatch: {path}")
        samples.append(sample)
    packed = {key: np.stack([s[key] for s in samples])
              for key in ("text", "audio", "vision", "text_bert")}
    view = tensor_view(packed, protocol["normalizers"], device)
    raw = predict(model, view)
    rows = []
    for sample, value in zip(samples, raw):
        intensity = 0. if abs(float(value)) <= float(checkpoint["tau"]) else float(value)
        rows.append({"id": str(sample["id"]),
                     "polarity": "Negative" if intensity < 0 else
                     "Positive" if intensity > 0 else "Neutral",
                     "intensity": intensity, "raw_intensity": float(value),
                     "model_sha256": sha256(checkpoint_path)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.compare:
        with args.compare.open(encoding="utf-8-sig", newline="") as stream:
            original = list(csv.DictReader(stream))
        if len(original) != len(rows):
            raise ValueError("Different prediction counts")
        differences = []
        for found, expected in zip(rows, original):
            if (found["id"], found["polarity"]) != (expected["id"], expected["polarity"]):
                raise ValueError(f"Prediction mismatch: {found['id']}")
            differences.append(abs(found["raw_intensity"] - float(expected["raw_intensity"])))
        print("All 20 classes identical; max raw intensity difference", max(differences))
    else:
        print("Predicted", len(rows))


if __name__ == "__main__":
    main()
