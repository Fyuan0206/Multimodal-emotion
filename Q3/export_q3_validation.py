"""Export frozen per-sample valid predictions and five-seed attribution stability."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from explain_q3 import batch_shapley
from run_q3 import MODES, Q3Model, predict, tensor_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    selected = set(protocol["selection_ids"])
    with args.attachment2.open("rb") as stream:
        source = pickle.load(stream)
    valid = source["valid"]
    device = torch.device(args.device)
    view = tensor_view(valid, protocol["normalizers"], device)
    train_x, train_mask = tensor_view(source["train"], protocol["normalizers"], device)
    refs = tuple(v[m].mean(dim=0) for v, m in zip(train_x, train_mask.unbind(1)))
    ids = [str(i) for i in valid["id"]]
    diagnostic_indices = [i for i, sample_id in enumerate(ids) if sample_id not in selected]
    diag_view = (tuple(v[diagnostic_indices] for v in view[0]),
                 view[1][diagnostic_indices])
    raw_rows, attribution_rows = [], []
    for record in results:
        checkpoint = torch.load(run / record["checkpoint"], map_location=device,
                                weights_only=True)
        model = Q3Model(record["architecture"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        raw = predict(model, view)
        tau = float(record["tau"])
        for i, value in enumerate(raw):
            intensity = 0. if abs(float(value)) <= tau else float(value)
            raw_rows.append({"id": ids[i], "group": ids[i].rsplit("$_$", 1)[0],
                             "set": "selection" if ids[i] in selected else "diagnostic",
                             "architecture": record["architecture"], "seed": record["seed"],
                             "tau": tau, "true_intensity": float(valid["regression_labels"][i]),
                             "true_class": int(valid["classification_labels"][i]),
                             "raw_intensity": float(value), "intensity": intensity,
                             "class": int(np.sign(intensity)) + 1})
        if record["architecture"] == "C2":
            shapley = batch_shapley(model, diag_view[0], diag_view[1], refs, tau)
            for position, explained in zip(diagnostic_indices, shapley):
                phi = np.asarray(explained["phi"])
                primary = MODES[int(np.argmax(abs(phi)))] if abs(phi).sum() > 1e-6 else "undetermined"
                attribution_rows.append({"id": ids[position], "seed": record["seed"],
                                         "predicted_class": explained["class"],
                                         "primary_modality": primary,
                                         **{f"phi_{m}": float(phi[j]) for j, m in enumerate(MODES)}})
        print(record["architecture"], record["seed"], flush=True)
    output = run / "validation_detail"
    output.mkdir(exist_ok=True)
    for name, rows in (("predictions.csv", raw_rows), ("c2_seed_attributions.csv", attribution_rows)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Rows", len(raw_rows), len(attribution_rows), flush=True)


if __name__ == "__main__":
    main()
