"""Evaluate fixed 10/20/30% diagnostic evidence budgets against random controls."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from explain_q3 import intervene, chosen_spans
from run_q3 import MODES, Q3Model, tensor_view


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    record = json.loads((run / "selection.json").read_text(encoding="utf-8"))["final_record"]
    checkpoint = torch.load(run / record["checkpoint"], map_location=args.device,
                            weights_only=True)
    model = Q3Model(checkpoint["architecture"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tau = float(checkpoint["tau"])
    with args.attachment2.open("rb") as stream:
        source = pickle.load(stream)
    train = tensor_view(source["train"], protocol["normalizers"], args.device)
    valid = tensor_view(source["valid"], protocol["normalizers"], args.device)
    references = tuple(values[mask].mean(dim=0) for values, mask in
                       zip(train[0], train[1].unbind(1)))
    ids = {str(sample_id): i for i, sample_id in enumerate(source["valid"]["id"])}
    summaries = csv_rows(run / "explanation" / "diagnostic" / "explanations.csv")
    candidates = {}
    for row in csv_rows(run / "explanation" / "diagnostic" / "local_candidate_scores.csv"):
        candidates.setdefault((row["id"], row["modality"]), []).append(row)
    classes = {}
    with (run / "explanation" / "diagnostic" / "shapley.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            classes[entry["id"]] = int(entry["class"])
    rng = np.random.default_rng(6208)
    output = []
    for row in summaries:
        sample_id = row["id"]
        mode_name = row["primary_modality"]
        if mode_name not in MODES:
            continue
        mode = MODES.index(mode_name)
        i = ids[sample_id]
        values = tuple(value[i] for value in valid[0])
        masks = valid[1][i]
        records = candidates[(sample_id, mode_name)]
        spans = [(int(r["feature_start"]), int(r["feature_end_exclusive"])) for r in records]
        deltas = np.asarray([float(r["signed_delta"]) for r in records])
        full = float(row["full_score"])
        klass = classes[sample_id]
        for fraction in (.1, .2, .3):
            budget = max(1, int(np.ceil(fraction * int(masks[mode].sum()))))
            selected = chosen_spans(spans, deltas, budget, supportive=True)
            if not selected:
                selected = chosen_spans(spans, deltas, budget, supportive=False)
            if not selected:
                continue
            labelled = [(mode, start, end, delta) for start, end, delta in selected]
            drop = full - intervene(model, values, masks, references, labelled, klass, tau)
            retained = intervene(model, values, masks, references, labelled, klass, tau, retain=True)
            widths = [end - start for start, end, _ in selected]
            random_drops = []
            for _ in range(10):
                picked, used = [], set()
                for width in widths:
                    possible = [(start, end) for start, end in spans if end - start == width
                                and not any(pos in used for pos in range(start, end))]
                    if not possible:
                        break
                    start, end = possible[int(rng.integers(len(possible)))]
                    picked.append((mode, start, end, 0.))
                    used.update(range(start, end))
                if len(picked) == len(widths):
                    random_drops.append(full - intervene(model, values, masks, references,
                                                         picked, klass, tau))
            output.append({"id": sample_id, "group": sample_id.rsplit("$_$", 1)[0],
                           "fraction_requested": fraction, "positions_budget": budget,
                           "positions_selected": sum(widths),
                           "selected_drop": drop, "random_mean_drop":
                           float(np.mean(random_drops)) if random_drops else "",
                           "random_repetitions": len(random_drops),
                           "retained_score": retained, "full_score": full})
        if len(output) % 75 == 0:
            print("Evaluated", sample_id, len(output), flush=True)
    target = run / "explanation" / "diagnostic" / "budget_curves.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {str(fraction): {"rows": sum(r["fraction_requested"] == fraction for r in output),
                               "selected_mean": float(np.mean([r["selected_drop"] for r in output
                                                                 if r["fraction_requested"] == fraction])),
                               "random_mean": float(np.mean([r["random_mean_drop"] for r in output
                                                               if r["fraction_requested"] == fraction
                                                               and r["random_mean_drop"] != ""])),
                               "mean_covered_positions": float(np.mean([r["positions_selected"] for r in output
                                                                           if r["fraction_requested"] == fraction]))}
               for fraction in (.1, .2, .3)}
    (target.parent / "budget_summary.json").write_text(json.dumps(summary, indent=2,
                                                                ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
