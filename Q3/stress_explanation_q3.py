"""Diagnostic-only reference and parameter sensitivity for Q3 Shapley values."""
import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from explain_q3 import SUBSETS, margin
from run_q3 import Q3Model, tensor_view


def phi_fixed(model, view, references, classes, tau, batch=64):
    x, masks = view
    parts = []
    model.eval()
    for begin in range(0, len(masks), batch):
        end = min(begin + batch, len(masks))
        base = tuple(v[begin:end] for v in x)
        subset_x = [tuple(base[j] if j in subset else
                          references[j][None, None, :].expand_as(base[j])
                          for j in range(3)) for subset in SUBSETS]
        joined = tuple(torch.cat([v[j] for v in subset_x], dim=0) for j in range(3))
        joined_masks = masks[begin:end].repeat(len(SUBSETS), 1, 1)
        with torch.inference_mode():
            raw = model(joined, joined_masks)[0].cpu().numpy().reshape(len(SUBSETS), end - begin)
        for i in range(end - begin):
            cls = int(classes[begin + i])
            scores = {s: margin(float(raw[k, i]), cls, tau)
                      for k, s in enumerate(SUBSETS)}
            phi = []
            for m in range(3):
                value = 0.
                for subset in SUBSETS:
                    if m not in subset:
                        expanded = tuple(sorted((*subset, m)))
                        value += {0: 1/3, 1: 1/6, 2: 1/3}[len(subset)] * (
                            scores[expanded] - scores[subset])
                phi.append(value)
            parts.append((float(raw[-1, i]), phi))
    return (np.asarray([p[0] for p in parts]),
            np.asarray([p[1] for p in parts]))


def comparison(base_raw, base_phi, changed_raw, changed_phi):
    base_primary = np.argmax(abs(base_phi), axis=1)
    other_primary = np.argmax(abs(changed_phi), axis=1)
    return {"primary_agreement": float(np.mean(base_primary == other_primary)),
            "phi_mean_absolute_difference": float(np.mean(abs(base_phi - changed_phi))),
            "raw_prediction_mean_absolute_difference": float(np.mean(abs(base_raw - changed_raw))),
            "phi_flat_pearson": float(np.corrcoef(base_phi.ravel(), changed_phi.ravel())[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / selection["final_record"]["checkpoint"],
                            map_location=args.device, weights_only=True)
    model = Q3Model(checkpoint["architecture"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with args.attachment2.open("rb") as stream:
        source = pickle.load(stream)
    train = tensor_view(source["train"], protocol["normalizers"], args.device)
    all_valid = tensor_view(source["valid"], protocol["normalizers"], args.device)
    diagnostic = set(protocol["diagnostic_ids"])
    indices = [i for i, value in enumerate(source["valid"]["id"])
               if str(value) in diagnostic]
    view = (tuple(v[indices] for v in all_valid[0]), all_valid[1][indices])
    mean_refs = tuple(v[m].mean(dim=0) for v, m in zip(train[0], train[1].unbind(1)))
    tau = float(checkpoint["tau"])
    with torch.inference_mode():
        base_prediction = model(view[0], view[1])[0].cpu().numpy()
    classes = np.where(base_prediction < -tau, 0,
                       np.where(base_prediction > tau, 2, 1))
    base_raw, base_phi = phi_fixed(model, view, mean_refs, classes, tau)
    rng = np.random.default_rng(6202)
    all_modes = train[1].any(dim=2).all(dim=1).cpu().numpy()
    donors = rng.choice(np.flatnonzero(all_modes), 16, replace=False)
    reference_results = []
    for donor in donors:
        refs = tuple(train[0][j][donor, int(torch.nonzero(train[1][donor, j])[0])]
                     for j in range(3))
        raw, phi = phi_fixed(model, view, refs, classes, tau)
        reference_results.append({"donor_train_index": int(donor),
                                  **comparison(base_raw, base_phi, raw, phi)})
        print("Reference", len(reference_results), "/16", flush=True)
    random_results = []
    torch.manual_seed(6203)
    random_model = copy.deepcopy(model)
    for stage in ("regression_head", "fusion", "encoders"):
        if stage == "regression_head":
            random_model.regression.reset_parameters()
        elif stage == "fusion":
            for module in random_model.fusion.modules():
                if module is not random_model.fusion and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        else:
            for module in random_model.encoders.modules():
                if module is not random_model.encoders and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        raw, phi = phi_fixed(random_model, view, mean_refs, classes, tau)
        random_results.append({"stage": stage,
                               **comparison(base_raw, base_phi, raw, phi)})
        print("Randomized", stage, flush=True)
    output = {"diagnostic_samples": len(indices), "original_class_fixed": True,
              "reference_method": "16 paired train sample row vectors; each modality uses first observed row",
              "reference_results": reference_results,
              "reference_primary_agreement_mean": float(np.mean([
                  r["primary_agreement"] for r in reference_results])),
              "parameter_randomization": random_results,
              "limitations": "Reference rows are drawn from observed train frames, not full paired donor sequences."}
    target = run / "explanation" / "sensitivity.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print("Complete", target, flush=True)


if __name__ == "__main__":
    main()
