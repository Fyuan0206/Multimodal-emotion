"""Explain frozen Q3 predictions and test feature-intervention faithfulness."""
import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from run_q3 import MODES, Q3Model, save_json, tensor_view, sha256


SUBSETS = tuple(itertools.chain.from_iterable(
    itertools.combinations(range(3), k) for k in range(4)))


def margin(raw, predicted_class, tau):
    if predicted_class == 0:
        return -raw - tau
    if predicted_class == 2:
        return raw - tau
    return tau - abs(raw)


def score_many(model, x, masks, klass, tau, batch=256):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(masks), batch):
            values = tuple(v[start:start + batch] for v in x)
            raw, _ = model(values, masks[start:start + batch])
            scores.extend(margin(float(v), klass, tau) for v in raw.cpu().numpy())
    return np.asarray(scores, dtype=np.float64)


def batch_shapley(model, x, masks, references, tau, batch=32):
    output = []
    for begin in range(0, len(masks), batch):
        end = min(begin + batch, len(masks))
        batch_x = tuple(v[begin:end] for v in x)
        batch_masks = masks[begin:end]
        n = len(batch_masks)
        subset_x = []
        for subset in SUBSETS:
            subset_x.append(tuple(
                batch_x[j] if j in subset else references[j][None, None, :].expand_as(batch_x[j])
                for j in range(3)))
        joined = tuple(torch.cat([v[j] for v in subset_x], dim=0) for j in range(3))
        joined_masks = batch_masks.repeat(len(SUBSETS), 1, 1)
        with torch.inference_mode():
            raws, _ = model(joined, joined_masks)
        raw = raws.cpu().numpy().reshape(len(SUBSETS), n)
        for i in range(n):
            full = float(raw[-1, i])
            klass = 0 if full < -tau else 2 if full > tau else 1
            subset_scores = {subset: margin(float(raw[k, i]), klass, tau)
                             for k, subset in enumerate(SUBSETS)}
            phi = np.zeros(3, dtype=np.float64)
            for mode in range(3):
                for subset in SUBSETS:
                    if mode in subset:
                        continue
                    weight = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}[len(subset)]
                    enlarged = tuple(sorted((*subset, mode)))
                    phi[mode] += weight * (subset_scores[enlarged] - subset_scores[subset])
            residual = sum(phi) - (subset_scores[(0, 1, 2)] - subset_scores[()])
            if abs(residual) > 1e-5 * (1 + abs(subset_scores[(0, 1, 2)])):
                raise ValueError("Shapley additivity failed")
            output.append({"raw": full, "class": klass, "phi": phi.tolist(),
                           "baseline": subset_scores[()], "full": subset_scores[(0, 1, 2)],
                           "subset_scores": {"+".join(MODES[j] for j in subset) or "none": score
                                             for subset, score in subset_scores.items()},
                           "residual": residual})
    return output


def candidate_scores(model, x, mask, references, mode, klass, tau, budget):
    positions = np.flatnonzero(mask[mode].cpu().numpy())
    candidates = []
    for width in range(1, min(4, budget) + 1):
        for left in range(len(positions) - width + 1):
            span = positions[left:left + width]
            if span[-1] - span[0] != width - 1:
                continue
            candidates.append((int(span[0]), int(span[-1] + 1)))
    if not candidates:
        return [], np.empty(0)
    xs = [v.unsqueeze(0).expand(len(candidates), -1, -1).clone() for v in x]
    for i, (start, end) in enumerate(candidates):
        xs[mode][i, start:end] = references[mode]
    repeated = mask.unsqueeze(0).expand(len(candidates), -1, -1)
    return candidates, score_many(model, xs, repeated, klass, tau)


def chosen_spans(candidates, deltas, budget, supportive=True):
    order = np.argsort(-deltas if supportive else -np.abs(deltas), kind="stable")
    selected, used = [], set()
    for index in order:
        start, end = candidates[int(index)]
        if len(selected) >= 3 or len(used) + end - start > budget:
            continue
        if supportive and deltas[int(index)] <= 0:
            continue
        if any(p in used for p in range(start, end)):
            continue
        selected.append((start, end, float(deltas[int(index)])))
        used.update(range(start, end))
    return selected


def intervene(model, x, mask, references, spans, klass, tau, retain=False):
    modified = [v.unsqueeze(0).clone() for v in x]
    mode = spans[0][0] if spans else 0
    if retain:
        for j in range(3):
            observed = mask[j]
            modified[j][0, observed] = references[j]
        for j, start, end, _ in spans:
            modified[j][0, start:end] = x[j][start:end]
    else:
        for j, start, end, _ in spans:
            modified[j][0, start:end] = references[j]
    score = score_many(model, tuple(modified), mask.unsqueeze(0), klass, tau)
    return float(score[0])


def analyze_local(model, view, explanations, references, tau, ids, output, rng):
    evidence, summary, local_rows = [], [], []
    for i, sample_id in enumerate(ids):
        x = tuple(v[i] for v in view[0])
        mask = view[1][i]
        explained = explanations[i]
        phi = np.asarray(explained["phi"])
        weights = np.abs(phi) / np.abs(phi).sum() if np.abs(phi).sum() >= 1e-6 else np.zeros(3)
        primary = int(np.argmax(np.abs(phi))) if np.abs(phi).sum() >= 1e-6 else -1
        supportive = int(np.argmax(phi)) if phi.max() > 0 else -1
        all_mode_spans = []
        candidates_by_mode = {}
        for mode in range(3):
            active = int(mask[mode].sum())
            if active == 0:
                continue
            budget = max(1, int(np.ceil(.2 * active)))
            candidates, values = candidate_scores(model, x, mask, references, mode,
                                                   explained["class"], tau, budget)
            deltas = explained["full"] - values
            candidates_by_mode[mode] = (candidates, deltas, budget)
            for (start, end), delta in zip(candidates, deltas):
                local_rows.append({"id": sample_id, "modality": MODES[mode],
                                   "feature_start": start, "feature_end_exclusive": end,
                                   "signed_delta": float(delta)})
            picked = chosen_spans(candidates, deltas, budget, supportive=True)
            if not picked:
                picked = chosen_spans(candidates, deltas, budget, supportive=False)
            all_mode_spans.append((mode, picked))
            for start, end, delta in picked:
                evidence.append({"evidence_id": f"{sample_id}:{MODES[mode]}:{start}:{end}",
                                 "id": sample_id, "modality": MODES[mode],
                                 "feature_start": start, "feature_end_exclusive": end,
                                 "signed_score": delta, "mapping_status": "feature_only",
                                 "text_excerpt": "", "start_sec": "", "end_sec": "",
                                 "video_frame_pts": "", "human_verified": False})
        row = {"id": sample_id, "primary_modality": MODES[primary] if primary >= 0 else "undetermined",
               "top_support_modality": MODES[supportive] if supportive >= 0 else "undetermined",
               "contribution_target": "original_class_margin",
               "baseline_score": explained["baseline"], "full_score": explained["full"],
               "explanation_status": "feature_contributions_complete_raw_mapping_pending"}
        for mode, name in enumerate(MODES):
            row[f"phi_{name}"] = float(phi[mode])
            row[f"weight_{name}"] = float(weights[mode])
        selected = next((spans for mode, spans in all_mode_spans if mode == primary), [])
        if selected:
            labelled = [(primary, start, end, delta) for start, end, delta in selected]
            removed = intervene(model, x, mask, references, labelled,
                                explained["class"], tau)
            row["primary_deletion_drop"] = explained["full"] - removed
            row["primary_retain_score"] = intervene(model, x, mask, references, labelled,
                                                      explained["class"], tau, retain=True)
            candidates, _, _ = candidates_by_mode[primary]
            widths = [end - start for start, end, _ in selected]
            random_drops = []
            for _ in range(10):
                random_spans, used = [], set()
                for width in widths:
                    feasible = [(s, e) for s, e in candidates if e - s == width and
                                not any(p in used for p in range(s, e))]
                    if not feasible:
                        break
                    start, end = feasible[int(rng.integers(len(feasible)))]
                    random_spans.append((primary, start, end, 0.))
                    used.update(range(start, end))
                if len(random_spans) == len(widths):
                    random_drops.append(explained["full"] - intervene(
                        model, x, mask, references, random_spans, explained["class"], tau))
            row["random_mean_drop"] = float(np.mean(random_drops)) if random_drops else ""
            row["random_repetitions"] = len(random_drops)
        else:
            row.update(primary_deletion_drop="", primary_retain_score="",
                       random_mean_drop="", random_repetitions=0)
        summary.append(row)
        if (i + 1) % 25 == 0:
            print("Explained local", i + 1, "/", len(ids), flush=True)
    def csv_write(path, rows):
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    csv_write(output / "explanations.csv", summary)
    csv_write(output / "evidence_feature_only.csv", evidence)
    csv_write(output / "local_candidate_scores.csv", local_rows)
    return summary


def plot_outputs(output, diagnostic_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.transparent": False})
    weights = np.asarray([[float(r[f"weight_{m}"]) for m in MODES]
                          for r in diagnostic_rows])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(MODES, weights.mean(axis=0), color=["#315b7d", "#d18146", "#688b68"])
    ax.set_ylabel("Mean absolute-contribution share")
    ax.set_ylim(0, 1)
    ax.set_title("Q3 validation diagnostic subset")
    fig.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(output / f"modality_contribution.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    paired = [(float(r["primary_deletion_drop"]), float(r["random_mean_drop"]))
              for r in diagnostic_rows if r["primary_deletion_drop"] != ""
              and r["random_mean_drop"] != ""]
    if paired:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.boxplot(np.asarray(paired), tick_labels=["Selected", "Random"],
                   patch_artist=False, showfliers=False)
        ax.set_ylabel("Drop in original class margin")
        ax.set_title("Equal-budget feature interventions")
        fig.tight_layout()
        for ext in ("png", "pdf", "eps"):
            fig.savefig(output / f"deletion_validation.{ext}", dpi=300,
                        facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--attachment4-aligned", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if sha256(args.attachment2) != protocol["attachment2_sha256"]:
        raise ValueError("Training source hash mismatch")
    checkpoint_path = run / selection["final_record"]["checkpoint"]
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Q3Model(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tau = float(checkpoint["tau"])
    with args.attachment2.open("rb") as stream:
        source = pickle.load(stream)
    valid = source["valid"]
    norms = protocol["normalizers"]
    train_x, train_masks = tensor_view(source["train"], norms, device)
    references = tuple(values[mask].mean(dim=0) for values, mask in
                       zip(train_x, train_masks.unbind(dim=1)))
    diagnostic_set = set(protocol["diagnostic_ids"])
    diagnostic_indices = [i for i, sample_id in enumerate(valid["id"])
                          if str(sample_id) in diagnostic_set]
    full_valid = tensor_view(valid, norms, device)
    diagnostic = (tuple(values[diagnostic_indices] for values in full_valid[0]),
                  full_valid[1][diagnostic_indices])
    paths = sorted(args.attachment4_aligned.glob("*.pkl"))
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        if str(sample["id"]) != path.stem:
            raise ValueError(path)
        samples.append(sample)
    packed = {key: np.stack([s[key] for s in samples])
              for key in ("text", "audio", "vision", "text_bert")}
    attachment = tensor_view(packed, norms, device)
    out = run / "explanation"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(6204)
    records = {}
    for name, view, ids in (("diagnostic", diagnostic,
                             [str(valid["id"][i]) for i in diagnostic_indices]),
                            ("attachment4", attachment, [str(s["id"]) for s in samples])):
        target = out / name
        target.mkdir(exist_ok=True)
        shapley = batch_shapley(model, view[0], view[1], references, tau)
        summary = analyze_local(model, view, shapley, references, tau, ids, target, rng)
        with (target / "shapley.jsonl").open("w", encoding="utf-8") as stream:
            for sample_id, values in zip(ids, shapley):
                stream.write(json.dumps({"id": sample_id, **values}, allow_nan=False) + "\n")
        records[name] = {"samples": len(ids), "primary_counts": {
            m: sum(r["primary_modality"] == m for r in summary)
            for m in (*MODES, "undetermined")}}
        if name == "diagnostic":
            plot_outputs(target, summary)
            differences = np.asarray([float(r["primary_deletion_drop"])
                                      - float(r["random_mean_drop"])
                                      for r in summary if r["primary_deletion_drop"] != ""
                                      and r["random_mean_drop"] != ""])
            records[name]["mean_selected_minus_random_drop"] = (
                float(differences.mean()) if len(differences) else None)
            records[name]["paired_samples"] = len(differences)
    save_json(out / "summary.json", {"model_sha256": sha256(checkpoint_path),
              "tau": tau, "reference": "train_mean_normalized_feature_rows",
              "interpretation": "feature interventions and original-class margin",
              "raw_video_mapping": "pending", "records": records})


if __name__ == "__main__":
    main()
