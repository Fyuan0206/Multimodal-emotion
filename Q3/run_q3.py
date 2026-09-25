"""Train Q3 on aligned_50 train, select on valid, predict aligned Attachment 4."""
import argparse
import copy
import csv
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODES = ("text", "audio", "vision")
DIMS = (768, 74, 35)
ARCHITECTURES = ("C0", "C1", "C2", "C3")
TAUS = (0, .05, .10, .15, .20, .25, .30)
SEEDS = range(2026, 2031)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(ids, labels):
    groups = np.asarray([str(i).rsplit("$_$", 1)[0] for i in ids])
    unique = np.unique(groups)
    whole = np.bincount(labels, minlength=3) / len(labels)
    rng = np.random.default_rng(6201)
    best = None
    for _ in range(100):
        shuffled = rng.permutation(unique)
        chosen = set(shuffled[:round(.6 * len(unique))])
        selection = np.asarray([g in chosen for g in groups])
        n = int(selection.sum())
        if n < 200 or n > len(ids) - 150:
            continue
        proportions = np.bincount(labels[selection], minlength=3) / n
        cost = abs(n / len(ids) - .6) + np.abs(proportions - whole).sum()
        if best is None or cost < best[0]:
            best = (cost, selection)
    if best is None:
        raise ValueError("Could not split validation groups")
    selected = np.flatnonzero(best[1])
    diagnostic = np.flatnonzero(~best[1])
    if set(groups[selected]) & set(groups[diagnostic]):
        raise AssertionError("Source group overlap")
    return selected, diagnostic


def make_arrays(split, norms=None):
    bert = np.asarray(split["text_bert"])
    attention = bert[:, 1].astype(bool)
    content = attention & ~np.isin(bert[:, 0], [0, 101, 102])
    arrays = {}
    observed = []
    for mode in MODES:
        raw = np.asarray(split[mode], dtype=np.float32)
        mask = content.copy()
        if mode != "text":
            mask &= np.any(raw != 0, axis=2)
        observed.append(mask)
        if mode == "text":
            arrays[mode] = raw
        else:
            if norms is None:
                continue
            mean = np.asarray(norms[mode]["mean"], dtype=np.float32)
            scale = np.asarray(norms[mode]["scale"], dtype=np.float32)
            arrays[mode] = np.clip((raw - mean) / scale, -10, 10)
    masks = np.stack(observed, axis=1)
    if not masks[:, 0].any(axis=1).all():
        raise ValueError("A sample has no text content")
    return arrays, masks


def fit_norms(train):
    _, masks = make_arrays(train)
    norms = {}
    for j, mode in enumerate(MODES[1:], 1):
        raw = np.asarray(train[mode], dtype=np.float32)
        values = raw[masks[:, j]]
        norms[mode] = {"mean": values.mean(axis=0).tolist(),
                       "scale": np.maximum(values.std(axis=0), 1e-3).tolist()}
    return norms


def tensor_view(split, norms, device):
    arrays, masks = make_arrays(split, norms)
    x = tuple(torch.as_tensor(arrays[m], dtype=torch.float32, device=device)
              for m in MODES)
    mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
    return x, mask


def take(view, indices):
    x, mask = view
    return tuple(value[indices] for value in x), mask[indices]


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, temporal, attention):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.temporal = temporal
        self.attention = attention
        if temporal:
            self.sequence = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 4, dim_feedforward=128,
                                           dropout=.15, batch_first=True,
                                           norm_first=False), num_layers=1,
                enable_nested_tensor=False)
            position = torch.arange(50, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, 64, 2, dtype=torch.float32)
                                  * (-np.log(10000.0) / 64))
            encoding = torch.zeros(50, 64)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer("position", encoding, persistent=False)
        if attention:
            self.score = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.empty = nn.Parameter(torch.zeros(64))

    def forward(self, values, mask):
        safe = mask.clone()
        missing = ~safe.any(dim=1)
        safe[missing, 0] = True
        hidden = self.project(values * mask.unsqueeze(-1))
        if self.temporal:
            hidden = self.sequence(hidden + self.position[None],
                                   src_key_padding_mask=~safe)
        if self.attention:
            scores = self.score(hidden).squeeze(-1).masked_fill(~safe, -1e4)
            weights = torch.softmax(scores, dim=1) * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(missing[:, None], self.empty[None], pooled)
        return pooled, weights


class Q3Model(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        self.architecture = architecture
        self.pool_only = architecture == "C1"
        if self.pool_only:
            self.encoders = nn.ModuleList([
                nn.Sequential(nn.Linear(d * 5, 64), nn.LayerNorm(64), nn.GELU())
                for d in DIMS])
        else:
            self.encoders = nn.ModuleList([
                ModalityEncoder(d, architecture in ("C0", "C2", "C3"),
                                architecture in ("C0", "C3")) for d in DIMS])
        count = 1 if architecture == "C0" else 3
        self.fusion = nn.Sequential(nn.Linear(count * 64 + count, 128), nn.GELU(),
                                    nn.Dropout(.2))
        self.regression = nn.Linear(128, 1)
        self.classification = nn.Linear(128, 3)

    @staticmethod
    def stat_pool(values, mask):
        valid = mask.unsqueeze(-1)
        n = valid.sum(dim=1).clamp_min(1)
        mean = (values * valid).sum(dim=1) / n
        variance = ((values - mean[:, None]) ** 2 * valid).sum(dim=1) / n
        positions = torch.arange(50, device=values.device)[None, :].expand_as(mask)
        ranks = mask.long().cumsum(dim=1) - 1
        length = mask.sum(dim=1).clamp_min(1)[:, None]
        third = torch.clamp(3 * ranks // length, 0, 2)
        thirds = []
        for k in range(3):
            selected = mask & (third == k) & (positions >= 0)
            thirds.append((values * selected.unsqueeze(-1)).sum(dim=1)
                          / selected.sum(dim=1, keepdim=True).clamp_min(1))
        return torch.cat([mean, torch.sqrt(variance + 1e-8), *thirds], dim=1)

    def forward(self, x, masks, return_weights=False):
        count = 1 if self.architecture == "C0" else 3
        parts, weights = [], []
        for i in range(count):
            if self.pool_only:
                part = self.encoders[i](self.stat_pool(x[i], masks[:, i]))
                weight = masks[:, i].float() / masks[:, i].sum(dim=1, keepdim=True).clamp_min(1)
            else:
                part, weight = self.encoders[i](x[i], masks[:, i])
            parts.append(part)
            weights.append(weight)
        coverage = masks[:, :count].float().mean(dim=2)
        joined = torch.cat([*parts, coverage], dim=1)
        hidden = self.fusion(joined)
        reg = 3 * torch.tanh(self.regression(hidden).squeeze(1))
        logits = self.classification(hidden)
        if return_weights:
            return reg, logits, weights
        return reg, logits


def scores(y, c, raw, tau):
    prediction = np.asarray(raw).copy()
    prediction[np.abs(prediction) <= tau] = 0
    classes = np.sign(prediction).astype(int) + 1
    f1 = []
    for label in range(3):
        tp = np.sum((c == label) & (classes == label))
        denominator = np.sum(c == label) + np.sum(classes == label)
        f1.append(2 * tp / denominator if denominator else 0.)
    corr = float(np.corrcoef(y, prediction)[0, 1]) if np.std(prediction) > 0 else None
    return {"accuracy": float(np.mean(c == classes)), "macro_f1": float(np.mean(f1)),
            "mae": float(np.mean(np.abs(y - prediction))), "pearson_r": corr}


def choose_tau(y, c, raw):
    candidates = []
    for tau in TAUS:
        metrics = scores(y, c, raw, tau)
        objective = metrics["mae"] + .6 * (1 - metrics["macro_f1"])
        candidates.append((objective, metrics["mae"], tau, metrics))
    return min(candidates, key=lambda item: item[:3])


def predict(model, view, batch_size=256):
    model.eval()
    result = []
    with torch.inference_mode():
        for begin in range(0, len(view[1]), batch_size):
            x, masks = take(view, slice(begin, begin + batch_size))
            reg, _ = model(x, masks)
            result.append(reg.cpu().numpy())
    return np.concatenate(result)


def train_one(architecture, seed, train, valid, y_train, c_train, y_valid,
              c_valid, selected, output, epochs, batch, smoke=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Q3Model(architecture).to(train[1].device)
    frequencies = np.bincount(c_train, minlength=3).astype(float)
    weights = np.sqrt(frequencies.sum() / (3 * frequencies))
    weights = torch.as_tensor(weights / weights.mean(), dtype=torch.float32,
                              device=train[1].device)
    labels = torch.as_tensor(y_train, dtype=torch.float32, device=train[1].device)
    classes = torch.as_tensor(c_train, dtype=torch.long, device=train[1].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, stale = None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(y_train))
        losses = []
        for begin in range(0, len(order), batch):
            indices = order[begin:begin + batch]
            x, masks = take(train, indices)
            reg, logits = model(x, masks)
            loss = nn.functional.smooth_l1_loss(reg, labels[indices], beta=.5)
            loss += .7 * nn.functional.cross_entropy(logits, classes[indices],
                                                       weight=weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {architecture}/{seed}/{epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if smoke:
                break
        raw = predict(model, valid)
        objective, mae, tau, metrics = choose_tau(y_valid[selected], c_valid[selected],
                                                   raw[selected])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "selection_objective": objective, "selection_tau": tau,
                        **metrics})
        key = (objective, mae, tau, epoch)
        if best is None or key < best[0]:
            best = (key, copy.deepcopy(model.state_dict()))
            stale = 0
        else:
            stale += 1
        print(architecture, seed, epoch, "loss", round(history[-1]["train_loss"], 4),
              "J", round(objective, 4), "tau", tau, flush=True)
        if stale >= 8 or smoke:
            break
    model.load_state_dict(best[1])
    checkpoint = output / f"{architecture}_{seed}.pt"
    torch.save({"architecture": architecture, "seed": seed, "tau": best[0][2],
                "epoch": best[0][3], "state_dict": best[1]}, checkpoint)
    save_json(output / f"{architecture}_{seed}_history.json", history)
    return {"architecture": architecture, "seed": seed, "checkpoint": checkpoint.name,
            "epoch": best[0][3], "tau": best[0][2],
            "selection_objective": best[0][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--attachment4-aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with args.attachment2.open("rb") as stream:
        source = pickle.load(stream)
    train, valid = source["train"], source["valid"]
    norms = fit_norms(train)
    selected, diagnostic = split_indices(valid["id"],
                                          np.asarray(valid["classification_labels"], dtype=int))
    device = torch.device(args.device)
    train_view = tensor_view(train, norms, device)
    valid_view = tensor_view(valid, norms, device)
    y_train = np.asarray(train["regression_labels"], dtype=np.float32)
    c_train = np.asarray(train["classification_labels"], dtype=int)
    y_valid = np.asarray(valid["regression_labels"], dtype=np.float32)
    c_valid = np.asarray(valid["classification_labels"], dtype=int)
    save_json(output / "protocol.json", {"attachment2_sha256": sha256(args.attachment2),
              "architecture_candidates": ARCHITECTURES, "seeds": list(SEEDS),
              "selection_ids": [str(valid["id"][i]) for i in selected],
              "diagnostic_ids": [str(valid["id"][i]) for i in diagnostic],
              "tau_candidates": TAUS, "normalizers": norms,
              "device": str(device), "smoke": args.smoke})
    records = []
    for arch in ARCHITECTURES:
        for seed in (list(SEEDS)[:1] if args.smoke else SEEDS):
            record = train_one(arch, seed, train_view, valid_view, y_train, c_train,
                               y_valid, c_valid, selected, output, 1 if args.smoke else 40,
                               32 if args.smoke else 128, args.smoke)
            checkpoint = torch.load(output / record["checkpoint"], map_location=device,
                                    weights_only=True)
            model = Q3Model(arch).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            raw = predict(model, valid_view)
            record["selection"] = scores(y_valid[selected], c_valid[selected],
                                          raw[selected], record["tau"])
            record["diagnostic"] = scores(y_valid[diagnostic], c_valid[diagnostic],
                                           raw[diagnostic], record["tau"])
            record["full_valid"] = scores(y_valid, c_valid, raw, record["tau"])
            records.append(record)
            save_json(output / "results.json", records)
    if args.smoke:
        print("Smoke run complete; no Attachment 4 prediction", flush=True)
        return
    means = {arch: float(np.mean([r["selection_objective"] for r in records
                                  if r["architecture"] == arch]))
             for arch in ARCHITECTURES[1:]}
    chosen = min(means, key=lambda arch: (means[arch], ARCHITECTURES.index(arch)))
    final_record = next(r for r in records if r["architecture"] == chosen and r["seed"] == 2026)
    checkpoint = torch.load(output / final_record["checkpoint"], map_location=device,
                            weights_only=True)
    model = Q3Model(chosen).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    paths = sorted(args.attachment4_aligned.glob("*.pkl"))
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        if str(sample["id"]) != path.stem:
            raise ValueError(f"ID mismatch: {path}")
        samples.append(sample)
    packed = {key: np.stack([s[key] for s in samples])
              for key in ("text", "audio", "vision", "text_bert")}
    attachment_view = tensor_view(packed, norms, device)
    raw = predict(model, attachment_view)
    rows = []
    for sample, value in zip(samples, raw):
        intensity = 0. if abs(float(value)) <= final_record["tau"] else float(value)
        rows.append({"id": str(sample["id"]), "polarity":
                     ("Negative" if intensity < 0 else "Positive" if intensity > 0 else "Neutral"),
                     "intensity": intensity, "raw_intensity": float(value),
                     "model_sha256": sha256(output / final_record["checkpoint"])})
    with (output / "attachment4_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "selection.json", {"mean_selection_objective": means,
              "chosen_architecture": chosen, "final_record": final_record,
              "attachment4_count": len(rows), "status": "prediction_complete_explanation_pending"})
    print("Finished", chosen, len(rows), "Attachment 4 predictions", flush=True)


if __name__ == "__main__":
    main()
