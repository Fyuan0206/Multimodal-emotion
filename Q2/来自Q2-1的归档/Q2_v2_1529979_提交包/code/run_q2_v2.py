"""Q2 paired local-span experiments. See EXPERIMENT_PLAN.md and README.md."""
import argparse
import copy
import itertools
import json
import os
import pickle
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import BertModel
from huggingface_hub import hf_hub_download

import run_q2 as base

MODES = base.MODES
COMBINATIONS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
POSITIONS = ("early", "middle", "late")
GRID = [(m, p, r) for m, p, r in itertools.product(COMBINATIONS, POSITIONS, (.15, .35, .55))]
VARIANTS = {"T0": (False, False, True), "T1": (True, False, True),
            "M0": (False, False, False), "M1": (True, False, False),
            "M2": (False, True, False), "M3": (True, True, False)}
THRESHOLDS = np.array([0, .05, .1, .15, .2, .25, .3])
FEATURE_KEYS = (*MODES, "coverage")
METRICS = ("accuracy", "macro_f1", "mae", "pearson_r")
SCHEMA = "q2-v2-paired-20260925-1"


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def save_npz(path, **arrays):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def group_ids(ids):
    # The supplied IDs have the stable video-ID$_$segment-ID convention.
    if not all("$_$" in str(i) and str(i).rsplit("$_$", 1)[1].isdigit() for i in ids):
        return np.asarray([str(i) for i in ids]), "sample (video identifier unavailable)"
    return np.asarray([str(i).rsplit("$_$", 1)[0] for i in ids]), "video prefix before $_$"


def span_masks(observed, eligible, seed, modes=None, rate=None, position=None):
    """No length shrinking or holes: infeasible sample/condition is explicit."""
    rng = np.random.default_rng(seed)
    masks = observed.copy()
    n = len(masks)
    valid = np.zeros(n, bool)
    starts = np.full(n, -1, np.int16)
    widths = np.zeros(n, np.int16)
    selected_bits = np.zeros(n, np.uint8)
    requested = np.zeros(n, np.float32)
    centers = np.full(n, np.nan, np.float32)
    choices_count = np.zeros(n, np.int16)
    for i in range(n):
        indices = np.flatnonzero(eligible[i])
        length = len(indices)
        chosen = modes if modes is not None else COMBINATIONS[int(rng.integers(7))]
        fraction = float(rate) if rate is not None else float(rng.uniform(.1, .7))
        width = max(1, int(round(fraction * length)))
        widths[i], requested[i] = width, fraction
        selected_bits[i] = sum(1 << j for j in chosen)
        candidates = []
        for start in range(max(0, length - width + 1)):
            center = (start + width / 2) / max(length, 1)
            if position is not None and min(2, int(3 * center)) != POSITIONS.index(position):
                continue
            interval = indices[start:start + width]
            # Require genuine removal and at least one original observation outside.
            removed = observed[i, interval][:, chosen].sum(axis=0)
            total = observed[i][:, chosen].sum(axis=0)
            if np.all(removed > 0) and np.all(removed < total):
                candidates.append(start)
        choices_count[i] = len(candidates)
        if not candidates:
            continue
        start = int(rng.choice(candidates))
        for j in chosen:
            masks[i, indices[start:start + width], j] = False
        valid[i], starts[i] = True, start
        centers[i] = (start + width / 2) / length
    removed = observed.sum(axis=1) - masks.sum(axis=1)
    manifest = dict(valid=valid, start=starts, width=widths, requested_rate=requested,
                    center=centers, mode_bits=selected_bits, feasible_starts=choices_count,
                    removed=removed, length=eligible.sum(1), original_count=observed.sum(1))
    return masks, manifest


def decode(reg, tau):
    intensity = np.clip(np.asarray(reg), -3, 3).copy()
    intensity[np.abs(intensity) <= tau] = 0
    return intensity, (np.sign(intensity).astype(np.int64) + 1)


def batch_metrics(y, classes, predictions, predicted_classes, valid):
    """Vectorized metrics per condition, never treating mask repetitions as new IDs."""
    pred = np.atleast_2d(predictions)
    cls = np.atleast_2d(predicted_classes)
    mask = np.broadcast_to(valid, pred.shape).astype(np.float64)
    target = np.broadcast_to(y, pred.shape)
    labels = np.broadcast_to(classes, pred.shape)
    n = mask.sum(1)
    denom = np.maximum(n, 1)
    acc = ((cls == labels) * mask).sum(1) / denom
    f1 = np.zeros(len(pred))
    for c in range(3):
        tp = ((cls == c) & (labels == c)) * mask
        totals = (((cls == c).astype(int) + (labels == c).astype(int)) * mask).sum(1)
        f1 += np.divide(2 * tp.sum(1), totals, out=np.zeros(len(pred)), where=totals > 0) / 3
    mae = (np.abs(pred - target) * mask).sum(1) / denom
    pm = (pred * mask).sum(1) / denom
    ym = (target * mask).sum(1) / denom
    covariance = ((pred - pm[:, None]) * (target - ym[:, None]) * mask).sum(1)
    variance = np.sqrt((((pred - pm[:, None]) ** 2) * mask).sum(1) *
                       (((target - ym[:, None]) ** 2) * mask).sum(1))
    pearson = np.divide(covariance, variance, out=np.full(len(pred), np.nan), where=variance > 1e-10)
    result = np.stack([acc, f1, mae, pearson], axis=1)
    result[n == 0] = np.nan
    return result


def select_tau(y, cls, raw, valid):
    best = None
    for tau in THRESHOLDS:
        reg, classes = decode(raw, tau)
        stats = batch_metrics(y, cls, reg, classes, valid)
        if np.isnan(stats[:, :3]).any():
            raise ValueError("Empty selection condition")
        score = stats[0, 2] + stats[1:, 2].mean() + .6 * (2 - stats[0, 1] - stats[1:, 1].mean())
        if best is None or score < best[0] - 1e-12:
            best = float(score), float(tau), stats
    return best


class EncoderCache:
    def __init__(self, root, norms, device, weight_path, batch_size=128):
        self.root, self.norms, self.device = root, norms, device
        self.batch_size = batch_size
        self.bert = BertModel.from_pretrained(str(weight_path.parent), local_files_only=True).to(device).eval()

    def encode(self, split, masks):
        parts = []
        n = len(masks)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            bert_inputs = np.asarray(split["text_bert"][start:stop], dtype=np.int64).copy()
            eligible, _ = base.masks_from_raw({k: np.asarray(split[k][start:stop]) for k in ("text_bert", "audio", "vision")})
            bert_inputs[:, 0][eligible & ~masks[start:stop, :, 0]] = 100
            with torch.inference_mode():
                text = self.bert(input_ids=torch.as_tensor(bert_inputs[:, 0], device=self.device),
                                 attention_mask=torch.as_tensor(bert_inputs[:, 1], device=self.device),
                                 token_type_ids=torch.as_tensor(bert_inputs[:, 2], device=self.device)).last_hidden_state.cpu().numpy()
            x = {"text": text}
            for mode in ("audio", "vision"):
                x[mode] = np.clip((np.asarray(split[mode][start:stop], dtype=np.float32) - self.norms[mode]["mean"]) /
                                  self.norms[mode]["scale"], -10, 10)
            parts.append(base.pool_features(dict(x=x, eligible=eligible, masks=masks[start:stop])))
        return base.combine_features(parts)

    def get(self, name, split, masks, manifest=None):
        path = self.root / (name + ".npz")
        if path.exists():
            with np.load(path) as saved:
                return {k: saved[k] for k in FEATURE_KEYS}
        start = time.monotonic()
        features = self.encode(split, masks)
        save_npz(path, **features)
        if manifest is not None:
            save_npz(self.root / (name + "_mask.npz"), masks=masks, **manifest)
        log(f"cache {name}: n={len(masks)}, {time.monotonic()-start:.1f}s")
        return features


def audit(data, output, source):
    record = {"source_sha256": base.sha256(source), "splits": {},
              "unk_policy": "100 treated as unavailable; no original UNK in train/valid if counts are zero; attachment3 encoding remains an explicit assumption",
              "time_unit": "aligned content positions, no seconds conversion",
              "test_split_used": False}
    groups = {}
    for name in ("train", "valid"):
        split = data[name]
        eligible, masks = base.masks_from_raw(split)
        ids = list(map(str, split["id"]))
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs")
        if np.any(np.abs(split["regression_labels"]) > 3):
            raise ValueError("Out of range labels")
        g, protocol = group_ids(ids)
        groups[name] = set(g)
        record["splits"][name] = dict(n=len(ids), groups=len(set(g)), group_protocol=protocol,
            unk_tokens=int((np.asarray(split["text_bert"])[:, 0] == 100).sum()),
            fully_missing_by_modality=(masks.sum(1) == 0).sum(0).tolist(),
            class_counts=np.bincount(split["classification_labels"], minlength=3).tolist(),
            content_length_min=int(eligible.sum(1).min()), content_length_max=int(eligible.sum(1).max()))
    record["train_valid_shared_video_groups"] = len(groups["train"] & groups["valid"])
    record["unk_sensitivity"] = "Treating original UNK as observed vs missing produces identical train/valid masks when original UNK count is zero; no attachment3-based policy tuning."
    json_write(output / "data_audit.json", record)
    log("audit " + json.dumps(record["splits"]))
    return record


def grid_cache(cache, split, seed):
    eligible, observed = base.masks_from_raw(split)
    clean = cache.get("valid_clean", split, observed)
    features, manifests = [clean], []
    for index, (modes, position, rate) in enumerate(GRID):
        name = f"valid_{seed}_{index:02d}"
        masks, manifest = span_masks(observed, eligible, seed + 1009 * index, modes, rate, position)
        features.append(cache.get(name, split, masks, manifest))
        manifests.append(manifest)
    valid = np.vstack([np.ones(len(observed), bool)] + [m["valid"] for m in manifests])
    return base.combine_features(features), valid, manifests


def gpu_predict(model, features, text_only=False, batch=2048):
    model.eval()
    regs, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(features["text"]), batch):
            inputs = [features[k][start:start + batch] for k in FEATURE_KEYS]
            if text_only:
                inputs = [inputs[0], torch.zeros_like(inputs[1]), torch.zeros_like(inputs[2]), inputs[3].clone()]
                inputs[3][:, 1:] = 0
            r, c = model(*inputs)
            regs.append(r.cpu().numpy())
            logits.append(c.cpu().numpy())
    return np.concatenate(regs), np.concatenate(logits)


def to_device(features, device):
    return {key: torch.as_tensor(value, device=device) for key, value in features.items()}


def train_one(variant, seed, features, data, selection, valid_mask, output, epochs, device, smoke=False):
    path = output / "checkpoints" / f"{variant}_{seed}.pt"
    meta_path = path.with_suffix(".json")
    selection_path = path.with_name(path.stem + "_selection.npz")
    if path.exists() and meta_path.exists() and selection_path.exists():
        log(f"resume complete {variant}/{seed}")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    augmented, gated, text_only = VARIANTS[variant]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = base.RobustFusion(learned_gate=gated).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    y = torch.as_tensor(np.tile(data["train"]["regression_labels"], 4), dtype=torch.float32, device=device)
    cls = torch.as_tensor(np.tile(data["train"]["classification_labels"], 4), dtype=torch.long, device=device)
    counts = np.bincount(data["train"]["classification_labels"], minlength=3)
    weights = torch.as_tensor(np.sqrt(counts.sum() / (3 * counts)), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best, state, stale, history = float("inf"), None, 0, []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / 128))):
            idx = torch.as_tensor(indices, device=device)
            # Clean arms repeat original views to match augmented-arm update budgets.
            feature_idx = idx if augmented else idx % len(data["train"]["id"])
            inputs = [features[k].index_select(0, feature_idx) for k in FEATURE_KEYS]
            if text_only:
                inputs = [inputs[0], torch.zeros_like(inputs[1]), torch.zeros_like(inputs[2]), inputs[3].clone()]
                inputs[3][:, 1:] = 0
            reg, logits = model(*inputs)
            loss = nn.functional.smooth_l1_loss(reg, y[idx], beta=.5)
            loss += .7 * nn.functional.cross_entropy(logits, cls[idx], weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        raw, _ = gpu_predict(model, selection, text_only)
        raw = raw.reshape(valid_mask.shape)
        score, tau, stats = select_tau(data["valid"]["regression_labels"], data["valid"]["classification_labels"], raw, valid_mask)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), score=score, tau=tau))
        log(f"train {variant}/{seed} epoch={epoch} score={score:.5f} tau={tau:.2f}")
        if score < best - 1e-4:
            best, best_tau, best_epoch, stale = score, tau, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    info = dict(variant=variant, seed=seed, selection_score=best, tau=best_tau, best_epoch=best_epoch,
                epochs_run=epoch, steps_per_epoch=int(np.ceil(len(y) / 128)),
                total_updates=int(epoch * np.ceil(len(y) / 128)), seconds=time.monotonic() - started,
                history=history, smoke=smoke)
    model.load_state_dict(state)
    raw, logits = gpu_predict(model, selection, text_only)
    save_npz(selection_path, raw=raw.reshape(valid_mask.shape), logits=logits.reshape((*valid_mask.shape, 3)), valid=valid_mask)
    torch.save(dict(state_dict=state, **info), path)
    json_write(meta_path, info)
    return info


def safe_stats(values):
    return {key: (float(value) if np.isfinite(value) else None) for key, value in zip(METRICS, values)}


def evaluate(cache, data, output, records, device, mask_seeds):
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    y, cls = data["valid"]["regression_labels"], data["valid"]["classification_labels"]
    for mask_seed in mask_seeds:
        if all((predictions / f"{r['variant']}_{r['seed']}_{mask_seed}.npz").exists() for r in records):
            log(f"resume evaluation {mask_seed}")
            continue
        features, valid, manifests = grid_cache(cache, data["valid"], mask_seed)
        tensors = to_device(features, device)
        save_npz(predictions / f"mask_{mask_seed}.npz", valid=valid,
                 **{key: np.stack([m[key] for m in manifests]) for key in manifests[0] if key != "valid"})
        del features
        for record in records:
            variant, seed, tau = record["variant"], record["seed"], record["tau"]
            path = predictions / f"{variant}_{seed}_{mask_seed}.npz"
            if path.exists():
                continue
            checkpoint = torch.load(output / "checkpoints" / f"{variant}_{seed}.pt", map_location="cpu", weights_only=False)
            model = base.RobustFusion(learned_gate=VARIANTS[variant][1]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            raw, logits = gpu_predict(model, tensors, VARIANTS[variant][2])
            raw, logits = raw.reshape(valid.shape), logits.reshape((*valid.shape, 3))
            reg, classes = decode(raw, tau)
            stats = batch_metrics(y, cls, reg, classes, valid)
            save_npz(path, raw=raw, logits=logits, reg=reg, classes=classes.astype(np.int8), stats=stats)
        del tensors
        log(f"diagnostic mask seed {mask_seed} complete")


def final_inference(cache, args, norms, records, audit_record, config_hash):
    by_variant = {v: np.mean([r["selection_score"] for r in records if r["variant"] == v]) for v in VARIANTS}
    selected = min((v for v in VARIANTS if v.startswith("M")), key=lambda v: by_variant[v])
    chosen = next(r for r in records if r["variant"] == selected and r["seed"] == 2026)
    checkpoint = torch.load(args.output / "checkpoints" / f"{selected}_2026.pt", map_location="cpu", weights_only=False)
    weight_path = Path(hf_hub_download(base.MODEL_ID, "model.safetensors", revision=base.MODEL_REVISION, local_files_only=True))
    frozen = dict(selected_variant=selected, final_seed=2026, tau=chosen["tau"], selection_means=by_variant,
                  config_sha256=config_hash, source_sha256=audit_record["source_sha256"],
                  bert_model=base.MODEL_ID, bert_revision=base.MODEL_REVISION, bert_weight_sha256=base.sha256(weight_path),
                  note="Frozen before reading Attachment3; selection uses validation only")
    json_write(args.output / "frozen_model.json", frozen)
    checkpoint.update(normalizers=norms, frozen=frozen)
    torch.save(checkpoint, args.output / "robust_fusion_v2.pt")
    model = base.RobustFusion(learned_gate=VARIANTS[selected][1]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    paths = sorted((args.data_root / "附件3-模态缺失特征样本" / "对齐版本").glob("附件3_*.pkl"))
    if len(paths) != 30:
        raise ValueError("Expected 30 Attachment3 samples")
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)["test"]
        for key, shape in (("text_bert", (1, 3, 50)), ("audio", (1, 50, 74)), ("vision", (1, 50, 35))):
            if np.shape(sample[key]) != shape or not np.isfinite(sample[key]).all():
                raise ValueError(f"Invalid {path.name}/{key}")
        samples.append(sample)
    split = {k: np.concatenate([s[k] for s in samples]) for k in ("text_bert", "audio", "vision")}
    _, masks = base.masks_from_raw(split)
    features = cache.encode(split, masks)
    raw, _ = base.predict(model, features)
    reg, cls = decode(raw, chosen["tau"])
    if not np.isfinite(reg).all() or not np.all(cls == np.sign(reg).astype(int) + 1):
        raise ValueError("Invalid decoded output")
    rows = [dict(sample_id=p.stem, pred_polarity=base.CLASS_NAMES[int(cls[i])], pred_intensity=float(reg[i])) for i, p in enumerate(paths)]
    base.write_csv(args.output / "attachment3_predictions_v2.csv", list(rows[0]), rows)
    coverage = [dict(sample_id=p.stem, **{m + "_coverage": float(features["coverage"][i, j]) for j, m in enumerate(MODES)}) for i, p in enumerate(paths)]
    base.write_csv(args.output / "attachment3_coverage.csv", list(coverage[0]), coverage)
    json_write(args.output / "attachment3_summary.json", dict(n=30, unique_ids=len(set(p.stem for p in paths)),
        predicted_counts={name: int((cls == i).sum()) for i, name in enumerate(base.CLASS_NAMES)},
        mean_intensity=float(reg.mean()), min_intensity=float(reg.min()), max_intensity=float(reg.max()),
        sha256={p.name: base.sha256(p) for p in paths}, unlabeled=True, **frozen))
    log(f"frozen {selected}, final inference 30/30 complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = args.data_root / "附件2-数据集特征文件" / "aligned_50.pkl"
    data = base.load_source(source)
    # Do not retain or process the official test split.
    data = {name: data[name] for name in ("train", "valid")}
    for split in data.values():
        split["classification_labels"] = np.asarray(split["classification_labels"], dtype=np.int64)
        split["regression_labels"] = np.asarray(split["regression_labels"], dtype=np.float32)
    audit_record = audit(data, args.output, source)
    if args.audit_only:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seeds = [2026] if args.smoke else list(range(2026, 2031))
    epochs = 2 if args.smoke else 30
    config = dict(schema=SCHEMA, seeds=seeds, epochs=epochs, patience=6, smoke=args.smoke,
        variants=VARIANTS, selection_mask_seed=4101, diagnostic_mask_seeds=[5101,5102,5103,5104,5105],
        thresholds=THRESHOLDS.tolist(), bert_model=base.MODEL_ID, bert_revision=base.MODEL_REVISION,
        augmentation_views=3, batch_size=128, learning_rate=.001, weight_decay=.0001,
        source_sha256=audit_record["source_sha256"], code_sha256=base.sha256(Path(__file__)),
        base_code_sha256=base.sha256(Path(base.__file__)), train_block_fraction_range=[.1,.7],
        grid=[dict(modes=list(m), position=p, rate=r) for m,p,r in GRID],
        mask_rule="contiguous, positive observed removal in every selected modality, at least one original observation retained; infeasible is flagged, not shortened",
        loss="SmoothL1(beta=.5)+.7*weighted_CE", decoding="regression threshold; neutral intensity zero")
    config_path = args.output / "config_v2.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
        raise ValueError("Output exists with a different configuration/code; use a new output directory")
    json_write(config_path, config)
    (args.output / "environment_server.txt").write_text(
        f"Python {platform.python_version()}\nPlatform {platform.platform()}\nTorch {torch.__version__}\nCUDA {torch.version.cuda}\n" +
        (f"GPU {torch.cuda.get_device_name(0)}\n" if torch.cuda.is_available() else "CPU\n"), encoding="utf-8")
    for folder in ("cache", "checkpoints", "predictions"):
        (args.output / folder).mkdir(exist_ok=True)
    norms = base.fit_normalizers(data["train"])
    weight_path = Path(hf_hub_download(base.MODEL_ID, "model.safetensors", revision=base.MODEL_REVISION, local_files_only=True))
    cache = EncoderCache(args.output / "cache", norms, args.device, weight_path)
    eligible, observed = base.masks_from_raw(data["train"])
    if args.smoke:
        for split in data:
            data[split] = {k: (v[:64] if hasattr(v, "__len__") else v) for k,v in data[split].items()}
        eligible, observed = base.masks_from_raw(data["train"])
    # Verify deterministic full-input re-encoding against the supplied BERT features.
    small = {k: np.asarray(v[:3]) for k,v in data["train"].items()}
    check = cache.encode(small, observed[:3])
    reference = base.pool_features(base.prepare(small, norms))
    difference = float(np.abs(check["text"] - reference["text"]).mean())
    json_write(args.output / "bert_reencoding_check.json", dict(mean_absolute_pooled_difference=difference, n=3))
    if difference > .005:
        raise ValueError("BERT re-encoding differs substantially from source")
    clean = cache.get("train_clean", data["train"], observed)
    selection_features, selection_mask, _ = grid_cache(cache, data["valid"], 4101)
    selection = to_device(selection_features, args.device)
    del selection_features
    records = []
    for seed in seeds:
        views = [clean]
        for index in range(3):
            masks, manifest = span_masks(observed, eligible, seed * 10 + index)
            views.append(cache.get(f"train_{seed}_{index}", data["train"], masks, manifest))
        features = to_device(base.combine_features(views), args.device)
        for variant in VARIANTS:
            records.append(train_one(variant, seed, features, data, selection, selection_mask,
                                     args.output, epochs, args.device, args.smoke))
        del features, views
        json_write(args.output / "training_records.json", records)
    del selection
    if args.smoke:
        json_write(args.output / "SMOKE_COMPLETE.json", dict(runs=len(records), passed=True))
        log("SMOKE complete; no Attachment3 access")
        return
    evaluate(cache, data, args.output, records, args.device, [5101,5102,5103,5104,5105])
    save_npz(args.output / "validation_targets.npz", y=data["valid"]["regression_labels"],
             classes=data["valid"]["classification_labels"], ids=np.asarray(data["valid"]["id"]),
             groups=group_ids(data["valid"]["id"])[0], raw_text=data["valid"]["raw_text"])
    final_inference(cache, args, norms, records, audit_record, base.sha256(config_path))
    json_write(args.output / "TRAINING_AND_INFERENCE_COMPLETE.json", dict(runs=len(records), predictions=150, status="complete"))


if __name__ == "__main__":
    main()
