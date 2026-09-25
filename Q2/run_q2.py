"""Train a local-missingness-robust sentiment model and predict Attachment 3.

Example: python Q2/run_q2.py --data-root ../E题数据 --output Q2/outputs/local
Only aligned_50.pkl train/valid labels are used. Attachment 3 is inference only.
"""

import argparse
import copy
import csv
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from scipy.stats import pearsonr
import torch
from torch import nn
from transformers import BertModel


SEED = 2026
MODES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path):
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not {"train", "valid"}.issubset(data):
        raise ValueError("aligned_50.pkl requires train and valid splits")
    for split_name in ("train", "valid"):
        split = data[split_name]
        required = {"id", "text", "text_bert", "audio", "vision", "classification_labels", "regression_labels"}
        if not required.issubset(split):
            raise ValueError(f"{split_name} missing {required - set(split)}")
        n = len(split["id"])
        shapes = {"text": (n, 50, 768), "text_bert": (n, 3, 50), "audio": (n, 50, 74), "vision": (n, 50, 35)}
        for key, expected in shapes.items():
            if np.shape(split[key]) != expected or not np.isfinite(split[key]).all():
                raise ValueError(f"{split_name}/{key} has invalid shape or values")
        y = np.asarray(split["regression_labels"], dtype=np.float32)
        labels = np.asarray(split["classification_labels"], dtype=np.int64)
        if y.shape != (n,) or labels.shape != (n,) or not np.isfinite(y).all():
            raise ValueError(f"{split_name} label count or values invalid")
        if not np.array_equal(labels, np.sign(y).astype(np.int64) + 1):
            raise ValueError(f"{split_name} polarity disagrees with intensity sign")
    if set(data["train"]["id"]) & set(data["valid"]["id"]):
        raise ValueError("train and valid IDs overlap")
    return data


def masks_from_raw(split):
    bert = np.asarray(split["text_bert"])
    token_ids = bert[:, 0, :].astype(np.int64)
    attn = bert[:, 1, :] > 0
    eligible = attn & ~np.isin(token_ids, [0, 101, 102])
    masks = np.stack(
        [eligible & (token_ids != 100),
         eligible & np.any(np.asarray(split["audio"]) != 0, axis=-1),
         eligible & np.any(np.asarray(split["vision"]) != 0, axis=-1)],
        axis=-1,
    )
    if np.any(eligible.sum(axis=1) == 0):
        raise ValueError("Sample without content positions")
    return eligible, masks


def fit_normalizers(train):
    eligible, masks = masks_from_raw(train)
    norms = {}
    for j, mode in enumerate(("audio", "vision"), 1):
        observed = np.asarray(train[mode], dtype=np.float32)[masks[:, :, j]]
        if not len(observed):
            raise ValueError(f"No observed {mode} rows")
        mean = observed.mean(axis=0)
        scale = observed.std(axis=0)
        scale = np.maximum(scale, 1e-3)
        norms[mode] = {"mean": mean, "scale": scale}
    return norms


def prepare(split, norms, include_labels=True):
    eligible, masks = masks_from_raw(split)
    x = {"text": np.asarray(split["text"], dtype=np.float32)}
    for mode in ("audio", "vision"):
        raw = np.asarray(split[mode], dtype=np.float32)
        normalized = np.clip((raw - norms[mode]["mean"]) / norms[mode]["scale"], -10, 10)
        x[mode] = normalized
    result = {"x": x, "eligible": eligible, "masks": masks}
    if include_labels:
        result.update({
            "id": list(split["id"]),
            "raw_text": np.asarray(split["raw_text"]),
            "y_reg": np.asarray(split["regression_labels"], dtype=np.float32),
            "y_cls": np.asarray(split["classification_labels"], dtype=np.int64),
        })
    return result


def augmented_masks(base, eligible, seed, modes=None, rate=None, position=None):
    rng = np.random.default_rng(seed)
    masks = base.copy()
    choices = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    for i in range(len(masks)):
        indices = np.flatnonzero(eligible[i])
        if len(indices) < 2:
            continue
        selected = tuple(modes) if modes is not None else choices[rng.integers(len(choices))]
        fraction = float(rate) if rate is not None else float(rng.uniform(0.1, 0.7))
        width = min(len(indices) - 1, max(1, int(round(fraction * len(indices)))))
        location = position or ("early", "middle", "late", "random")[rng.integers(4)]
        max_start = len(indices) - width
        starts = {"early": 0, "middle": max_start // 2, "late": max_start}
        start = starts[location] if location in starts else int(rng.integers(max_start + 1))
        for modality in selected:
            masks[i, indices[start:start + width], modality] = False
    return masks


def pool_features(prepared, masks=None):
    masks = prepared["masks"] if masks is None else masks
    eligible = prepared["eligible"]
    n, length = eligible.shape
    rank = np.cumsum(eligible, axis=1) - 1
    count = np.maximum(eligible.sum(axis=1, keepdims=True), 1)
    bin_id = np.minimum(2, 3 * rank // count)
    result = {}
    coverage = np.zeros((n, 3), dtype=np.float32)
    for j, mode in enumerate(MODES):
        values = prepared["x"][mode]
        observed = masks[:, :, j]
        coverage[:, j] = observed.sum(axis=1) / count[:, 0]
        weights = observed.astype(np.float32)
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1)
        mean = np.einsum("ntd,nt->nd", values, weights) / denom
        second = np.einsum("ntd,nt->nd", values * values, weights) / denom
        std = np.sqrt(np.maximum(second - mean * mean, 0))
        parts = [mean, std]
        for k in range(3):
            bin_weights = (observed & (bin_id == k)).astype(np.float32)
            bin_denom = np.maximum(bin_weights.sum(axis=1, keepdims=True), 1)
            parts.append(np.einsum("ntd,nt->nd", values, bin_weights) / bin_denom)
        result[mode] = np.concatenate(parts, axis=1).astype(np.float32)
    result["coverage"] = coverage
    return result


def combine_features(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in (*MODES, "coverage")}


def text_only(features):
    reduced = {key: value.copy() for key, value in features.items()}
    for mode in ("audio", "vision"):
        reduced[mode].fill(0)
    reduced["coverage"][:, 1:] = 0
    return reduced


class RobustFusion(nn.Module):
    def __init__(self, learned_gate=True):
        super().__init__()
        self.learned_gate = learned_gate
        self.encoders = nn.ModuleDict({
            mode: nn.Sequential(nn.Linear(5 * dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.15))
            for mode, dim in (("text", 768), ("audio", 74), ("vision", 35))
        })
        self.gate = nn.Linear(64 * 3 + 3, 3)
        self.fusion = nn.Sequential(nn.Linear(64 * 4 + 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        self.regression = nn.Linear(128, 1)
        self.classification = nn.Linear(128, 3)

    def forward(self, text, audio, vision, coverage):
        vectors = [self.encoders[name](value) for name, value in zip(MODES, (text, audio, vision))]
        stacked = torch.stack(vectors, dim=1)
        if self.learned_gate:
            gate_logits = self.gate(torch.cat(vectors + [coverage], dim=1))
            gate_logits = gate_logits + torch.log(coverage.clamp_min(1e-5))
            weights = torch.softmax(gate_logits.masked_fill(coverage <= 0, -1e4), dim=1)
        else:
            weights = coverage / coverage.sum(dim=1, keepdim=True).clamp_min(1e-5)
        weighted = stacked * weights.unsqueeze(-1)
        fused = self.fusion(torch.cat([weighted.flatten(1), weighted.sum(dim=1), coverage], dim=1))
        return 3 * torch.tanh(self.regression(fused).squeeze(-1)), self.classification(fused)


def predict(model, features, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    predictions, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(features["text"]), batch_size):
            end = start + batch_size
            inputs = [torch.from_numpy(features[key][start:end]).to(device) for key in (*MODES, "coverage")]
            reg, cls = model(*inputs)
            predictions.append(reg.cpu().numpy())
            logits.append(cls.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(logits)


def metrics(y_reg, y_cls, reg, logits, neutral_bias=0.0):
    adjusted = logits.copy()
    adjusted[:, 1] += neutral_bias
    classes = adjusted.argmax(axis=1)
    correlation = float(pearsonr(y_reg, reg).statistic) if np.std(reg) > 0 else 0.0
    return {
        "n": len(y_reg),
        "accuracy": float(accuracy_score(y_cls, classes)),
        "macro_f1": float(f1_score(y_cls, classes, labels=[0, 1, 2], average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_reg, reg)),
        "pearson_r": correlation,
    }


def choose_neutral_bias(y_cls, logits):
    options = np.round(np.arange(-0.6, 1.21, 0.1), 2)
    return float(max(options, key=lambda bias: (
        f1_score(y_cls, (logits + np.array([0, bias, 0])).argmax(1), average="macro", zero_division=0),
        accuracy_score(y_cls, (logits + np.array([0, bias, 0])).argmax(1)),
        -abs(bias),
    )))


def fit_variant(train_features, y_reg, y_cls, valid_features, valid_stress, valid_reg, valid_cls,
                learned_gate, seed, device, epochs=30, patience=6):
    torch.manual_seed(seed)
    model = RobustFusion(learned_gate=learned_gate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(y_cls, minlength=3)
    class_weights = torch.as_tensor(np.sqrt(counts.sum() / (3 * counts)), dtype=torch.float32, device=device)
    features = {key: torch.from_numpy(value).to(device) for key, value in train_features.items()}
    targets_reg = torch.from_numpy(y_reg).to(device)
    targets_cls = torch.from_numpy(y_cls).to(device)
    rng = np.random.default_rng(seed)
    best_score, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in np.array_split(rng.permutation(len(y_reg)), max(1, int(np.ceil(len(y_reg) / 128)))):
            batch_indices = torch.as_tensor(indices, dtype=torch.long, device=device)
            reg, cls = model(*(features[key].index_select(0, batch_indices) for key in (*MODES, "coverage")))
            loss = nn.functional.smooth_l1_loss(reg, targets_reg.index_select(0, batch_indices), beta=0.5)
            loss = loss + 0.7 * nn.functional.cross_entropy(
                cls, targets_cls.index_select(0, batch_indices), weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        clean_reg, clean_logits = predict(model, valid_features)
        stress_reg, stress_logits = predict(model, valid_stress)
        clean = metrics(valid_reg, valid_cls, clean_reg, clean_logits)
        stress = metrics(valid_reg, valid_cls, stress_reg, stress_logits)
        score = clean["mae"] + stress["mae"] + 0.6 * (2 - clean["macro_f1"] - stress["macro_f1"])
        if score < best_score - 1e-4:
            best_score, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "selection_score": best_score}


def encode_attachment3(data_dir, norms, device):
    paths = sorted(data_dir.glob("附件3_*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 aligned Attachment 3 files, got {len(paths)}")
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if set(payload) != {"test"} or set(payload["test"]) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"Unexpected Attachment 3 keys: {path}")
        sample = payload["test"]
        if np.shape(sample["text_bert"]) != (1, 3, 50) or np.shape(sample["audio"]) != (1, 50, 74) or np.shape(sample["vision"]) != (1, 50, 35):
            raise ValueError(f"Unexpected Attachment 3 shape: {path}")
        if any(not np.isfinite(sample[key]).all() for key in ("text_bert", "audio", "vision")):
            raise ValueError(f"Non-finite Attachment 3 values: {path}")
        samples.append(sample)
    bert_inputs = np.concatenate([s["text_bert"] for s in samples], axis=0).astype(np.int64)
    if bert_inputs.min() < 0 or bert_inputs[:, 0, :].max() >= 30522:
        raise ValueError("Invalid BERT token IDs")
    model_path = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION,
                                      local_files_only=True)).parent
    bert = BertModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    bert_outputs = []
    with torch.inference_mode():
        for start in range(0, len(samples), 8):
            batch = bert_inputs[start:start + 8]
            encoded = bert(
                input_ids=torch.from_numpy(batch[:, 0, :]).to(device),
                attention_mask=torch.from_numpy(batch[:, 1, :]).to(device),
                token_type_ids=torch.from_numpy(batch[:, 2, :]).to(device),
            ).last_hidden_state
            bert_outputs.append(encoded.cpu().numpy())
    sample_split = {
        "text_bert": bert_inputs,
        "text": np.concatenate(bert_outputs),
        "audio": np.concatenate([s["audio"] for s in samples]),
        "vision": np.concatenate([s["vision"] for s in samples]),
    }
    return paths, prepare(sample_split, norms, include_labels=False)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_attachment3(data_dir, output, norms, model, neutral_bias, device):
    paths, test = encode_attachment3(data_dir, norms, device)
    features = pool_features(test)
    reg, logits = predict(model, features)
    logits[:, 1] += neutral_bias
    classes = logits.argmax(axis=1)
    predictions = [
        {"sample_id": path.stem, "pred_polarity": CLASS_NAMES[classes[i]],
         "pred_intensity": round(float(reg[i]), 6)}
        for i, path in enumerate(paths)
    ]
    write_csv(output / "attachment3_predictions.csv",
              ["sample_id", "pred_polarity", "pred_intensity"], predictions)
    diagnostics = [
        {"sample_id": path.stem, "text_coverage": float(features["coverage"][i, 0]),
         "audio_coverage": float(features["coverage"][i, 1]),
         "vision_coverage": float(features["coverage"][i, 2])}
        for i, path in enumerate(paths)
    ]
    write_csv(output / "attachment3_coverage.csv", list(diagnostics[0]), diagnostics)
    counts = {name: int(np.sum(classes == i)) for i, name in enumerate(CLASS_NAMES)}
    summary = {"n": len(paths), "predicted_polarity_counts": counts,
               "mean_predicted_intensity": float(reg.mean()),
               "min_predicted_intensity": float(reg.min()),
               "max_predicted_intensity": float(reg.max()),
               "note": "Unlabeled inference; these are prediction summaries, not test accuracy."}
    (output / "attachment3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(CLASS_NAMES, [counts[name] for name in CLASS_NAMES], color=["#E45756", "#72B7B2", "#4C78A8"])
    axes[0].set(ylabel="Predicted sample count", title="Attachment 3 polarity")
    axes[1].scatter(np.arange(1, len(reg) + 1), reg, color="#4C78A8", s=28)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Attachment 3 file number", ylabel="Predicted intensity", ylim=(-3.1, 3.1))
    fig.tight_layout()
    fig.savefig(output / "attachment3_overview.png", dpi=200)
    plt.close(fig)
    return paths


def plot_validation(output, true, pred, classes):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    matrix = confusion_matrix(true["cls"], classes, labels=[0, 1, 2])
    image = axes[0].imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        axes[0].text(j, i, str(count), ha="center", va="center")
    axes[0].set_xticks(range(3), CLASS_NAMES, rotation=25)
    axes[0].set_yticks(range(3), CLASS_NAMES)
    axes[0].set_xlabel("Predicted polarity")
    axes[0].set_ylabel("True polarity")
    fig.colorbar(image, ax=axes[0], shrink=0.8)
    axes[1].scatter(true["reg"], pred, s=9, alpha=0.35)
    axes[1].plot([-3, 3], [-3, 3], color="black", linewidth=1)
    axes[1].set(xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    fig.tight_layout()
    fig.savefig(output / "validation_overview.png", dpi=200)
    plt.close(fig)


def plot_missingness(output, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"text": "#4C78A8", "audio": "#F58518", "vision": "#54A24B"}
    for mode in MODES:
        for metric, axis in (("mae", axes[0]), ("macro_f1", axes[1])):
            points = [row for row in rows if row["missing_mode"] == mode and row["position"] == "middle"]
            axis.plot([row["missing_rate"] for row in points], [row[metric] for row in points],
                      marker="o", label=mode, color=colors[mode])
    axes[0].set(xlabel="Missing span fraction", ylabel="MAE")
    axes[1].set(xlabel="Missing span fraction", ylabel="Macro F1")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "missing_rate_effect.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    for k, position in enumerate(("early", "middle", "late")):
        values = [next(row["mae"] for row in rows if row["missing_mode"] == mode
                       and row["position"] == position and row["missing_rate"] == 0.35)
                  for mode in MODES]
        axis.bar(x + (k - 1) * 0.24, values, width=0.23, label=position)
    axis.set_xticks(x, MODES)
    axis.set(xlabel="Missing modality", ylabel="MAE at 35% missing span")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "missing_position_effect.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--epochs", type=int, help="Optional override of config max_epochs")
    parser.add_argument("--checkpoint", type=Path, help="Reuse trained parameters for Attachment 3 prediction only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed_config = {
        "schema_version": "q2-local-missing-1", "data_version": "aligned_50", "seed": SEED,
        "bert_model": MODEL_ID, "bert_revision": MODEL_REVISION,
        "patience": 6, "batch_size": 128, "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "SmoothL1(beta=0.5) + 0.7*weighted_cross_entropy",
        "train_augmentation_copies": 3, "train_block_fraction_range": [0.1, 0.7],
        "validation_grid_block_fractions": [0.15, 0.35, 0.55],
    }
    if any(config.get(key) != value for key, value in fixed_config.items()):
        raise ValueError("Config differs from the fixed model implementation")
    epochs = args.epochs if args.epochs is not None else int(config["max_epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}", flush=True)
    torch.set_num_threads(4)
    np.random.seed(SEED)
    if not args.output.is_dir():
        raise FileNotFoundError(f"Output directory must already exist: {args.output}")
    source = args.data_root / "附件2-数据集特征文件" / "aligned_50.pkl"
    attachment3 = args.data_root / "附件3-模态缺失特征样本" / "对齐版本"
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_id"] != MODEL_ID or checkpoint["model_revision"] != MODEL_REVISION:
            raise ValueError("Checkpoint BERT identity does not match this script")
        weight_path = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION,
                                      local_files_only=True)
        if checkpoint["bert_weight_sha256"] != sha256(weight_path):
            raise ValueError("Cached BERT weight hash differs from trained checkpoint")
        model = RobustFusion(learned_gate=checkpoint["learned_gate"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        predict_attachment3(attachment3, args.output, checkpoint["normalizers"], model,
                            checkpoint["neutral_bias"], device)
        return
    data = load_source(source)
    norms = fit_normalizers(data["train"])
    train = prepare(data["train"], norms)
    valid = prepare(data["valid"], norms)
    clean_train = pool_features(train)
    clean_valid = pool_features(valid)
    augmented = [clean_train] + [pool_features(train, augmented_masks(train["masks"], train["eligible"], SEED + i)) for i in range(1, 4)]
    robust_train = combine_features(augmented)
    robust_y_reg = np.tile(train["y_reg"], len(augmented))
    robust_y_cls = np.tile(train["y_cls"], len(augmented))
    stress_valid = pool_features(valid, augmented_masks(valid["masks"], valid["eligible"], SEED + 100, rate=0.45))
    variants = {
        "robust_gate": (robust_train, robust_y_reg, robust_y_cls, clean_valid, stress_valid, True),
        "no_block_augmentation": (clean_train, train["y_reg"], train["y_cls"], clean_valid, stress_valid, True),
        "no_reliability_gate": (robust_train, robust_y_reg, robust_y_cls, clean_valid, stress_valid, False),
        "text_only": (text_only(robust_train), robust_y_reg, robust_y_cls, text_only(clean_valid), text_only(stress_valid), True),
    }
    trained, report = {}, {}
    for index, (name, spec) in enumerate(variants.items()):
        print(f"Training {name}", flush=True)
        fit_features, y_reg, y_cls, eval_clean, eval_stress, gate = spec
        model, fit = fit_variant(fit_features, y_reg, y_cls, eval_clean, eval_stress,
                                 valid["y_reg"], valid["y_cls"], gate, SEED + index, device, epochs=epochs)
        clean_reg, clean_logits = predict(model, eval_clean)
        stress_reg, stress_logits = predict(model, eval_stress)
        neutral_bias = choose_neutral_bias(np.r_[valid["y_cls"], valid["y_cls"]],
                                           np.concatenate([clean_logits, stress_logits]))
        report[name] = {
            **fit, "neutral_bias": neutral_bias,
            "clean": metrics(valid["y_reg"], valid["y_cls"], clean_reg, clean_logits, neutral_bias),
            "stress": metrics(valid["y_reg"], valid["y_cls"], stress_reg, stress_logits, neutral_bias),
        }
        trained[name] = model
        print(name, report[name], flush=True)
    # Select on the validation set only, balancing clean and simulated missingness.
    # Text-only and no-augmentation runs are ablations, not candidates for the final multimodal model.
    selected = min(("robust_gate", "no_reliability_gate"),
                   key=lambda name: report[name]["selection_score"])
    selected_model = trained[selected]
    selected_bias = report[selected]["neutral_bias"]
    model_uses_text_only = selected == "text_only"
    selected_clean = text_only(clean_valid) if model_uses_text_only else clean_valid
    pred_reg, pred_logits = predict(selected_model, selected_clean)
    pred_logits[:, 1] += selected_bias
    pred_cls = pred_logits.argmax(axis=1)

    grid = []
    missing_sets = {"text": (0,), "audio": (1,), "vision": (2,),
                    "text+audio": (0, 1), "text+vision": (0, 2),
                    "audio+vision": (1, 2), "text+audio+vision": (0, 1, 2)}
    for mode, modal_indices in missing_sets.items():
        for position in ("early", "middle", "late"):
            for rate in (0.15, 0.35, 0.55):
                changed = augmented_masks(valid["masks"], valid["eligible"], SEED + 200,
                                          modes=modal_indices, rate=rate, position=position)
                features = pool_features(valid, changed)
                if model_uses_text_only:
                    features = text_only(features)
                reg, logits = predict(selected_model, features)
                removed_steps = np.stack([
                    (valid["masks"][:, :, j] & ~changed[:, :, j]).sum(axis=1)
                    for j in modal_indices
                ], axis=1).mean(axis=1)
                row = {"missing_mode": mode, "position": position, "missing_rate": rate,
                       "mean_removed_steps": float(removed_steps.mean()),
                       **metrics(valid["y_reg"], valid["y_cls"], reg, logits, selected_bias)}
                grid.append(row)
    write_csv(args.output / "missingness_grid.csv",
              ["missing_mode", "position", "missing_rate", "mean_removed_steps", "n", "accuracy", "macro_f1", "mae", "pearson_r"], grid)
    plot_validation(args.output, {"cls": valid["y_cls"], "reg": valid["y_reg"]}, pred_reg, pred_cls)
    plot_missingness(args.output, grid)
    error_rows = []
    for i in np.argsort(np.abs(valid["y_reg"] - pred_reg))[::-1][:30]:
        error_rows.append({"sample_id": valid["id"][i], "true_intensity": float(valid["y_reg"][i]),
                           "pred_intensity": float(pred_reg[i]), "true_polarity": CLASS_NAMES[valid["y_cls"][i]],
                           "pred_polarity": CLASS_NAMES[pred_cls[i]],
                           "absolute_error": float(abs(valid["y_reg"][i] - pred_reg[i])),
                           "raw_text": str(valid["raw_text"][i])})
    write_csv(args.output / "validation_top_errors.csv", list(error_rows[0]), error_rows)
    class_error = {}
    for label, name in enumerate(CLASS_NAMES):
        subset = valid["y_cls"] == label
        class_error[name] = {"n": int(subset.sum()),
                             "mae": float(mean_absolute_error(valid["y_reg"][subset], pred_reg[subset])),
                             "accuracy": float(accuracy_score(valid["y_cls"][subset], pred_cls[subset]))}
    error_summary = {"confusion_matrix": confusion_matrix(valid["y_cls"], pred_cls, labels=[0, 1, 2]).tolist(),
                     "by_true_polarity": class_error,
                     "prediction_sign_disagreement_count": int(np.sum((pred_cls == 0) & (pred_reg > 0)) +
                                                               np.sum((pred_cls == 2) & (pred_reg < 0)))}
    (args.output / "validation_error_analysis.json").write_text(
        json.dumps(error_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    bert_weight_sha256 = sha256(hf_hub_download(MODEL_ID, "model.safetensors",
                                                    revision=MODEL_REVISION, local_files_only=True))
    torch.save({"state_dict": selected_model.state_dict(), "learned_gate": selected_model.learned_gate,
                "neutral_bias": selected_bias, "text_only": model_uses_text_only,
                "normalizers": norms, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "bert_weight_sha256": bert_weight_sha256},
               args.output / "robust_fusion.pt")
    paths = predict_attachment3(attachment3, args.output, norms, selected_model, selected_bias, device)
    run = {
        "seed": SEED, "device": str(device), "torch_version": torch.__version__,
        "data_version": "aligned_50", "train_n": len(train["y_reg"]),
        "valid_n": len(valid["y_reg"]), "attachment3_n": len(paths),
        "selected_variant": selected, "selection_rule": "minimum validation clean+stress MAE + 0.6*(2-clean_macroF1-stress_macroF1) among robust multimodal variants",
        "variants": report, "source_sha256": sha256(source),
        "attachment3_sha256": {path.name: sha256(path) for path in paths},
        "bert_model": MODEL_ID, "bert_revision": MODEL_REVISION,
        "bert_weight_sha256": bert_weight_sha256, "config": config,
        "notes": ["Attachment 2 test split not used", "Attachment 3 labels unavailable and not used",
                  "Synthetic missing text is zeroed in frozen BERT features; Attachment 3 text is re-encoded with [UNK] holes"],
    }
    (args.output / "metrics.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Selected", selected, "wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
