"""Inference only with a trusted frozen Q2 v2 checkpoint; never tunes thresholds."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_q2_v2 as v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frozen = checkpoint["frozen"]
    if (frozen["bert_model"], frozen["bert_revision"]) != (v2.base.MODEL_ID, v2.base.MODEL_REVISION):
        raise ValueError("BERT identity mismatch")
    weight = Path(hf_hub_download(frozen["bert_model"], "model.safetensors", revision=frozen["bert_revision"], local_files_only=True))
    if v2.base.sha256(weight) != frozen["bert_weight_sha256"]:
        raise ValueError("BERT weight hash mismatch")
    model = v2.base.RobustFusion(v2.VARIANTS[frozen["selected_variant"]][1]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    paths = sorted((args.data_root / "附件3-模态缺失特征样本" / "对齐版本").glob("附件3_*.pkl"))
    if len(paths) != 30:
        raise ValueError("Expected exactly 30 input files")
    samples = []
    for path in paths:
        with path.open("rb") as stream:
            sample = pickle.load(stream)["test"]
        for name, shape in (("text_bert", (1, 3, 50)), ("audio", (1, 50, 74)), ("vision", (1, 50, 35))):
            if np.shape(sample[name]) != shape or not np.isfinite(sample[name]).all():
                raise ValueError(f"Invalid input: {path.name}/{name}")
        samples.append(sample)
    split = {key: np.concatenate([s[key] for s in samples]) for key in ("text_bert", "audio", "vision")}
    _, masks = v2.base.masks_from_raw(split)
    cache = v2.EncoderCache(args.output.parent, checkpoint["normalizers"], args.device, weight)
    features = cache.encode(split, masks)
    raw, _ = v2.base.predict(model, features)
    reg, cls = v2.decode(raw, frozen["tau"])
    rows = [dict(sample_id=p.stem, pred_polarity=v2.base.CLASS_NAMES[int(cls[i])], pred_intensity=float(reg[i])) for i,p in enumerate(paths)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    v2.base.write_csv(args.output, list(rows[0]), rows)
    print(f"Predicted {len(rows)} samples with frozen {frozen['selected_variant']}, tau={frozen['tau']}")


if __name__ == "__main__":
    main()
