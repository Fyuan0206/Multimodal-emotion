"""Check Q3 prediction, explanation, evidence, image and provenance artifacts."""
import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    checkpoint = run / selection["final_record"]["checkpoint"]
    model_hash = sha256(checkpoint)
    root = run / "explanation" / "attachment4"
    predictions = rows(run / "attachment4_predictions.csv")
    explanations = rows(root / "explanations.csv")
    combined = rows(root / "attachment4_predictions_explanations.csv")
    evidence = rows(root / "attachment4_evidence.csv")
    expected = {f"{i:02d}" for i in range(1, 21)}
    for name, group in (("predictions", predictions), ("explanations", explanations),
                        ("combined", combined)):
        if len(group) != 20 or {r["id"] for r in group} != expected:
            raise ValueError(f"Incomplete/duplicate {name}")
    for row in predictions:
        intensity = float(row["intensity"])
        raw = float(row["raw_intensity"])
        cls = "Negative" if intensity < 0 else "Positive" if intensity > 0 else "Neutral"
        if not -3 <= intensity <= 3 or not math.isfinite(raw) or row["polarity"] != cls:
            raise ValueError(f"Prediction inconsistency: {row['id']}")
        if row["model_sha256"] != model_hash:
            raise ValueError("Model hash mismatch")
    for row in explanations:
        phi = [float(row[f"phi_{mode}"]) for mode in ("text", "audio", "vision")]
        weights = [float(row[f"weight_{mode}"]) for mode in ("text", "audio", "vision")]
        if not all(math.isfinite(v) for v in (*phi, *weights)):
            raise ValueError(f"Nonfinite attribution: {row['id']}")
        if abs(sum(weights) - 1) > 1e-5 or abs(sum(phi) -
             (float(row["full_score"]) - float(row["baseline_score"]))) > 1e-5:
            raise ValueError(f"Contribution inconsistency: {row['id']}")
    if len(evidence) != 116 or len({r["evidence_id"] for r in evidence}) != len(evidence):
        raise ValueError("Evidence count/ID mismatch")
    source_text = {}
    for path in args.aligned.glob("*.pkl"):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        source_text[path.stem] = str(sample["raw_text"])
    candidate, unresolved = 0, []
    for row in evidence:
        start, end = int(row["char_start"]), int(row["char_end_exclusive"])
        if not 0 <= start < end <= len(source_text[row["id"]]):
            raise ValueError(f"Character bounds invalid: {row['evidence_id']}")
        if source_text[row["id"]][start:end] != row["text_excerpt"]:
            raise ValueError(f"Text excerpt mismatch: {row['evidence_id']}")
        if row["human_verified"].lower() != "false":
            raise ValueError("Unperformed human review claimed")
        if row["mapping_status"] == "text_exact_ctc_time_candidate":
            begin, finish = float(row["start_sec"]), float(row["end_sec"])
            if not 0 <= begin < finish:
                raise ValueError(f"Time interval invalid: {row['evidence_id']}")
            candidate += 1
            if row["modality"] == "vision" and not (root / row["frame_path"]).is_file():
                raise ValueError(f"Missing keyframe: {row['evidence_id']}")
        else:
            unresolved.append(row["evidence_id"])
            if row["start_sec"] or row["end_sec"]:
                raise ValueError("Unresolved interval has fake time")
    if candidate != 116 or unresolved or len(list((root / "cards").glob("*.md"))) != 20:
        raise ValueError("Mapping/card count mismatch")
    plots = sorted(run.rglob("*.eps"))
    if len(plots) < 8 or any(not path.read_bytes().startswith(b"%!PS-Adobe") for path in plots):
        raise ValueError("EPS figure missing or malformed")
    result = {"prediction_rows": len(predictions), "explanation_rows": len(explanations),
              "evidence_rows": len(evidence), "candidate_time_rows": candidate,
              "unresolved": unresolved, "keyframes": len(list((root / "frames").glob("*.png"))),
              "eps_figures": len(plots), "model_sha256": model_hash, "passed": True}
    (run / "verification.json").write_text(json.dumps(result, ensure_ascii=False,
                                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
