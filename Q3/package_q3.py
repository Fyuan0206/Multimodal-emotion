"""Create an anonymous, lightweight Q3-only archive from a verified run."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    if not run.is_relative_to(root) or args.output.resolve().is_relative_to(run):
        raise ValueError("Unexpected run or output location")
    verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    if not verification["passed"]:
        raise ValueError("Run did not pass integrity verification")
    selected = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    files = [
        *(root / "Q3").glob("*.py"), *(root / "Q3").glob("*.sbatch"),
        root / "Q3" / "README.md", root / "Q3" / "DESIGN.md",
        root / "Q3" / "RESULTS.md",
        root / "Q3" / "audit" / "initial" / "audit.json",
        root / "Q3" / "audit" / "initial" / "sample_inventory.csv",
        run / selected["final_record"]["checkpoint"],
        *(run / name for name in ("protocol.json", "selection.json", "results.json",
                                  "attachment4_predictions.csv", "verification.json",
                                  "environment_packages.txt")),
        run / "report" / "report_metrics.json",
        run / "validation_detail" / "predictions.csv",
        run / "validation_detail" / "c2_seed_attributions.csv",
        run / "alignment" / "candidate_alignment.json",
        run / "explanation" / "summary.json",
        run / "explanation" / "final_audit.json",
        run / "explanation" / "sensitivity.json",
        run / "explanation" / "text_mask_check" / "summary.json",
        run / "explanation" / "text_mask_check" / "sample_results.csv",
        run / "explanation" / "diagnostic" / "explanations.csv",
        run / "explanation" / "diagnostic" / "shapley.jsonl",
        run / "explanation" / "diagnostic" / "budget_curves.csv",
        run / "explanation" / "diagnostic" / "budget_summary.json",
        run / "explanation" / "attachment4" / "explanations.csv",
        run / "explanation" / "attachment4" / "attachment4_predictions_explanations.csv",
        run / "explanation" / "attachment4" / "attachment4_evidence.csv",
        run / "explanation" / "attachment4" / "shapley.jsonl",
        root / "Q2" / "outputs" / "v2_1529979" / "bert_vocab.txt",
        *(run / "report").glob("*.eps"),
        *(run / "explanation" / "diagnostic").glob("*.eps"),
        *(run / "explanation" / "attachment4" / "cards").glob("*.md"),
    ]
    # Keep just representative frames in this Q3-specific lightweight bundle.
    frames = sorted((run / "explanation" / "attachment4" / "frames").glob("14_vision_*.png"))[:3]
    files.extend(frames)
    files = sorted(set(files))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("A selected submission artifact is missing")
    private = (("C:" + chr(92) + "Users" + chr(92)).encode(),
               ("/home/bingxing2/home/" + "scx" + "6706").encode(),
               ("scx" + "6706@").encode())
    for path in files:
        if path.suffix.lower() in (".py", ".md", ".json", ".jsonl", ".csv", ".sbatch", ".txt"):
            payload = path.read_bytes()
            if any(value in payload for value in private):
                raise ValueError(f"Private path/user reference in {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise ValueError("Zip integrity failure")
    manifest = {"archive": args.output.name, "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output), "files": len(files),
                "selected_checkpoint": selected["final_record"]["checkpoint"],
                "scope": "Q3-only lightweight package; overall Q1+Q2+Q3 size must be checked separately",
                "excluded": "original videos, full local result history, most keyframe images, external pretrained BERT/CTC weights"}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2,
                                                            ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
