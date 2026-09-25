"""Read-only integrity checks for a completed Q2 v2 output directory."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    records = json.loads((root / "training_records.json").read_text(encoding="utf-8"))
    expected = {(v, s) for v in ("T0", "T1", "M0", "M1", "M2", "M3") for s in range(2026, 2031)}
    assert {(r["variant"], r["seed"]) for r in records} == expected
    assert len(records) == 30 and not any(r["smoke"] for r in records)
    assert len({r["steps_per_epoch"] for r in records}) == 1
    config = json.loads((root / "config_v2.json").read_text(encoding="utf-8"))
    assert config["seeds"] == list(range(2026, 2031)) and not config["smoke"]
    report = json.loads((root / "REPORT_COMPLETE.json").read_text(encoding="utf-8"))
    assert report["bootstrap_repeats"] == 2000
    for filename, count in (("main_results.csv", 60), ("main_summary.csv", 12),
                            ("missingness_grid_v2.csv", 9450), ("missingness_common_samples.csv", 9450),
                            ("decoding_comparison.csv", 180), ("paired_comparisons.csv", 24),
                            ("error_cases.csv", 12), ("attachment3_predictions_v2.csv", 30)):
        rows = read_csv(root / filename)
        assert len(rows) == count, (filename, len(rows), count)
        for row in rows:
            for key in ("accuracy", "macro_f1"):
                if key in row:
                    assert 0 <= float(row[key]) <= 1
            if "mae" in row:
                assert 0 <= float(row["mae"]) <= 6
    attachment = read_csv(root / "attachment3_predictions_v2.csv")
    assert len({r["sample_id"] for r in attachment}) == 30
    for row in attachment:
        value = float(row["pred_intensity"])
        assert np.isfinite(value) and -3 <= value <= 3
        expected_label = "Negative" if value < 0 else "Positive" if value > 0 else "Neutral"
        assert row["pred_polarity"] == expected_label
    checked = 0
    for seed in range(5101, 5106):
        with np.load(root / "predictions" / f"mask_{seed}.npz") as mask:
            valid = mask["valid"]
            assert valid.shape == (64, 728) and valid[0].all()
            for grid_index, cell in enumerate(config["grid"]):
                ok = valid[grid_index + 1]
                original = mask["original_count"][grid_index][:, cell["modes"]]
                removed = mask["removed"][grid_index][:, cell["modes"]]
                assert np.all(removed[ok] > 0) and np.all(removed[ok] < original[ok])
                assert np.all(mask["start"][grid_index][~ok] == -1)
        for variant, train_seed in sorted(expected):
            with np.load(root / "predictions" / f"{variant}_{train_seed}_{seed}.npz") as p:
                assert p["reg"].shape == (64, 728)
                assert np.isfinite(p["reg"]).all()
                assert np.array_equal(p["classes"], np.sign(p["reg"]).astype(int) + 1)
            checked += 1
    print(json.dumps(dict(status="PASS", training_runs=30, diagnostic_prediction_files=checked,
                         grid_rows=9450, attachment3_rows=30, bootstrap_repeats=2000,
                         steps_per_epoch=records[0]["steps_per_epoch"]), indent=2))


if __name__ == "__main__":
    main()
