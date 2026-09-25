"""Summarize Q3 validation and export publication figures, including EPS."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


METRICS = ("accuracy", "macro_f1", "mae", "pearson_r")
ARCH = ("C0", "C1", "C2", "C3")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def score(rows):
    target = np.asarray([float(r["true_intensity"]) for r in rows])
    pred = np.asarray([float(r["intensity"]) for r in rows])
    truth = np.asarray([int(r["true_class"]) for r in rows])
    classes = np.asarray([int(r["class"]) for r in rows])
    f1 = []
    for c in range(3):
        n = int(np.sum(classes == c) + np.sum(truth == c))
        f1.append(2 * np.sum((classes == c) & (truth == c)) / n if n else 0.)
    corr = float(np.corrcoef(target, pred)[0, 1]) if np.std(pred) else float("nan")
    return {"accuracy": float(np.mean(truth == classes)),
            "macro_f1": float(np.mean(f1)),
            "mae": float(np.mean(np.abs(target - pred))),
            "pearson_r": corr}


def save_figure(fig, path):
    fig.tight_layout()
    for suffix in ("png", "pdf", "eps"):
        fig.savefig(path.with_suffix("." + suffix), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    report = run / "report"
    report.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.transparent": False})
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    validation = read_csv(run / "validation_detail" / "predictions.csv")
    seeds = read_csv(run / "validation_detail" / "c2_seed_attributions.csv")
    tests = read_csv(run / "attachment4_predictions.csv")
    final_audit = json.loads((run / "explanation" / "final_audit.json").read_text(encoding="utf-8"))
    if len(validation) != 20 * 728 or len(seeds) != 5 * 291 or len(tests) != 20:
        raise ValueError("Incomplete Q3 results")
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    for ax, metric in zip(axes.flat, METRICS):
        groups = [[float(r["diagnostic"][metric]) for r in results
                   if r["architecture"] == arch] for arch in ARCH]
        averages = [np.mean(group) for group in groups]
        deviations = [np.std(group, ddof=1) for group in groups]
        ax.bar(ARCH, averages, color=["#687d9e", "#a99a82", "#416b93", "#829976"])
        ax.errorbar(ARCH, averages, yerr=deviations, color="black", fmt="none", capsize=3)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Diagnostic subset, 5 seeds")
        if metric != "mae":
            ax.set_ylim(0, min(1, max(averages) + max(deviations) + .15))
    save_figure(figure, report / "validation_five_seed_metrics")
    final = [r for r in validation if r["architecture"] == selection["chosen_architecture"]
             and int(r["seed"]) == 2026 and r["set"] == "diagnostic"]
    controls = [r for r in validation if r["architecture"] == "C0"
                and int(r["seed"]) == 2026 and r["set"] == "diagnostic"]
    if {r["id"] for r in final} != {r["id"] for r in controls}:
        raise ValueError("Model/control diagnostic ID mismatch")
    matrix = np.zeros((3, 3), dtype=int)
    for r in final:
        matrix[int(r["true_class"]), int(r["class"])] += 1
    figure, ax = plt.subplots(figsize=(5, 4))
    scale = max(int(matrix.max()), 1)
    for i in range(3):
        for j in range(3):
            shade = matrix[i, j] / scale
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=plt.cm.Blues(.15 + .8 * shade),
                                   edgecolor="white", linewidth=2))
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(2.5, -.5)
    ax.set_xticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    save_figure(figure, report / "validation_confusion")
    figure, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([float(r["true_intensity"]) for r in final],
               [float(r["intensity"]) for r in final], s=12, alpha=1,
               color="#315b7d")
    ax.plot([-3, 3], [-3, 3], color="#9d5c4c", linewidth=1)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="True intensity",
           ylabel="Predicted intensity", title="Final C2, diagnostic subset")
    save_figure(figure, report / "validation_intensity_scatter")
    local = read_csv(run / "explanation" / "attachment4" / "local_candidate_scores.csv")
    for sample_id in ("01", "05", "14"):
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7.5, 5.8))
        for axis, mode, color in zip(axes, ("text", "audio", "vision"),
                                     ("#315b7d", "#d18146", "#688b68")):
            values = [r for r in local if r["id"] == sample_id and
                      r["modality"] == mode and
                      int(r["feature_end_exclusive"]) - int(r["feature_start"]) == 1]
            axis.bar([int(r["feature_start"]) for r in values],
                     [float(r["signed_delta"]) for r in values], color=color,
                     width=.86)
            axis.axhline(0, color="black", linewidth=.6)
            axis.set_ylabel(mode + "\nmargin drop")
        axes[-1].set_xlabel("Aligned feature position (0-based)")
        axes[0].set_title(f"Attachment 4 sample {sample_id}: single-position occlusion")
        save_figure(figure, report / f"local_importance_{sample_id}")
    budget_path = run / "explanation" / "diagnostic" / "budget_curves.csv"
    budget_data = None
    if budget_path.exists():
        budget_data = json.loads((budget_path.parent / "budget_summary.json").read_text(encoding="utf-8"))
        fractions = [.1, .2, .3]
        figure, ax = plt.subplots(figsize=(6.3, 3.8))
        for field, label, color in (("selected_mean", "Selected", "#315b7d"),
                                    ("random_mean", "Random", "#d18146")):
            ax.plot([100 * f for f in fractions],
                    [budget_data[str(f)][field] for f in fractions],
                    marker="o", linewidth=2, color=color, label=label)
        ax.set(xlabel="Requested evidence budget (%)",
               ylabel="Mean drop in original class margin",
               title="Diagnostic subset, fixed evidence ranking")
        ax.legend(frameon=False)
        save_figure(figure, report / "evidence_budget_curve")
    rng = np.random.default_rng(6206)
    groups = sorted({r["group"] for r in final})
    by_arch = {arch: {r["id"]: r for r in rows} for arch, rows in (("C2", final), ("C0", controls))}
    group_ids = {g: [r["id"] for r in final if r["group"] == g] for g in groups}
    draws = {metric: [] for metric in METRICS}
    for _ in range(2000):
        chosen_groups = rng.choice(groups, len(groups), replace=True)
        chosen_ids = [item for group in chosen_groups for item in group_ids[group]]
        new = score([by_arch["C2"][sample_id] for sample_id in chosen_ids])
        old = score([by_arch["C0"][sample_id] for sample_id in chosen_ids])
        for metric in METRICS:
            draws[metric].append(new[metric] - old[metric])
    gaps = {metric: {"difference": score(final)[metric] - score(controls)[metric],
                     "95pct": np.nanquantile(draws[metric], [.025, .975]).tolist()}
            for metric in METRICS}
    by_id = {}
    for row in seeds:
        by_id.setdefault(row["id"], []).append(row)
    agreement_all = np.mean([len({r["primary_modality"] for r in rows}) == 1
                             for rows in by_id.values()])
    same_class = [rows for rows in by_id.values()
                  if len({r["predicted_class"] for r in rows}) == 1]
    agreement_same = (float(np.mean([len({r["primary_modality"] for r in rows}) == 1
                                     for rows in same_class])) if same_class else None)
    distribution = Counter(r["polarity"] for r in tests)
    report_data = {"diagnostic_count": len(final), "diagnostic_groups": len(groups),
                   "final_metrics": score(final), "C0_metrics": score(controls),
                   "C2_minus_C0_seed2026_group_bootstrap": gaps,
                   "five_seed_primary_modality_agreement_all": float(agreement_all),
                   "five_seed_prediction_consistent_samples": len(same_class),
                   "five_seed_primary_modality_agreement_prediction_consistent": agreement_same,
                   "confusion_matrix": matrix.tolist(), "attachment4_distribution": dict(distribution),
                   "deletion_validation": final_audit["validation_deletion_vs_random"]}
    if budget_data is not None:
        report_data["evidence_budgets"] = budget_data
    (report / "report_metrics.json").write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False))


if __name__ == "__main__":
    main()
