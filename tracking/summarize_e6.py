#!/usr/bin/env python3
"""Audit and summarize the completed E6 budget-control experiments.

This script is offline. It reads the two verified ``extract_results`` caches,
recomputes sequence-averaged trapz AUC, normalized precision and precision,
and writes CSV/JSON/Markdown/LaTeX artifacts. It never trains a model, runs a
tracker, or modifies prediction files.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


RUNS = (
    {
        "key": "10k30",
        "parameter_name": "baseline_medium",
        "display_name": "MMTrack 10Kx30",
        "schedule": "10K x 30",
        "samples_per_epoch": 10000,
        "epochs": 30,
        "total_pairs": 300000,
        "gpus": 2,
        "batch_per_gpu": 16,
        "global_batch": 32,
        "steps_per_epoch": 312,
    },
    {
        "key": "60k30",
        "parameter_name": "baseline_full_30",
        "display_name": "MMTrack 60Kx30",
        "schedule": "60K x 30",
        "samples_per_epoch": 60000,
        "epochs": 30,
        "total_pairs": 1800000,
        "gpus": 2,
        "batch_per_gpu": 32,
        "global_batch": 64,
        "steps_per_epoch": 937,
    },
    {
        "key": "10k180",
        "parameter_name": "baseline_fullpool_10k_ep180",
        "display_name": "MMTrack 10Kx180 singleGPU",
        "schedule": "10K x 180",
        "samples_per_epoch": 10000,
        "epochs": 180,
        "total_pairs": 1800000,
        "gpus": 1,
        "batch_per_gpu": 32,
        "global_batch": 32,
        "steps_per_epoch": 312,
    },
)

CACHES = {
    "tnl2k_lang": ("e6_final_check_tnl2k_lang", 700, "TNL2K"),
    "lasot_lang": ("e6_final_check_lasot_lang", 280, "LaSOT"),
}


def project_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "lib" / "test").is_dir() and (candidate / "tracking").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the MMTrack project root")


def trapz(values, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


def threshold_index(data, key, target, fallback):
    values = np.asarray(data.get(key, []), dtype=np.float64)
    if values.size == 0:
        return fallback
    found = np.flatnonzero(np.isclose(values, target))
    if len(found) != 1:
        raise RuntimeError(f"Cannot uniquely locate {target} in {key}")
    return int(found[0])


def read_cache(root, dataset_name, cache_name, expected):
    path = root / "output" / "test" / "result_plots" / cache_name / "eval_data.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)

    valid = np.asarray(data["valid_sequence"], dtype=bool)
    if valid.size != expected or int(valid.sum()) != expected:
        raise RuntimeError(
            f"{dataset_name}: expected {expected}/{expected} valid sequences, "
            f"got {int(valid.sum())}/{valid.size}"
        )
    if len(data["trackers"]) != len(RUNS):
        raise RuntimeError(
            f"{dataset_name}: expected {len(RUNS)} trackers, got {len(data['trackers'])}"
        )

    actual_names = [item.get("disp_name", "") for item in data["trackers"]]
    expected_names = [item["display_name"] for item in RUNS]
    if actual_names != expected_names:
        raise RuntimeError(
            f"{dataset_name}: tracker order mismatch: {actual_names} != {expected_names}"
        )

    success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=np.float64)
    precision = np.asarray(data["ave_success_rate_plot_center"], dtype=np.float64)
    pnorm = np.asarray(data["ave_success_rate_plot_center_norm"], dtype=np.float64)
    thresholds = np.asarray(data["threshold_set_overlap"], dtype=np.float64)
    p_index = threshold_index(data, "threshold_set_center", 20.0, 20)
    pn_index = threshold_index(data, "threshold_set_center_norm", 0.20, 20)

    seq_auc = trapz(success[valid], thresholds, axis=2) * 100.0
    seq_p = precision[valid, :, p_index] * 100.0
    seq_pn = pnorm[valid, :, pn_index] * 100.0

    rows = []
    for index, run in enumerate(RUNS):
        row = dict(run)
        row.update(
            {
                "dataset": dataset_name,
                "paired_sequences": int(valid.sum()),
                "auc_trapz": float(seq_auc[:, index].mean()),
                "pnorm": float(seq_pn[:, index].mean()),
                "precision": float(seq_p[:, index].mean()),
                "optimizer_updates": int(run["steps_per_epoch"] * run["epochs"]),
            }
        )
        rows.append(row)

    audit = {
        "dataset": dataset_name,
        "cache": str(path),
        "valid_sequences": int(valid.sum()),
        "total_sequences": int(valid.size),
        "tracker_names": actual_names,
    }
    return rows, audit


def write_csv(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows, dataset, key):
    return next(row for row in rows if row["dataset"] == dataset and row["key"] == key)


def make_deltas(rows):
    result = []
    for dataset in CACHES:
        short = find_row(rows, dataset, "10k30")
        long = find_row(rows, dataset, "10k180")
        full = find_row(rows, dataset, "60k30")
        for comparison, lhs, rhs in (
            ("10Kx180 - 10Kx30", long, short),
            ("10Kx180 - 60Kx30", long, full),
        ):
            result.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "auc_delta": lhs["auc_trapz"] - rhs["auc_trapz"],
                    "pnorm_delta": lhs["pnorm"] - rhs["pnorm"],
                    "precision_delta": lhs["precision"] - rhs["precision"],
                }
            )
    return result


def write_latex(path, rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Training-budget controls for MMTrack on TNL2K and LaSOT. All runs stochastically sample from the same complete training pool and use the same standard augmentations. The 60K$\times$30 and 10K$\times$180 settings have the same total sampled-pair exposure (1.8M), but use different global batch sizes (64 vs. 32) and numbers of optimizer updates. AUC is computed by trapezoidal integration.}",
        r"\label{tab:mmtrack_budget_control}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{TNL2K} & \multicolumn{3}{c}{LaSOT} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Setting & Schedule & Total pairs & AUC & $P_{\mathrm{Norm}}$ & P & AUC & $P_{\mathrm{Norm}}$ & P \\",
        r"\midrule",
    ]
    labels = {
        "10k30": ("Reduced per-epoch budget", r"10K$\times$30"),
        "10k180": ("Exposure matched", r"10K$\times$180"),
        "60k30": ("Full per-epoch budget", r"60K$\times$30"),
    }
    for key in ("10k30", "10k180", "60k30"):
        tnl = find_row(rows, "tnl2k_lang", key)
        lasot = find_row(rows, "lasot_lang", key)
        total = f"{tnl['total_pairs'] / 1_000_000:.1f}M"
        setting, schedule = labels[key]
        lines.append(
            f"{setting} & {schedule} & {total} & {tnl['auc_trapz']:.2f} & {tnl['pnorm']:.2f} & "
            f"{tnl['precision']:.2f} & {lasot['auc_trapz']:.2f} & {lasot['pnorm']:.2f} & "
            f"{lasot['precision']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path, rows, deltas):
    lines = [
        "# E6 final budget-control summary",
        "",
        "All values below were recomputed from the verified evaluation caches.",
        "AUC uses trapezoidal integration.",
        "",
        "| Schedule | Total pairs | TNL2K AUC | PNorm | P | LaSOT AUC | PNorm | P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("10k30", "60k30", "10k180"):
        tnl = find_row(rows, "tnl2k_lang", key)
        lasot = find_row(rows, "lasot_lang", key)
        lines.append(
            f"| {tnl['schedule']} | {tnl['total_pairs']/1_000_000:.1f}M | "
            f"{tnl['auc_trapz']:.4f} | {tnl['pnorm']:.4f} | {tnl['precision']:.4f} | "
            f"{lasot['auc_trapz']:.4f} | {lasot['pnorm']:.4f} | {lasot['precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| Dataset | Comparison | AUC delta | PNorm delta | P delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['auc_delta']:+.4f} | "
            f"{row['pnorm_delta']:+.4f} | {row['precision_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Execution metadata",
            "",
            "| Schedule | GPUs | Batch/GPU | Global batch | Steps/epoch | Approx. updates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in RUNS:
        lines.append(
            f"| {run['schedule']} | {run['gpus']} | {run['batch_per_gpu']} | "
            f"{run['global_batch']} | {run['steps_per_epoch']} | "
            f"{run['steps_per_epoch'] * run['epochs']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/final_revision/e6")
    args = parser.parse_args()

    root = project_root()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    rows, audits = [], []
    for dataset, (cache_name, expected, _) in CACHES.items():
        dataset_rows, audit = read_cache(root, dataset, cache_name, expected)
        rows.extend(dataset_rows)
        audits.append(audit)

    deltas = make_deltas(rows)
    write_csv(output / "e6_budget_metrics.csv", rows)
    write_csv(output / "e6_budget_deltas.csv", deltas)
    write_latex(output / "e6_budget_table.tex", rows)
    write_summary(output / "e6_summary.md", rows, deltas)
    (output / "e6_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for audit in audits:
        print(
            f"{audit['dataset']}: {audit['valid_sequences']}/{audit['total_sequences']} "
            "paired sequences"
        )
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
