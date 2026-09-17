#!/usr/bin/env python3
"""Create the final E1 reliability summary from reliability_frames.csv.

This script is intentionally offline and read-only with respect to tracker results.
It recomputes all reported metrics from the frame-level evidence, writes auditable
CSV/JSON files, and creates the two-panel figure proposed for the paper.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr, rankdata, spearmanr


REQUIRED_COLUMNS = {"sequence", "frame_id", "reliability", "iou"}


def binary_auroc(y_true, score):
    """AUROC from average ranks, including the standard tie correction."""
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both positive and negative samples")
    ranks = rankdata(score, method="average")
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def binary_average_precision(y_true, score):
    """Average precision with tied scores evaluated at distinct thresholds."""
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_positive = int((y_true == 1).sum())
    if n_positive == 0:
        raise ValueError("Average precision requires positive samples")

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_true = y_true[order]
    group_end = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(score) - 1]
    cumulative_positive = np.cumsum(sorted_true)[group_end]
    cumulative_total = group_end + 1
    precision = cumulative_positive / cumulative_total
    recall = cumulative_positive / n_positive
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def sha256sum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_sequence_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def latest_trace(trace_dir, sequence_name):
    pattern = str(Path(trace_dir) / (safe_sequence_name(sequence_name) + "_*.tsv"))
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_boxes(path):
    boxes = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    return boxes[:, :4]


def xywh_iou(box_a, box_b):
    ax, ay, aw, ah = [float(value) for value in box_a]
    bx, by, bw, bh = [float(value) for value in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return math.nan
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else math.nan


def build_frames_from_trace(dataset_name, trace_dir, results_dir, pos_thr, neg_thr):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from lib.test.evaluation import get_dataset

    dataset = get_dataset(dataset_name)
    records = []
    missing_trace = []
    missing_result = []
    selected_traces = []
    for sequence in dataset:
        trace_path = latest_trace(trace_dir, sequence.name)
        result_path = Path(results_dir) / (sequence.name + ".txt")
        if trace_path is None:
            missing_trace.append(sequence.name)
            continue
        if not result_path.is_file():
            missing_result.append(sequence.name)
            continue

        trace = pd.read_csv(trace_path, sep="\t", na_values=["None"])
        required_trace_columns = {"frame_id", "should_update_rel", "pred_rel"}
        missing_columns = required_trace_columns - set(trace.columns)
        if missing_columns:
            raise RuntimeError(
                f"Trace {trace_path} lacks columns {sorted(missing_columns)}"
            )
        trace = trace.loc[
            (pd.to_numeric(trace["should_update_rel"], errors="coerce") == 1)
            & pd.to_numeric(trace["pred_rel"], errors="coerce").notna()
        ].copy()
        prediction = load_boxes(result_path)
        ground_truth = np.asarray(
            sequence.ground_truth_rect, dtype=np.float64
        ).reshape(-1, 4)
        max_frame = min(len(prediction), len(ground_truth))
        selected_traces.append(
            {
                "sequence": sequence.name,
                "trace_file": str(Path(trace_path).resolve()),
                "result_file": str(result_path.resolve()),
                "prediction_frames": int(len(prediction)),
                "ground_truth_frames": int(len(ground_truth)),
            }
        )
        for row in trace.itertuples(index=False):
            frame_id = int(row.frame_id)
            if frame_id < 0 or frame_id >= max_frame:
                continue
            reliability = float(row.pred_rel)
            if not np.isfinite(reliability):
                continue
            iou = xywh_iou(prediction[frame_id], ground_truth[frame_id])
            if not np.isfinite(iou):
                continue
            if iou >= pos_thr:
                hard_label = 1.0
            elif iou <= neg_thr:
                hard_label = 0.0
            else:
                hard_label = math.nan
            records.append(
                {
                    "dataset": dataset_name,
                    "sequence": sequence.name,
                    "frame_id": frame_id,
                    "reliability": reliability,
                    "iou": iou,
                    "hard_label": hard_label,
                    "trace_file": Path(trace_path).name,
                }
            )

    frames = pd.DataFrame(records)
    audit = {
        "dataset": dataset_name,
        "num_sequences_in_dataset": int(len(dataset)),
        "num_sequences_with_records": int(frames["sequence"].nunique())
        if not frames.empty
        else 0,
        "missing_trace_count": int(len(missing_trace)),
        "missing_result_count": int(len(missing_result)),
        "missing_traces": missing_trace,
        "missing_results": missing_result,
        "selected_traces": selected_traces,
    }
    return frames, audit


def equal_width_calibration(y_true, probability, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(
        np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1
    )
    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count:
            confidence = float(probability[mask].mean())
            positive_rate = float(y_true[mask].mean())
            gap = abs(confidence - positive_rate)
            ece += count / len(y_true) * gap
        else:
            confidence = math.nan
            positive_rate = math.nan
            gap = math.nan
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": count,
                "mean_confidence": confidence,
                "positive_rate": positive_rate,
                "absolute_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def equal_frequency_calibration(y_true, probability, n_bins):
    # qcut preserves score ordering and drops duplicate boundaries if necessary.
    # These bins are only for displaying the saturated score range; ECE remains
    # the standard equal-width ECE computed above.
    bin_ids = pd.qcut(
        pd.Series(probability), q=n_bins, labels=False, duplicates="drop"
    ).to_numpy()
    rows = []
    for bin_id in sorted(np.unique(bin_ids)):
        mask = bin_ids == bin_id
        confidence = float(probability[mask].mean())
        positive_rate = float(y_true[mask].mean())
        rows.append(
            {
                "bin": int(bin_id) + 1,
                "score_min": float(probability[mask].min()),
                "score_max": float(probability[mask].max()),
                "count": int(mask.sum()),
                "mean_confidence": confidence,
                "positive_rate": positive_rate,
                "absolute_gap": abs(confidence - positive_rate),
            }
        )
    return pd.DataFrame(rows)


def distribution_row(name, values):
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "class": name,
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(np.max(values)),
    }


def ecdf(values, max_points=2500):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) > max_points:
        indices = np.linspace(0, len(ordered) - 1, max_points).astype(int)
        x = ordered[indices]
        y = (indices + 1) / len(ordered)
    else:
        x = ordered
        y = np.arange(1, len(ordered) + 1) / len(ordered)
    return x, y


def save_main_figure(
    output_dir,
    model_label,
    positive,
    negative,
    quantile_calibration,
    metrics,
    zoom_min,
):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#2F6B9A"
    orange = "#D9792B"
    purple = "#6A4C93"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.55), constrained_layout=True)

    combined = np.concatenate([negative, positive])
    display_low = max(zoom_min, float(np.quantile(combined, 0.005)) - 0.00015)
    display_high = min(1.0, float(np.quantile(combined, 0.9995)) + 0.00010)
    bin_edges = np.linspace(display_low, display_high, 76)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    negative_hist, _ = np.histogram(negative, bins=bin_edges)
    positive_hist, _ = np.histogram(positive, bins=bin_edges)
    negative_pct = 100.0 * negative_hist / len(negative)
    positive_pct = 100.0 * positive_hist / len(positive)

    ax = axes[0]
    ax.plot(
        bin_centers,
        negative_pct,
        color=blue,
        linewidth=2.0,
        label=f"Poor localization, IoU <= 0.3 (n={len(negative):,})",
    )
    ax.fill_between(bin_centers, negative_pct, color=blue, alpha=0.12)
    ax.plot(
        bin_centers,
        positive_pct,
        color=orange,
        linewidth=2.0,
        label=f"Good localization, IoU >= 0.5 (n={len(positive):,})",
    )
    ax.fill_between(bin_centers, positive_pct, color=orange, alpha=0.12)
    ax.set_xlim(display_low, display_high)
    ax.set_ylim(
        0.0,
        1.14 * max(float(negative_pct.max()), float(positive_pct.max())),
    )
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Predicted reliability")
    ax.set_ylabel("Samples per bin (%)")
    ax.set_title("(a) Scores by localization quality")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(loc="upper left", frameon=True, fontsize=8.0)
    ax = axes[1]
    qx = quantile_calibration["mean_confidence"].to_numpy()
    qy = quantile_calibration["positive_rate"].to_numpy()
    group_ids = np.arange(1, len(qx) + 1)
    ax.fill_between(group_ids, qy, qx, color=purple, alpha=0.09)
    ax.plot(
        group_ids,
        qx,
        marker="s",
        markersize=4.6,
        color=orange,
        linewidth=1.8,
        label="Mean predicted reliability",
    )
    ax.plot(
        group_ids,
        qy,
        marker="o",
        markersize=4.8,
        color=purple,
        linewidth=1.8,
        label="Empirical positive rate",
    )
    ax.axhline(
        metrics["positive_prevalence"],
        linestyle="--",
        color="0.45",
        linewidth=1.1,
        label=f"Positive prevalence = {metrics['positive_prevalence']:.4f}",
    )
    ax.set_xlim(0.65, len(group_ids) + 0.35)
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks(group_ids)
    ax.set_xlabel("Equal-frequency reliability group (low to high)")
    ax.set_ylabel("Mean score / empirical positive rate")
    ax.set_title("(b) Calibration across reliability groups")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=7.7)
    ax.text(
        0.03,
        0.91,
        "AUROC = {:.4f}, AUPRC = {:.4f}\nECE-EW10 = {:.4f}, Brier = {:.4f}\nPearson r = {:.4f}, Spearman rho = {:.4f}".format(
            metrics["auroc"],
            metrics["auprc"],
            metrics["ece_equal_width_10bin"],
            metrics["brier_score"],
            metrics["pearson_r"],
            metrics["spearman_rho"],
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.78", "alpha": 0.94},
    )

    png_path = output_dir / "figure_e1_reliability_quality.png"
    pdf_path = output_dir / "figure_e1_reliability_quality.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_summary(path, args, metrics, figure_name):
    status = (
        "This evidence is suitable for the final paper model only if the model label "
        "and checkpoint provenance match the method reported in the paper."
    )
    lines = [
        "# E1 reliability-analysis result summary",
        "",
        f"- Dataset: `{args.dataset_label}`",
        f"- Model: `{args.model_label}`",
        f"- Source note: {args.source_note}",
        f"- Sequences: {metrics['num_sequences']}",
        f"- Fresh reliability samples: {metrics['num_fresh_reliability_samples']:,}",
        f"- Hard-label samples: {metrics['num_hard_label_samples']:,}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| AUROC | {metrics['auroc']:.4f} |",
        f"| AUPRC | {metrics['auprc']:.4f} |",
        f"| AUPRC random baseline / positive prevalence | {metrics['positive_prevalence']:.4f} |",
        f"| ECE, 10 equal-width bins | {metrics['ece_equal_width_10bin']:.4f} |",
        f"| Brier score | {metrics['brier_score']:.4f} |",
        f"| Pearson r | {metrics['pearson_r']:.4f} |",
        f"| Spearman rho | {metrics['spearman_rho']:.4f} |",
        f"| Mean score, positive | {metrics['positive_mean_reliability']:.6f} |",
        f"| Mean score, negative | {metrics['negative_mean_reliability']:.6f} |",
        "",
        "## Interpretation boundary",
        "",
        "The reliability score has useful ranking evidence only when AUROC is above 0.5 "
        "and AUPRC is above the positive-class prevalence. A large ECE/Brier score and "
        "scores concentrated near one indicate overconfidence rather than good absolute calibration.",
        "",
        status,
        "",
        f"Main figure: `{figure_name}`",
        "",
        "The calibration points in the figure use equal-frequency groups only to reveal "
        "the ordering inside the saturated score range. The reported ECE is still computed "
        "with ten fixed equal-width bins on [0, 1].",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--frames-csv", help="Existing reliability_frames.csv"
    )
    source_group.add_argument(
        "--trace-dir", help="Trace directory; requires --dataset and --results-dir"
    )
    parser.add_argument("--dataset", choices=["tnl2k_lang", "lasot_lang"])
    parser.add_argument("--results-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="FRTFTrack")
    parser.add_argument("--dataset-label", default="TNL2K")
    parser.add_argument("--source-note", default="not supplied")
    parser.add_argument("--expected-sequences", type=int, default=700)
    parser.add_argument("--pos-thr", type=float, default=0.5)
    parser.add_argument("--neg-thr", type=float, default=0.3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--zoom-min", type=float, default=0.98)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.frames_csv:
        frames_path = Path(args.frames_csv).expanduser().resolve()
        if not frames_path.is_file():
            raise FileNotFoundError(frames_path)
        frames = pd.read_csv(frames_path)
        source_audit = {
            "source_mode": "frames_csv",
            "frames_csv": str(frames_path),
            "frames_csv_sha256": sha256sum(frames_path),
        }
    else:
        if not args.dataset or not args.results_dir:
            parser.error("--trace-dir requires --dataset and --results-dir")
        frames, source_audit = build_frames_from_trace(
            args.dataset,
            args.trace_dir,
            args.results_dir,
            args.pos_thr,
            args.neg_thr,
        )
        if frames.empty:
            raise RuntimeError("No matched reliability samples were found")
        frames_path = output_dir / "reliability_frames.csv"
        frames.to_csv(frames_path, index=False)
        source_audit["source_mode"] = "trace_and_results"
        source_audit["trace_dir"] = str(Path(args.trace_dir).resolve())
        source_audit["results_dir"] = str(Path(args.results_dir).resolve())
        source_audit["generated_frames_csv"] = str(frames_path)

    with (output_dir / "source_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(source_audit, handle, indent=2, ensure_ascii=False)

    missing = REQUIRED_COLUMNS - set(frames.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    frames = frames.copy()
    for column in ("frame_id", "reliability", "iou"):
        frames[column] = pd.to_numeric(frames[column], errors="coerce")
    if frames[["frame_id", "reliability", "iou"]].isna().any().any():
        raise RuntimeError("Non-numeric or missing frame_id/reliability/IoU values found")
    if frames.duplicated(["sequence", "frame_id"]).any():
        examples = frames.loc[
            frames.duplicated(["sequence", "frame_id"], keep=False),
            ["sequence", "frame_id"],
        ].head()
        raise RuntimeError(f"Duplicate sequence/frame_id records found:\n{examples}")
    if not frames["reliability"].between(0.0, 1.0).all():
        raise RuntimeError("Reliability values outside [0, 1]")
    if not frames["iou"].between(0.0, 1.0).all():
        raise RuntimeError("IoU values outside [0, 1]")

    num_sequences = int(frames["sequence"].nunique())
    if args.expected_sequences > 0 and num_sequences != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} sequences but found {num_sequences}"
        )

    calculated_label = np.full(len(frames), np.nan, dtype=np.float64)
    calculated_label[frames["iou"].to_numpy() >= args.pos_thr] = 1.0
    calculated_label[frames["iou"].to_numpy() <= args.neg_thr] = 0.0
    if "hard_label" in frames.columns:
        supplied = pd.to_numeric(frames["hard_label"], errors="coerce").to_numpy()
        mismatch = ~(
            (np.isnan(supplied) & np.isnan(calculated_label))
            | np.isclose(supplied, calculated_label, equal_nan=True)
        )
        if mismatch.any():
            raise RuntimeError(
                f"hard_label disagrees with IoU thresholds in {int(mismatch.sum())} rows"
            )
    frames["hard_label"] = calculated_label

    reliability = frames["reliability"].to_numpy(dtype=np.float64)
    iou = frames["iou"].to_numpy(dtype=np.float64)
    hard_mask = np.isfinite(calculated_label)
    y_true = calculated_label[hard_mask].astype(np.int64)
    probability = reliability[hard_mask]
    if set(np.unique(y_true)) != {0, 1}:
        raise RuntimeError("Hard-label subset must contain both positive and negative samples")

    positive = probability[y_true == 1]
    negative = probability[y_true == 0]
    pearson_value = float(pearsonr(reliability, iou)[0])
    spearman_value = float(spearmanr(reliability, iou)[0])
    ece, equal_width = equal_width_calibration(y_true, probability, args.bins)
    equal_frequency = equal_frequency_calibration(y_true, probability, args.bins)

    metrics = {
        "dataset_label": args.dataset_label,
        "model_label": args.model_label,
        "num_sequences": num_sequences,
        "num_fresh_reliability_samples": int(len(frames)),
        "num_hard_label_samples": int(hard_mask.sum()),
        "num_positive": int((y_true == 1).sum()),
        "num_negative": int((y_true == 0).sum()),
        "num_ignored_mid_iou": int((~hard_mask).sum()),
        "positive_prevalence": float(y_true.mean()),
        "auroc": float(binary_auroc(y_true, probability)),
        "auprc": float(binary_average_precision(y_true, probability)),
        "ece_equal_width_10bin": ece,
        "brier_score": float(np.mean((probability - y_true) ** 2)),
        "pearson_r": pearson_value,
        "spearman_rho": spearman_value,
        "positive_mean_reliability": float(positive.mean()),
        "negative_mean_reliability": float(negative.mean()),
        "mean_reliability_difference": float(positive.mean() - negative.mean()),
        "reliability_min": float(reliability.min()),
        "reliability_max": float(reliability.max()),
        "reliability_mean": float(reliability.mean()),
        "fraction_score_at_least_0_5": float((reliability >= 0.5).mean()),
        "pos_threshold": args.pos_thr,
        "neg_threshold": args.neg_thr,
        "calibration_display": "10 equal-frequency score groups",
        "ece_definition": "10 fixed equal-width bins on [0, 1]",
    }

    equal_width.to_csv(output_dir / "calibration_equal_width_10bin.csv", index=False)
    equal_frequency.to_csv(
        output_dir / "calibration_equal_frequency_10group.csv", index=False
    )
    pd.DataFrame(
        [
            distribution_row("negative_iou_le_0.3", negative),
            distribution_row("positive_iou_ge_0.5", positive),
        ]
    ).to_csv(output_dir / "score_distribution_summary.csv", index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(output_dir / "table_e1_reliability_quality.csv", index=False)
    with (output_dir / "metrics_recomputed.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    png_path, pdf_path = save_main_figure(
        output_dir,
        args.model_label,
        positive,
        negative,
        equal_frequency,
        metrics,
        args.zoom_min,
    )
    write_summary(
        output_dir / "e1_result_summary.md", args, metrics, png_path.name
    )

    manifest = {
        "input_frames_csv": str(frames_path),
        "input_sha256": sha256sum(frames_path),
        "source_mode": source_audit["source_mode"],
        "output_dir": str(output_dir),
        "model_label": args.model_label,
        "dataset_label": args.dataset_label,
        "source_note": args.source_note,
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
