import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.test.evaluation import get_dataset


def xywh_iou(box_a, box_b):
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return np.nan
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else np.nan


def expected_calibration_error(y_true, prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": bin_id,
                "lower": edges[bin_id],
                "upper": edges[bin_id + 1],
                "count": 0,
                "mean_confidence": np.nan,
                "positive_rate": np.nan,
            })
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += count / len(y_true) * abs(confidence - accuracy)
        rows.append({
            "bin": bin_id,
            "lower": edges[bin_id],
            "upper": edges[bin_id + 1],
            "count": count,
            "mean_confidence": confidence,
            "positive_rate": accuracy,
        })
    return float(ece), pd.DataFrame(rows)


def safe_sequence_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def latest_trace(trace_dir, sequence_name):
    pattern = os.path.join(trace_dir, safe_sequence_name(sequence_name) + "_*.tsv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_boxes(path):
    boxes = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    return boxes[:, :4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tnl2k_lang", "lasot_lang"])
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing one prediction txt per sequence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pos-thr", type=float, default=0.5)
    parser.add_argument("--neg-thr", type=float, default=0.3)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = get_dataset(args.dataset)
    records = []
    missing_trace = []
    missing_result = []

    for seq in dataset:
        trace_path = latest_trace(args.trace_dir, seq.name)
        result_path = os.path.join(args.results_dir, seq.name + ".txt")

        if trace_path is None:
            missing_trace.append(seq.name)
            continue
        if not os.path.isfile(result_path):
            missing_result.append(seq.name)
            continue

        trace = pd.read_csv(trace_path, sep="\t", na_values=["None"])
        required = {"frame_id", "should_update_rel", "pred_rel"}
        if not required.issubset(trace.columns):
            raise RuntimeError(
                "Trace {} lacks columns {}".format(
                    trace_path, sorted(required - set(trace.columns))
                )
            )

        trace = trace[
            (trace["should_update_rel"] == 1) & trace["pred_rel"].notna()
        ].copy()

        pred_boxes = load_boxes(result_path)
        gt_boxes = np.asarray(seq.ground_truth_rect, dtype=np.float64).reshape(-1, 4)
        max_frame = min(len(pred_boxes), len(gt_boxes))

        for row in trace.itertuples(index=False):
            frame_id = int(row.frame_id)
            if frame_id < 0 or frame_id >= max_frame:
                continue
            reliability = float(row.pred_rel)
            if not np.isfinite(reliability):
                continue
            iou = xywh_iou(pred_boxes[frame_id], gt_boxes[frame_id])
            if not np.isfinite(iou):
                continue

            if iou >= args.pos_thr:
                hard_label = 1.0
            elif iou <= args.neg_thr:
                hard_label = 0.0
            else:
                hard_label = np.nan

            records.append({
                "dataset": args.dataset,
                "sequence": seq.name,
                "frame_id": frame_id,
                "reliability": float(np.clip(reliability, 0.0, 1.0)),
                "iou": float(iou),
                "hard_label": hard_label,
                "trace_file": os.path.basename(trace_path),
            })

    frames = pd.DataFrame(records)
    if frames.empty:
        raise RuntimeError("No matched reliability samples were found.")

    frames.to_csv(os.path.join(args.output_dir, "reliability_frames.csv"), index=False)

    pearson_value, pearson_p = pearsonr(frames["reliability"], frames["iou"])
    spearman_value, spearman_p = spearmanr(frames["reliability"], frames["iou"])

    valid = frames.dropna(subset=["hard_label"]).copy()
    y_true = valid["hard_label"].astype(int).to_numpy()
    prob = valid["reliability"].to_numpy()
    if len(np.unique(y_true)) != 2:
        raise RuntimeError("AUROC requires both positive and negative samples.")

    auroc = roc_auc_score(y_true, prob)
    auprc = average_precision_score(y_true, prob)
    brier = brier_score_loss(y_true, prob)
    ece, calibration = expected_calibration_error(y_true, prob, args.bins)
    calibration.to_csv(os.path.join(args.output_dir, "calibration_bins.csv"), index=False)

    frames["reliability_bin"] = pd.cut(
        frames["reliability"],
        bins=np.linspace(0.0, 1.0, args.bins + 1),
        include_lowest=True,
    )
    reliability_bins = frames.groupby("reliability_bin", observed=False).agg(
        count=("iou", "size"),
        mean_reliability=("reliability", "mean"),
        mean_iou=("iou", "mean"),
        median_iou=("iou", "median"),
    ).reset_index()
    reliability_bins["reliability_bin"] = reliability_bins["reliability_bin"].astype(str)
    reliability_bins.to_csv(os.path.join(args.output_dir, "reliability_bins.csv"), index=False)

    positive = valid[valid["hard_label"] == 1]
    negative = valid[valid["hard_label"] == 0]
    metrics = {
        "dataset": args.dataset,
        "num_sequences_in_dataset": len(dataset),
        "num_sequences_with_samples": int(frames["sequence"].nunique()),
        "num_fresh_reliability_samples": int(len(frames)),
        "num_hard_label_samples": int(len(valid)),
        "num_positive": int((y_true == 1).sum()),
        "num_negative": int((y_true == 0).sum()),
        "positive_mean_reliability": float(positive["reliability"].mean()),
        "negative_mean_reliability": float(negative["reliability"].mean()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "ece_10bin": float(ece),
        "brier_score": float(brier),
        "pearson_r": float(pearson_value),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_value),
        "spearman_p": float(spearman_p),
        "missing_trace_count": len(missing_trace),
        "missing_result_count": len(missing_result),
        "missing_traces": missing_trace,
        "missing_results": missing_result,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(6.2, 5.0))
    plt.hexbin(frames["reliability"], frames["iou"], gridsize=30,
               mincnt=1, cmap="viridis")
    plt.colorbar(label="Frame count")
    plt.xlabel("Predicted reliability")
    plt.ylabel("Tracking IoU")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "reliability_iou_hexbin.png"), dpi=300)
    plt.close()

    nonempty = calibration[calibration["count"] > 0]
    plt.figure(figsize=(5.5, 5.0))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(nonempty["mean_confidence"], nonempty["positive_rate"],
             marker="o", label="FRTFTrack RE")
    plt.xlabel("Mean predicted reliability")
    plt.ylabel("Empirical positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "calibration_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6.2, 4.5))
    bins = np.linspace(0, 1, 21)
    plt.hist(negative["reliability"], bins=bins, alpha=0.65,
             density=True, label="Negative: IoU <= {:.1f}".format(args.neg_thr))
    plt.hist(positive["reliability"], bins=bins, alpha=0.65,
             density=True, label="Positive: IoU >= {:.1f}".format(args.pos_thr))
    plt.xlabel("Predicted reliability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "reliability_histogram.png"), dpi=300)
    plt.close()

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
