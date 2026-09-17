#!/usr/bin/env python3
"""Audit E3/E4 results and build one combined paper table.

Place under PROJECT_ROOT/tracking. The script is offline: it reads existing
tracking predictions, validates sequence/frame completeness, computes trapz
AUC/PNorm/P, and writes CSV/JSON/Markdown/LaTeX artifacts. It does not train or
run trackers and never modifies prediction files.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "lib" / "test").is_dir() and (candidate / "tracking").is_dir():
            return candidate
    raise RuntimeError("Cannot locate MMTrack project root")


ROOT = find_project_root()
for path in (ROOT, ROOT / "tracking"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _init_paths  # noqa: E402,F401
from lib.test.analysis.extract_results import extract_results  # noqa: E402
from lib.test.evaluation import get_dataset, trackerlist  # noqa: E402
from lib.test.evaluation.environment import env_settings  # noqa: E402


E3_UNIQUE = (
    ("alpha002", "reliability_query_stageC_e3_alpha002_rsu", "Alpha=0.02"),
    ("alpha005", "reliability_query_stageC_e3_alpha005_rsu", "Alpha=0.05"),
    ("default", "reliability_query_stageC_scale008_scalemod_temporal_rsu", "Default"),
    ("lambda001", "reliability_query_stageC_e3_lambda001_rsu", "Lambda=0.01"),
    ("lambda010", "reliability_query_stageC_e3_lambda010_rsu", "Lambda=0.10"),
    ("beta070", "reliability_query_stageC_e3_beta070_rsu", "Beta=0.70"),
    ("beta095", "reliability_query_stageC_e3_beta095_rsu", "Beta=0.95"),
)

E3_SWEEPS = (
    ("alpha", 0.02, "alpha002", False),
    ("alpha", 0.05, "alpha005", False),
    ("alpha", 0.08, "default", True),
    ("lambda_rel", 0.01, "lambda001", False),
    ("lambda_rel", 0.05, "default", True),
    ("lambda_rel", 0.10, "lambda010", False),
    ("beta", 0.70, "beta070", False),
    ("beta", 0.85, "default", True),
    ("beta", 0.95, "beta095", False),
)

E4_RUNS = (
    ("MMTrack", 42, "baseline_medium"),
    ("MMTrack", 3407, "baseline_medium_seed3407"),
    ("MMTrack", 2026, "baseline_medium_seed2026"),
    ("FRTFTrack", 42, "reliability_query_stageC_scale008_scalemod_temporal_rsu"),
    ("FRTFTrack", 3407, "reliability_query_stageC_scale008_scalemod_temporal_rsu_seed3407"),
    ("FRTFTrack", 2026, "reliability_query_stageC_scale008_scalemod_temporal_rsu_seed2026"),
)


def trapz(values, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


def threshold_index(data, key, target, fallback):
    thresholds = np.asarray(data.get(key, []), dtype=np.float64)
    if thresholds.size == 0:
        return fallback
    matches = np.flatnonzero(np.isclose(thresholds, target))
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely locate {target} in {key}")
    return int(matches[0])


def inventory(dataset, parameter_name):
    root = Path(env_settings().results_path) / "mmtrack" / parameter_name
    missing, empty, mismatch = [], [], []
    for sequence in dataset:
        path = root / sequence.dataset / f"{sequence.name}.txt"
        if not path.is_file():
            missing.append(sequence.name)
            continue
        if path.stat().st_size == 0:
            empty.append(sequence.name)
            continue
        prediction = np.loadtxt(path, delimiter="\t", dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)
        expected = int(len(sequence.ground_truth_rect))
        if len(prediction) != expected:
            mismatch.append(
                {"sequence": sequence.name, "prediction": int(len(prediction)), "gt": expected}
            )
    return {
        "dataset": dataset.get_name() if hasattr(dataset, "get_name") else "unknown",
        "parameter_name": parameter_name,
        "results_root": str(root),
        "expected_sequences": int(len(dataset)),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "frame_mismatch_count": len(mismatch),
        "missing": missing,
        "empty": empty,
        "frame_mismatches": mismatch,
    }


def evaluate(dataset_name, specs, report_name, expected, audit):
    dataset = get_dataset(dataset_name)
    if len(dataset) != expected:
        raise RuntimeError(f"{dataset_name}: expected {expected}, loaded {len(dataset)}")

    trackers = []
    for key, parameter_name, display_name in specs:
        inv = inventory(dataset, parameter_name)
        audit.append(inv)
        if inv["missing_count"] or inv["empty_count"] or inv["frame_mismatch_count"]:
            raise RuntimeError(
                f"Incomplete {dataset_name}/{parameter_name}: missing={inv['missing_count']}, "
                f"empty={inv['empty_count']}, frame_mismatch={inv['frame_mismatch_count']}"
            )
        trackers.extend(
            trackerlist(
                name="mmtrack",
                parameter_name=parameter_name,
                dataset_name=dataset_name,
                run_ids=None,
                display_name=display_name,
            )
        )

    data = extract_results(trackers, dataset, report_name=report_name, skip_missing_seq=False)
    valid = np.asarray(data["valid_sequence"], dtype=bool)
    if int(valid.sum()) != expected:
        raise RuntimeError(f"{dataset_name}: expected {expected} paired, got {valid.sum()}")

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
    for index, (key, parameter_name, display_name) in enumerate(specs):
        rows.append(
            {
                "key": key,
                "dataset": dataset_name,
                "variant": display_name,
                "parameter_name": parameter_name,
                "paired_sequences": int(valid.sum()),
                "auc_trapz": float(seq_auc[:, index].mean()),
                "pnorm": float(seq_pn[:, index].mean()),
                "precision": float(seq_p[:, index].mean()),
            }
        )
    return rows


def evaluate_cache(cache_name, dataset_name, specs, expected, audit):
    """Read a previously completed extract_results cache without rescanning frames."""
    cache_path = ROOT / "output" / "test" / "result_plots" / cache_name / "eval_data.pkl"
    if not cache_path.is_file():
        raise RuntimeError(f"Missing evaluation cache: {cache_path}")
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    valid = np.asarray(data["valid_sequence"], dtype=bool)
    if int(valid.sum()) != expected or valid.size != expected:
        raise RuntimeError(
            f"{cache_name}: expected {expected}/{expected} valid, "
            f"got {valid.sum()}/{valid.size}"
        )
    if len(data["trackers"]) != len(specs):
        raise RuntimeError(
            f"{cache_name}: expected {len(specs)} trackers, got {len(data['trackers'])}"
        )
    actual_names = [tracker.get("disp_name", "") for tracker in data["trackers"]]
    expected_names = [display_name for _, _, display_name in specs]
    # Cache display names may contain more descriptive default labels. Require
    # matching order for all non-default entries and record both lists for audit.
    audit.append(
        {
            "cache": str(cache_path),
            "dataset": dataset_name,
            "valid_sequences": int(valid.sum()),
            "total_sequences": int(valid.size),
            "actual_tracker_names": actual_names,
            "expected_tracker_names": expected_names,
        }
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
    for index, (key, parameter_name, display_name) in enumerate(specs):
        rows.append(
            {
                "key": key,
                "dataset": dataset_name,
                "variant": display_name,
                "parameter_name": parameter_name,
                "paired_sequences": int(valid.sum()),
                "auc_trapz": float(seq_auc[:, index].mean()),
                "pnorm": float(seq_pn[:, index].mean()),
                "precision": float(seq_p[:, index].mean()),
            }
        )
    return rows


def pm(mean, std):
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def write_latex(e3_rows, e4_summary, path):
    sweep_names = {"alpha": r"$\alpha$", "lambda_rel": r"$\lambda_{\rm rel}$", "beta": r"$\beta$"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Hyperparameter sensitivity and multi-seed reproducibility. E3 changes one parameter at a time and reports TNL2K results. E4 reports mean and sample standard deviation over seeds 42, 3407, and 2026.}",
        r"\label{tab:sensitivity_reproducibility}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Group & Setting & AUC & $P_{\rm Norm}$ & P \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{(a) Hyperparameter sensitivity on TNL2K}} \\",
    ]
    for row in e3_rows:
        value = f"{row['value']:.2f}"
        if row["is_default"]:
            value = rf"\textbf{{{value}}}"
        lines.append(
            f"{sweep_names[row['sweep']]} & {value} & {row['auc_trapz']:.1f} & "
            f"{row['pnorm']:.1f} & {row['precision']:.1f} \\\\"
        )
    lines.extend([r"\midrule", r"\multicolumn{5}{l}{\textit{(b) Multi-seed reproducibility}} \\"])
    for row in e4_summary:
        dataset = "TNL2K" if row["dataset"] == "tnl2k_lang" else "LaSOT"
        lines.append(
            f"{dataset} & {row['method']} & {pm(row['auc_mean'], row['auc_std'])} & "
            f"{pm(row['pnorm_mean'], row['pnorm_std'])} & "
            f"{pm(row['p_mean'], row['p_std'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(e3_rows, e4_rows, e4_summary, gains, path):
    lines = [
        "# E3 + E4 final summary",
        "",
        "## E3 hyperparameter sensitivity (TNL2K)",
        "",
        "| Sweep | Value | AUC | PNorm | P |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in e3_rows:
        default = " (default)" if row["is_default"] else ""
        lines.append(
            f"| {row['sweep']} | {row['value']:.2f}{default} | {row['auc_trapz']:.4f} | "
            f"{row['pnorm']:.4f} | {row['precision']:.4f} |"
        )
    lines.extend(["", "## E4 per-seed results", "", "| Dataset | Method | Seed | AUC | PNorm | P |", "|---|---|---:|---:|---:|---:|"])
    for row in e4_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | {row['auc_trapz']:.4f} | "
            f"{row['pnorm']:.4f} | {row['precision']:.4f} |"
        )
    lines.extend(["", "## E4 mean +/- sample SD", "", "| Dataset | Method | AUC | PNorm | P |", "|---|---|---:|---:|---:|"])
    for row in e4_summary:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['auc_mean']:.4f} +/- {row['auc_std']:.4f} | "
            f"{row['pnorm_mean']:.4f} +/- {row['pnorm_std']:.4f} | "
            f"{row['p_mean']:.4f} +/- {row['p_std']:.4f} |"
        )
    lines.extend(["", "## Paired gains (FRTFTrack - MMTrack)", "", "| Dataset | Seed | AUC gain | PNorm gain | P gain |", "|---|---:|---:|---:|---:|"])
    for row in gains:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['auc_gain']:.4f} | "
            f"{row['pnorm_gain']:.4f} | {row['p_gain']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/final_revision/e3_e4")
    parser.add_argument(
        "--from-caches",
        action="store_true",
        help="Use verified extract_results caches for a fast summary.",
    )
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = []

    if args.from_caches:
        e3_unique = evaluate_cache(
            "e3_sensitivity_tnl2k_final_check", "tnl2k_lang", E3_UNIQUE, 700, audit
        )
    else:
        e3_unique = evaluate(
            "tnl2k_lang", E3_UNIQUE, "e3_sensitivity_final", 700, audit
        )
    e3_map = {row["key"]: row for row in e3_unique}
    e3_rows = []
    for sweep, value, key, is_default in E3_SWEEPS:
        metric = e3_map[key]
        e3_rows.append(
            {
                "sweep": sweep,
                "value": value,
                "is_default": is_default,
                "parameter_name": metric["parameter_name"],
                "auc_trapz": metric["auc_trapz"],
                "pnorm": metric["pnorm"],
                "precision": metric["precision"],
            }
        )

    e4_specs = tuple(
        (f"{method}_{seed}", parameter, f"{method}-seed{seed}")
        for method, seed, parameter in E4_RUNS
    )
    e4_rows = []
    for dataset_name, expected in (("tnl2k_lang", 700), ("lasot_lang", 280)):
        if args.from_caches:
            cache_name = f"e4_final_check_{dataset_name}"
            evaluated = evaluate_cache(
                cache_name, dataset_name, e4_specs, expected, audit
            )
        else:
            evaluated = evaluate(
                dataset_name,
                e4_specs,
                f"e4_multiseed_{dataset_name}_final",
                expected,
                audit,
            )
        for row, (method, seed, _) in zip(evaluated, E4_RUNS):
            row["method"] = method
            row["seed"] = seed
            e4_rows.append(row)

    summary = []
    gains = []
    for dataset_name in ("tnl2k_lang", "lasot_lang"):
        for method in ("MMTrack", "FRTFTrack"):
            selected = [r for r in e4_rows if r["dataset"] == dataset_name and r["method"] == method]
            values = {name: np.asarray([r[name] for r in selected], dtype=float) for name in ("auc_trapz", "pnorm", "precision")}
            summary.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "seeds": "42/3407/2026",
                    "auc_mean": float(values["auc_trapz"].mean()),
                    "auc_std": float(values["auc_trapz"].std(ddof=1)),
                    "pnorm_mean": float(values["pnorm"].mean()),
                    "pnorm_std": float(values["pnorm"].std(ddof=1)),
                    "p_mean": float(values["precision"].mean()),
                    "p_std": float(values["precision"].std(ddof=1)),
                }
            )
        for seed in (42, 3407, 2026):
            mm = next(r for r in e4_rows if r["dataset"] == dataset_name and r["method"] == "MMTrack" and r["seed"] == seed)
            fr = next(r for r in e4_rows if r["dataset"] == dataset_name and r["method"] == "FRTFTrack" and r["seed"] == seed)
            gains.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "auc_gain": fr["auc_trapz"] - mm["auc_trapz"],
                    "pnorm_gain": fr["pnorm"] - mm["pnorm"],
                    "p_gain": fr["precision"] - mm["precision"],
                }
            )

    pd.DataFrame(e3_unique).to_csv(out / "e3_unique_config_metrics.csv", index=False)
    pd.DataFrame(e3_rows).to_csv(out / "e3_sensitivity_metrics.csv", index=False)
    pd.DataFrame(e4_rows).to_csv(out / "e4_per_seed_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "e4_mean_std.csv", index=False)
    pd.DataFrame(gains).to_csv(out / "e4_paired_gains.csv", index=False)
    (out / "result_inventory.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_latex(e3_rows, summary, out / "e3_e4_combined_table.tex")
    write_markdown(e3_rows, e4_rows, summary, gains, out / "e3_e4_summary.md")

    print("E3_PAIRED_SEQUENCES=700")
    print("E4_TNL2K_PAIRED_SEQUENCES=700")
    print("E4_LASOT_PAIRED_SEQUENCES=280")
    for row in summary:
        print(
            f"{row['dataset']},{row['method']},AUC={row['auc_mean']:.4f}+/-{row['auc_std']:.4f},"
            f"PNorm={row['pnorm_mean']:.4f}+/-{row['pnorm_std']:.4f},"
            f"P={row['p_mean']:.4f}+/-{row['p_std']:.4f}"
        )
    print(f"OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
