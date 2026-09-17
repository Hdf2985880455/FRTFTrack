#!/usr/bin/env python3
"""E5 offline analysis: five pre-specified attributes plus sequence length.

This script does not train or run a tracker. It reads existing MMTrack and
FRTFTrack prediction files through the project's official evaluation code.
All 17 TNL2K attributes are computed for auditability, while the paper figure
shows only the five attributes fixed before inspecting the results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), here.parent, *here.parents]
    for candidate in candidates:
        if (candidate / "lib" / "test" / "evaluation").is_dir():
            return candidate
    raise RuntimeError("Cannot locate MMTrack project root (missing lib/test/evaluation).")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.analysis.extract_results import extract_results  # noqa: E402
from lib.test.evaluation import get_dataset, trackerlist  # noqa: E402


TRACKER_SPECS: Tuple[Tuple[str, str], ...] = (
    ("baseline_medium", "MMTrack"),
    (
        "reliability_query_stageC_scale008_scalemod_temporal",
        "FRTFTrack",
    ),
)

ATTRIBUTE_SPECS: Tuple[Tuple[str, str], ...] = (
    ("CM", "Camera Motion"),
    ("ROT", "Rotation"),
    ("DEF", "Deformation"),
    ("FOC", "Full Occlusion"),
    ("IV", "Illumination Variation"),
    ("OV", "Out-of-View"),
    ("POC", "Partial Occlusion"),
    ("VC", "Viewpoint Change"),
    ("SV", "Scale Variation"),
    ("BC", "Background Clutter"),
    ("MB", "Motion Blur"),
    ("ARC", "Aspect Ratio Change"),
    ("LR", "Low Resolution"),
    ("FM", "Fast Motion"),
    ("AS", "Adversarial Samples"),
    ("TC", "Thermal Crossover"),
    ("MS", "Modality Switch"),
)

# Fixed before inspecting any attribute result. Do not reorder or replace based
# on measured gains.
MAIN5: Tuple[str, ...] = ("FOC", "OV", "FM", "SV", "VC")
LENGTH_GROUPS: Tuple[str, ...] = ("Short", "Medium", "Long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tnl2k_attribute_dir",
        type=Path,
        required=True,
        help="Directory containing <sequence>_attribute.txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/e5_main5"),
    )
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument(
        "--figure_only",
        action="store_true",
        help="Regenerate the figure from existing CSV files without reevaluation.",
    )
    return parser.parse_args()


def trapz(values: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


def read_attribute_vector(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    vector = np.asarray([float(value) for value in values], dtype=np.float64)
    if vector.size != len(ATTRIBUTE_SPECS):
        raise ValueError(
            f"{path}: expected {len(ATTRIBUTE_SPECS)} values, got {vector.size}"
        )
    return vector


def build_trackers(dataset_name: str):
    trackers = []
    for parameter_name, display_name in TRACKER_SPECS:
        trackers.extend(
            trackerlist(
                "mmtrack",
                parameter_name,
                dataset_name,
                display_name=display_name,
            )
        )
    return trackers


def evaluate_dataset(dataset_name: str, report_name: str) -> Dict[str, object]:
    dataset = get_dataset(dataset_name)
    trackers = build_trackers(dataset_name)
    data = extract_results(
        trackers,
        dataset,
        report_name,
        skip_missing_seq=False,
        plot_bin_gap=0.05,
        exclude_invalid_frames=False,
    )

    sequence_names = [sequence.name for sequence in dataset]
    if list(data["sequences"]) != sequence_names:
        raise RuntimeError(f"{dataset_name}: evaluation cache sequence order mismatch")
    if not all(bool(value) for value in data["valid_sequence"]):
        raise RuntimeError(f"{dataset_name}: one or more sequences are invalid")

    params = [tracker["param"] for tracker in data["trackers"]]
    expected = [item[0] for item in TRACKER_SPECS]
    if params != expected:
        raise RuntimeError(
            f"{dataset_name}: tracker order mismatch; expected {expected}, got {params}"
        )

    return {"dataset": dataset, "eval": data}


def metric_arrays(bundle: Dict[str, object]) -> Dict[str, np.ndarray]:
    data = bundle["eval"]
    success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=np.float64)
    precision = np.asarray(data["ave_success_rate_plot_center"], dtype=np.float64)
    norm_precision = np.asarray(
        data["ave_success_rate_plot_center_norm"], dtype=np.float64
    )
    overlap_thresholds = np.asarray(data["threshold_set_overlap"], dtype=np.float64)
    center_thresholds = np.asarray(data["threshold_set_center"], dtype=np.float64)
    center_norm_thresholds = np.asarray(
        data["threshold_set_center_norm"], dtype=np.float64
    )

    expected_shape = (len(bundle["dataset"]), len(TRACKER_SPECS))
    if success.shape[:2] != expected_shape:
        raise RuntimeError(f"Unexpected success array shape: {success.shape}")
    for name, array in (
        ("success", success),
        ("precision", precision),
        ("normalized precision", norm_precision),
    ):
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite values found in {name} array")

    p_indices = np.flatnonzero(np.isclose(center_thresholds, 20.0))
    pnorm_indices = np.flatnonzero(np.isclose(center_norm_thresholds, 0.20))
    if p_indices.size != 1 or pnorm_indices.size != 1:
        raise RuntimeError("Cannot locate P@20 or PNorm@0.20 threshold")

    auc = trapz(success, overlap_thresholds, axis=2) * 100.0
    p = precision[:, :, int(p_indices[0])] * 100.0
    pnorm = norm_precision[:, :, int(pnorm_indices[0])] * 100.0
    return {
        "auc": auc,
        "p": p,
        "pnorm": pnorm,
        "success": success,
        "overlap_thresholds": overlap_thresholds,
    }


def paired_bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    batch_size: int = 1000,
) -> Tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64).reshape(-1)
    if differences.size == 0:
        return math.nan, math.nan
    estimates: List[np.ndarray] = []
    remaining = repeats
    while remaining > 0:
        current = min(batch_size, remaining)
        draw = rng.integers(
            0,
            differences.size,
            size=(current, differences.size),
            endpoint=False,
        )
        estimates.append(differences[draw].mean(axis=1))
        remaining -= current
    distribution = np.concatenate(estimates)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def summarize_indices(
    indices: np.ndarray,
    metrics: Dict[str, np.ndarray],
    rng: np.random.Generator,
    repeats: int,
) -> Dict[str, float]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return {
            "mmtrack_auc": math.nan,
            "frtftrack_auc": math.nan,
            "delta_auc": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "mmtrack_pnorm": math.nan,
            "frtftrack_pnorm": math.nan,
            "delta_pnorm": math.nan,
            "mmtrack_p": math.nan,
            "frtftrack_p": math.nan,
            "delta_p": math.nan,
        }

    auc = metrics["auc"][indices]
    pnorm = metrics["pnorm"][indices]
    p = metrics["p"][indices]
    ci_low, ci_high = paired_bootstrap_ci(
        auc[:, 1] - auc[:, 0], rng=rng, repeats=repeats
    )
    return {
        "mmtrack_auc": float(auc[:, 0].mean()),
        "frtftrack_auc": float(auc[:, 1].mean()),
        "delta_auc": float((auc[:, 1] - auc[:, 0]).mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mmtrack_pnorm": float(pnorm[:, 0].mean()),
        "frtftrack_pnorm": float(pnorm[:, 1].mean()),
        "delta_pnorm": float((pnorm[:, 1] - pnorm[:, 0]).mean()),
        "mmtrack_p": float(p[:, 0].mean()),
        "frtftrack_p": float(p[:, 1].mean()),
        "delta_p": float((p[:, 1] - p[:, 0]).mean()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def attribute_rows(
    dataset,
    metrics: Dict[str, np.ndarray],
    attribute_dir: Path,
    rng: np.random.Generator,
    repeats: int,
) -> List[Dict[str, object]]:
    vectors: List[np.ndarray] = []
    missing: List[str] = []
    for sequence in dataset:
        path = attribute_dir / f"{sequence.name}_attribute.txt"
        if not path.is_file():
            missing.append(sequence.name)
            continue
        vectors.append(read_attribute_vector(path))
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} TNL2K attribute files; examples: {preview}"
        )
    matrix = np.vstack(vectors)
    if matrix.shape != (len(dataset), len(ATTRIBUTE_SPECS)):
        raise RuntimeError(f"Unexpected attribute matrix shape: {matrix.shape}")

    rows: List[Dict[str, object]] = []
    for column, (code, full_name) in enumerate(ATTRIBUTE_SPECS):
        indices = np.flatnonzero(matrix[:, column] > 0)
        summary = summarize_indices(indices, metrics, rng, repeats)
        rows.append(
            {
                "attribute": code,
                "full_name": full_name,
                "num_sequences": int(indices.size),
                "is_main5": code in MAIN5,
                **summary,
            }
        )
    return rows


def length_rows(
    dataset_name: str,
    dataset,
    metrics: Dict[str, np.ndarray],
    rng: np.random.Generator,
    repeats: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    lengths = np.asarray(
        [len(sequence.ground_truth_rect) for sequence in dataset], dtype=np.int64
    )
    names = np.asarray([sequence.name for sequence in dataset], dtype=object)
    # Primary key: length; deterministic tie-breaker: sequence name.
    order = np.lexsort((names, lengths))
    split_indices = np.array_split(order, len(LENGTH_GROUPS))

    metric_rows: List[Dict[str, object]] = []
    boundary_rows: List[Dict[str, object]] = []
    for group_name, indices in zip(LENGTH_GROUPS, split_indices):
        selected_lengths = lengths[indices]
        summary = summarize_indices(indices, metrics, rng, repeats)
        common = {
            "dataset": dataset_name,
            "length_group": group_name,
            "num_sequences": int(indices.size),
            "min_frames": int(selected_lengths.min()),
            "max_frames": int(selected_lengths.max()),
            "total_frames": int(selected_lengths.sum()),
        }
        metric_rows.append({**common, **summary})
        boundary_rows.append(common)
    return metric_rows, boundary_rows


def render_figure(
    main5_rows: Sequence[Dict[str, object]],
    tnl_rows: Sequence[Dict[str, object]],
    lasot_rows: Sequence[Dict[str, object]],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figure_dir.mkdir(parents=True, exist_ok=True)
    blue = "#0072B2"
    orange = "#D55E00"
    negative_color = "#999999"
    fig = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=(1.12, 1.0))
    ax_attr = fig.add_subplot(grid[:, 0])
    ax_tnl = fig.add_subplot(grid[0, 1])
    ax_lasot = fig.add_subplot(grid[1, 1])

    x_attr = np.arange(len(main5_rows), dtype=float)
    deltas = np.asarray([float(row["delta_auc"]) for row in main5_rows])
    bar_colors = [orange if value >= 0 else negative_color for value in deltas]
    ax_attr.bar(
        x_attr,
        deltas,
        width=0.62,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    labels = [
        f'{row["attribute"]}\n(n={int(row["num_sequences"])})'
        for row in main5_rows
    ]
    ax_attr.set_xticks(x_attr, labels)
    ax_attr.set_ylabel("Delta AUC (percentage points)")
    ax_attr.set_title("(a) Five pre-specified TNL2K attributes", loc="left")
    ax_attr.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    upper_attr = float(max(deltas.max(), 1.0))
    ax_attr.set_ylim(0.0, upper_attr * 1.28)
    for xi, delta in zip(x_attr, deltas):
        ax_attr.annotate(
            f"{delta:.2f}",
            xy=(xi, delta),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            clip_on=True,
        )
    ax_attr.text(
        0.02,
        0.985,
        "FRTFTrack - MMTrack",
        transform=ax_attr.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#444444",
    )

    def plot_length_panel(ax, rows, title):
        x = np.arange(len(rows), dtype=float)
        mm_auc = np.asarray([float(row["mmtrack_auc"]) for row in rows])
        fr_auc = np.asarray([float(row["frtftrack_auc"]) for row in rows])
        ax.plot(x, mm_auc, color=blue, marker="o", linewidth=2, label="MMTrack")
        ax.plot(x, fr_auc, color=orange, marker="s", linewidth=2, label="FRTFTrack")
        ax.set_xticks(x, [row["length_group"] for row in rows])
        ax.set_ylabel("AUC (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.set_xlim(-0.55, len(rows) - 1 + 0.55)
        values = np.concatenate([mm_auc, fr_auc])
        lower = float(values.min())
        upper = float(values.max())
        span_local = max(upper - lower, 2.0)
        ax.set_ylim(lower - 0.18 * span_local, upper + 0.40 * span_local)
        for xi, row, first, second in zip(x, rows, mm_auc, fr_auc):
            if xi == 0:
                value_offset = (-6, 4)
                value_alignment = "right"
            else:
                value_offset = (6, 4)
                value_alignment = "left"
            ax.annotate(
                f"{first:.1f}",
                xy=(xi, first),
                xytext=value_offset,
                textcoords="offset points",
                color=blue,
                ha=value_alignment,
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
            ax.annotate(
                f"{second:.1f}",
                xy=(xi, second),
                xytext=value_offset,
                textcoords="offset points",
                color=orange,
                ha=value_alignment,
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
            ax.text(
                xi,
                0.94,
                f'Delta={float(row["delta_auc"]):.2f}',
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                clip_on=True,
            )

    plot_length_panel(ax_tnl, tnl_rows, "(b) TNL2K by sequence length")
    plot_length_panel(ax_lasot, lasot_rows, "(c) LaSOT by sequence length")
    ax_tnl.legend(loc="lower left", frameon=False)

    png_path = figure_dir / "e5_main5_and_length.png"
    pdf_path = figure_dir / "e5_main5_and_length.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: Iterable[Dict[str, object]], first_column: str) -> str:
    rows = list(rows)
    header = (
        f"| {first_column} | n | MMTrack AUC | FRTFTrack AUC | Delta | 95% CI |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        "| {name} | {n} | {mm:.2f} | {fr:.2f} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n".format(
            name=row["attribute"] if "attribute" in row else row["length_group"],
            n=int(row["num_sequences"]),
            mm=float(row["mmtrack_auc"]),
            fr=float(row["frtftrack_auc"]),
            delta=float(row["delta_auc"]),
            lo=float(row["ci_low"]),
            hi=float(row["ci_high"]),
        )
        for row in rows
    )
    return header + body


def write_summary(
    path: Path,
    main5_rows: Sequence[Dict[str, object]],
    tnl_rows: Sequence[Dict[str, object]],
    lasot_rows: Sequence[Dict[str, object]],
    overall: Dict[str, Dict[str, float]],
) -> None:
    text = [
        "# E5 analysis results\n\n",
        "All AUC values use trapezoidal integration of the success curve. ",
        "Each sequence has equal weight. Delta is FRTFTrack minus MMTrack.\n\n",
        "## Overall sanity check\n\n",
        "| Dataset | MMTrack AUC | FRTFTrack AUC | Delta |\n",
        "|---|---:|---:|---:|\n",
    ]
    for dataset_name in ("TNL2K", "LaSOT"):
        row = overall[dataset_name]
        text.append(
            f'| {dataset_name} | {row["mmtrack_auc"]:.2f} | '
            f'{row["frtftrack_auc"]:.2f} | {row["delta_auc"]:+.2f} |\n'
        )
    text.extend(
        [
            "\n## Five pre-specified TNL2K attributes\n\n",
            markdown_table(main5_rows, "Attribute"),
            "\n## TNL2K length groups\n\n",
            markdown_table(tnl_rows, "Length group"),
            "\n## LaSOT length groups\n\n",
            markdown_table(lasot_rows, "Length group"),
        ]
    )
    path.write_text("".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    attribute_dir = args.tnl2k_attribute_dir.expanduser().resolve()
    if not attribute_dir.is_dir():
        raise FileNotFoundError(f"Attribute directory does not exist: {attribute_dir}")

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    csv_dir = output_dir / "csv"
    figure_dir = output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.figure_only:
        main5_rows = read_csv(csv_dir / "tnl2k_attribute_metrics_main5.csv")
        tnl_length_rows = read_csv(csv_dir / "tnl2k_length_metrics.csv")
        lasot_length_rows = read_csv(csv_dir / "lasot_length_metrics.csv")
        render_figure(main5_rows, tnl_length_rows, lasot_length_rows, figure_dir)
        print(f"Figure regenerated: {figure_dir / 'e5_main5_and_length.png'}")
        return

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"ATTRIBUTE_DIR={attribute_dir}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"MAIN5={','.join(MAIN5)}")
    print("Computing paired TNL2K evaluation cache...")
    tnl_bundle = evaluate_dataset("tnl2k_lang", "e5_main5_tnl2k_paired")
    print("Computing paired LaSOT evaluation cache...")
    lasot_bundle = evaluate_dataset("lasot_lang", "e5_main5_lasot_paired")

    if len(tnl_bundle["dataset"]) != 700:
        raise RuntimeError(f'TNL2K sequence count is {len(tnl_bundle["dataset"])}, expected 700')
    if len(lasot_bundle["dataset"]) != 280:
        raise RuntimeError(f'LaSOT sequence count is {len(lasot_bundle["dataset"])}, expected 280')

    tnl_metrics = metric_arrays(tnl_bundle)
    lasot_metrics = metric_arrays(lasot_bundle)
    rng = np.random.default_rng(args.bootstrap_seed)

    all_attribute_rows = attribute_rows(
        tnl_bundle["dataset"],
        tnl_metrics,
        attribute_dir,
        rng,
        args.bootstrap_repeats,
    )
    row_by_code = {row["attribute"]: row for row in all_attribute_rows}
    main5_rows = [row_by_code[code] for code in MAIN5]

    tnl_length_rows, tnl_boundaries = length_rows(
        "TNL2K",
        tnl_bundle["dataset"],
        tnl_metrics,
        rng,
        args.bootstrap_repeats,
    )
    lasot_length_rows, lasot_boundaries = length_rows(
        "LaSOT",
        lasot_bundle["dataset"],
        lasot_metrics,
        rng,
        args.bootstrap_repeats,
    )

    write_csv(csv_dir / "tnl2k_attribute_metrics_all17.csv", all_attribute_rows)
    write_csv(csv_dir / "tnl2k_attribute_metrics_main5.csv", main5_rows)
    write_csv(csv_dir / "tnl2k_length_metrics.csv", tnl_length_rows)
    write_csv(csv_dir / "lasot_length_metrics.csv", lasot_length_rows)
    write_csv(
        csv_dir / "length_group_boundaries.csv",
        [*tnl_boundaries, *lasot_boundaries],
    )

    overall = {}
    for display_name, metrics in (("TNL2K", tnl_metrics), ("LaSOT", lasot_metrics)):
        all_indices = np.arange(metrics["auc"].shape[0])
        summary = summarize_indices(
            all_indices,
            metrics,
            np.random.default_rng(args.bootstrap_seed),
            args.bootstrap_repeats,
        )
        overall[display_name] = summary

    render_figure(main5_rows, tnl_length_rows, lasot_length_rows, figure_dir)
    write_summary(
        output_dir / "e5_result_summary.md",
        main5_rows,
        tnl_length_rows,
        lasot_length_rows,
        overall,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "tracker_specs": TRACKER_SPECS,
                "attribute_order_all17": [item[0] for item in ATTRIBUTE_SPECS],
                "main5_pre_specified": MAIN5,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_repeats": args.bootstrap_repeats,
                "auc": "sequence-equal success curve; trapezoidal integration",
                "length_grouping": "dataset-specific equal-count tertiles; ties by sequence name",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nOverall Trapz AUC sanity check:")
    for dataset_name, row in overall.items():
        print(
            f'  {dataset_name}: MMTrack={row["mmtrack_auc"]:.3f}, '
            f'FRTFTrack={row["frtftrack_auc"]:.3f}, Delta={row["delta_auc"]:+.3f}'
        )
    print(f"\nDone. Summary: {output_dir / 'e5_result_summary.md'}")
    print(f"Figure: {figure_dir / 'e5_main5_and_length.png'}")


if __name__ == "__main__":
    main()
