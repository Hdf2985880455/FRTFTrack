from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualize_tnl2k_cases import (
    BASE_1OF6,
    COLORS,
    FAILURE_CASES,
    OURS_1OF6,
    OUT_ROOT,
    SUCCESS_CASES,
    crop_region_all,
    find_sequence_dir,
    fit_text,
    image_files,
    read_boxes,
    read_language,
    transform_box,
)


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"


def _color01(rgb):
    return tuple(v / 255.0 for v in rgb)


def _load_crop_and_boxes(seq_dir, frame_idx, gt_boxes, base_boxes, ours_boxes):
    files = image_files(seq_dir)
    idx = min(max(frame_idx - 1, 0), len(files) - 1)
    img = Image.open(files[idx]).convert("RGB")
    width, height = img.size

    gt = gt_boxes[idx]
    base = base_boxes[idx]
    ours = ours_boxes[idx]
    crop = crop_region_all([gt, base, ours], width, height)
    cropped = img.crop(crop)
    cw, ch = cropped.size

    boxes = {
        "GT": transform_box(gt, crop, 1.0, 1.0),
        "Baseline": transform_box(base, crop, 1.0, 1.0),
        "FRTFTrack": transform_box(ours, crop, 1.0, 1.0),
    }
    return np.asarray(cropped), boxes, (cw, ch)


def _draw_box(ax, box, color, lw=0.9):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1),
        max(x2 - x1, 1e-3),
        max(y2 - y1, 1e-3),
        linewidth=lw,
        edgecolor=_color01(color),
        facecolor="none",
        joinstyle="miter",
    )
    ax.add_patch(rect)


def render_vector_pdf(cases, outfile, row_letters=True):
    n_rows = len(cases)
    n_cols = max(len(c["frames"]) for c in cases)

    # Strict paper-style grid: every sequence row has four equally sized frames.
    frame_w = 2.62
    frame_h = 1.36
    gap_x = 0.08
    row_h = 1.82
    left_w = 0.34
    right_pad = 0.05
    top_pad = 0.05
    legend_h = 0.34

    fig_w = left_w + n_cols * frame_w + (n_cols - 1) * gap_x + right_pad
    fig_h = top_pad + n_rows * row_h + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for row, case in enumerate(cases):
        seq_dir = find_sequence_dir(case["name"])
        gt = read_boxes(seq_dir / "groundtruth.txt")
        base = read_boxes(BASE_1OF6 / f"{case['name']}.txt")
        ours = read_boxes(OURS_1OF6 / f"{case['name']}.txt")
        language = fit_text(read_language(seq_dir), 120)

        y_top = fig_h - top_pad - row * row_h
        lang_y = (y_top - 0.16) / fig_h
        frame_y = (y_top - 0.30 - frame_h) / fig_h

        if row_letters:
            fig.text(
                0.025,
                lang_y,
                f"({chr(ord('a') + row)})",
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="center",
            )

        fig.text(
            (left_w + (n_cols * frame_w + (n_cols - 1) * gap_x) / 2.0) / fig_w,
            lang_y,
            f'Language description:  "{language}"',
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

        for col, frame_idx in enumerate(case["frames"]):
            x = left_w + col * (frame_w + gap_x)
            ax = fig.add_axes([x / fig_w, frame_y, frame_w / fig_w, frame_h / fig_h])
            image, boxes, (cw, ch) = _load_crop_and_boxes(
                seq_dir, frame_idx, gt, base, ours
            )
            ax.imshow(
                image,
                extent=[0, cw, ch, 0],
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_xlim(0, cw)
            ax.set_ylim(ch, 0)
            ax.set_aspect("auto")
            ax.axis("off")

            for key in ["GT", "Baseline", "FRTFTrack"]:
                _draw_box(ax, boxes[key], COLORS[key], lw=0.9)

            ax.text(
                0.012,
                0.955,
                f"#{frame_idx:03d}",
                transform=ax.transAxes,
                fontsize=8.0,
                color="#f5dc00",
                fontweight="bold",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            )

    legend_y = 0.16 / fig_h
    legend_items = ["GT", "Baseline", "FRTFTrack"]
    total_w = 0.46
    start_x = 0.5 - total_w / 2
    step = total_w / len(legend_items)
    for i, key in enumerate(legend_items):
        x = start_x + i * step
        rect = patches.Rectangle(
            (x, legend_y),
            0.025,
            0.018,
            transform=fig.transFigure,
            linewidth=1.4,
            edgecolor=_color01(COLORS[key]),
            facecolor="none",
        )
        fig.add_artist(rect)
        fig.text(
            x + 0.033,
            legend_y + 0.009,
            key,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222",
        )

    out_path = OUT_ROOT / outfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.004)
    plt.close(fig)
    return out_path


def main():
    success = render_vector_pdf(
        SUCCESS_CASES,
        "tnl2k_success_cases_1of6_vector_overlay_aligned.pdf",
    )
    failure = render_vector_pdf(
        FAILURE_CASES,
        "tnl2k_failure_cases_1of6_vector_overlay_aligned.pdf",
    )
    print(success)
    print(failure)


if __name__ == "__main__":
    main()
