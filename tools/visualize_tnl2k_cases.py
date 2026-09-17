from pathlib import Path
import re

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/tnl2k/test"
OUT_ROOT = ROOT / "output/visualizations/tnl2k_cases"

BASE_1OF6 = ROOT / "output/test/tracking_results/mmtrack/baseline_medium/tnl2k_lang"
OURS_1OF6 = ROOT / "output/test/tracking_results/mmtrack/reliability_query_stageC_scale008_scalemod_temporal/tnl2k_lang"

COLORS = {
    "GT": (40, 180, 80),
    "Baseline": (220, 60, 55),
    "FRTFTrack": (55, 115, 230),
}


SUCCESS_CASES = [
    {
        "name": "Transform_video_X22-Done",
        "label": "Transform X22",
        "frames": [318, 389, 575, 815],
        "note": "low-data, head target",
    },
    {
        "name": "monitor_manypeople",
        "label": "Multi-person",
        "frames": [149, 261, 459, 570],
        "note": "similar distractors",
    },
    {
        "name": "Transform_video_X24-Done",
        "label": "Transform X24",
        "frames": [729, 968, 1085, 1513],
        "note": "low-data, local target",
    },
    {
        "name": "Chess_video_16",
        "label": "Chess occlusion",
        "frames": [58, 333, 420, 672],
        "note": "occlusion, clutter",
    },
]

FAILURE_CASES = [
    {
        "name": "Baseball_video_03-Done",
        "label": "Baseball",
        "frames": [173, 195, 220, 250],
        "note": "fast motion",
    },
    {
        "name": "test_023_Marvel_video_01_done",
        "label": "Marvel",
        "frames": [23, 65, 115, 171],
        "note": "weak language cue",
    },
]


def read_boxes(path):
    boxes = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            vals = [float(x) for x in re.split(r"[\s,]+", line.strip()) if x]
            if len(vals) >= 4:
                boxes.append(vals[:4])
    return boxes


def find_sequence_dir(name):
    matches = list(DATA_ROOT.glob(f"TNL2K_test_subset_*/*/{name}"))
    if matches:
        return matches[0]
    for p in DATA_ROOT.glob("TNL2K_test_subset_*/*"):
        if p.is_dir() and p.name == name:
            return p
    raise FileNotFoundError(f"Cannot find sequence directory: {name}")


def image_files(seq_dir):
    imgs = seq_dir / "imgs"
    files = sorted(
        [p for p in imgs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not files:
        raise FileNotFoundError(f"No images found under {imgs}")
    return files


def _box_center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def crop_region(gt, base, ours, width, height, margin_ratio=1.35):
    valid = [b for b in [gt, ours] if b and b[2] > 1 and b[3] > 1]
    if not valid:
        valid = [b for b in [gt, base, ours] if b and b[2] > 1 and b[3] > 1]
    if not valid:
        return 0, 0, width, height

    # Use a target-centric crop. A very poor baseline prediction can be far away
    # and would otherwise force a full-frame crop where all useful boxes become tiny.
    if base and base[2] > 1 and base[3] > 1 and gt and gt[2] > 1 and gt[3] > 1:
        gcx, gcy = _box_center(gt)
        bcx, bcy = _box_center(base)
        gdiag = max((gt[2] ** 2 + gt[3] ** 2) ** 0.5, 1)
        if ((bcx - gcx) ** 2 + (bcy - gcy) ** 2) ** 0.5 < 3.0 * gdiag:
            valid.append(base)

    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[0] + b[2] for b in valid)
    y2 = max(b[1] + b[3] for b in valid)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    margin = max(bw, bh) * margin_ratio

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = min(width, max(bw + 2 * margin, 180))
    crop_h = min(height, max(bh + 2 * margin, 140))

    left = max(0, min(width - crop_w, cx - crop_w / 2))
    top = max(0, min(height - crop_h, cy - crop_h / 2))
    right = min(width, left + crop_w)
    bottom = min(height, top + crop_h)
    return int(left), int(top), int(right), int(bottom)


def transform_box(box, crop, scale_x, scale_y):
    left, top, _, _ = crop
    x, y, w, h = box
    return [
        (x - left) * scale_x,
        (y - top) * scale_y,
        (x + w - left) * scale_x,
        (y + h - top) * scale_y,
    ]


def draw_box(draw, box, color, image_size, width=4):
    w, h = image_size
    x1 = max(0, min(w - 1, box[0]))
    y1 = max(0, min(h - 1, box[1]))
    x2 = max(0, min(w - 1, box[2]))
    y2 = max(0, min(h - 1, box[3]))
    if x2 <= 0 or y2 <= 0 or x1 >= w - 1 or y1 >= h - 1 or x2 <= x1 or y2 <= y1:
        return
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def font(size=18, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()

def trim_white_margins(img, pad=8):
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.size[0], bbox[2] + pad)
    bottom = min(img.size[1], bbox[3] + pad)
    return img.crop((left, top, right, bottom))


def render_frame(seq_dir, frame_idx, gt_boxes, base_boxes, ours_boxes, thumb=(260, 180)):
    files = image_files(seq_dir)
    idx = min(max(frame_idx - 1, 0), len(files) - 1)
    img = Image.open(files[idx]).convert("RGB")
    w, h = img.size

    gt = gt_boxes[idx]
    base = base_boxes[idx]
    ours = ours_boxes[idx]
    crop = crop_region_all([gt, base, ours], w, h)
    cropped = img.crop(crop)
    cw, ch = cropped.size
    resized = cropped.resize(thumb, Image.BICUBIC)
    sx, sy = thumb[0] / cw, thumb[1] / ch

    draw = ImageDraw.Draw(resized)
    draw_box(draw, transform_box(gt, crop, sx, sy), COLORS["GT"], thumb, 2)
    draw_box(draw, transform_box(base, crop, sx, sy), COLORS["Baseline"], thumb, 2)
    draw_box(draw, transform_box(ours, crop, sx, sy), COLORS["FRTFTrack"], thumb, 2)

    label = f"#{frame_idx:03d}"
    label_font = font(15, True)
    tw = int(draw.textlength(label, font=label_font)) + 10
    draw.rectangle([0, 0, tw, 24], fill=(255, 255, 255))
    draw.text((5, 3), label, fill=(245, 220, 0), font=label_font)
    return resized


def draw_legend(canvas, x, y):
    draw = ImageDraw.Draw(canvas)
    f = font(18, True)
    cursor = x
    for key in ["GT", "Baseline", "FRTFTrack"]:
        draw.rectangle([cursor, y + 5, cursor + 28, y + 21], outline=COLORS[key], width=3)
        draw.text((cursor + 36, y), key, fill=(30, 30, 30), font=f)
        cursor += 170


def read_language(seq_dir):
    path = seq_dir / "language.txt"
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore").strip().replace("\n", " ")
    return " ".join(text.split())


def fit_text(text, max_chars=96):
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def render_grid(cases, title, outfile):
    thumb_w, thumb_h = 278, 172
    left_w = 44
    gap = 8
    side_pad = 14
    top_pad = 10
    lang_h = 30
    row_gap = 12
    legend_h = 50
    row_h = lang_h + thumb_h + row_gap
    max_frames = max(len(c["frames"]) for c in cases)
    width = side_pad * 2 + left_w + max_frames * thumb_w + (max_frames - 1) * gap
    height = top_pad + len(cases) * row_h + legend_h + 6
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    letter_font = font(18, True)
    lang_font = font(18, True)
    quote_font = font(18, False)

    for row, case in enumerate(cases):
        y = top_pad + row * row_h
        seq_dir = find_sequence_dir(case["name"])
        gt = read_boxes(seq_dir / "groundtruth.txt")
        base = read_boxes(BASE_1OF6 / f"{case['name']}.txt")
        ours = read_boxes(OURS_1OF6 / f"{case['name']}.txt")
        language = fit_text(read_language(seq_dir), 120)

        letter = f"({chr(ord('a') + row)})"
        draw.text((side_pad + 2, y + lang_h - 23), letter, fill=(20, 20, 20), font=letter_font)

        prefix = "Language description:  "
        lang_line = f'{prefix}"{language}"'
        lang_x = side_pad + left_w + max(0, (max_frames * thumb_w + (max_frames - 1) * gap - int(draw.textlength(lang_line, font=lang_font))) // 2)
        draw.text((lang_x, y + 2), prefix, fill=(15, 15, 15), font=lang_font)
        draw.text((lang_x + int(draw.textlength(prefix, font=lang_font)), y + 2), f'"{language}"', fill=(15, 15, 15), font=quote_font)

        for col, frame_idx in enumerate(case["frames"]):
            x = side_pad + left_w + col * (thumb_w + gap)
            tile = render_frame(seq_dir, frame_idx, gt, base, ours, (thumb_w, thumb_h))
            canvas.paste(tile, (x, y + lang_h))

    legend_y = top_pad + len(cases) * row_h + 8
    draw.line([(side_pad, legend_y - 8), (width - side_pad, legend_y - 8)], fill=(230, 230, 230), width=1)
    legend_width = 3 * 170
    draw_legend(canvas, (width - legend_width) // 2, legend_y)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / outfile
    canvas = trim_white_margins(canvas, pad=6)
    canvas.save(path)
    return path


def crop_region_all(boxes, width, height, margin_ratio=0.25):
    valid = [b for b in boxes if b and b[2] > 1 and b[3] > 1]
    if not valid:
        return 0, 0, width, height

    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[0] + b[2] for b in valid)
    y2 = max(b[1] + b[3] for b in valid)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    margin = max(bw, bh) * margin_ratio

    left = max(0, x1 - margin)
    top = max(0, y1 - margin)
    right = min(width, x2 + margin)
    bottom = min(height, y2 + margin)
    return int(left), int(top), int(right), int(bottom)


def draw_legend_compact(draw, x, y):
    f = font(20, True)
    cursor = x
    for key in ["GT", "Baseline", "FRTFTrack"]:
        draw.rectangle([cursor, y + 7, cursor + 34, y + 25], outline=COLORS[key], width=3)
        draw.text((cursor + 44, y), key, fill=(30, 30, 30), font=f)
        cursor += 190


def render_single_case_frame(case, seq_dir, frame_idx, gt_boxes, base_boxes, ours_boxes, out_dir):
    files = image_files(seq_dir)
    idx = min(max(frame_idx - 1, 0), len(files) - 1)
    img = Image.open(files[idx]).convert("RGB")
    w, h = img.size
    gt = gt_boxes[idx]
    base = base_boxes[idx]
    ours = ours_boxes[idx]

    crop = crop_region_all([gt, base, ours], w, h)
    cropped = img.crop(crop)
    cw, ch = cropped.size

    canvas_w, canvas_h = 1100, 760
    top_h, bottom_h, pad = 86, 70, 26
    max_w = canvas_w - 2 * pad
    max_h = canvas_h - top_h - bottom_h - 2 * pad
    scale = min(max_w / cw, max_h / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cropped.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    lang = fit_text(read_language(seq_dir), 130)
    lang_prefix = "Language description:  "
    lang_font = font(23, True)
    quote_font = font(23, False)
    lang_w = draw.textlength(lang_prefix, font=lang_font) + draw.textlength(f'"{lang}"', font=quote_font)
    lang_x = max(pad, int((canvas_w - lang_w) / 2))
    draw.text((lang_x, 18), lang_prefix, fill=(15, 15, 15), font=lang_font)
    draw.text((lang_x + int(draw.textlength(lang_prefix, font=lang_font)), 18), f'"{lang}"', fill=(15, 15, 15), font=quote_font)

    img_x = (canvas_w - new_w) // 2
    img_y = top_h
    canvas.paste(resized, (img_x, img_y))
    sx, sy = new_w / cw, new_h / ch

    def box_on_canvas(box):
        bx = transform_box(box, crop, sx, sy)
        return [bx[0] + img_x, bx[1] + img_y, bx[2] + img_x, bx[3] + img_y]

    draw_box(draw, box_on_canvas(gt), COLORS["GT"], (canvas_w, canvas_h), 2)
    draw_box(draw, box_on_canvas(base), COLORS["Baseline"], (canvas_w, canvas_h), 2)
    draw_box(draw, box_on_canvas(ours), COLORS["FRTFTrack"], (canvas_w, canvas_h), 2)

    label = f"#{frame_idx:03d}"
    label_font = font(22, True)
    tw = int(draw.textlength(label, font=label_font)) + 16
    draw.rectangle([img_x, img_y, img_x + tw, img_y + 34], fill=(255, 255, 255))
    draw.text((img_x + 8, img_y + 4), label, fill=(245, 220, 0), font=label_font)

    legend_y = canvas_h - bottom_h + 16
    draw.line([(pad, legend_y - 14), (canvas_w - pad, legend_y - 14)], fill=(230, 230, 230), width=1)
    draw_legend_compact(draw, (canvas_w - 3 * 190) // 2, legend_y)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name}_frame_{frame_idx:04d}.png"
    canvas.save(out_path)
    return out_path


def render_per_frame_cases(cases, split_name):
    out_dir = OUT_ROOT / "per_frame" / split_name
    paths = []
    for case in cases:
        seq_dir = find_sequence_dir(case["name"])
        gt = read_boxes(seq_dir / "groundtruth.txt")
        base = read_boxes(BASE_1OF6 / f"{case['name']}.txt")
        ours = read_boxes(OURS_1OF6 / f"{case['name']}.txt")
        for frame_idx in case["frames"]:
            paths.append(render_single_case_frame(case, seq_dir, frame_idx, gt, base, ours, out_dir))
    return paths


def main():
    success = render_grid(
        SUCCESS_CASES,
        "TNL2K Success Cases under 1/6 Training Data",
        "tnl2k_success_cases_1of6.png",
    )
    failure = render_grid(
        FAILURE_CASES,
        "TNL2K Failure Cases under 1/6 Training Data",
        "tnl2k_failure_cases_1of6.png",
    )
    per_success = render_per_frame_cases(SUCCESS_CASES, "success")
    per_failure = render_per_frame_cases(FAILURE_CASES, "failure")
    print(success)
    print(failure)
    print(f"Generated {len(per_success)} success frame images under {OUT_ROOT / 'per_frame/success'}")
    print(f"Generated {len(per_failure)} failure frame images under {OUT_ROOT / 'per_frame/failure'}")


if __name__ == "__main__":
    main()
