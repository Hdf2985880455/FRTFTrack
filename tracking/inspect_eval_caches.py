import glob
import os
import pickle

import numpy as np


for path in sorted(glob.glob("output/test/result_plots/**/eval_data.pkl", recursive=True)):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        trackers = data.get("trackers", [])
        curves = np.asarray(data.get("ave_success_rate_plot_overlap", []), dtype=float)
        pcurves = np.asarray(data.get("ave_success_rate_plot_center", []), dtype=float)
        ncurves = np.asarray(data.get("ave_success_rate_plot_center_norm", []), dtype=float)
        valid = np.asarray(data.get("valid_sequence", []), dtype=bool)
        thresholds = np.asarray(data.get("threshold_set_overlap", []), dtype=float)
        if curves.ndim != 3 or valid.size != curves.shape[0]:
            continue
        for i, tracker in enumerate(trackers):
            ident = " ".join(str(tracker.get(k, "")) for k in ("name", "param", "disp_name"))
            low = ident.lower()
            if not any(token in low for token in ("rtf", "rgre", "stagec", "ftqm", "reliability")):
                continue
            curve = curves[valid, i].mean(axis=0) * 100
            discrete = float(curve.mean())
            trapz = float(np.trapz(curve, thresholds) / (thresholds[-1] - thresholds[0]))
            precision = float(pcurves[valid, i].mean(axis=0)[20] * 100)
            norm_precision = float(ncurves[valid, i].mean(axis=0)[20] * 100)
            if 49.0 <= discrete <= 55.0 or 49.0 <= trapz <= 55.0:
                print(
                    f"{path}\n  {ident}\n"
                    f"  n={valid.sum()} discrete={discrete:.4f} trapz={trapz:.4f} "
                    f"P={precision:.4f} PNorm={norm_precision:.4f}"
                )
    except Exception as exc:
        print(f"ERROR {path}: {exc}")
