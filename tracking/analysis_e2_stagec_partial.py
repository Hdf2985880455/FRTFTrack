import _init_paths
import numpy as np

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist


dataset_name = "tnl2k_lang"
params = [
    ("reliability_query_stageC_scale008_scalemod_temporal", "Normal feedback"),
    ("reliability_query_stageC_scale008_scalemod_temporal_fixed_r05", "Fixed r=0.5"),
]
trackers = []
for parameter_name, display_name in params:
    trackers.extend(trackerlist(
        name="mmtrack",
        parameter_name=parameter_name,
        dataset_name=dataset_name,
        run_ids=None,
        display_name=display_name,
    ))

data = extract_results(
    trackers,
    get_dataset(dataset_name),
    report_name="e2_stagec_partial",
    skip_missing_seq=True,
)

valid = np.asarray(data["valid_sequence"], dtype=bool)
success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
precision = np.asarray(data["ave_success_rate_plot_center"], dtype=float)
norm_precision = np.asarray(data["ave_success_rate_plot_center_norm"], dtype=float)
thresholds = np.asarray(data["threshold_set_overlap"], dtype=float)

print(f"PAIRED_SEQUENCES={valid.sum()}")
for i, tracker in enumerate(data["trackers"]):
    success_curve = success[valid, i].mean(axis=0)
    auc = np.trapz(success_curve, thresholds) * 100.0
    p = precision[valid, i].mean(axis=0)[20] * 100.0
    pnorm = norm_precision[valid, i].mean(axis=0)[20] * 100.0
    print(
        f'{tracker["disp_name"]}: '
        f'AUC_trapz={auc:.4f}, PNorm={pnorm:.4f}, P={p:.4f}'
    )
