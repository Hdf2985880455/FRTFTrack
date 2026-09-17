import _init_paths

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


dataset_name = "tnl2k_lang"
trackers = []
trackers.extend(
    trackerlist(
        name="mmtrack",
        parameter_name="rtftrack_lite_ftqm_1of6",
        dataset_name=dataset_name,
        run_ids=None,
        display_name="Normal feedback",
    )
)
trackers.extend(
    trackerlist(
        name="mmtrack",
        parameter_name="rtftrack_lite_ftqm_1of6_fixed_r05",
        dataset_name=dataset_name,
        run_ids=None,
        display_name="Fixed r=0.5",
    )
)

dataset = get_dataset(dataset_name)
print_results(
    trackers,
    dataset,
    dataset_name,
    merge_results=True,
    plot_types=("success", "norm_prec", "prec"),
)
