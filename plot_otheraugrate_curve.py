from typing import Dict, List, Literal, cast, Tuple, Optional
import pathlib as pl
import re, json, os, argparse
from dataclasses import dataclass
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import scienceplots


AugMethod = Literal["balance_upsample", "replace", "standard", "none"]

AUGMETHOD2PRETTY: Dict[AugMethod, str] = {
    "balance_upsample": "upsample and balance",
    "replace": "replace",
}


@dataclass
class XPParams:
    aug_method: AugMethod
    aug_rate: float


def get_file_xp_params(path: str) -> Optional[XPParams]:
    m = re.match(
        r"global_results_([^0-9]+)_([^_]+)_[0-9]+\.json", os.path.basename(path)
    )
    if m is None:
        raise ValueError(os.path.basename(path))
    aug_method = m.group(1)
    if not aug_method in ["balance_upsample", "replace", "standard", "none"]:
        return None
    aug_method = cast(AugMethod, aug_method)
    aug_rate = float(m.group(2))
    return XPParams(aug_method, aug_rate)


def metrics_from_file(path: str) -> Dict[Literal["precision", "recall", "f1"], float]:
    with open(path) as f:
        metrics = json.load(f)
    return metrics


def load_metrics(
    directory: pl.Path, metric: Literal["precision", "recall", "f1"]
) -> Dict[AugMethod, Dict[float, List[float]]]:

    metrics = {
        "balance_upsample": defaultdict(list),
        "replace": defaultdict(list),
        "standard": defaultdict(list),
        "none": defaultdict(list),
    }

    for path in directory.glob("global_results_*.json"):
        xpparams = get_file_xp_params(str(path))
        if xpparams is None:
            continue
        xpmetrics = metrics_from_file(str(path))
        metrics[xpparams.aug_method][xpparams.aug_rate].append(xpmetrics[metric])

    return {
        key: {**values, **metrics["none"]}
        for key, values in metrics.items()
        if key != "none"
    }  # type: ignore


def confidence_interval(metric: List[float]) -> Tuple[float, float]:
    """
    :return: ``(mean - low_ci_bound, high_ci_bound - mean)``
    """
    metric_mean = mean(metric)
    low, high = stats.t.interval(
        0.95,
        len(metric) - 1,
        loc=metric_mean,
        scale=stats.sem(metric),
    )
    return (metric_mean - low, high - metric_mean)


def plot_errorbar(ax, metrics_list: Dict[float, List[float]], metric: str):
    xy = sorted(metrics_list.items(), key=lambda kv: kv[0])
    x = [x for x, _ in xy]
    y = [y for _, y in xy]
    ci = np.array([confidence_interval(e) for e in y]).swapaxes(0, 1)
    ax.errorbar(
        x,
        [mean(e) for e in y],
        ci,
        elinewidth=2,
        capsize=3,
        linewidth=1.5,
        label=metric,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pl.Path)
    args = parser.parse_args()

    FONTSIZE = 8
    COLUMN_WIDTH_IN = 6.3
    ASPECT_RATIO = 0.35

    plt.style.use(["science", "grid"])
    plt.rcParams.update({"font.size": FONTSIZE})

    fig, axs = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * ASPECT_RATIO)
    )

    f1s = load_metrics(pl.Path("./metrics/phdthesis"), "f1")
    precisions = load_metrics(pl.Path("./metrics/phdthesis"), "precision")
    recalls = load_metrics(pl.Path("./metrics/phdthesis"), "recall")

    for aug_method, ax in [
        ("balance_upsample", axs[0]),
        ("replace", axs[1]),
    ]:
        plot_errorbar(ax, f1s[aug_method], "F1")
        plot_errorbar(ax, precisions[aug_method], "Precision")
        plot_errorbar(ax, recalls[aug_method], "Recall")
        ax.set_title(AUGMETHOD2PRETTY[aug_method])

    fig.text(0.5, 0.00, "Augmentation rate", ha="center")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fancybox=True,
    )

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
