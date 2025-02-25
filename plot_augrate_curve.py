from typing import Dict, List, Literal, cast, Tuple
import pathlib as pl
import re, json, os, argparse
from dataclasses import dataclass
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import scienceplots


Aug = Literal["none", "conll", "wgold", "the_elder_scrolls", "dekker_fantasy"]

AUG2PRETTY: Dict[Aug, str] = {
    "none": "",
    "conll": "conll",
    "wgold": "wgold",
    "the_elder_scrolls": "fantasy",
    "dekker_fantasy": "novelties",
}


@dataclass
class XPParams:
    aug: Aug
    aug_rate: float


def get_file_xp_params(path: str) -> XPParams:
    m = re.match(
        r"global_results_([^0-9]+)_([^_]+)_[0-9]+\.json", os.path.basename(path)
    )
    if m is None:
        return XPParams("none", 0)
    aug = m.group(1)
    # assert aug in ["none", "conll", "wgold", "the_elder_scrolls", "dekker_fantasy"]
    aug = cast(Aug, aug)
    aug_rate = float(m.group(2))
    return XPParams(aug, aug_rate)


def metrics_from_file(path: str) -> Dict[Literal["precision", "recall", "f1"], float]:
    with open(path) as f:
        metrics = json.load(f)
    return metrics


def load_metrics(
    directory: pl.Path, metric: Literal["precision", "recall", "f1"]
) -> Dict[Aug, Dict[float, List[float]]]:
    metrics = {
        "none": defaultdict(list),
        "conll": defaultdict(list),
        "wgold": defaultdict(list),
        "the_elder_scrolls": defaultdict(list),
        "dekker_fantasy": defaultdict(list),
    }

    for path in directory.glob("global_results_*.json"):
        xpparams = get_file_xp_params(str(path))
        xpmetrics = metrics_from_file(str(path))
        try:
            metrics[xpparams.aug][xpparams.aug_rate].append(xpmetrics[metric])
        except KeyError:
            continue

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
    ASPECT_RATIO = 0.7

    plt.style.use(["science", "grid"])
    plt.rcParams.update({"font.size": FONTSIZE})

    fig, axs = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * ASPECT_RATIO)
    )

    f1s = load_metrics(pl.Path("./metrics/phdthesis"), "f1")
    precisions = load_metrics(pl.Path("./metrics/phdthesis"), "precision")
    recalls = load_metrics(pl.Path("./metrics/phdthesis"), "recall")

    for aug, ax in [
        ("conll", axs[0][0]),
        ("wgold", axs[0][1]),
        ("the_elder_scrolls", axs[1][0]),
        ("dekker_fantasy", axs[1][1]),
    ]:
        plot_errorbar(ax, f1s[aug], "F1")
        plot_errorbar(ax, precisions[aug], "Precision")
        plot_errorbar(ax, recalls[aug], "Recall")
        ax.set_title(AUG2PRETTY[aug])

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
