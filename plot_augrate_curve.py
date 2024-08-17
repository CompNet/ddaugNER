from typing import Dict, List, Literal, cast
import pathlib as pl
import re, json, os
from dataclasses import dataclass
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
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
        "morrowind": defaultdict(list),
    }

    for path in directory.glob("global_results_*.json"):
        xpparams = get_file_xp_params(str(path))
        xpmetrics = metrics_from_file(str(path))
        metrics[xpparams.aug][xpparams.aug_rate].append(xpmetrics[metric])

    return {
        key: {**values, **metrics["none"]}
        for key, values in metrics.items()
        if key != "none"
    }  # type: ignore


def plot_errorbar(
    ax,
    metrics_list: Dict[float, List[float]],
    metric: Literal["precision", "recall", "f1"],
):
    xy = sorted(metrics_list.items(), key=lambda kv: kv[0])
    ax.errorbar(
        [x for x, _ in xy],
        [mean(y) for _, y in xy],
        yerr=[stdev(y) for _, y in xy],
        elinewidth=2,
        capsize=6,
        linewidth=3,
        label=metric,
    )


if __name__ == "__main__":

    plt.style.use("science")
    plt.rcParams.update({"font.size": 22})

    fig, axs = plt.subplots(2, 2)

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
        ax.grid()

    fig.text(0.5, 0.05, "Augmentation rate", ha="center")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fancybox=True)
    plt.show()
