from typing import Dict, List, Literal, cast
import pathlib as pl
import re, json, os
from dataclasses import dataclass
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
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


def get_file_xp_params(path: str) -> XPParams:
    m = re.match(
        r"global_results_([^0-9]+)_([^_]+)_[0-9]+\.json", os.path.basename(path)
    )
    if m is None:
        raise ValueError(os.path.basename(path))
    aug_method = m.group(1)
    assert aug_method in ["balance_upsample", "replace", "standard", "none"]
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
        xpmetrics = metrics_from_file(str(path))
        metrics[xpparams.aug_method][xpparams.aug_rate].append(xpmetrics[metric])

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

    fig, axs = plt.subplots(1, 2)

    f1s = load_metrics(pl.Path("./metrics/phdthesis"), "f1")
    precisions = load_metrics(pl.Path("./metrics/phdthesis"), "precision")
    recalls = load_metrics(pl.Path("./metrics/phdthesis"), "recall")

    for aug_method, ax in [
        ("balance_upsample", axs[0]),
        ("replace", axs[1]),
    ]:
        plot_errorbar(ax, f1s[aug_method], "f1")
        plot_errorbar(ax, precisions[aug_method], "precision")
        plot_errorbar(ax, recalls[aug_method], "recall")
        ax.set_title(AUGMETHOD2PRETTY[aug_method])
        ax.grid()

    fig.text(0.5, 0.05, "Augmentation rate", ha="center")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fancybox=True)
    plt.show()
