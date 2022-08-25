from typing import List
import argparse, json, os
import matplotlib.pyplot as plt
from rich import print
from ddaugner.datas.dekker import groups


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-if", "--input-files", nargs="*", default=[])
    parser.add_argument("-pg", "--plot-group", type=str, default=None)
    parser.add_argument(
        "-oi", "--only-include", nargs="*", default=None, help="A list of books"
    )
    parser.add_argument("-m", "--metric", type=str, default="f1")
    args = parser.parse_args()

    metrics_list: List[dict] = []

    for input_file in args.input_files:
        with open(input_file) as f:
            if args.only_include:
                metrics_list.append(
                    {
                        book: metrics
                        for book, metrics in json.load(f).items()
                        if book in args.only_include
                    }
                )
            elif args.plot_group:
                metrics_list.append(
                    {
                        book: metrics
                        for book, metrics in json.load(f).items()
                        if book in groups[args.plot_group]
                    }
                )
            else:
                metrics_list.append(json.load(f))

    fig, ax = plt.subplots()
    system_metrics = {}

    for i, metrics in enumerate(metrics_list):

        system_name = os.path.splitext(os.path.basename(args.input_files[i]))[0]

        bar = ax.bar(
            list(range(i, len(metrics) * len(metrics_list), len(metrics_list))),
            [
                m[args.metric] if not m[args.metric] is None else 0
                for _, m in sorted(metrics.items(), key=lambda t: t[0])
            ],
            tick_label=list(sorted(metrics.keys())),
            label=system_name,
        )

        mean_metric = sum(
            [
                m[args.metric] if not m[args.metric] is None else 0
                for m in metrics.values()
            ]
        ) / len(metrics)
        ax.axline((0, mean_metric), (1, mean_metric), c=bar.patches[0]._facecolor)
        system_metrics[system_name] = mean_metric

    print(system_metrics)

    plt.xticks(rotation="vertical")
    plt.legend(loc="upper right")
    plt.show()
