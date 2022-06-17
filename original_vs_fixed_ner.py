import glob, os
from rich import print
from tqdm import tqdm
from ddaugner.datas import BookDataset
from ddaugner.score import score_ner


script_dir = f"{os.path.dirname(os.path.abspath(__file__))}"

if __name__ == "__main__":

    old_paths = [
        {"original": path, "fixed": f"{path}.fixed"}
        for path in glob.glob(f"{script_dir}/ner/old/*.conll")
    ]
    new_paths = [
        {"original": path, "fixed": f"{path}.fixed"}
        for path in glob.glob(f"{script_dir}/ner/new/*.conll")
    ]

    results = {}

    for paths in tqdm(new_paths + old_paths):
        original = BookDataset(paths["original"])
        fixed = BookDataset(paths["fixed"])
        precision, recall, f1 = score_ner(original.sents, [s.tags for s in fixed.sents])
        book_name = os.path.splitext(os.path.basename(paths["original"]))[0]
        results[book_name] = {"precision": precision, "recall": recall, "f1": f1}

    mean_precision = sum(
        [v["precision"] if not v["precision"] is None else 0 for v in results.values()]
    ) / len(results)
    mean_recall = sum(
        [v["recall"] if not v["recall"] is None else 0 for v in results.values()]
    ) / len(results)
    mean_f1 = sum(
        [v["f1"] if not v["f1"] is None else 0 for v in results.values()]
    ) / len(results)

    results["mean_precision"] = mean_precision
    results["mean_recall"] = mean_recall
    results["mean_f1"] = mean_f1

    print(results)
