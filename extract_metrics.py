from ddaugner.datas.datas import EnsembleDataset
from ddaugner.score import score_ner
import argparse, json, os, glob, re

from rich import print
from tqdm import tqdm
from transformers import BertForTokenClassification

from ddaugner.predict import predict
from ddaugner.ner_utils import prediction_errors
from ddaugner.datas import BookDataset
from ddaugner.book_groups import groups


script_dir = f"{os.path.dirname(os.path.abspath(__file__))}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model-path", type=str)
    parser.add_argument("-bz", "--batch-size", type=int, default=4)
    parser.add_argument("-bm", "--batch-mode", action="store_true")
    parser.add_argument("-cs", "--context-size", type=int, default=0)
    parser.add_argument("-bg", "--book-group", type=str, default=None)
    parser.add_argument("-gm", "--global-metrics", action="store_true")
    parser.add_argument("-of", "--output-file", type=str)
    args = parser.parse_args()

    print("running with config")
    print(vars(args))

    old_paths = glob.glob(f"{script_dir}/ner/old/*.conll.fixed")
    new_paths = glob.glob(f"{script_dir}/ner/new/*.conll.fixed")

    if args.book_group:

        def book_name(path: str) -> str:
            return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore

        old_paths = [p for p in old_paths if book_name(p) in groups[args.book_group]]
        new_paths = [p for p in new_paths if book_name(p) in groups[args.book_group]]

    model = BertForTokenClassification.from_pretrained(args.model_path)

    if args.global_metrics:
        dataset = EnsembleDataset(
            [
                BookDataset(path, context_size=args.context_size)
                for path in tqdm(old_paths + new_paths)
            ],
        )

        predictions = predict(model, dataset, args.batch_size, quiet=args.batch_mode)

        precision, recall, f1 = score_ner(
            dataset.sents, predictions, ignored_classes={"MISC", "ORG", "LOC"}
        )

        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
        print(metrics_dict)
        with open(args.output_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        exit(0)

    book_metrics = {}

    for path in tqdm(old_paths + new_paths, disable=args.batch_mode):

        book_name = re.search(r"[^.]*", os.path.basename(path)).group(0)  # type: ignore

        dataset = BookDataset(path, context_size=args.context_size)
        predictions = predict(model, dataset, args.batch_size, quiet=True)

        precision, recall, f1 = score_ner(
            dataset.sents, predictions, ignored_classes={"MISC", "ORG", "LOC"}
        )
        book_metrics[book_name] = {"precision": precision, "recall": recall, "f1": f1}

        precision_errors, recall_errors = prediction_errors(
            dataset.sents, predictions, ignored_classes={"MISC", "ORG", "LOC"}
        )

        book_metrics[book_name] = {
            **book_metrics[book_name],
            **{
                "precision_errors": precision_errors,
                "recall_errors": recall_errors,
            },
        }

    with open(args.output_file, "w") as f:
        json.dump(book_metrics, f, indent=4)
