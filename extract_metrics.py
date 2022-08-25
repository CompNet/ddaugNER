from typing import cast, List
from ddaugner.score import score_ner
import argparse, json, os, re, os

from rich import print
from tqdm import tqdm
from transformers import BertForTokenClassification  # type: ignore

from ddaugner.predict import predict
from ddaugner.ner_utils import prediction_errors
from ddaugner.datas import BookDataset
from ddaugner.datas.dekker import load_dekker_books, load_dekker_dataset

script_dir = f"{os.path.dirname(os.path.abspath(__file__))}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model-path", type=str)
    parser.add_argument("-bz", "--batch-size", type=int, default=4)
    parser.add_argument("-bm", "--batch-mode", action="store_true")
    parser.add_argument("-cs", "--context-size", type=int, default=0)
    parser.add_argument("-bg", "--book-group", type=str, default=None)
    parser.add_argument("-gm", "--global-metrics", action="store_true")
    parser.add_argument("-fst", "--fix-sent-tokenization", action="store_true")
    parser.add_argument("-of", "--output-file", type=str)
    args = parser.parse_args()

    print("running with config")
    print(vars(args))

    model = BertForTokenClassification.from_pretrained(args.model_path)

    if args.global_metrics:

        dataset = load_dekker_dataset(
            "./ner",
            args.book_group,
            args.context_size,
            args.fix_sent_tokenization,
            quiet=args.batch_mode,
        )

        predictions = predict(model, dataset, args.batch_size, quiet=args.batch_mode)
        predictions = cast(List[List[str]], predictions)

        precision, recall, f1 = score_ner(
            dataset.sents, predictions, ignored_classes={"MISC", "ORG", "LOC"}
        )

        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
        print(metrics_dict)
        if not args.output_file is None:
            with open(args.output_file, "w") as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"saved metrics at {args.output_file}")

        exit(0)

    book_datasets = load_dekker_books(
        "./ner", args.book_group, args.context_size, args.fix_sent_tokenization
    )

    book_metrics = {}

    for dataset in tqdm(book_datasets, disable=args.batch_mode):

        book_name = re.search(r"[^.]*", os.path.basename(dataset.path)).group(0)  # type: ignore

        predictions = predict(model, dataset, args.batch_size, quiet=True)
        predictions = cast(List[List[str]], predictions)

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

    print(book_metrics)
    if not args.output_file is None:
        with open(args.output_file, "w") as f:
            json.dump(book_metrics, f, indent=4)
        print(f"saved metrics at {args.output_file}")
