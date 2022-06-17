import json, re
from ddaugner.score import score_ner
from typing import List, Dict, Optional
import glob, os, argparse, re
from rich import print
from tqdm import tqdm

from ddaugner.datas import NERDataset, NERSentence, BookDataset
from ddaugner.book_groups import groups


script_dir = f"{os.path.dirname(os.path.abspath(__file__))}"


def clean_dekker_etal_tags(tags: List[str]) -> List[str]:
    """Clean tags, removing non-per tags and converting to BIO tags"""
    tags = ["I-PER" if t == "I-PERSON" else "O" for t in tags]
    new_tags = tags

    in_entity = False
    for i, tag in enumerate(tags):
        if tag == "I-PER":
            if not in_entity:
                new_tags[i] = "B-PER"
                in_entity = True
        else:
            in_entity = False

    return new_tags


class DekkerBookDataset(NERDataset):
    """"""

    def __init__(self, path: str) -> None:

        tokens = []
        tags = []
        sents = []
        with open(path) as f:
            for line in f:
                splitted = line.strip().split(" ")
                try:
                    tags.append(splitted[2])
                    tokens.append(splitted[0])
                    if tokens[-1] in {".", "?", "!"}:
                        sents.append(NERSentence(tokens, clean_dekker_etal_tags(tags)))
                        tokens = []
                        tags = []
                except IndexError:
                    continue
        if len(tokens) != 0:
            sents.append(NERSentence(tokens, clean_dekker_etal_tags(tags)))

        super().__init__(sents, {"B-PER", "I-PER", "O"})


def metrics_dict(
    dekker_dataset: DekkerBookDataset,
    book_dataset: BookDataset,
    tagger: str,
    book_name: str,
) -> Dict[str, Optional[float]]:
    try:
        p, r, f1 = score_ner(book_dataset.sents, [s.tags for s in dekker_dataset.sents])
    except AssertionError:
        print(f"Could not compute score for book '{book_name}' and tagger '{tagger}'")
        return {"precision": None, "recall": None, "f1": None}
    return {"precision": p, "recall": r, "f1": f1}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-derp", "--dekker-etal-repo-path", type=str)
    parser.add_argument("-od", "--output-directory", type=str)
    parser.add_argument("-bg", "--book-group", type=str, default=None)
    args = parser.parse_args()

    old_paths = glob.glob(f"{script_dir}/ner/old/*.conll.fixed")
    new_paths = glob.glob(f"{script_dir}/ner/new/*.conll.fixed")

    if args.book_group:

        def book_name(path: str) -> str:
            return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore

        old_paths = [p for p in old_paths if book_name(p) in groups[args.book_group]]
        new_paths = [p for p in new_paths if book_name(p) in groups[args.book_group]]

    scores = {
        "illinois": {},
        "ixa": {},
        "booknlp": {},
        "stanford": {},
    }

    for path in tqdm(old_paths + new_paths):

        book_set = path.split("/")[-2]
        book_name = re.search(r"[^.]*", os.path.basename(path)).group(0)  # type: ignore

        book_dataset = BookDataset(path)

        illinois_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.illconllout"
        illinois_dataset = DekkerBookDataset(illinois_path)
        scores["illinois"][book_name] = metrics_dict(
            illinois_dataset, book_dataset, "illinois", book_name
        )

        ixa_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.ixa-conllout"
        ixa_dataset = DekkerBookDataset(ixa_path)
        scores["ixa"][book_name] = metrics_dict(
            ixa_dataset, book_dataset, "ixa", book_name
        )

        booknlp_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.booknlpoutput"
        booknlp_dataset = DekkerBookDataset(booknlp_path)
        scores["booknlp"][book_name] = metrics_dict(
            booknlp_dataset, book_dataset, "booknlp", book_name
        )

        stanford_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.stanfordout"
        stanford_dataset = DekkerBookDataset(stanford_path)
        scores["stanford"][book_name] = metrics_dict(
            stanford_dataset, book_dataset, "stanford", book_name
        )

    for key, metrics in scores.items():
        print(key)
        print(metrics)
        with open(f"{args.output_directory}/{key}.json", "w") as f:
            json.dump(metrics, f, indent=4)
