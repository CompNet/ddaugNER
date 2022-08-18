import json, re
from ddaugner.score import score_ner
from typing import List, Dict, Optional
import glob, os, argparse, re
from rich import print
from tqdm import tqdm

from ddaugner.datas import NERDataset, NERSentence, BookDataset
from ddaugner.datas.dekker import load_dekker_books


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

    scores = {
        "illinois": {},
        "ixa": {},
        "booknlp": {},
        "stanford": {},
    }

    book_datasets = load_dekker_books("./ner", args.book_group)

    for book_dataset in tqdm(book_datasets):

        book_set = book_dataset.path.split("/")[-2]
        book_name = re.search(r"[^.]*", os.path.basename(path)).group(0)  # type: ignore

        illinois_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.illconllout"
        illinois_dataset = BookDataset(illinois_path)
        scores["illinois"][book_name] = metrics_dict(
            illinois_dataset, book_dataset, "illinois", book_name
        )

        ixa_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.ixa-conllout"
        ixa_dataset = BookDataset(ixa_path)
        scores["ixa"][book_name] = metrics_dict(
            ixa_dataset, book_dataset, "ixa", book_name
        )

        booknlp_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.booknlpoutput"
        booknlp_dataset = BookDataset(booknlp_path)
        scores["booknlp"][book_name] = metrics_dict(
            booknlp_dataset, book_dataset, "booknlp", book_name
        )

        stanford_path = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.stanfordout"
        stanford_dataset = BookDataset(stanford_path)
        scores["stanford"][book_name] = metrics_dict(
            stanford_dataset, book_dataset, "stanford", book_name
        )

    for key, metrics in scores.items():
        print(key)
        print(metrics)
        with open(f"{args.output_directory}/{key}.json", "w") as f:
            json.dump(metrics, f, indent=4)
