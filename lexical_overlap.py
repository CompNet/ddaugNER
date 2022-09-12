import os, json, copy
import nltk
from typing import Dict, List, Set, Tuple
import functools
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from nameparser.config.titles import TITLES
from transformers import BertTokenizer
from rich import print
from ddaugner.datas.conll import CoNLLDataset
from ddaugner.utils import entities_from_bio_tags, flattened
from ddaugner.datas.dekker import load_dekker_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))


def exact_match(nameset1: Set[str], nameset2: Set[str]) -> Set[str]:
    """Compute the proportion of mentions of ``nameset1`` that have an
    exact match in ``nameset2``
    """
    return {name for name in nameset1 if name in nameset2}


@functools.lru_cache(maxsize=None)
def is_title(token: str) -> bool:
    return token.lower().replace(".", "") in TITLES


@functools.lru_cache(maxsize=None)
def is_stopword(token: str) -> bool:
    nltk.download("stopwords", quiet=True)
    return token in stopwords.words("english")


tokenizer = None


def get_tokenizer() -> BertTokenizer:
    global tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer


@functools.lru_cache(maxsize=None)
def wp_tokenize(name: str) -> List[str]:
    tokenizer = get_tokenizer()
    return tokenizer.tokenize(name)


def wp_partial_match(nameset1: Set[str], nameset2: Set[str]) -> Set[str]:
    """Compute the proportion of mentions of ``nameset1`` that have a
    partial match in ``nameset2``, taking wordpiece"""

    def names_partially_match(name1: str, name2: str) -> bool:

        if is_title(name1) or is_title(name2):
            return False

        if is_stopword(name1) or is_stopword(name2):
            return False

        wp_name1 = wp_tokenize(name1)
        wp_name2 = wp_tokenize(name2)

        for wp1 in wp_name1:
            if wp1 in wp_name2:
                return True

        return False

    return {
        name1
        for name1 in nameset1
        if any([names_partially_match(name1, name2) for name2 in nameset2])
    }


def overlap_subsets(target_set: Set[str], source_set: Set[str]) -> Dict[str, float]:

    new_set = copy.copy(target_set)
    exact_matchs_set = exact_match(new_set, source_set)
    new_set -= exact_matchs_set
    wp_partial_matchs_set = wp_partial_match(new_set, source_set)
    new_set -= wp_partial_matchs_set

    return {
        "exact matchs": len(exact_matchs_set) / len(target_set),
        "wordpiece partial matchs": len(wp_partial_matchs_set) / len(target_set),
        "new": len(new_set) / len(target_set),
    }


if __name__ == "__main__":

    dekker_dataset = load_dekker_dataset("./ner", "fantasy")
    dekker_entities = entities_from_bio_tags(
        flattened([s.tokens for s in dekker_dataset.sents]),
        flattened([s.tags for s in dekker_dataset.sents]),
    )
    dekker_names = set(flattened([e.tokens for e in dekker_entities]))

    conll_dataset = CoNLLDataset.train_dataset({}, {})
    conll_entities = entities_from_bio_tags(
        flattened([s.tokens for s in conll_dataset.sents]),
        flattened([s.tags for s in conll_dataset.sents]),
    )
    conll_names = set(flattened([e.tokens for e in conll_entities if e.tag == "PER"]))

    with open(f"{script_dir}/ddaugner/resources/wgold.json") as f:
        wgold_names = json.load(f)
    # TODO
    wgold_names = set(
        flattened([([t for t in name.split(" ") if t]) for name in wgold_names])
    )

    with open(f"{script_dir}/ddaugner/resources/morrowind_names.json") as f:
        morrowind_names = json.load(f)
    morrowind_names = set(
        flattened([[t for t in name.split(" ") if t] for name in morrowind_names])
    )  # TODO

    with open(f"{script_dir}/ddaugner/resources/the_elder_scrolls_names.json") as f:
        tes_data = json.load(f)
        tes_names = (
            tes_data["first_names"]
            + tes_data["last_names"]
            + tes_data["suffixs"]
            + tes_data["prefixs"]
        )
    # TODO
    tes_names = set(
        flattened([[t for t in name.split(" ") if t] for name in tes_names])
    )

    namesets = {
        "dekker": dekker_names,
        "conll": conll_names,
        "wgold": wgold_names,
        "morrowind": morrowind_names,
        "the elder scrolls": tes_names,
    }

    overlaps = {}
    for nameset1, names1 in tqdm(namesets.items()):
        for nameset2, names2 in tqdm(namesets.items()):
            overlaps[f"{nameset1} {nameset2}"] = overlap_subsets(names1, names2)

    rows = []

    for nameset in set(namesets.keys()) - {"dekker"}:
        rows.append(
            pd.Series(
                {
                    "exact matchs": overlaps[f"dekker {nameset}"]["exact matchs"],
                    "wordpiece partial matchs": overlaps[f"dekker {nameset}"][
                        "wordpiece partial matchs"
                    ],
                    "new": overlaps[f"dekker {nameset}"]["new"],
                },
                name=nameset,
            )
        )

    df = pd.DataFrame(
        rows,
        columns=["exact matchs", "wordpiece partial matchs", "new"],
    )
    print("Overlap with dekker fantasy dataset")
    print(df)

    rows = []

    for nameset in set(namesets.keys()) - {"dekker"}:
        rows.append(
            pd.Series(
                {
                    "exact matchs": overlaps[f"conll {nameset}"]["exact matchs"],
                    "wordpiece partial matchs": overlaps[f"conll {nameset}"][
                        "wordpiece partial matchs"
                    ],
                    "new": overlaps[f"conll {nameset}"]["new"],
                },
                name=nameset,
            )
        )

    df = pd.DataFrame(
        rows,
        columns=["exact matchs", "wordpiece partial matchs", "new"],
    )

    print("Overlap with conll train dataset")
    print(df)
