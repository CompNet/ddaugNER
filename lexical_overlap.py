import re, glob, os, json, copy, rich
import nltk
from typing import List, Set, Tuple
import functools
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from nameparser.config.titles import TITLES
from transformers import BertTokenizer
from ddaugner.datas.conll import CoNLLDataset
from ddaugner.utils import entities_from_bio_tags, flattened
from ddaugner.datas.dekker import load_dekker_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))


dekker_dataset = load_dekker_dataset("./ner", "fantasy")
dekker_entities = entities_from_bio_tags(
    flattened([s.tokens for s in dekker_dataset.sents]),
    flattened([s.tags for s in dekker_dataset.sents]),
)
dekker_names = set([tuple(e.tokens) for e in dekker_entities])

conll_dataset = CoNLLDataset.train_dataset({}, {})
conll_entities = entities_from_bio_tags(
    flattened([s.tokens for s in conll_dataset.sents]),
    flattened([s.tags for s in conll_dataset.sents]),
)
conll_names = set([tuple(e.tokens) for e in conll_entities if e.tag == "PER"])


with open(f"{script_dir}/ddaugner/resources/wgold.json") as f:
    wgold_names = json.load(f)
# TODO
wgold_names = set([tuple([t for t in name.split(" ") if t]) for name in wgold_names])

with open(f"{script_dir}/ddaugner/resources/morrowind_names.json") as f:
    morrowind_names = json.load(f)
morrowind_names = set(
    [tuple([t for t in name.split(" ") if t]) for name in morrowind_names]
)  # TODO


def exact_match(
    nameset1: Set[Tuple[str, ...]], nameset2: Set[Tuple[str, ...]]
) -> Set[Tuple[str, ...]]:
    """Compute the proportion of mentions of ``nameset1`` that have an
    exact match in ``nameset2``

    :return: a `float` between 0 and 1
    """
    return {name for name in nameset1 if name in nameset2}


@functools.lru_cache(maxsize=None)
def is_title(token: str) -> bool:
    return token.lower().replace(".", "") in TITLES


@functools.lru_cache(maxsize=None)
def is_stopword(token: str) -> bool:
    nltk.download("stopwords", quiet=True)
    return token in stopwords.words("english")


def partial_match(
    nameset1: Set[Tuple[str, ...]], nameset2: Set[Tuple[str, ...]]
) -> Set[Tuple[str, ...]]:
    """Compute the proportion of mentions of ``nameset1`` that have a
    partial match in ``nameset2``

    .. note::

        titles are *not* taken into account when partial matching

    :return: a `float` between 0 and 1
    """

    def names_partially_match(name1: Tuple[str, ...], name2: Tuple[str, ...]) -> bool:
        for tok in name1:
            if len(name1) > 1 and is_title(tok):
                continue
            if len(name1) > 1 and is_stopword(tok):
                continue
            if tok in name2:
                return True
        return False

    return {
        name1
        for name1 in nameset1
        if any([names_partially_match(name1, name2) for name2 in nameset2])
    }


def wp_partial_match(
    nameset1: Set[Tuple[str, ...]], nameset2: Set[Tuple[str, ...]]
) -> Set[Tuple[str, ...]]:
    """Compute the proportion of mentions of ``nameset1`` that have a
    partial match in ``nameset2``, taking wordpiece"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @functools.lru_cache(maxsize=None)
    def subtokenize(name: Tuple[str, ...]) -> Tuple[str, ...]:
        """HACK"""
        return tuple(
            [
                tokenizer.decode([e])
                for e in tokenizer.encode(
                    name, is_split_into_words=True, add_special_tokens=False
                )
            ]
        )

    def names_partially_match(name1: Tuple[str, ...], name2: Tuple[str, ...]) -> bool:

        if len(name1) > 1:
            name1 = tuple([t for t in name1 if not is_title(t) and not is_stopword(t)])

        if len(name2) > 1:
            name2 = tuple([t for t in name2 if not is_title(t) and not is_stopword(t)])

        if len(name1) == 0 or len(name2) == 0:
            return False

        wp_name1 = subtokenize(name1)
        wp_name2 = subtokenize(name2)

        for wp1 in wp_name1:
            if wp1 in wp_name2:
                return True

        return False

    return {
        name1
        for name1 in nameset1
        if any([names_partially_match(name1, name2) for name2 in nameset2])
    }


def overlap_subsets(
    target_set: Set[Tuple[str, ...]], source_set: Set[Tuple[str, ...]]
) -> dict:

    new_set = copy.copy(target_set)
    exact_matchs_set = exact_match(new_set, source_set)
    new_set -= exact_matchs_set
    partial_matchs_set = partial_match(new_set, source_set)
    new_set -= partial_matchs_set
    wp_partial_matchs_set = wp_partial_match(new_set, source_set)
    new_set -= wp_partial_matchs_set

    return {
        "exact matchs": len(exact_matchs_set) / len(target_set),
        "partial matchs": len(partial_matchs_set) / len(target_set),
        "wordpiece partial matchs": len(wp_partial_matchs_set) / len(target_set),
        "new": len(new_set) / len(target_set),
    }


namesets = {
    "dekker": dekker_names,
    "conll": conll_names,
    "wgold": wgold_names,
    "morrowind": morrowind_names,
}

overlaps = {}
for nameset1, names1 in tqdm(namesets.items()):
    for nameset2, names2 in namesets.items():
        overlaps[f"{nameset1} {nameset2}"] = overlap_subsets(names1, names2)


# rich.print(overlaps)


rows = []

for nameset in ("conll", "wgold", "morrowind"):
    # rows.append(
    #     pd.Series(
    #         {
    #             "exact matchs": overlaps[f"{nameset} dekker"]["exact matchs"],
    #             "partial matchs": overlaps[f"{nameset} dekker"]["partial matchs"],
    #             "wordpiece partial matchs": overlaps[f"{nameset} dekker"][
    #                 "wordpiece partial matchs"
    #             ],
    #             "new": overlaps[f"{nameset} dekker"]["new"],
    #         },
    #         name=nameset,
    #     )
    # )
    rows.append(
        pd.Series(
            {
                "exact matchs": overlaps[f"dekker {nameset}"]["exact matchs"],
                "partial matchs": overlaps[f"dekker {nameset}"]["partial matchs"],
                "wordpiece partial matchs": overlaps[f"dekker {nameset}"][
                    "wordpiece partial matchs"
                ],
                "new": overlaps[f"dekker {nameset}"]["new"],
            },
            name=nameset,
        )
    )

df = pd.DataFrame(
    rows, columns=["exact matchs", "partial matchs", "wordpiece partial matchs", "new"]
)
print("Overlap with dekker fantasy dataset")
print(df)


rows = []

for nameset in ("conll", "wgold", "morrowind"):
    # rows.append(
    #     pd.Series(
    #         {
    #             "exact matchs": overlaps[f"{nameset} conll"]["exact matchs"],
    #             "partial matchs": overlaps[f"{nameset} conll"]["partial matchs"],
    #             "wordpiece partial matchs": overlaps[f"{nameset} conll"][
    #                 "wordpiece partial matchs"
    #             ],
    #             "new": overlaps[f"{nameset} conll"]["new"],
    #         },
    #         name=nameset,
    #     )
    # )
    rows.append(
        pd.Series(
            {
                "exact matchs": overlaps[f"conll {nameset}"]["exact matchs"],
                "partial matchs": overlaps[f"conll {nameset}"]["partial matchs"],
                "wordpiece partial matchs": overlaps[f"conll {nameset}"][
                    "wordpiece partial matchs"
                ],
                "new": overlaps[f"conll {nameset}"]["new"],
            },
            name=nameset,
        )
    )

df = pd.DataFrame(
    rows, columns=["exact matchs", "partial matchs", "wordpiece partial matchs", "new"]
)

print("Overlap with conll train dataset")
print(df)
