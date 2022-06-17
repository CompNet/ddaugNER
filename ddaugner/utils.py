from typing import Any, Callable, Dict, Tuple, TypeVar, List, Optional
import copy, time
from dataclasses import dataclass
from more_itertools import windowed


@dataclass(frozen=True)
class NEREntity:
    """"""

    #: tokens composing the entity
    tokens: List[str]

    #: NER tag (class identifier such as ``"PER"``, not token class
    #: such as ``"B-PER"``)
    tag: str

    #: token start end index
    start_idx: int

    #: token end index, inclusive
    end_idx: int

    def __hash__(self) -> int:
        return hash(tuple(self.tokens) + (self.tag, self.start_idx, self.end_idx))


T = TypeVar("T")


def flattened(lst: List[List[T]]) -> List[T]:
    out_lst = []
    for in_lst in lst:
        for elt in in_lst:
            out_lst.append(elt)
    return out_lst


def get_tokenizer():
    """resiliently try to get a tokenizer from the transformers library"""
    from transformers import BertTokenizerFast

    tokenizer = None
    for i in range(10):
        try:
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        except ValueError:
            print(f"could not load tokenizer (try {i}). ")
            time.sleep(10)
            continue
        break

    if tokenizer is None:
        raise RuntimeError("could not load tokenizer.")

    return tokenizer


def search_ner_pattern(
    pattern: List[Tuple[str, str]], tokens: List[str], tags: List[str]
) -> List[Tuple[int, int]]:
    """
    :param pattern: a list of tuple of form : `(token, tag)`
    """
    assert len(tokens) == len(tags)

    pattern_tokens = tuple([p[0] for p in pattern])
    pattern_tags = tuple([p[1] for p in pattern])

    idxs = []

    for i, (wtokens, wtags) in enumerate(
        zip(windowed(tokens, len(pattern)), windowed(tags, len(pattern)))
    ):
        if wtokens == pattern_tokens and wtags == pattern_tags:
            idxs.append((i, i + len(pattern) - 1))

    return idxs


def majority_voting(tokens: List[str], tags: List[str]) -> List[str]:
    """
    TODO: fix
    """

    new_tags = copy.copy(tags)

    entities = entities_from_bio_tags(tokens, tags)
    entities_tokens = [e.tokens for e in entities]  # type: ignore

    for entity_tokens in entities_tokens:

        per_matchs = search_ner_pattern(
            [(entity_tokens[0], "B-PER")] + [(t, "I-PER") for t in entity_tokens[1:]],  # type: ignore
            tokens,
            tags,
        )
        o_matchs = search_ner_pattern([(t, "O") for t in entity_tokens], tokens, tags)

        for match in per_matchs + o_matchs:
            if len(per_matchs) > len(o_matchs):
                new_tags[match[0] : match[1] + 1] = ["B-PER"] + ["I-PER"] * (
                    len(entity_tokens) - 1
                )
            else:
                new_tags[match[0] : match[1] + 1] = ["O"] * len(entity_tokens)

    return new_tags


def entities_from_bio_tags(
    tokens: List[str],
    bio_tags: List[str],
    quiet: bool = True,
    resolve_inconsistencies: bool = True,
) -> List[NEREntity]:
    """
    :param resolve_inconsistencies: if ``True``, will try to resolve
        inconsistencies (cases where an entity starts with
        ``"I-PER"`` instead of ``"B-PER"``)
    """
    assert len(bio_tags) == len(tokens)
    entities = []

    current_tag: Optional[str] = None
    current_tag_start_idx: Optional[int] = None

    for i, tag in enumerate(bio_tags):

        if not current_tag is None and not tag.startswith("I-"):
            assert not current_tag_start_idx is None
            entities.append(
                NEREntity(
                    tokens[current_tag_start_idx:i],
                    current_tag,
                    current_tag_start_idx,
                    i - 1,
                )
            )
            current_tag = None
            current_tag_start_idx = None

        if tag.startswith("B-"):
            current_tag = tag[2:]
            current_tag_start_idx = i

        elif tag.startswith("I-"):
            if current_tag is None and resolve_inconsistencies:
                if not quiet:
                    print(f"[warning] inconsistant bio tags. Will try to procede.")
                current_tag = tag[2:]
                current_tag_start_idx = i
                continue

    if not current_tag is None:
        assert not current_tag_start_idx is None
        entities.append(
            NEREntity(
                tokens[current_tag_start_idx : len(tokens)],
                current_tag,
                current_tag_start_idx,
                len(bio_tags) - 1,
            )
        )

    return entities


def entities_to_bio_tags(entities: List[NEREntity], tags_nb: int) -> List[str]:
    """
    :param entities:
    :param tags_nb: total number of tags to generate
    :return: a list of tags, of len ``tags_nb``
    """
    tags = ["O" for _ in range(tags_nb)]
    for entity in entities:
        tags[entity.start_idx] = f"B-{entity.tag}"
        for i in range(entity.start_idx + 1, entity.end_idx + 1):
            tags[i] = f"I-{entity.tag}"
    return tags


K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")


def valmap(d: Dict[K, V], fn: Callable[[V], R]) -> Dict[K, R]:
    """Map on a dictionary values"""
    return {k: fn(v) for k, v in d.items()}
