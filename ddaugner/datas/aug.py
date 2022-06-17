from typing import List, Optional, Set, Tuple
import random, re, copy
from ddaugner.utils import search_ner_pattern
from ddaugner.datas import NERSentence
from ddaugner.resources.french_names import FrenchNamesGazetteer
from ddaugner.resources.morrowind import (
    MorrowindLocationsGazetteer,
    MorrowindNamesGazetteer,
)
from ddaugner.resources.word_names import WordNamesGazetteer
from ddaugner.resources.conll_entities import ConllGazetteer
from ddaugner.resources.wgold_names import WGoldNamesGazetteer
from ddaugner.resources.dekker import DekkerFantasyPERGazetteer


def rand_sent_entity_indices(
    sent_tags: List[str], entity_type: str
) -> Optional[Tuple[int, int]]:
    """
    :param sent_tags:
    :param entity_type: the type of the entity to retrieve (PER, LOC...)

    :return: a tuple representing the indices of a random entity of type
        ``entity_type``, or ``None`` if no entity of that type was present
    """
    if not f"B-{entity_type}" in sent_tags:
        return None

    entity_idxs = [i for i, tag in enumerate(sent_tags) if tag == f"B-{entity_type}"]
    start_idx = random.choice(entity_idxs)
    end_idx = start_idx
    for i, tag in enumerate(sent_tags[start_idx + 1 :]):
        if tag != f"I-{entity_type}":
            break
        end_idx = start_idx + i + 1
    return (start_idx, end_idx)


def replace_sent_entity(
    sent: NERSentence,
    entity_tokens: List[str],
    entity_type: str,
    new_entity_tokens: List[str],
    new_entity_type: str,
) -> NERSentence:
    assert len(entity_tokens) > 0
    assert len(new_entity_tokens) > 0

    entity_tags = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(entity_tokens) - 1)
    idxs = search_ner_pattern(
        [(tok, tag) for tok, tag in zip(entity_tokens, entity_tags)],
        sent.tokens,
        sent.tags,
    )

    if len(idxs) == 0:
        return sent

    new_entity_tags = [f"B-{new_entity_type}"] + [f"I-{new_entity_type}"] * (
        len(new_entity_tokens) - 1
    )

    new_tokens = []
    new_tags = []
    cur_start_idx = 0
    for start_idx, end_idx in idxs:
        new_tokens += sent.tokens[cur_start_idx:start_idx] + new_entity_tokens
        new_tags += sent.tags[cur_start_idx:start_idx] + new_entity_tags
        cur_start_idx = end_idx + 1
    new_tokens += sent.tokens[cur_start_idx:]
    new_tags += sent.tags[cur_start_idx:]

    return NERSentence(new_tokens, new_tags)


class NERAugmenter:
    """"""

    def __init__(self) -> None:
        pass

    def __call__(self, sent: NERSentence, *args, **kwargs) -> Optional[NERSentence]:
        """Perform augmention on input sentence.

        :param sent: sentence in which to perform the replacement

        :return: if a new, different sentence can't be returned,
                 should return ``None``.
        """
        raise RuntimeError("must be implemented by subclass")


class LabelWiseNERAugmenter(NERAugmenter):
    """"""

    def __init__(self, repl_entity_types: Set[str]) -> None:
        self.repl_entity_types = list(repl_entity_types)

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        """
        :return: a tuple ``(tokens, new entity type)``
        """
        raise RuntimeError("must be implemented by subclass")

    def __call__(
        self, sent: NERSentence, prev_entity_type: Optional[str] = None
    ) -> Optional[NERSentence]:
        """Perform mention replacement on input sentence.

        :param sent: sentence in which to perform the replacement
        :param prev_entity_type: type of the entity to replace.  If
            specified, must be present in ``self.repl_entity_types``.
            If ``None``, the entity type will be picked at random in
            ``self.repl_entity_types``.

        :return: a new ``NERSentence`` where the replacement has been
                 performed.  If the replacement could not be
                 performed, returns ``None``.
        """

        if prev_entity_type:
            assert prev_entity_type in self.repl_entity_types
        else:
            prev_entity_type = random.choices(list(self.repl_entity_types), k=1)[0]

        indices = rand_sent_entity_indices(sent.tags, prev_entity_type)
        if indices is None:
            return None
        start_idx, end_idx = indices
        entity_tokens = sent.tokens[start_idx : end_idx + 1]

        new_entity_tokens, new_entity_type = self.replacement_entity(
            entity_tokens, prev_entity_type
        )

        # TODO: context ?
        return replace_sent_entity(
            sent,
            entity_tokens,
            prev_entity_type,
            new_entity_tokens,
            new_entity_type,
        )


class FrenchNamesAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.gazeteer = FrenchNamesGazetteer()
        super().__init__({"PER"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.gazeteer.random_name()
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, "PER"


class WordNamesAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.gazeteer = WordNamesGazetteer()
        super().__init__({"PER"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.gazeteer.random_name()
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, "PER"


class MorrowindAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.name_gazeteer = MorrowindNamesGazetteer()
        # NOTE: unused
        self.loc_gazeteer = MorrowindLocationsGazetteer()
        super().__init__({"PER"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.name_gazeteer.random_name()
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, "PER"


class CapitalizationAugmenter(NERAugmenter):
    """"""

    def __init__(self) -> None:
        pass

    def __call__(self, sent: NERSentence) -> Optional[NERSentence]:
        upped = NERSentence([t.upper() for t in sent.tokens], sent.tags)
        if upped == sent:
            return None
        return upped


class CoNLLAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.gazetteer = ConllGazetteer()
        super().__init__({"PER", "ORG", "LOC", "MISC"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.gazetteer.random_name(prev_entity_type)
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, prev_entity_type


class WGoldAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.name_gazetter = WGoldNamesGazetteer()
        super().__init__({"PER"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.name_gazetter.random_name()
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, "PER"


class DekkerFantasyAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        self.name_gazetter = DekkerFantasyPERGazetteer()
        super().__init__({"PER"})

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        random_name = self.name_gazetter.random_name()
        splitted_name = re.split(r" ", random_name)
        splitted_name = [t for t in splitted_name if not t in {" ", ""}]
        return splitted_name, "PER"


all_augmenters = {
    "conll": CoNLLAugmenter,
    "wgold": WGoldAugmenter,
    "capitalization": CapitalizationAugmenter,
    "morrowind": MorrowindAugmenter,
    "french": FrenchNamesAugmenter,
    "word_names": WordNamesAugmenter,
    "dekker_fantasy": DekkerFantasyAugmenter,
}
