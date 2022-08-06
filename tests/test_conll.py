from typing import List, Optional
import unittest
from hypothesis import given, strategies as st
from hypothesis import assume
from tests.strategies import ner_sentence
from ddaugner.datas.aug import rand_sent_entity_indices, replace_sent_entity
from ddaugner.datas.datas import NERSentence
from ddaugner.datas.aug import NERAugmenter
import ddaugner.datas.conll.conll as conll


class IdentityAugmenter(NERAugmenter):
    def __call__(self, sent: NERSentence, *args, **kwargs) -> Optional[NERSentence]:
        return sent


class PERConstantAugmenter(NERAugmenter):
    def __init__(self, replacement: List[str]) -> None:
        self.replacement = replacement

    def __call__(self, sent: NERSentence, *args, **kwargs) -> Optional[NERSentence]:
        indices = rand_sent_entity_indices(sent.tags, "PER")
        if indices is None:
            return None
        return replace_sent_entity(
            sent,
            sent.tokens[indices[0] : indices[1] + 1],
            "PER",
            self.replacement,
            "PER",
        )


class TestConllAugment(unittest.TestCase):
    """"""

    @given(
        sents=st.lists(ner_sentence(conll.CONLL_NER_CLASSES), min_size=1),
        aug_freq=st.floats(min_value=0.0, exclude_min=True, max_value=1.0),
    )
    def test_augmented_sents_are_more_numerous(
        self, sents: List[NERSentence], aug_freq: float
    ):
        augmented = conll._augment(
            sents, {"PER": [IdentityAugmenter()]}, {"PER": [aug_freq]}
        )
        self.assertGreater(len(augmented), len(sents))


class TestConllAugmentReplace(unittest.TestCase):
    """"""

    @given(
        sents=st.lists(ner_sentence(conll.CONLL_NER_CLASSES), min_size=1),
        aug_ratio=st.floats(min_value=0.0, exclude_min=True, max_value=1.0),
    )
    def test_replacing_sents_doesnt_add_new_ones(
        self, sents: List[NERSentence], aug_ratio: float
    ):
        augmented = conll._augment_replace(
            sents, {"PER": [IdentityAugmenter()]}, {"PER": [aug_ratio]}
        )
        self.assertEqual(len(augmented), len(sents))

    @given(
        sents=st.lists(ner_sentence(conll.CONLL_NER_CLASSES, min_size=1), min_size=1),
        aug_ratio=st.floats(min_value=0.0, exclude_min=True, max_value=1.0),
    )
    def test_replacing_sents_yield_some_new_sents(
        self, sents: List[NERSentence], aug_ratio: float
    ):
        assume(any(["B-PER" in sent.tags for sent in sents]))
        augmented = conll._augment_replace(
            sents, {"PER": [PERConstantAugmenter(["ENTITY"])]}, {"PER": [aug_ratio]}
        )
        self.assertNotEqual(sents, augmented)
        self.assertTrue(any(["ENTITY" in sent.tokens for sent in augmented]))
