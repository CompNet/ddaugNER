from typing import List, Tuple, Optional
import unittest
from hypothesis import given, strategies as st
from hypothesis import assume
from ddaugner.ner_utils import ner_classes_ratios
from tests.strategies import ner_sentence
from ddaugner.datas.aug import LabelWiseNERAugmenter
from ddaugner.datas.datas import NERSentence
import ddaugner.datas.conll.conll as conll


class IdentityAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self) -> None:
        super().__init__(conll.CONLL_NER_CLASSES)

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        return (prev_entity_tokens, prev_entity_type)


class PERConstantAugmenter(LabelWiseNERAugmenter):
    """"""

    def __init__(self, replacement: List[str]) -> None:
        self.repl_entity_types = {"PER"}
        self.replacement = replacement

    def replacement_entity(
        self, prev_entity_tokens: List[str], prev_entity_type: str
    ) -> Tuple[List[str], str]:
        return (self.replacement, "PER")


class TestConllAugment(unittest.TestCase):
    """"""

    @given(
        sents=st.lists(ner_sentence(["PER"]), min_size=1, max_size=16),
        aug_freq=st.floats(min_value=0.0, exclude_min=True, max_value=1.0),
    )
    def test_augmented_sents_are_more_numerous(
        self, sents: List[NERSentence], aug_freq: float
    ):
        assume(any(["B-PER" in sent.tags for sent in sents]))
        augmented = conll._augment(
            sents, {"PER": [IdentityAugmenter()]}, {"PER": [aug_freq]}
        )
        self.assertGreater(len(augmented), len(sents))


class TestConllAugmentBalance(unittest.TestCase):
    """"""

    @given(
        sents=st.lists(ner_sentence(["PER"]), min_size=1, max_size=16),
        aug_freq=st.floats(min_value=0.0, exclude_min=True, max_value=1.0),
    )
    def test_augmented_sents_are_more_numerous(
        self, sents: List[NERSentence], aug_freq: float
    ):
        assume(any(["B-PER" in sent.tags for sent in sents]))
        augmented = conll._augment_balance(
            sents, {"PER": [IdentityAugmenter()]}, {"PER": [aug_freq]}
        )
        self.assertGreater(len(augmented), len(sents))

    def test_augmented_sents_ner_classes_ratio_is_balanced(self):
        conll_dataset = conll.CoNLLDataset.train_dataset({}, {})
        original_ratios = ner_classes_ratios(
            conll_dataset.sents, conll.CONLL_NER_CLASSES
        )
        assert not original_ratios is None

        augmented = conll._augment_balance(
            conll_dataset.sents, {"PER": [IdentityAugmenter()]}, {"PER": [0.1]}
        )
        augmented_ratios = ner_classes_ratios(augmented, conll.CONLL_NER_CLASSES)
        assert not augmented_ratios is None

        for ner_class, class_ratio in augmented_ratios.items():
            original_ratio = original_ratios[ner_class]
            self.assertAlmostEqual(original_ratio, class_ratio, places=3)


class TestConllAugmentReplace(unittest.TestCase):
    """"""

    @given(
        sents=st.lists(
            ner_sentence(list(conll.CONLL_NER_CLASSES)),
            min_size=1,
            max_size=16,
        ),
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
        sents=st.lists(
            ner_sentence(list(conll.CONLL_NER_CLASSES), min_size=1),
            min_size=1,
            max_size=16,
        ),
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
