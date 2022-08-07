from typing import List
import unittest
from tests.strategies import ner_sentence
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.control import assume
from ddaugner.ner_utils import ner_classes_ratios, prediction_errors
from ddaugner.utils import entities_from_bio_tags
from ddaugner.datas import NERSentence


class TestPredictionErrors(unittest.TestCase):
    """"""

    @given(sent=ner_sentence(["PER"]))
    def test_no_prediction(self, sent: NERSentence):
        precision_errors, recall_errors = prediction_errors([sent], [["O"] * len(sent)])
        all_entities = entities_from_bio_tags(sent.tokens, sent.tags)
        self.assertEqual(len(precision_errors), 0)
        self.assertEqual(sum([v for v in recall_errors.values()]), len(all_entities))


class TestNERClassesRatios(unittest.TestCase):
    """"""

    @given(sents=st.lists(ner_sentence(["PER", "LOC"]), min_size=1))
    def test_sum_of_ratios_is_one_and_all_classes_are_present(
        self, sents: List[NERSentence]
    ):
        classes_ratios = ner_classes_ratios(sents, {"PER", "LOC"})
        assume(not classes_ratios is None)
        assert not classes_ratios is None
        self.assertEqual(sum(classes_ratios.values()), 1.0)
        self.assertEqual(set(classes_ratios.keys()), {"PER", "LOC"})

    def test_no_entities_means_undefined_ratio(self):
        self.assertIsNone(
            ner_classes_ratios([NERSentence(["TOKEN"], ["O"])], {"PER", "LOC"})
        )

    def test_ratio_is_correct(self):
        self.assertEqual(
            ner_classes_ratios(
                [NERSentence(["TOKEN"], ["B-PER"]), NERSentence(["TOKEN"], ["B-LOC"])],
                {"PER", "LOC"},
            ),
            {"PER": 0.5, "LOC": 0.5},
        )
