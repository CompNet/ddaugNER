from typing import List
import unittest
from strategies import ner_sentence
from hypothesis import given
from ddaugner.ner_utils import prediction_errors
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
