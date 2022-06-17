from typing import List
import unittest
from strategies import bio_sequence
from hypothesis import given
from ddaugner.ner_utils import prediction_errors
from ddaugner.utils import entities_from_bio_tags, flattened
from ddaugner.datas import NERSentence


class TestPredictionErrors(unittest.TestCase):
    """"""

    @given(tags=bio_sequence("PER"))
    def test_no_prediction(self, tags: List[str]):
        precision_errors, recall_errors = prediction_errors(
            [NERSentence(tags, tags)], [["O"] * len(tags)]
        )
        all_entities = entities_from_bio_tags(tags, tags)
        self.assertEqual(len(precision_errors), 0)
        self.assertEqual(sum([v for v in recall_errors.values()]), len(all_entities))
