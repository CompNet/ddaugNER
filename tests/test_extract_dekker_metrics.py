import unittest
from extract_dekker_metrics import clean_dekker_etal_tags


class TestCleanDekkerTags(unittest.TestCase):
    """"""

    def test_entity_at_end(self):
        self.assertEqual(
            clean_dekker_etal_tags(["O", "I-PERSON", "I-PERSON"]),
            ["O", "B-PER", "I-PER"],
        )

    def test_entity_at_beginning(self):
        self.assertEqual(
            clean_dekker_etal_tags(["I-PERSON", "I-PERSON", "O"]),
            ["B-PER", "I-PER", "O"],
        )

    def test_several_entities(self):
        self.assertEqual(
            clean_dekker_etal_tags(
                ["O", "I-PERSON", "I-PERSON", "O", "I-PERSON", "I-PERSON", "O"]
            ),
            ["O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O"],
        )
