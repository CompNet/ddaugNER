import unittest
from hypothesis import given
from ddaugner.datas.aug import all_augmenters
from ddaugner.datas.datas import NERSentence
from tests.strategies import ner_sentence


class TestAugmenters(unittest.TestCase):
    """"""

    @given(
        sent=ner_sentence(["PER"]),
    )
    def test_augmented_sent_is_different(self, sent: NERSentence):
        augmenters = [aug_class() for aug_class in all_augmenters.values()]
        for augmenter in augmenters:
            self.assertNotEqual(sent, augmenter(sent))
