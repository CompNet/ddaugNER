from typing import List
import unittest
from hypothesis import given, strategies as st
from ddaugner.datas.aug import all_augmenters
from ddaugner.datas.datas import NERSentence


class TestAugmenters(unittest.TestCase):
    """"""

    @given(
        tags=st.lists(st.sampled_from(("B-PER", "I-PER", "O"))),
    )
    def test_augmented_sent_is_different(self, tags: List[str]):
        tokens = ["A"] * len(tags)
        augmenters = [aug_class() for aug_class in all_augmenters.values()]
        for augmenter in augmenters:
            original = NERSentence(tokens, tags)
            self.assertNotEqual(original, augmenter(original))
