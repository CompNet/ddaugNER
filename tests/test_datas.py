from typing import List
import unittest
from hypothesis import given, strategies as st
from strategies import bio_sequence
from ddaugner.datas.datas import NERSentence


class TestSentsWithSurroundingContext(unittest.TestCase):
    """"""

    @given(
        tags=bio_sequence(("PER", "LOC"), min_length=1, max_length=16),
        context_size=st.integers(min_value=0, max_value=16),
    )
    def test_context_is_of_right_size(self, tags: List[str], context_size: int):
        sent = NERSentence(["A"] * len(tags), tags)
        sents_with_ctx = NERSentence.sents_with_surrounding_context(
            [sent] * (context_size + 2), context_size=context_size
        )
        self.assertEqual(len(sents_with_ctx[-1].left_context), context_size)
        self.assertEqual(len(sents_with_ctx[0].right_context), context_size)

    @given(
        tags=bio_sequence(("PER", "LOC"), min_length=1, max_length=16),
        context_size=st.integers(min_value=0, max_value=16),
        sents_nb=st.integers(min_value=0, max_value=16)
    )
    def test_same_number_of_generated_sents_as_input_sents(self, tags: List[str], context_size: int, sents_nb: int):
        sent = NERSentence(["A"] * len(tags), tags)
        sents_with_ctx = NERSentence.sents_with_surrounding_context(
            [sent] * sents_nb, context_size=context_size
        )
        self.assertEqual(sents_nb, len(sents_with_ctx))
