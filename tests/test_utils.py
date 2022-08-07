from typing import List, Tuple
import unittest
from hypothesis import given, strategies as st
from tests.strategies import bio_sequence
from ddaugner.utils import (
    entities_from_bio_tags,
    entities_to_bio_tags,
    search_ner_pattern,
    majority_voting,
)


class TestNerPattern(unittest.TestCase):
    """"""

    @given(
        tags=st.lists(st.sampled_from(("B-PER", "I-PER", "O"))),
        pattern=st.lists(
            st.tuples(st.just("TOKEN"), st.sampled_from(("B-PER", "I-PER", "O"))),
            min_size=1,
        ),
    )
    def test_pattern_is_present(self, tags: List[str], pattern: List[Tuple[str, str]]):
        tags += [p[1] for p in pattern]
        tokens = ["TOKEN" for _ in tags]
        self.assertGreaterEqual(len(search_ner_pattern(pattern, tokens, tags)), 1)


class TestEntitiesFromBioTags(unittest.TestCase):
    """"""

    @given(tags=bio_sequence(("PER", "MISC")))
    def test_entity_properties(self, tags: List[str]):
        self.assertEqual(
            [
                {"start_idx": i, "tag": tag[2:]}
                for i, tag in enumerate(tags)
                if tag.startswith("B-")
            ],
            [
                {"start_idx": e.start_idx, "tag": e.tag}
                for e in entities_from_bio_tags(tags, tags)
            ],
        )


class TestBioTagsEntityConversion(unittest.TestCase):
    """"""

    @given(tags=bio_sequence(("PER", "MISC")))
    def test_to_entities_to_bio(self, tags: List[str]):
        entities = entities_from_bio_tags(tags, tags)
        out_tags = entities_to_bio_tags(entities, len(tags))
        self.assertEqual(tags, out_tags)

    @given(tags=st.lists(st.sampled_from(("B-PER", "I-PER", "O"))))
    def test_to_entities_to_bio_with_inconsistencies(self, tags: List[str]):
        entities = entities_from_bio_tags(tags, tags, resolve_inconsistencies=True)
        out_tags = entities_to_bio_tags(entities, len(tags))
        self.assertEqual(
            [t if t == "O" else t[2:] for t in tags],
            [t if t == "O" else t[2:] for t in out_tags],
        )


class TestMajorityVoting(unittest.TestCase):
    """"""

    def test_should_be_per(self):
        self.assertEqual(
            majority_voting(
                ["A", "A", "A", "A", "A", "A"],
                ["B-PER", "I-PER", "O", "O", "B-PER", "I-PER"],
            ),
            ["B-PER", "I-PER", "B-PER", "I-PER", "B-PER", "I-PER"],
        )

    def test_should_be_o(self):
        self.assertEqual(
            majority_voting(
                ["A", "A", "A", "A", "A", "A"],
                ["B-PER", "I-PER", "O", "O", "O", "O"],
            ),
            ["O", "O", "O", "O", "O", "O"],
        )


if __name__ == "__main__":
    unittest.main()
