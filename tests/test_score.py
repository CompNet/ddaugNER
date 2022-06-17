import os
from random import shuffle
import subprocess, re
from typing import List
import unittest
from hypothesis import given, strategies as st
from strategies import bio_sequence
from hypothesis.control import assume
from ddaugner.datas.datas import NERSentence
from ddaugner.score import score_ner_old


class TestScoreNER(unittest.TestCase):
    """"""

    def test_empty_prediction_and_reference_has_none_score(self):
        self.assertEqual(score_ner_old([], []), (None, None, None))

    @given(tags=st.lists(st.sampled_from(("B-PER", "I-PER", "O")), min_size=1))
    def test_perfect_prediction_has_perfect_score(self, tags: List[str]):
        assume("B-PER" in tags)
        self.assertEqual(score_ner_old([NERSentence(tags, tags)], [tags]), (1, 1, 1))

    @given(tags=st.lists(st.sampled_from(("B-PER", "O")), min_size=1))
    def test_predicting_everything_has_perfect_recall(self, tags: List[str]):
        assume("B-PER" in tags)
        self.assertEqual(
            score_ner_old(
                [NERSentence(tags, tags)], [["B-PER" if t == "O" else t for t in tags]]
            )[1],
            1,
        )

    @given(tags=st.lists(st.sampled_from(("B-PER", "I-PER", "O")), min_size=1))
    def test_predicting_nothing_has_zero_recall(self, tags: List[str]):
        assume("B-PER" in tags)
        self.assertEqual(
            score_ner_old([NERSentence(tags, tags)], [["O" for _ in tags]])[1], 0
        )

    @given(st.data())
    def test_equal_conlleval(self, data):
        tags = data.draw(st.lists(st.sampled_from(("B-PER", "I-PER", "O")), min_size=1))
        predictions = data.draw(
            st.lists(
                st.sampled_from(tuple(set(tags))),
                min_size=len(tags),
                max_size=len(tags),
            )
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_score = score_ner_old(
            [NERSentence(tags, tags)], [predictions], resolve_inconsistencies=True
        )
        with open(f"{script_dir}/test.conll", "w") as f:
            for tag, prediction in zip(tags, predictions):
                f.write(f"{tag} {tag} {prediction}\n")

        out = subprocess.check_output(
            f"{script_dir}/conlleval.pl < {script_dir}/test.conll", shell=True
        )
        conll_score = (None, None, None)
        for line in out.decode("utf-8").split("\n"):
            m = re.match(
                r"^[ \t]*PER: precision: *([0-9\.]*)%; recall: *([0-9\.]*)%; FB1: *([0-9\.]*) .*$",
                line,
            )
            if m:
                conll_score = (
                    float(m.group(1)) / 100,
                    float(m.group(2)) / 100,
                    float(m.group(3)) / 100,
                )

        local_score = (
            local_score[0] if not local_score[0] is None else 0,
            local_score[1] if not local_score[1] is None else 0,
            local_score[2] if not local_score[2] is None else 0,
        )

        conll_score = (
            conll_score[0] if not conll_score[0] is None else 0,
            conll_score[1] if not conll_score[1] is None else 0,
            conll_score[2] if not conll_score[2] is None else 0,
        )

        self.assertAlmostEqual(local_score[0], conll_score[0], places=3)
        self.assertAlmostEqual(local_score[1], conll_score[1], places=3)
        self.assertAlmostEqual(local_score[2], conll_score[2], places=3)


if __name__ == "__main__":
    unittest.main()
