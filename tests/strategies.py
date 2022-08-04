import string
from hypothesis import strategies as st
from hypothesis.strategies import composite
from typing import List, Tuple
from ddaugner.datas.datas import NERSentence


@composite
def bio_sequence(
    draw, classes: List[str], min_size: int = 0, max_size: int = 32
) -> List[str]:
    seq_len = draw(st.integers(min_value=min_size, max_value=max_size))
    seq = ["O"] * seq_len
    i = 0
    while i < seq_len:
        if draw(st.integers(min_value=1, max_value=10)) == 10:
            cls = draw(st.sampled_from(classes))
            seq[i] = f"B-{cls}"
            for _ in range(draw(st.integers(min_value=0, max_value=3))):
                i += 1
                if i == seq_len:
                    break
                seq[i] = f"I-{cls}"
        i += 1
    return seq


@composite
def ner_sentence(
    draw, ner_classes: List[str], min_size: int = 0, max_size: int = 32
) -> NERSentence:
    """Generate a random ``NERSentence``

    :param ner_classes: possible NER classes
    :param min_size: min sentence size
    :param max_size: max sentence size
    """
    tags = draw(bio_sequence(ner_classes, min_size=min_size, max_size=max_size))
    tokens = draw(
        st.lists(
            st.text(alphabet=string.ascii_letters),
            min_size=len(tags),
            max_size=len(tags),
        )
    )
    return NERSentence(tokens, tags)
