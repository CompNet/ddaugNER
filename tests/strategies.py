from hypothesis import strategies as st
from hypothesis.strategies import composite
from typing import List, Tuple


@composite
def bio_sequence(
    draw, classes: Tuple[str], min_length: int = 0, max_length: int = 128
) -> List[str]:
    seq_len = draw(st.integers(min_value=min_length, max_value=max_length))
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
