from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import os, random, math, copy
from ddaugner.datas import NERDataset, NERSentence
from ddaugner.datas.aug import NERAugmenter
from ddaugner.utils import flattened

script_dir = os.path.dirname(os.path.abspath(__file__))


CONLL_NER_CLASSES = {"PER", "LOC", "ORG", "MISC"}


def _augment(
    sents: List[NERSentence],
    augmenters: Dict[str, List[NERAugmenter]],
    aug_frequencies: Dict[str, List[float]],
) -> List[NERSentence]:
    """Augment the input sentences by adding new sentences

    :param sents: a list of sentences to augment
    :param augmenters: a mapping of NER class to some augmenters to
        use for entities of this class.
    :param aug_frequencies: a mapping of NER class to a list of
        augmentation frequency to use for each augmenter of
        ``augmenter``.

    :return: a list of sents, with some of them replaced according to
             ``augmenters``
    """
    for ner_class, local_augmenters in augmenters.items():
        assert len(local_augmenters) == len(aug_frequencies[ner_class])

    new_sents = copy.copy(sents)

    for ner_class, local_augmenters in augmenters.items():

        for i, augmenter in enumerate(local_augmenters):

            aug_freq = aug_frequencies[ner_class][i]
            augmented_sents: List[NERSentence] = []

            while len(augmented_sents) < len(sents) * aug_freq:
                sent = random.choice(sents)
                # kind of a hack - NER class is passed in case of
                # LabelWiseNERAugmenter
                augmented = augmenter(sent, prev_entity_type=ner_class)
                if not augmented is None:
                    augmented_sents.append(augmented)

            new_sents += augmented_sents

    return new_sents


def _augment_replace(
    sents: List[NERSentence],
    augmenters: Dict[str, List[NERAugmenter]],
    replacement_ratios: Dict[str, List[float]],
) -> List[NERSentence]:
    """Augment the input sentences by replacing some of them.

    .. warning::

        Behaviour regarding replacement ratio is not well-defined
        if more than one augmenter is specified. This is because
        augmenters might replace already replaced sentences.

    :param sents: a list of sentences to augment by replacement
    :param augmenters: a mapping of NER class to some augmenters to
        use for entities of this class.
    :param replacement_ratios: a mapping of NER class to a list of
        replacement ratios (between 0 and 1) to use for each augmenter
        of ``augmenter``.

    :return: a list of sents, with some of them replaced according to
             ``augmenters``
    """
    for ner_class, local_augmenters in augmenters.items():
        assert len(local_augmenters) == len(replacement_ratios[ner_class])
    for ratio in replacement_ratios.values():
        assert all([ratio > 0 and ratio <= 1.0 for ratio in ratio])
    # TODO: to remove when warning is fixed
    assert len(flattened([list(a) for a in augmenters.values()])) <= 1

    new_sents = copy.copy(sents)

    for ner_class, local_augmenters in augmenters.items():

        for i, augmenter in enumerate(local_augmenters):

            aug_ratio = replacement_ratios[ner_class][i]

            shuffled_idx = list(range(len(sents)))
            random.shuffle(shuffled_idx)
            augmented_sents_max_nb = math.ceil(len(sents) * aug_ratio)
            augmented_i_and_sents: List[Tuple[int, NERSentence]] = []

            for sent_i in shuffled_idx:

                if len(augmented_i_and_sents) >= augmented_sents_max_nb:
                    break

                sent = sents[sent_i]
                # kind of a hack - NER class is passed in case of
                # LabelWiseNERAugmenter
                augmented = augmenter(sent, prev_entity_type=ner_class)
                if not augmented is None:
                    augmented_i_and_sents.append((i, augmented))

            for sent_i, augmented_sent in augmented_i_and_sents:
                new_sents[sent_i] = augmented_sent

    return new_sents


class CoNLLDataset(NERDataset):
    """A class representing a CoNLL-2003 dataset"""

    def __init__(
        self,
        path: str,
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        usage_percentage: float = 1.0,
        keep_only_classes: Optional[Set[str]] = None,
        context_size: int = 0,
        data_aug_replace: bool = False,
    ) -> None:
        """
        :param data_aug_replace: if ``True``, replace existing
            examples instead of adding new ones
        """
        assert len(augmenters) == len(aug_frequencies)

        # Dataset loading
        with open(path) as f:
            raw_datas = f.read().strip()

        self.sents = []
        for sent in raw_datas.split("\n\n"):
            self.sents.append(NERSentence([], []))
            for line in sent.split("\n"):
                self.sents[-1].tokens.append(line.split(" ")[0])
                tag = line.split(" ")[1]
                if keep_only_classes:
                    self.sents[-1].tags.append(
                        tag if tag[2:] in keep_only_classes else "O"
                    )
                else:
                    self.sents[-1].tags.append(tag)

        self.sents = self.sents[: int(len(self.sents) * usage_percentage)]

        self.sents = NERSentence.sents_with_surrounding_context(
            self.sents, context_size=context_size
        )

        # Data augmentation
        self.augmenters = augmenters
        self.aug_frequencies = aug_frequencies

        if data_aug_replace:
            self.sents = _augment_replace(
                self.sents, self.augmenters, self.aug_frequencies
            )
        else:
            self.sents = _augment(self.sents, self.augmenters, self.aug_frequencies)

        # Init
        super().__init__(
            self.sents,
            set([tag for sent in self.sents for tag in sent.tags]),
        )

    @staticmethod
    def train_dataset(
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        context_size: int = 0,
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/train2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            **kwargs,
        )

    @staticmethod
    def test_dataset(
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        context_size: int = 0,
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/test2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            **kwargs,
        )

    @staticmethod
    def valid_dataset(
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        context_size: int = 0,
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/valid2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            **kwargs,
        )
