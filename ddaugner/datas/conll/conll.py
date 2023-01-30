from __future__ import annotations
from typing import Dict, List, Literal, Optional, Set, Tuple
import os, random, math, copy
from collections import Counter
import numpy as np
from scipy import linalg
from ddaugner.datas import NERDataset, NERSentence
from ddaugner.datas.aug import NERAugmenter
from ddaugner.ner_utils import ner_classes_appearances_nb, ner_classes_ratios
from ddaugner.utils import entities_from_bio_tags, flattened


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

    :return: a list of sents, with some of them being augmented
             version of one of the given sents according to
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
                sent_i = random.choice(range(len(sents)))
                sent = sents[sent_i]
                # kind of a hack - NER class is passed in case of
                # LabelWiseNERAugmenter
                augmented = augmenter(sent, prev_entity_type=ner_class)
                if not augmented is None:
                    augmented_sents.append(augmented)

            new_sents += augmented_sents

    return new_sents


def _augment_balance(
    sents: List[NERSentence],
    augmenters: Dict[str, List[NERAugmenter]],
    aug_frequencies: Dict[str, List[float]],
):
    """
    Augment the input sentences, and restore the classes ratio as it
    was originally by upsampling

    :param sents: a list of sentences to augment
    :param augmenters: a mapping of NER class to some augmenters to
        use for entities of this class.
    :param aug_frequencies: a mapping of NER class to a list of
        augmentation frequency to use for each augmenter of
        ``augmenter``.

    :return: a list of sents, with some of them being augmented
             version of one of the given sents according to
             ``augmenters``
    """
    # augment sentences according to augmenters
    augmented_sents = _augment(sents, augmenters, aug_frequencies)

    # Find the number of entities needed to balance the dataset by
    # upsampling.
    #
    # Let :
    # - r be the original ratio vectors (sents_cls_ratios_v)
    # - a be the vector of the number of entities per class in the
    #   augmented sents (augmented_classes_appearances_nb_v)
    # - e the number of entities in the augmented sents
    # - c be the number of classes (len(CONLL_NER_CLASSES))
    #
    # We want to reequilibrate the ratio of classes of the augmented
    # dataset such that it is equal to s by upsampling. We want to
    # find c real numbers n_1, n_2, ..., n_c such that, for each
    # class, adding those numbers of entities will reequilibrate the
    # dataset. Let ∑n be the sum of n_1, n_2, ..., n_c. In terms of
    # classes ratio, we are trying to solve :
    #
    # [(a_1 + n_1) / (e + ∑n) , (a_2 + n_2) / (e + ∑n) , ..., (a_c + n_c) / (e + ∑n)] = r
    #
    # which can be written as a system :
    #
    # (a_1 + n_1) / (e + ∑n) = r_1
    # (a_2 + n_2) / (e + ∑n) = r_2
    # ...
    # (a_c + n_c) / (e + ∑n) = r_c
    #
    # which we can re-arrange as :
    # n_1 (r_1 - 1) + n_2 r_1 + ... + n_c r_1 = a_1 - e r_1
    # n_1 r_2 + n_2 (r_2 - 1) + ... + n_c r_2 = a_2 - e r_2
    # ...
    # n_1 r_c + n_2 r_c + ... + n_c (r_c - 1) = a_c - e r_c
    #
    # We also know that the n_i = 0 for the majority classes, since we
    # won't add any new examples for those
    sents_cls_ratios = ner_classes_ratios(sents, CONLL_NER_CLASSES)
    if sents_cls_ratios is None:
        return augmented_sents
    # matrix wont be invertible in the case where a class
    # was alone
    if any([ratio == 1.0 for ratio in sents_cls_ratios.values()]):
        return augmented_sents

    aug_cls_nb = ner_classes_appearances_nb(augmented_sents)
    sorted_cls = sorted(CONLL_NER_CLASSES)

    r = [sents_cls_ratios[cls] for cls in sorted_cls]
    a = np.array([aug_cls_nb[cls] for cls in sorted_cls])
    c = len(CONLL_NER_CLASSES)
    e = sum(aug_cls_nb.values())

    # indices of the majority classes in our system, we want to ignore
    # the lines and the column corresponding to majority classes
    majority_classes_idx = np.argwhere(a == np.amax(a)).flatten()
    sorted_non_majority_classes = [
        cls for i, cls in enumerate(sorted_cls) if not i in majority_classes_idx
    ]
    majority_mask = np.array(
        [
            [
                not j in majority_classes_idx and not i in majority_classes_idx
                for j in range(c)
            ]
            for i in range(c)
        ]
    )

    A = np.array([[r[i] - 1 if i == j else r[i] for j in range(c)] for i in range(c)])
    A = A[majority_mask].reshape(
        c - len(majority_classes_idx), c - len(majority_classes_idx)
    )
    b = np.array([a[i] - e * r[i] for i in range(c) if not i in majority_classes_idx])
    n = linalg.solve(A, b)  # type: ignore

    entities_to_add_per_class = {k: 0 for k in CONLL_NER_CLASSES}
    for i in range(len(sorted_non_majority_classes)):
        ner_class = sorted_non_majority_classes[i]
        number_to_add = int(n[i])
        entities_to_add_per_class[ner_class] = number_to_add

    # Try to add the correct number of entities per class in the datasets
    while any([v > 0 for v in entities_to_add_per_class.values()]):
        for sent in sents:
            entities = entities_from_bio_tags(sent.tokens, sent.tags)
            entities_counter = Counter([e.tag for e in entities])
            if not all(
                [entities_to_add_per_class[k] >= v for k, v in entities_counter.items()]
            ):
                continue
            augmented_sents.append(sent)
            for entity in entities:
                entities_to_add_per_class[entity.tag] -= 1

    return augmented_sents


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
                    augmented_i_and_sents.append((sent_i, augmented))

            for sent_i, augmented_sent in augmented_i_and_sents:
                new_sents[sent_i] = augmented_sent

    return new_sents


script_dir = os.path.dirname(os.path.abspath(__file__))


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
        aug_method: Optional[Literal["standard", "replace", "balance_upsample"]] = None,
    ) -> None:
        """
        :param aug_method: one of :

                - ``'standard'`` : add new examples by creating them
                  according to ``augmenters``

                - ``'replace'`` : replace old examples by new one,
                  created using ``augmenters``

                - ``'balance_upsample'`` : add new examples by
                  creating them according to ``augmenters``, and
                  upsample if necessary to rebalance the dataset.
                  Classes ratio will be the closest possible to the
                  pre-augmentation ratio.
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
        self.original_sents_nb = len(self.sents)

        # Data augmentation
        self.augmenters = augmenters
        self.aug_frequencies = aug_frequencies

        if aug_method is None:
            assert len(self.augmenters) == 0
            assert len(self.aug_frequencies) == 0
        else:
            if aug_method == "standard":
                self.sents = _augment(self.sents, self.augmenters, self.aug_frequencies)
            elif aug_method == "replace":
                self.sents = _augment_replace(
                    self.sents, self.augmenters, self.aug_frequencies
                )
            elif aug_method == "balance_upsample":
                self.sents = _augment_balance(
                    self.sents, self.augmenters, self.aug_frequencies
                )
            else:
                raise ValueError(f"Unknown data augmentation method : {aug_method}")

        self.augmented_sents_nb = len(self.sents)

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
        aug_method: Optional[
            Literal["standard", "replace", "balance_upsample"]
        ] = "standard",
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/train2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            aug_method=aug_method,
            **kwargs,
        )

    @staticmethod
    def test_dataset(
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        context_size: int = 0,
        aug_method: Optional[Literal["standard", "replace", "balance_upsample"]] = None,
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/test2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            aug_method=aug_method,
            **kwargs,
        )

    @staticmethod
    def valid_dataset(
        augmenters: Dict[str, List[NERAugmenter]],
        aug_frequencies: Dict[str, List[float]],
        context_size: int = 0,
        aug_method: Optional[Literal["standard", "replace", "balance_upsample"]] = None,
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(
            f"{script_dir}/valid2.txt",
            augmenters,
            aug_frequencies,
            context_size=context_size,
            aug_method=aug_method,
            **kwargs,
        )
