from __future__ import annotations
from typing import Dict, List, Optional, Set
import os, random
from ddaugner.datas import NERDataset, NERSentence
from ddaugner.datas.aug import NERAugmenter

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
    ) -> None:
        """ """
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

        augmented_sents = []

        for ner_class, local_augmenters in self.augmenters.items():

            for i, augmenter in enumerate(local_augmenters):

                local_augmented_sents = []
                aug_freq = self.aug_frequencies[ner_class][i]

                while len(local_augmented_sents) < len(self.sents) * aug_freq:
                    sent = random.choice(self.sents)
                    # kind of a hack - NER class is passed in case of
                    # LabelWiseNERAugmenter
                    augmented = augmenter(sent, prev_entity_type=ner_class)
                    if not augmented is None:
                        local_augmented_sents.append(augmented)

                augmented_sents += local_augmented_sents

        self.sents += augmented_sents

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
