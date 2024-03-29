from __future__ import annotations
from typing import List, Dict, Set, Union, Optional, cast
from collections import defaultdict
from dataclasses import dataclass, field

from itertools import chain
from more_itertools import windowed
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from ddaugner.utils import flattened, get_tokenizer


@dataclass(frozen=True)
class NERSentence:
    tokens: List[str]
    tags: List[str]
    left_context: List[NERSentence] = field(default_factory=lambda: [])
    right_context: List[NERSentence] = field(default_factory=lambda: [])

    def __len__(self) -> int:
        assert len(self.tokens) == len(self.tags)
        return len(self.tokens)

    @staticmethod
    def sents_with_surrounding_context(
        sents: List[NERSentence], context_size: int = 1
    ) -> List[NERSentence]:
        """Set the context of each sent of the given list, according to surrounding sentence

        :param sents:

        :param context_size: number of surrounding sents to take into account, left and right.
            (a value of 1 means taking one sentence left for ``left_context`` and one sentence
            right for ``right_context``.)

        :return: a list of sentences, with ``left_context`` and ``right_context`` set with
            surrounding sentences.
        """
        if len(sents) == 0:
            return []

        new_sents: List[NERSentence] = []

        window_size = 1 + 2 * context_size
        padding = [None] * context_size
        for window_sents in windowed(chain(padding, sents, padding), window_size):
            center_idx = window_size // 2
            center_sent = window_sents[center_idx]
            assert not center_sent is None
            left_ctx = [s for s in window_sents[:center_idx] if not s is None]
            right_ctx = [s for s in window_sents[center_idx + 1 :] if not s is None]
            new_sents.append(
                NERSentence(
                    center_sent.tokens,
                    center_sent.tags,
                    left_context=left_ctx,
                    right_context=right_ctx,
                )
            )

        return new_sents


def batch_to_device(batch: BatchEncoding, device: torch.device) -> BatchEncoding:
    """Send a batch to a torch device, even when containing non-tensor variables"""
    if isinstance(batch, BatchEncoding) and all(
        [isinstance(v, torch.Tensor) for v in batch.values()]
    ):
        return batch.to(device)
    return BatchEncoding(
        {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        },
        encoding=batch.encodings,
    )


def align_tokens_labels_(
    batch_encoding: BatchEncoding, labels: List[str], all_labels: Dict[str, int]
) -> BatchEncoding:
    """Modify a huggingface single batch encoding by adding tokens labels, taking wordpiece into account

    .. note::

        Adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner

    :param batch_encoding: a ``'labels'`` key will be added. It must be a single batch.
    :param labels: list of per-token labels. ``None`` labels will be given -100 label,
        in order to be ignored by torch loss functions.
    :param all_labels: mapping of a label to its index
    :return: the modified batch encoding
    """
    labels_ids: List[int] = []
    word_ids = batch_encoding.word_ids(batch_index=0)
    for word_idx in word_ids:
        if word_idx is None:
            labels_ids.append(-100)
            continue
        if labels[word_idx] is None:
            labels_ids.append(-100)
            continue
        labels_ids.append(all_labels[labels[word_idx]])
    batch_encoding["labels"] = labels_ids
    return batch_encoding


class DataCollatorForTokenClassificationWithBatchEncoding:
    """Same as ``transformers.DataCollatorForTokenClassification``, except it :

    - correctly returns a ``BatchEncoding`` object with correct ``encodings``
        attribute.
    - wont try to convert the key ``'tokens_labels_mask'`` that is used to
        determine

    Don't know why this is not the default ?
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = -100

    def __call__(self, features) -> Union[dict, BatchEncoding]:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )
        # keep encodings info dammit
        batch._encodings = [f.encodings[0] for f in features]

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        # ignore "tokens_labels_mask"
        return BatchEncoding(
            {
                k: torch.tensor(v, dtype=torch.int64)
                if not k in {"tokens_labels_mask"}
                else v
                for k, v in batch.items()
            },
            encoding=batch.encodings,
        )


class NERDataset(Dataset):
    """
    :ivar sents: ``List[NERSentence]``
    :ivar tags: the set of all possible entity classes
    :ivar tags_nb: number of tags
    :ivar tags_to_id: ``Dict[tag: str, id: int]``
    """

    def __init__(
        self,
        sents: List[NERSentence],
        tags: Optional[Set[str]] = None,
    ) -> None:
        """
        :param sents:
        :param tags:
        """
        self.sents = sents

        if tags is None:
            self.tags = {tag for sent in sents for tag in sent.tags}
        else:
            self.tags = tags
        self.tags_nb = len(self.tags)
        self.tag_to_id: Dict[str, int] = {
            tag: i for i, tag in enumerate(sorted(list(self.tags)))
        }

        self.tokenizer: BertTokenizerFast = get_tokenizer()

    def tag_frequencies(self) -> Dict[str, float]:
        """
        :return: a mapping from token to its frequency
        """
        tags_count = defaultdict(int)
        for sent in self.sents:
            for tag in sent.tags:
                tags_count[tag] += 1
        total_count = sum(tags_count.values())
        return {tag: count / total_count for tag, count in tags_count.items()}

    def tag_weights(self) -> List[float]:
        """
        :return: a list of weights, ordered by ``self.tags_to_id``.
            Each tag weight is computed as ``max_tags_frequency / tag_frequency``.
        """
        weights = [0.0] * len(self.tags)
        frequencies = self.tag_frequencies()
        max_frequency = max(frequencies.values())
        for tag, frequency in frequencies.items():
            weights[self.tag_to_id[tag]] = max_frequency / frequency
        return weights

    def __getitem__(self, index: int) -> Dict[str, List[str]]:
        """Get a BatchEncoding representing sentence at index, with its context

        .. note::

            As an addition to the classic huggingface BatchEncoding keys,
            a "tokens_labels_mask" is added to the outputed BatchEncoding.
            This masks denotes the difference between a sentence context
            (previous and next context) and the sentence itself. when
            concatenating a sentence and its context sentence, we obtain :

            ``[l1, l2, l3, ...] + [s1, s2, s3, ...] + [r1, r2, r3, ...]``

            with li being a token of the left context, si a token of the
            sentence and ri a token of the right context. The
            "tokens_labels_mask" is thus :

            ``[0, 0, 0, ...] + [1, 1, 1, ...] + [0, 0, 0, ...]``

            This mask is produced *before* tokenization by a huggingface
            tokenizer, and therefore corresponds to *tokens* and not to
            *wordpieces*.

        :param index:
        :return:
        """
        sent = self.sents[index]

        batch = self.tokenizer(
            flattened([s.tokens for s in sent.left_context])
            + sent.tokens
            + flattened([s.tokens for s in sent.right_context]),
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )

        batch["tokens_labels_mask"] = [0] * len(
            flattened([s.tags for s in sent.left_context])
        )
        batch["tokens_labels_mask"] += [1] * len(sent.tags)
        batch["tokens_labels_mask"] += [0] * len(
            flattened([s.tags for s in sent.right_context])
        )

        assert len([i for i in batch["tokens_labels_mask"] if i == 1]) == len(
            self.sents[index].tags
        )

        return align_tokens_labels_(
            batch,
            flattened([s.tags for s in sent.left_context])
            + sent.tags
            + flattened([s.tags for s in sent.right_context]),
            self.tag_to_id,
        )

    def __len__(self) -> int:
        return len(self.sents)


class BookDataset(NERDataset):
    """"""

    def __init__(
        self, path: str, context_size: int = 0, fix_sent_tokenization: bool = False
    ) -> None:
        """
        :param path:
        :param context_size:
        :param fix_sent_tokenization:
        """
        self.path = path

        sents = []
        with open(path) as f:
            sent = NERSentence([], [])
            for line in f:
                token, tag = line.strip().split(" ")
                sent.tokens.append(token)
                sent.tags.append(tag)
                if token in [".", "?", "!"] or (
                    fix_sent_tokenization and token == "''"
                ):
                    sents.append(sent)
                    sent = NERSentence([], [])

        if not fix_sent_tokenization:
            sents = NERSentence.sents_with_surrounding_context(sents, context_size)
            super().__init__(sents, {"O", "B-PER", "I-PER"})
            return

        for sent in sents:
            for token_i, token in enumerate(sent.tokens):

                # replace nltk's double quotes by conventional ones
                if token in ["``", "''"]:
                    sent.tokens[token_i] = '"'

                # replace nltk's single quotes with conventional ones
                if token == "`":
                    sent.tokens[token_i] = "'"

                # fix parentheses
                if token == "-LRB-":
                    sent.tokens[token_i] = "("
                elif token == "-RRB-":
                    sent.tokens[token_i] = ")"

                # fix brackets
                if token == "-LSB-":
                    sent.tokens[token_i] = "["
                elif token == "-RSB-":
                    sent.tokens[token_i] = "]"

        sents = NERSentence.sents_with_surrounding_context(sents, context_size)
        super().__init__(sents, {"O", "B-PER", "I-PER"})


class EnsembleDataset(NERDataset):
    """"""

    def __init__(self, datasets: List[NERDataset]) -> None:
        assert len(datasets) > 0
        super().__init__(
            flattened([d.sents for d in datasets]),
            set.union(*[d.tags for d in datasets]),
        )
