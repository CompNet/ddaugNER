from typing import List, Tuple, Dict, Dict, Set, Optional
from collections import Counter
from ddaugner.datas import NERSentence
from ddaugner.utils import flattened, entities_from_bio_tags


def prediction_errors(
    ref_sents: List[NERSentence],
    pred_bio_tags: List[List[str]],
    ignored_classes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    :param ref_sents: list of reference sents
    :param pred_bio_tags: list of sentences tag predictions
    :param ignored_classes:
    :return: ``(precision errors, recall errors)``
    """
    tokens = flattened([s.tokens for s in ref_sents])

    ref_entities = entities_from_bio_tags(
        tokens,
        flattened([s.tags for s in ref_sents]),
    )
    pred_entities = entities_from_bio_tags(tokens, flattened(pred_bio_tags))
    if not ignored_classes is None:
        ref_entities = [e for e in ref_entities if not e.tag in ignored_classes]
        pred_entities = [e for e in pred_entities if not e.tag in ignored_classes]

    precision_errors = Counter()
    for pred_entity in pred_entities:
        if not pred_entity in ref_entities:
            precision_errors[" ".join(pred_entity.tokens)] += 1

    recall_errors = Counter()
    for ref_entity in ref_entities:
        if not ref_entity in pred_entities:
            recall_errors[" ".join(ref_entity.tokens)] += 1

    return (dict(precision_errors), dict(recall_errors))


def conll_export(
    path: str,
    ref_sents: List[NERSentence],
    pred_bio_tags: Optional[List[List[str]]] = None,
):
    """Export a list of NER sentences and predictions from a tagger to a conll formatted file

    :param ref_sents:
    :param pred_bio_tags:
    :param path: export path
    """
    with open(path, "w") as f:
        if not pred_bio_tags is None:
            for sent, predicted_tags in zip(ref_sents, pred_bio_tags):
                for i in range(len(sent)):
                    f.write(
                        "{} {} {}\n".format(
                            sent.tokens[i], sent.tags[i], predicted_tags[i]
                        )
                    )
        else:
            for sent in ref_sents:
                for token, tag in zip(sent.tokens, sent.tags):
                    f.write(f"{token} {tag}\n")


def ner_classes_appearances_nb(sents: List[NERSentence]) -> Counter:
    """Compute the number of appearance of each NER class in a list of sentences"""
    return Counter(
        [
            e.tag
            for e in flattened(
                [entities_from_bio_tags(sent.tokens, sent.tags) for sent in sents]
            )
        ]
    )


def ner_classes_ratios(
    sents: List[NERSentence], ner_classes: Set[str]
) -> Optional[Dict[str, float]]:
    """Compute the proportion of each specified NER class in the given
    sentences

    :param sent: a list of ``NERSentence``
    :param ner_classes: a list of NER classes

    :return: a dict mapping a NER class to its appearance ratio, or
             ``None`` if it this ratio is undefined
    """
    counter = ner_classes_appearances_nb(sents)
    if len(counter) == 0:
        return None
    total = sum(counter.values())
    return {k: counter[k] / total for k in ner_classes.union(set(counter.keys()))}
