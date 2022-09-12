from typing import List, cast, Set
import argparse, json
from collections import Counter
from transformers import BertForTokenClassification  # type: ignore
from rich import print
from ddaugner.datas.dekker import load_dekker_dataset
from ddaugner.datas.aug import TheElderScrollsAugmenter
from ddaugner.train import train_ner_model
from ddaugner.predict import predict
from ddaugner.datas import NERDataset
from ddaugner.datas.conll import CoNLLDataset
from ddaugner.utils import NEREntity, flattened, entities_from_bio_tags


def train_and_predict(
    train_dataset: NERDataset, dekker_dataset: NERDataset
) -> List[NEREntity]:
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=train_dataset.tags_nb,
        label2id=train_dataset.tag_to_id,
        id2label={v: k for k, v in train_dataset.tag_to_id.items()},
    )
    model = train_ner_model(model, train_dataset, train_dataset, epochs_nb=2)
    preds = predict(model, dekker_dataset)
    preds = cast(List[List[str]], preds)
    return entities_from_bio_tags(
        flattened([s.tokens for s in dekker_dataset.sents]), flattened(preds)
    )


def entities_names(entities: Set[NEREntity]) -> List[str]:
    return [" ".join(e.tokens) for e in entities]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-rn", "--repeats-nb", type=int, default=3)
    parser.add_argument("-cs", "--context-size", type=int, default=0)
    parser.add_argument("-of", "--output-file", type=str, default=None)
    parser.add_argument("-fst", "--fix-sent-tokenization", action="store_true")
    args = parser.parse_args()
    assert args.repeats_nb >= 1
    assert not args.output_file is None

    dekker_dataset = load_dekker_dataset(
        "./ner",
        book_group="fantasy",
        context_size=args.context_size,
        fix_sent_tokenization=args.fix_sent_tokenization,
    )
    dekker_tokens = flattened([s.tokens for s in dekker_dataset.sents])
    gold_tags = flattened([s.tags for s in dekker_dataset.sents])
    gold_entities = set(entities_from_bio_tags(dekker_tokens, gold_tags))

    # noaug training
    noaug_train_dataset = CoNLLDataset.train_dataset(
        {}, {}, context_size=args.context_size
    )
    noaug_pred_entities = set(train_and_predict(noaug_train_dataset, dekker_dataset))
    for i in range(args.repeats_nb - 1):
        pred_entities = train_and_predict(noaug_train_dataset, dekker_dataset)
        noaug_pred_entities = noaug_pred_entities.intersection(set(pred_entities))

    # tes aug training
    tes_train_dataset = CoNLLDataset.train_dataset(
        {"PER": [TheElderScrollsAugmenter()]}, {"PER": [0.5]}, 0, "standard"
    )
    tes_pred_entities = set(train_and_predict(tes_train_dataset, dekker_dataset))
    for i in range(args.repeats_nb - 1):
        pred_entities = train_and_predict(tes_train_dataset, dekker_dataset)
        tes_pred_entities = tes_pred_entities.intersection(pred_entities)

    better_recalled = (
        tes_pred_entities.intersection(gold_entities) - noaug_pred_entities
    )
    counter = Counter(entities_names(better_recalled))
    srted = sorted([(k, v) for k, v in counter.items()], key=lambda kv: kv[1])
    print(srted)

    if not args.output_file is None:
        with open(args.output_file, "w") as f:
            json.dump(srted, f, indent=4)
