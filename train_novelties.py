import argparse, json
from transformers import BertForTokenClassification
from rich import print
from ddaugner.predict import predict
from ddaugner.score import score_ner
from ddaugner.train import train_ner_model
from ddaugner.datas.conll.conll import _augment, _augment_replace, _augment_balance
from ddaugner.datas.novelties import load_novelties_dataset
from ddaugner.datas.aug import all_augmenters
from ddaugner.utils import flattened


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-en", "--epochs-nb", type=float, default=2.0)
    parser.add_argument(
        "-den",
        "--dynamic-epochs-nb",
        action="store_true",
        help="If specified, the number of epochs (which is a float) will be modified according to the number of augmented examples added to the model, so that the model sees as much examples in total as he would have seen without augmentation.",
    )
    parser.add_argument("-bz", "--batch-size", type=int, default=4)
    parser.add_argument("-bm", "--batch-mode", action="store_true")
    parser.add_argument("-mp", "--model-path", type=str, default="./ner_model")
    parser.add_argument(
        "-cw",
        "--custom-weights",
        type=json.loads,
        help="Custom class weights, as a json dictionay. Exemple : python train.py -cw '{\"B-PER\": 2}'",
        default=None,
    )
    parser.add_argument("-cs", "--context-size", type=int, default=0)
    parser.add_argument(
        "-koc",
        "--keep-only-classes",
        nargs="*",
        default=None,
        help="A list of classes to keep at training time, separated by spaces. Exemple : 'PER ORG'",
    )
    parser.add_argument("-tmp", "--test-metrics-path", type=str, default=None)
    parser.add_argument(
        "-das",
        "--data-aug-strategies",
        default="{}",
        help=f"a json dictionary mapping a NER class to a list of replacement strategies (available strategies : {list(all_augmenters.keys())})",
    )
    parser.add_argument(
        "-daf",
        "--data-aug-frequencies",
        default="{}",
        help="a json dictionary mapping a NER class to a list of frequencies for the given replacement strategies, in order",
    )
    parser.add_argument(
        "-dam",
        "--data-aug-method",
        default="standard",
        type=str,
        help="augmentation method. One of 'standard', 'replace'or 'balance_upsample'",
    )
    parser.add_argument("-bg", "--book-group", type=str, default="nofantasy")
    args = parser.parse_args()

    print("running with config : ")
    print(vars(args))

    # augmentation parsing
    data_aug_strategies = json.loads(args.data_aug_strategies)
    augmenters = {}
    for ner_class, strategies in data_aug_strategies.items():
        assert all([strategy in all_augmenters.keys()] for strategy in strategies)
        augmenters[ner_class] = [all_augmenters[strategy]() for strategy in strategies]

    data_aug_frequencies = json.loads(args.data_aug_frequencies)
    for ner_class, frequencies in data_aug_frequencies.items():
        assert len(frequencies) == len(augmenters[ner_class])

    # * dataset loading
    # ** train dataset
    train = load_novelties_dataset(
        "./ner/Novelties", book_group=args.book_group, context_size=args.context_size
    )
    # HACK: and now for a bit of fun! Augmentation was only
    # implemented for CoNLL dataset (see the __init__ method for
    # details), but we can easily reproduce it here.
    train.original_sents_nb = len(train.sents)  # type: ignore
    train.augmenters = augmenters  # type: ignore
    train.aug_frequencies = data_aug_frequencies  # type: ignore
    if args.data_aug_method is None:
        assert len(train.augmenters) == 0  # type: ignore
        assert len(train.aug_frequencies) == 0  # type: ignore
    else:
        if args.data_aug_method == "standard":
            train.sents = _augment(train.sents, train.augmenters, train.aug_frequencies)  # type: ignore
        elif args.data_aug_method == "replace":
            train.sents = _augment_replace(
                train.sents, train.augmenters, train.aug_frequencies  # type: ignore
            )
        elif args.data_aug_method == "balance_upsample":
            train.sents = _augment_balance(
                train.sents, train.augmenters, train.aug_frequencies  # type: ignore
            )
        else:
            raise ValueError(
                f"Unknown data augmentation method : {args.data_aug_method}"
            )

    train.augmented_sents_nb = len(train.sents)  # type: ignore

    # *** dynamic epochs nb
    if args.dynamic_epochs_nb:
        args.epochs_nb = args.epochs_nb / (
            train.augmented_sents_nb / train.original_sents_nb  # type: ignore
        )
    # ** valid dataset
    valid = train
    # ** test dataset
    test = train

    # model loading
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=train.tags_nb,
        label2id=train.tag_to_id,
        id2label={v: k for k, v in train.tag_to_id.items()},
    )

    if args.custom_weights:
        weights = [1.0 for _ in train.tags]
        for tag, weight in args.custom_weights.items():
            weights[train.tag_to_id[tag]] = weight

    # training
    model = train_ner_model(
        model,
        train,
        valid,
        epochs_nb=args.epochs_nb,
        batch_size=args.batch_size,
        quiet=args.batch_mode,
        custom_weights=weights if args.custom_weights else None,  # type: ignore
    )

    # test inference
    predictions = predict(model, test, args.batch_size, quiet=args.batch_mode)
    precision, recall, f1 = score_ner(
        test.sents, predictions, ignored_classes={"MISC", "ORG", "LOC"}
    )
    metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
    print("test metrics : ")
    print(metrics_dict)
    if not args.test_metrics_path is None:
        with open(args.test_metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

    # save model
    model.save_pretrained(args.model_path)  # type: ignore
