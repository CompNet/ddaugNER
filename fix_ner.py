from typing import Dict, List, Optional, Tuple
import os, glob, itertools, argparse

from transformers import BertForTokenClassification
from nameparser.config.titles import TITLES
from more_itertools import windowed
from rich import print
from rich.table import Table
from rich.prompt import Prompt

from ddaugner.predict import predict
from ddaugner.utils import entities_from_bio_tags, flattened
from ddaugner.datas.datas import BookDataset


parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--bert-model", type=str, default="dslim/bert-base-NER")
args = parser.parse_args()

# 1. load corpus
script_dir = f"{os.path.dirname(os.path.abspath(__file__))}"

old_paths = glob.glob(f"{script_dir}/ner/old/*.conll")
new_paths = glob.glob(f"{script_dir}/ner/new/*.conll")

book_names = [
    os.path.splitext(os.path.basename(path))[0] for path in old_paths + new_paths
]

# a list of titles taken from nameparser
# we add some titles seen in the books for better performance
titles = TITLES | {"monsieur", "mr", "mr.", "mrs", "mrs.", "m."}

tokens: Dict[str, List[str]] = {}
tags: Dict[str, List[str]] = {}
entities = {}
characters = {}

for path in old_paths + new_paths:
    book_name = os.path.splitext(os.path.basename(path))[0]
    tokens[book_name] = []
    tags[book_name] = []
    entities[book_name] = []
    with open(path) as f:
        for line in f:
            splitted = line.split(" ")
            tokens[book_name].append(splitted[0])
            tags[book_name].append(splitted[1].strip())
    entities[book_name] = entities_from_bio_tags(tokens[book_name], tags[book_name])

    # we create a set of characters name with
    # - characters full name
    # - components of those names that are not thought
    #   to be titles
    with open(f"{os.path.dirname(path)}/{book_name}.characters") as f:
        book_characters = set(f.read().split("\n"))
    name_components = set()
    for character in book_characters:
        for i in range(len(character.split(" "))):
            for name_component in itertools.combinations(character.split(" "), i):
                name_component = " ".join(name_component)
                if (
                    not name_component == ""
                    and not name_component[0].islower()
                    and not name_component.lower() in titles
                ):
                    name_components.add(name_component)
    characters[book_name] = book_characters.union(name_components)


# 2. fix errors
nb_errors_fixed = 0

# {book_name => [decision]}
# each decision is of the form ([tokens + original tags], [chosen tags])
saved_decisions: Dict[str, List[Tuple[List[str], List[str]]]] = {
    book_name: [] for book_name in book_names
}


def get_saved_decision(
    book_name: str, tokens: List[str], tags: List[str]
) -> Optional[List[str]]:
    """
    :return: the list of fixed, or ``None`` if no fix about
        ``tokens`` was saved.
    """
    for saved_decision in saved_decisions[book_name]:
        if saved_decision[0] == tokens + tags:
            return saved_decision[1]
    return None


def save(book_name: str, book_path: str):
    fixed_path = f"{os.path.dirname(book_path)}/{os.path.basename(book_path)}.fixed"
    with open(fixed_path, "w") as f:
        f.write(
            "\n".join(
                [
                    f"{token} {tag}"
                    for token, tag in zip(tokens[book_name], tags[book_name])
                ]
            )
        )


def fix(book_name: str, start_index: int, end_index: int, new_tag: List[str]):
    tags[book_name][start_index : end_index + 1] = new_tag
    global nb_errors_fixed
    nb_errors_fixed += 1
    print(f"corrected error [bold]#{nb_errors_fixed}[/bold]")


def ask_to_fix(
    book_name: str, start_index: int, end_index: int, new_tag: List[str], reason: str
):
    """Ask the user about a possible error fix

    .. note::

        If a fix is found in ``saved_fixs``, it will be applied instead of
        asking the user.

    :param book_name:
    :param start_index: index of the first tag to correct
    :param end_index: index of the last tag to correct
    :param new_tag: replacement tag, of len ``end_index - start_index``
    :param reason: possible error reason
    """
    old_tag = tags[book_name][start_index : end_index + 1]
    assert len(old_tag) == len(new_tag)

    # check if a remembered previous decision should trigger
    # auto-accept or auto-refuse
    target_tokens = tokens[book_name][start_index : end_index + 1]
    saved_decision = get_saved_decision(book_name, target_tokens, old_tag)
    if not saved_decision is None:
        if saved_decision == old_tag:
            print(
                f"[red]auto-refuse fix : {target_tokens} {new_tag} (will keep {old_tag})[/red]"
            )
            return
        print(
            f"[green]auto-accept fix : {target_tokens} [strike]{old_tag}[/strike] -> {new_tag}[/green]"
        )
        fix(book_name, start_index, end_index, saved_decision)
        return

    # print context to assist decision
    ctx_start = max(start_index - 10, 0)
    ctx_end = min(end_index + 11, len(tokens[book_name]) - 1)
    table = Table(title=f"book : [bold]{book_name}[/bold]")
    table.add_column("tokens")
    table.add_column("tags")
    for i in range(ctx_start, ctx_end):
        if i >= start_index and i <= end_index:
            table.add_row(
                f"{tokens[book_name][i]}",
                f"[red strike]{tags[book_name][i]}[/red strike] -> [green]{new_tag[i - start_index]}[/green]",
            )
        else:
            table.add_row(tokens[book_name][i], tags[book_name][i])
    print(table)

    # prompt for user choice
    while True:
        print(f"possible error reason : [italic]{reason}[/italic]")
        print("[blue]- [bold]n[/bold] : refuse fix[/blue]")
        print("[blue]- [bold]nr[/bold] : refuse and [cyan]r[/cyan]emember fix[/blue]")
        print("[blue]- [bold]y[/bold] : accept fix[/blue]")
        print("[blue]- [bold]yr[/bold] : accept and [cyan]r[/cyan]emember fix[/blue]")
        print(
            "[blue]- [bold]c[/bold] : see [cyan]c[/cyan]haracter list (includes generated aliases)[/blue]"
        )
        answer = Prompt.ask("> ", choices=["y", "yr", "n", "nr", "c"], default="n")
        if answer == "c":
            print(characters[book_name])
            continue
        break

    # user refused the fix
    if answer.startswith("n"):
        if answer.endswith("r"):
            saved_decisions[book_name].append((target_tokens + old_tag, old_tag))
        return

    # user accepted the fix
    if answer.endswith("r"):
        saved_decisions[book_name].append((target_tokens + old_tag, new_tag))

    fix(book_name, start_index, end_index, new_tag)


# rules to find false negatives
for book_path in old_paths + new_paths:

    book_name = os.path.splitext(os.path.basename(book_path))[0]

    # check when a suite of tokens is not marked as PER
    # when it exists in the list of characters
    for i in range(5, 0, -1):  # match names of up to 5 tokens
        for j, (toks, tagz) in enumerate(
            zip(windowed(tokens[book_name], i), windowed(tags[book_name], i))
        ):
            if " ".join(toks) in characters[book_name] and any(
                [tag == "O" for tag in tagz]
            ):
                fix(
                    book_name,
                    j,
                    j + i - 1,
                    ["B-PER"] + ["I-PER"] * (i - 1),
                )

    save(book_name, book_path)


# rules to find false positives
for book_path in old_paths + new_paths:

    book_name = os.path.splitext(os.path.basename(book_path))[0]

    # check when a PER is not in the list of characters
    for entity in entities[book_name]:
        mention = " ".join(entity.tokens)
        if mention not in characters[book_name] and not mention.lower() in titles:
            ask_to_fix(
                book_name,
                entity.start_idx,
                entity.end_idx,
                ["O"] * (entity.end_idx - entity.start_idx + 1),
                f"[bold]{mention}[/bold] was not found in the list of characters",
            )

    # check when a suite of non-capitalized tokens were
    # marked as PER
    for i in range(5, 0, -1):  # match up to 5 tokens
        for j, (toks, tagz) in enumerate(
            zip(windowed(tokens[book_name], i), windowed(tags[book_name], i))
        ):
            if all([tok.islower() for tok in toks]) and (
                tagz[0] == "B-PER" and all([tag.endswith("PER") for tag in tagz[1:]])
            ):
                ask_to_fix(
                    book_name,
                    j,
                    j + i - 1,
                    ["O"] * i,
                    f"non-capitalized token(s) {toks} marked as [bold]PER[/bold]",
                )

    save(book_name, book_path)


# find errors using BERT
bert = BertForTokenClassification.from_pretrained(args.bert_model)


for book_path in old_paths + new_paths:

    book_name = os.path.splitext(os.path.basename(book_path))[0]
    book_dataset = BookDataset(book_path)

    predictions = flattened(predict(bert, book_dataset))
    predictions = [t if t in {"B-PER", "I-PER"} else "O" for t in predictions]
    truth = tags[book_name]
    # TODO:
    if len(predictions) != len(truth):
        print(
            f"[DEBUG] len(predictions) doesn't match len(truth) for {book_name}. Skipping..."
        )
        continue

    error_start: Optional[int] = None
    error_end: Optional[int] = None
    for i, (pred_tag, true_tag) in enumerate(zip(predictions, truth)):
        if error_start is None:
            if pred_tag != true_tag:
                error_start = i
        # potential error boundary : fix error if needed
        if not error_start is None and (
            i + 1 == len(predictions) or predictions[i + 1] == truth[i + 1]
        ):
            error_end = i
            predicted_tags = predictions[error_start : error_end + 1]
            ask_to_fix(
                book_name,
                error_start,
                error_end,
                predicted_tags,
                f"BERT predicted token [bold]{predicted_tags}[/bold]",
            )
            error_start = None

    save(book_name, book_path)


print(f"fixed {nb_errors_fixed} errors")
