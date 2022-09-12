from collections import defaultdict
import argparse, json
from rich import print
from rich.prompt import Prompt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--input-file", type=str)
    parser.add_argument("-of", "--output-file", type=str)
    args = parser.parse_args()
    assert not args.input_file is None
    assert not args.output_file is None

    with open(args.input_file) as f:
        specific_errors = json.load(f)

    categorised_errors = defaultdict(list)

    for i, err_dict in enumerate(specific_errors):
        print(f"{i+1}/{len(specific_errors)}")
        entity = err_dict["entity"]
        context = err_dict["context"]
        print('"' + " ".join(entity) + '"')
        print('"' + " ".join(context) + '"')
        print("[blue]- a(mbiguous)[/blue]")
        print("[blue]- n(ot ambiguous)[/blue]")
        answer = Prompt.ask("> ", choices=["a", "n"])
        if answer == "a":
            categorised_errors["ambiguous"].append([entity, context])
        elif answer == "n":
            categorised_errors["not ambiguous"].append([entity, context])
        else:
            raise ValueError

    ambiguous_nb = len(categorised_errors["ambiguous"])
    not_ambiguous_nb = len(categorised_errors["not ambiguous"])
    print(f"ambigous entities : {ambiguous_nb}")
    print(f"non ambiguous entities : {not_ambiguous_nb}")

    with open(args.output_file, "w") as f:
        json.dump(categorised_errors, f, indent=4)
