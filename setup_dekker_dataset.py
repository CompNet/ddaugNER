from typing import List
import os, glob, argparse, re
from rich import print


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-derp",
        "--dekker-etal-repo-path",
        type=str,
        default=None,
        help="root directory of dekker et al's repository. If not specified, will try to clone it using git.",
    )
    args = parser.parse_args()

    if args.dekker_etal_repo_path is None:
        os.system(
            "git clone https://github.com/Niels-Dekker/Out-with-the-Old-and-in-with-the-Novel.git"
        )
        args.dekker_etal_repo_path = "./Out-with-the-Old-and-in-with-the-Novel"
    args.dekker_etal_repo_path = args.dekker_etal_repo_path.rstrip("/")

    old_paths = glob.glob("./ner/old/*.conll*")
    new_paths = glob.glob("./ner/new/*.conll*")

    for path in old_paths + new_paths:

        book_set = path.split("/")[-2]
        book_name = re.search(r"[^.]*", os.path.basename(path)).group(0)  # type: ignore

        dekker_file = f"{args.dekker_etal_repo_path}/NER_Experiments/{book_set.capitalize()}/{book_name}.ixa-conllout"

        # get original tokens
        tokens: List[str] = []
        with open(dekker_file) as f:
            for line in f:
                tokens.append(line.split(" ")[0])

        # combine tokens from dekker et al and tags from this repository
        tokens_and_tags = []
        book_has_ner_tags = False
        with open(path) as f:
            for token, line in zip(tokens, f):
                if len(line.split(" ")) > 1:
                    book_has_ner_tags = True
                    break
                tokens_and_tags.append((token, line.strip("\n")))
        if book_has_ner_tags:
            print(f"Book '{book_name}' already has NER tags. Skipping...")
            continue

        # write tokens + tags
        with open(path, "w") as f:
            for token, tag in tokens_and_tags:
                f.write(f"{token} {tag}\n")
