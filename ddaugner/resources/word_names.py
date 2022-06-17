import os, json, random


script_dir = os.path.dirname(os.path.abspath(__file__))


class WordNamesGazetteer:
    """"""

    def __init__(self) -> None:
        with open(f"{script_dir}/word_names.json") as f:
            self.names: List[str] = json.load(f)

    def random_name(self) -> str:
        return random.choice(self.names)
