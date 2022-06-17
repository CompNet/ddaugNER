import os, json, random


script_dir = os.path.dirname(os.path.abspath(__file__))


class ConllGazetteer:
    """"""

    def __init__(self) -> None:
        self.names = {}

        with open(f"{script_dir}/conll_per.json") as f:
            self.names["PER"] = json.load(f)

        with open(f"{script_dir}/conll_org.json") as f:
            self.names["ORG"] = json.load(f)

        with open(f"{script_dir}/conll_loc.json") as f:
            self.names["LOC"] = json.load(f)

        with open(f"{script_dir}/conll_misc.json") as f:
            self.names["MISC"] = json.load(f)

    def random_name(self, tag: str) -> str:
        return random.choice(self.names[tag])
