import os, json, random


script_dir = os.path.dirname(os.path.abspath(__file__))


class DekkerFantasyPERGazetteer:
    """"""

    def __init__(self):
        with open(f"{script_dir}/dekker_fantasy.json") as f:
            self.names = json.load(f)

    def random_name(self) -> str:
        return random.choice(self.names)
