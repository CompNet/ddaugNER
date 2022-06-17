import os, json, random


script_dir = os.path.dirname(os.path.abspath(__file__))


class WGoldNamesGazetteer:
    """"""

    def __init__(self) -> None:
        with open(f"{script_dir}/wgold.json") as f:
            self.names = json.load(f)

    def random_name(self) -> str:
        return random.choice(self.names)
