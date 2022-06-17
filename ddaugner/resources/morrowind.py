import json
import random, os


script_dir = os.path.dirname(os.path.abspath(__file__))


class MorrowindNamesGazetteer:
    """"""

    def __init__(self) -> None:
        with open(f"{script_dir}/morrowind_names.json") as f:
            self.names = json.load(f)

    def random_name(self) -> str:
        return random.choice(self.names)


class MorrowindLocationsGazetteer:
    """"""

    def __init__(self) -> None:
        with open(f"{script_dir}/morrowind_locs.json") as f:
            self.locations = json.load(f)

    def random_loc(self) -> str:
        return random.choice(self.locations)
