import os, json, random


script_dir = os.path.dirname(os.path.abspath(__file__))


class FrenchNamesGazetteer:
    """"""

    def __init__(self) -> None:
        with open(f"{script_dir}/french_names.json") as f:
            datas = json.load(f)
            self.men_first_names = datas["firstnames"]["men"]
            self.men_honorifics = datas["honorifics"]["men"]
            self.women_first_names = datas["firstnames"]["women"]
            self.women_honorifics = datas["honorifics"]["women"]
            self.surnames = datas["surnames"]

    def random_name(self) -> str:
        return random.choice(
            [
                # first name
                lambda: random.choice(self.men_first_names + self.women_first_names),
                # first name + surname
                lambda: f"{random.choice(self.men_first_names + self.women_first_names)} {random.choice(self.surnames)}",
                # honorifics + surname
                lambda: f"{random.choice(self.men_honorifics + self.women_honorifics)} {random.choice(self.surnames)}",
                # men : honorific + first name + surname
                lambda: f"{random.choice(self.men_honorifics)} {random.choice(self.men_first_names)} {random.choice(self.surnames)}",
                # women : honorific + first name + surname
                lambda: f"{random.choice(self.women_honorifics)} {random.choice(self.women_first_names)} {random.choice(self.surnames)}",
            ]
        )()
