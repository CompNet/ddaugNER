import json, random, os
from typing import List, Tuple


script_dir = os.path.dirname(os.path.abspath(__file__))


class TheElderScrollsNamesGazetteer:
    def __init__(self) -> None:
        with open(f"{script_dir}/the_elder_scrolls_names.json") as f:
            self.names = json.load(f)
        self.first_names: List[str] = self.names["first_names"]
        self.last_names: List[str] = self.names["last_names"]
        self.prefixs: List[str] = self.names["prefixs"]
        self.suffixs: List[str] = self.names["suffixs"]

        total = (
            len(self.first_names)
            + len(self.last_names)
            + len(self.prefixs)
            + len(self.suffixs)
        )
        self.first_name_prob = len(self.first_names) / total
        self.last_name_prob = len(self.last_names) / total
        self.prefix_prob = len(self.prefixs) / total
        self.suffix_prob = len(self.suffixs) / total

    def random_name_form(self) -> Tuple[bool, bool, bool, bool]:
        """Return a valid name form

        :return: ``(has_prefix, has_first, has_last, has_suffix)``
        """
        rnd = random.random()
        if rnd <= 0.25:
            # first
            return (False, True, False, False)
        if rnd <= 0.5:
            # first last
            return (False, True, True, False)
        if rnd <= 0.75:
            # last
            return (False, False, True, False)
        if rnd <= 0.85:
            # first suffix
            return (False, True, False, True)
        if rnd <= 0.95:
            # first last suffix
            return (False, True, True, True)
        if rnd <= 0.96:
            # last suffix
            return (False, False, True, True)
        if rnd <= 0.97:
            # prefix first last suffix
            return (True, True, True, True)
        if rnd <= 0.98:
            # prefix first last
            return (True, True, True, False)
        if rnd <= 0.99:
            # prefix first suffix
            return (True, True, False, True)
        if rnd <= 1.0:
            # prefix first
            return (True, True, False, False)
        raise RuntimeError

    def random_name(self) -> str:
        name = []

        (
            has_prefix,
            has_first_name,
            has_last_name,
            has_suffix,
        ) = self.random_name_form()

        if has_prefix:
            name.append(random.choice(self.prefixs))
        if has_first_name:
            name.append(random.choice(self.first_names))
        if has_last_name:
            name.append(random.choice(self.last_names))
        if has_suffix:
            name.append(random.choice(self.suffixs))

        return " ".join(name)
