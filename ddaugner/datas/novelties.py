from typing import Optional
import glob
from tqdm import tqdm
from ddaugner.datas import EnsembleDataset, BookDataset

groups = {
    "fantasy": {
        "TheFellowshipOfTheRing",
        "TheWheelOfTime",
        "TheWayOfShadows",
        "TheBladeItself",
        "Elantris",
        "ThePaintedMan",
        "GardensOfTheMoon",
        "Magician",
        "BlackPrism",
        "TheBlackCompany",
        "Mistborn",
        "AGameOfThrones",
        "AssassinsApprentice",
        "TheNameOfTheWind",
        "TheColourOfMagic",
        "TheWayOfKings",
        "TheLiesOfLockeLamora",
    },
    "nofantasy": {
        "1984",
        "AliceInWonderland",
        "AStudyInScarlet",
        "BraveNewWorld",
        "DavidCopperfield",
        "Dracula",
        "Emma",
        "Frankenstein",
        "HarryPotter",
        "HuckleberryFinn",
        "JekyllAndHyde",
        "MobyDick",
        "OliverTwist",
        "PrideAndPrejudice",
        "StormFront",
        "TheCallOfTheWild",
        "TheCountOfMonteCristo",
        "TheGunslinger",
        "TheThreeMusketeers",
        "TheWayWeLiveNow",
        "TinkerTailorSoldierSpy",
        "Ulysses",
        "VanityFair",
    },
}


def load_novelties_books(
    dataset_root: str,
    book_group: Optional[str] = None,
    context_size: int = 0,
    fix_sent_tokenization: bool = False,
    quiet: bool = False,
):
    dataset_root = dataset_root.rstrip("/")

    if book_group:
        paths = []
        for name in groups[book_group]:
            paths.append(f"{dataset_root}/{name}/chapter_1.conll")
    else:
        paths = glob.glob(f"{dataset_root}/**/*.conll")

    return [
        BookDataset(
            path,
            context_size=context_size,
            fix_sent_tokenization=fix_sent_tokenization,
            tags={"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"},
        )
        for path in tqdm(paths, disable=quiet)
    ]


def load_novelties_dataset(
    dataset_root: str,
    book_group: Optional[str] = None,
    context_size: int = 0,
    fix_sent_tokenization: bool = False,
    **kwargs,
):
    return EnsembleDataset(
        load_novelties_books(
            dataset_root,
            book_group=book_group,
            context_size=context_size,
            fix_sent_tokenization=fix_sent_tokenization,
            **kwargs,
        )
    )
