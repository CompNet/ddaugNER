from typing import List, Optional
import re, glob, os
from tqdm import tqdm
from ddaugner.datas import EnsembleDataset, BookDataset


groups = {
    "fantasy": {
        "TheFellowshipoftheRing",
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
    }
}


def book_name(path: str) -> str:
    return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore


def load_dekker_books(
    dataset_root: str,
    book_group: Optional[str] = None,
    context_size: int = 0,
    fix_sent_tokenization: bool = False,
    quiet: bool = False,
) -> List[BookDataset]:

    dataset_root = dataset_root.rstrip("/")
    old_paths = glob.glob(f"{dataset_root}/old/*.conll.fixed")
    new_paths = glob.glob(f"{dataset_root}/new/*.conll.fixed")

    if not book_group is None:

        def book_name(path: str) -> str:
            return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore

        old_paths = [p for p in old_paths if book_name(p) in groups[book_group]]
        new_paths = [p for p in new_paths if book_name(p) in groups[book_group]]

    return [
        BookDataset(
            path,
            context_size=context_size,
            fix_sent_tokenization=fix_sent_tokenization,
        )
        for path in tqdm(old_paths + new_paths, disable=quiet)
    ]


def load_dekker_dataset(
    dataset_root: str,
    book_group: Optional[str] = None,
    context_size: int = 0,
    fix_sent_tokenization: bool = False,
    **kwargs,
) -> EnsembleDataset:

    return EnsembleDataset(
        load_dekker_books(
            dataset_root,
            book_group=book_group,
            context_size=context_size,
            fix_sent_tokenization=fix_sent_tokenization,
            **kwargs,
        )
    )
