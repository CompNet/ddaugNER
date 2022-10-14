# Domain Data Augmentation for NER


# Setup and dependencies

The project uses [Poetry](https://python-poetry.org/) to manage dependencies. Having poetry installed, you can install all dependencies using :

```sh
poetry install
```

Further commands in this /README/ assume that you activated the resulting python environment, either manually or by using `poetry shell`.

Alternatively, you can manage everything in your own environment using the provided `requirements.txt` file (use `pip install -r requirements.txt` to install dependencies).

## Literary test corpus

We re-annotated the corpus of [Dekker et al., 2019](https://peerj.com/articles/cs-189/#named-entity-recognition-experiments-and-results) to fix some errors. Due to copyright issues, tokens from the datasets are not directly available in this repository, but can be retrieved through a script :

```sh
python setup_dekker_dataset.py --dekker-etal-repo-path /path/to/dekker/repository
```

If you don't specify a path to Dekker et al repository, the script will attempt to download it automatically using git.


# Training a model

Use the `train.py` script to train a model. To see all the possible options, use `python train.py --help`.


# Evaluating a model

The `extract_metrics.py` script can be used to train a model. See `python extract_metrics.py --help` for more infos.


# Published Articles

## BERT meets d'Artagnan : Data Augmentation for Robust Character Detection in Novels

The following command trains a model without any augmentation :

```sh
poetry run python train.py\
       --epochs-nb 2\
       --batch-size 4\
       --context-size 1\
       --model-path model.pth
```

While the following trains a model with our *morrowind* augmentation as in the article :

```sh
poetry run python train.py\
       --epochs-nb 2\
       --batch-size 4\
       --context-size 1\
       --data-aug-strategies '{"PER": ["morrowind"]}'\
       --data-aug-frequencies '{"PER": [0.1]}'\
       --model-path augmented_model.pth
```

Replace `morrowind` with `word_names` to use our word names augmentation.

After training a model, you can see its performance on the dataset with the `extract_metrics.py` script :

```sh
poetry run python extract_metrics.py\
       --model-path model.pth\
       --global-metrics\
       --context-size 1\
       --book-group "fantasy"\
       --output-file results.json
```

### Citation

Please cite this work as follows :

```bibtex
@InProceedings{amalvy:hal-03617722,
  title = {{BERT meets d'Artagnan: Data Augmentation for Robust Character Detection in Novels}},
  author = {Amalvy, Arthur and Labatut, Vincent and Dufour, Richard},
  url = {https://hal.archives-ouvertes.fr/hal-03617722},
  booktitle = {{Workshop on Computational Methods in the Humanities (COMHUM)}},
  year = {2022},
  hal_id = {hal-03617722},
  hal_version = {v2},
}
```


## Remplacement de mentions pour l'adaptation d'un corpus de reconnaissance d'entités nommées à un domaine cible

All augmentation configurations can be tested as in the article :

```sh
for i in 0.05 0.1 0.5 1.0; do

    for aug in conll wgold morrowind dekker; do

		poetry run python train.py\
			--epochs-nb 2\
			--batch-size 4\
			--context-size 1\
			--data-aug-strategies "{\"PER\": [\"${aug}\"]}"\
			--data-aug-frequencies "{\"PER\": [${i}]}"\
			--model-path augmented_model.pth

		poetry run python extract_metrics.py\
			--model-path model.pth\
			--global-metrics\
			--context-size 1\
			--book-group "fantasy"\
			--output-file "results_${aug}_${i}.json"

    done

done
```

### Citation

Please cite this work as follows :

```bibtex
@InProceedings{amalvy:hal-03651510,
  title = {{Remplacement de mentions pour l'adaptation d'un corpus de reconnaissance d'entit{\'e}s nomm{\'e}es {\`a} un domaine cible}},
  author = {Amalvy, Arthur and Labatut, Vincent and Dufour, Richard},
  url = {https://hal.archives-ouvertes.fr/hal-03651510},
  booktitle = {{29{\`e}me Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles (TALN)}},
  year = {2022},
  hal_id = {hal-03651510},
  hal_version = {v3},
}
```


# About the CoNLL-2003 Corpus

This repository contains a modified version of the CoNLL-2003 corpus. Copyrights are defined at the [Reuters Corpus page](https://trec.nist.gov/data/reuters/reuters.html). See more details at [the page of the CoNLL-2003 shared task](https://www.clips.uantwerpen.be/conll2003/ner/).
