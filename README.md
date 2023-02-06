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

## Generating documentation
 
Some API documentation is available using `sphinx`. Go the the `docs` directory, and run `make html` to generate documentation under `docs/_build/html`.


# Training a model

Use the `train.py` script to train a model. To see all the possible options, use `python train.py --help`.


# Evaluating a model

The `extract_metrics.py` script can be used to evaluate a model. See `python extract_metrics.py --help` for more infos.


# Published Articles


## Data Augmentation for Robust Character Detection in Fantasy Novels

### Main Results

The following command trains a model without any augmentation:

```sh
python train.py\
       --epochs-nb 2\
       --batch-size 4\
       --context-size 1\
       --model-path model.pth
```

While the following trains a model with our `The Elder Scrolls` augmentation as in the article:

```sh
for aug_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

    python train.py\
	   --epochs-nb 2\
	   --dynamic-epochs-nb\
	   --batch-size 4\
	   --context-size 0\
	   --data-aug-strategies '{"PER": ["the_elder_scrolls"]}'\
	   --data-aug-frequencies "{\"PER\": [${aug_rate}]}"\
	   --model-path "augmented_model_${aug_rate}.pth"

done
```

After training a model, you can see its performance on the dataset with the `extract_metrics.py` script:

```sh
python extract_metrics.py\
       --model-path model.pth\
       --global-metrics\
       --context-size 0\
       --book-group "fantasy"\
       --fix-sent-tokenization\
       --output-file results.json
```

### Alternative augmentation methods

You can reproduce results shown in Figure 3 using the following:

```sh
for aug_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

    for aug_method in 'balance_upsample' 'replace'; do

	python train.py\
	    --epochs-nb 2\
	    --dynamic-epochs-nb\
	    --batch-size 4\
	    --context-size 0\
	    --data-aug-strategies '{"PER": ["the_elder_scrolls"]}'\
	    --data-aug-frequencies "{\"PER\": [${aug_rate}]}"\
	    --data-aug-method "${aug_method}"
	    --model-path "augmented_model_${aug_method}_${aug_rate}.pth"

    done

done
```

### Context size

Results in Figure 4 can be reproduced using:

```sh
for context_size in 1 2 3 4 5 6 7 8 9; do

    python train.py\
	--epochs-nb 2\
	--dynamic-epochs-nb\
	--batch-size 4\
	--context-size ${context_size}\
	--data-aug-strategies '{"PER": ["the_elder_scrolls"]}'\
	--data-aug-frequencies "{\"PER\": [0.6]}"\
	--model-path "augmented_model_${context_size}.pth"

done
```


### Citation

Please cite this work as follows:

```bibtex
@InProceedings{amalvy:hal-03972448,
  title	       = {{Data Augmentation for Robust Character Detection in
                  Fantasy Novels}},
  author       = {Amalvy, Arthur and Labatut, Vincent and Dufour,
                  Richard},
  url	       = {https://hal.science/hal-03972448},
  booktitle    = {{Workshop on Computational Methods in the Humanities
                  2022}},
  year	       = {2022},
  hal_id       = {hal-03972448},
  hal_version  = {v1},
}
```



## Remplacement de mentions pour l'adaptation d'un corpus de reconnaissance d'entités nommées à un domaine cible

All augmentation configurations can be tested as in the article :

```sh
for i in 0.05 0.1 0.5 1.0; do

    for aug in conll wgold morrowind dekker; do

	python train.py\
	       --epochs-nb 2\
	       --batch-size 4\
	       --context-size 1\
	       --data-aug-strategies "{\"PER\": [\"${aug}\"]}"\
	       --data-aug-frequencies "{\"PER\": [${i}]}"\
	       --model-path augmented_model.pth

	python extract_metrics.py\
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


## BERT meets d'Artagnan : Data Augmentation for Robust Character Detection in Novels

The following command trains a model without any augmentation:

```sh
poetry run python train.py\
       --epochs-nb 2\
       --batch-size 4\
       --context-size 1\
       --model-path model.pth
```

While the following trains a model with our *morrowind* augmentation as in the article:

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

After training a model, you can see its performance on the dataset with the `extract_metrics.py` script:

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


# About the CoNLL-2003 Corpus

This repository contains a modified version of the CoNLL-2003 corpus. Copyrights are defined at the [Reuters Corpus page](https://trec.nist.gov/data/reuters/reuters.html). See more details at [the page of the CoNLL-2003 shared task](https://www.clips.uantwerpen.be/conll2003/ner/).
