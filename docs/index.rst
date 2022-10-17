.. ddaugNER documentation master file, created by
   sphinx-quickstart on Mon Oct 17 10:09:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ddaugNER's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Train
=====

Training a model
----------------

Here is a full exemple of training a model on the CoNLL dataset :

.. code-block:: python

    from transformers import BertForTokenClassification
    from ddaugner.train import train_ner_model
    from ddaugner.datas.conll import CoNLLDataset

    train = CoNLLDataset.train_dataset(
        {}, # first argument represent the augmenters to use for each class
        {}, # second argument represent augmentation frequency
    ) # other optional arguments exist
    valid = CoNLLDataset.valid_dataset({}, {})

    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=train.tags_nb,
        label2id=train.tag_to_id,
        id2label={v: k for k, v in train.tag_to_id.items()},
    )

    model = train_ner_model(model, train, valid, epochs_nb=2, batch_size=4)
    model.save_pretrained("./my_model")


Module reference
----------------

.. automodule:: ddaugner.train
   :members:


Score
=====

Module reference
----------------

.. automodule:: ddaugner.score
   :members:


Predict
=======

Module reference
----------------

.. automodule:: ddaugner.predict
   :members:


Datas
=====

Module reference
----------------

.. automodule:: ddaugner.datas.datas
   :members:


Aug
===

Creating new augmenters
-----------------------

An augmenter is an object used to generate a new synthetic example
sentence from a given sentence. An augmenter is a class inheriting
from :class:`NERAugmenter` and overriding its ``__call__``
method. This method takes as input the original :class:`NERSentence`,
and returns either ``None`` if it cannot augment the input sentence,
or a new :class:`NERSentence`. Here is a simple (silly) example, where
we replace any occurence of the token ``"duck"`` by the token
``"horse"``:

.. code-block:: python

    class MyAugmenter(NERAugmenter):

	def __call__(self, sent: NERSentence, *args, **kwargs) -> Optional[NERSentence]:
	    if not "duck" in sent.tokens:
	        return None
	    return NERSentence(
	        ["horse" if token == "duck" else token for token in sent.tokens],
	        sent.tags,
	    )

Augmenters for mention replacement should inherit
:class:`LabelWiseNERAugmenter`, and can use specific helper functions
to help implementation. A :class:`LabelWiseNERAugmenter` should only
override the ``__init__`` and :func:`replacement_entity` methods. In
``__init__``, one should pass a set of supported entity types.
:func:`replacement_entity` takes as input the tokens of the entity to
replace, and its type and should return the replacement tokens and the
new entity type. The following :class:`LabelWiseNERAugmenter` replace
any ``"PER"`` entity with ``"Gandalf"``:

.. code-block:: python

    class MyAugmenter(LabelWiseNERAugmenter):

        def __init__(self):
	    super().__init__({"PER"})

        def replacement_entity(
	    self, prev_entity_tokens: List[str], prev_entity_type: str
	) -> Tuple[List[str], str]:
	    return (["Gandalf"], "PER")

This augmenter can then be passed when creating a CoNLL dataset:

.. code-block:: python

    from ddaugner.datas.conll import CoNLLDataset

    train = CoNLLDataset.train_dataset({"PER": [MyAugmenter()]}, {"PER": [1.0]})
	   

Module reference
----------------

.. automodule:: ddaugner.datas.aug
   :members:


Dekker
======

Module reference
----------------

.. automodule:: ddaugner.datas.dekker
   :members:


CoNLL
=====

Module reference
----------------

.. automodule:: ddaugner.datas.conll.conll
   :members:
   :private-members:


NER utils
=========

Module reference
----------------

.. automodule:: ddaugner.ner_utils
   :members:


Utils		
=====

Module reference
----------------

.. automodule:: ddaugner.utils
   :members:
