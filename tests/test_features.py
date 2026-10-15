import numpy as np

from src.features import lexicon_features


def test_positive_scores_higher():
    assert lexicon_features("i love this, amazing and happy")[0] > 0


def test_negative_scores_lower():
    assert lexicon_features("i hate this terrible awful thing")[0] < 0


def test_negation_flips():
    assert lexicon_features("not good")[0] < lexicon_features("good")[0]


def test_shape():
    assert lexicon_features("hello world").shape == (13,)
    assert lexicon_features("").shape == (13,)
