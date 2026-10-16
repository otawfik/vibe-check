import pytest
from sklearn.metrics import accuracy_score

from src import data, evaluate, features, models


@pytest.mark.parametrize("variant", list(models.VARIANTS))
def test_every_variant_trains(variant):
    x, y = data.toy("sentiment", n_per_class=15)
    m = models.build(variant).fit(x, y)
    assert set(m.predict(x)) <= set(data.LABELS["sentiment"])


def test_full_model_beats_rules():
    x, y = data.toy("emotion", seed=1)
    xt, yt = data.toy("emotion", seed=2)
    learned = accuracy_score(yt, models.build(models.FULL, C=4).fit(x, y).predict(xt))
    rules = accuracy_score(yt, models.LexiconBaseline("emotion").fit(x, y).predict(xt))
    assert learned > rules


def test_sentiment_rules_obvious_cases():
    b = models.LexiconBaseline("sentiment").fit(["x"], ["neutral"])
    assert b.predict(["i love this, amazing", "i hate this, awful", "bus at nine"]) == ["positive", "negative", "neutral"]


def test_emotion_rules_fall_back_to_majority():
    b = models.LexiconBaseline("emotion").fit(["a", "b", "c"], ["joy", "joy", "anger"])
    assert b.predict_one("the bus is at nine") == "joy"


def test_feature_blocks_switch():
    assert [n for n, _ in features.build(chars=False, lexicon=False).transformer_list] == ["words"]
    assert [n for n, _ in features.build().transformer_list] == ["words", "chars", "lexicon"]


def test_official_metric_per_task():
    y = ["a", "a", "b", "b"]
    p = ["a", "a", "a", "b"]
    s = evaluate.score("sentiment", y, p, ["a", "b"])
    assert s["official"] == s["macro_recall"] == 0.75
    e = evaluate.score("emotion", y, p, ["a", "b"])
    assert e["official"] == e["macro_f1"]
    assert s["confusion"] == [[2, 0], [1, 1]]
