from src.data import generate_dataset, train_test_split
from src.evaluate import accuracy
from src.models import LexiconBaseline, VibeClassifier


def test_baseline_obvious_cases():
    b = LexiconBaseline()
    assert b.predict_one("i love this amazing thing") in ("positive", "chill", "hype")
    assert b.predict_one("meeting at three tomorrow") == "neutral"
    assert b.predict_one("i hate this terrible thing") == "negative"


def test_classifier_beats_baseline():
    texts, labels = generate_dataset(n_per_class=80, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, seed=7)
    base_acc = accuracy(yte, LexiconBaseline().predict(Xte))
    clf = VibeClassifier().fit(Xtr, ytr)
    ml_acc = accuracy(yte, clf.predict(Xte))
    assert ml_acc >= 0.80
    assert ml_acc >= base_acc


def test_predict_proba_sums():
    texts, labels = generate_dataset(n_per_class=20, seed=7)
    clf = VibeClassifier().fit(texts, labels)
    proba = clf.predict_proba(["i love this"])[0]
    assert abs(sum(proba.values()) - 1.0) < 1e-6
