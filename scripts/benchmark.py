"""Benchmark lexicon baseline vs learned classifier on held-out split."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_dataset, train_test_split
from src.evaluate import accuracy, per_class_report
from src.models import LexiconBaseline, VibeClassifier


def main():
    texts, labels = generate_dataset(n_per_class=160, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, seed=7)
    base = LexiconBaseline()
    base_pred = base.predict(Xte)
    clf = VibeClassifier().fit(Xtr, ytr)
    ml_pred = clf.predict(Xte)
    out = {
        "n_train": len(Xtr), "n_test": len(Xte),
        "baseline_accuracy": round(accuracy(yte, base_pred), 4),
        "classifier_accuracy": round(accuracy(yte, ml_pred), 4),
        "classifier_per_class": per_class_report(yte, ml_pred),
        "dataset": "synthetic template corpus, seed=7",
    }
    path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
