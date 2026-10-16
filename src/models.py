"""Models: a rule baseline and logistic regression over the feature blocks.

VARIANTS is the ablation ladder the benchmark climbs, one block at a time, so
each row of the results table shows what that block is worth on its own.
"""
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import features
from .features import lexicon_features

VARIANTS = {
    "tfidf": dict(social=False, chars=False, lexicon=False),
    "+social": dict(social=True, chars=False, lexicon=False),
    "+chars": dict(social=True, chars=True, lexicon=False),
    "+lexicon": dict(social=True, chars=True, lexicon=True),
}
VARIANT_NAMES = {
    "lexicon-rules": "Lexicon rules (no learning)",
    "tfidf": "TF-IDF words, default tokenizer",
    "+social": "+ social-text tokenizer",
    "+chars": "+ character 2-5 grams",
    "+lexicon": "+ lexicon features (full model)",
}
FULL = "+lexicon"


def build(variant=FULL, C=1.0):
    """Unfitted pipeline: features -> class-balanced multinomial logistic regression.

    Class weights are balanced because both official metrics are macro-averaged:
    every class counts equally, however rare.
    """
    return Pipeline([
        ("features", features.build(**VARIANTS[variant])),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


class LexiconBaseline:
    """No learning beyond the majority class: lexicon counts -> label by rule."""

    def __init__(self, task):
        self.task = task

    def fit(self, texts, labels):
        self.fallback_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict_one(self, text):
        f = lexicon_features(text)
        if self.task == "sentiment":
            return "positive" if f[0] > 0.5 else "negative" if f[0] < -0.5 else "neutral"
        joy, anger, sadness, fear, surprise, disgust = f[4:10]
        votes = {"anger": anger + disgust, "joy": joy + surprise, "sadness": sadness + fear,
                 "optimism": 1.0 if f[0] > 0 and joy == 0 else 0.0}
        best = max(votes, key=votes.get)
        return best if votes[best] > 0 else self.fallback_

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]
