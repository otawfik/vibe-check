"""Classifiers: lexicon baseline + TF-IDF logistic regression.

The baseline is deliberately dumb (sum lexicon scores, threshold) so the
learned model has to earn its accuracy gain. Both share fit/predict.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import VIBE_LABELS
from .features import TfidfFeaturizer, lexicon_features, stack_features
from .lexicon import SENTIMENT
from .preprocess import tokenize


class LexiconBaseline:
    """Rule baseline: sentiment sum + punctuation energy -> vibe label."""

    def fit(self, texts, labels=None):
        return self

    def predict_one(self, text: str) -> str:
        tokens = tokenize(text)
        score = sum(SENTIMENT.get(t, 0) for t in tokens)
        feats = lexicon_features(text)
        excl, caps = feats[10], feats[11]
        if score >= 3 and (excl >= 1 or caps):
            return "hype"
        if score > 0:
            return "chill" if score <= 3 and excl == 0 else "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]


class VibeClassifier:
    """TF-IDF + lexicon features -> multinomial logistic regression."""

    def __init__(self):
        self.featurizer = TfidfFeaturizer()
        self.model = LogisticRegression(max_iter=1000, C=4.0)
        self.labels_ = list(VIBE_LABELS)

    def fit(self, texts, labels):
        texts = list(texts)
        X = stack_features(self.featurizer.fit_transform(texts), texts)
        self.model.fit(X, labels)
        self.labels_ = list(self.model.classes_)
        return self

    def _features(self, texts):
        texts = list(texts)
        return stack_features(self.featurizer.transform(texts), texts)

    def predict(self, texts):
        return list(self.model.predict(self._features(texts)))

    def predict_proba(self, texts):
        proba = self.model.predict_proba(self._features(texts))
        return [dict(zip(self.labels_, row)) for row in proba]
