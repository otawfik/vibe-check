"""Classifiers: lexicon baseline (v1). Learned model lands in next commit."""
from .features import lexicon_features
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
