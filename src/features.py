"""Feature pipelines for short social text.

Three blocks, each switchable so the benchmark can ablate them:

  words    TF-IDF over unigrams + bigrams, via the social-text tokenizer
           (emoji -> words, slang expansion, elongation, negation marking)
  chars    TF-IDF over character 2-5 grams inside word boundaries, which
           catches misspellings, hashtags and elongations the word view misses
  lexicon  13 interpretable counts: lexicon sentiment, emotion triggers,
           exclamations, caps, length
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler

from .lexicon import EMOTIONS, INTENSIFIERS, NEGATIONS, SENTIMENT
from .preprocess import exclamation_count, has_caps_shout, tokenize

_WORD_TO_EMOTION = {w: e for e, ws in EMOTIONS.items() for w in ws}
LEXICON_FEATURES = ["sentiment_sum", "sentiment_mean", "pos_hits", "neg_hits", "joy", "anger",
                    "sadness", "fear", "surprise", "disgust", "exclamations", "caps", "n_tokens"]


def lexicon_features(text: str) -> np.ndarray:
    """The 13 values named in LEXICON_FEATURES, with negation and intensifiers applied."""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(len(LEXICON_FEATURES), dtype=float)
    score = 0.0
    pos = neg = 0
    negate = 0
    mult = 1.0
    hits = {e: 0 for e in EMOTIONS}
    for tok in tokens:
        if tok in NEGATIONS or tok.endswith("n't"):
            negate = 2
            continue
        if tok in INTENSIFIERS:
            mult = INTENSIFIERS[tok]
            continue
        if tok in SENTIMENT:
            s = SENTIMENT[tok] * mult * (-1 if negate else 1)
            score += s
            pos += s > 0
            neg += s < 0
        if tok in _WORD_TO_EMOTION:
            hits[_WORD_TO_EMOTION[tok]] += 1
        if negate:
            negate -= 1
        mult = 1.0
    n = len(tokens)
    return np.array([score, score / n, pos, neg, hits["joy"], hits["anger"], hits["sadness"],
                     hits["fear"], hits["surprise"], hits["disgust"], exclamation_count(text),
                     1.0 if has_caps_shout(text) else 0.0, n], dtype=float)


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping lexicon_features, max-abs scaled."""

    def fit(self, texts, y=None):
        self.scaler_ = MaxAbsScaler().fit(self._raw(texts))
        return self

    def transform(self, texts):
        return self.scaler_.transform(self._raw(texts))

    def get_feature_names_out(self, input_features=None):
        return np.array(LEXICON_FEATURES, dtype=object)

    @staticmethod
    def _raw(texts):
        return np.vstack([lexicon_features(t) for t in texts])


def word_vectorizer(social=True, max_features=60000):
    if social:
        return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                               ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)


def char_vectorizer(max_features=150000):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, lowercase=True,
                           max_features=max_features, sublinear_tf=True, dtype=np.float32)


def build(social=True, chars=True, lexicon=True):
    """FeatureUnion of the enabled blocks."""
    parts = [("words", word_vectorizer(social=social))]
    if chars:
        parts.append(("chars", char_vectorizer()))
    if lexicon:
        parts.append(("lexicon", LexiconFeatures()))
    return FeatureUnion(parts)
