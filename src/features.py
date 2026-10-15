"""Feature extraction: TF-IDF (scikit-learn) + hand-crafted lexicon features."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexicon import EMOTIONS, INTENSIFIERS, NEGATIONS, SENTIMENT
from .preprocess import exclamation_count, has_caps_shout, tokenize


class TfidfFeaturizer:
    """Wraps TfidfVectorizer over our custom tokenizer (slang/emoji aware)."""

    def __init__(self, max_features: int = 3000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None,
            lowercase=False, max_features=max_features,
            ngram_range=(1, 2), min_df=1,
        )

    def fit(self, texts):
        self.vectorizer.fit(list(texts))
        return self

    def transform(self, texts):
        return self.vectorizer.transform(list(texts))

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(list(texts))


def lexicon_features(text: str) -> np.ndarray:
    """[sentiment_sum, sentiment_mean, pos_hits, neg_hits, joy, anger,
    sadness, fear, surprise, disgust, excl, caps, n_tokens]"""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(13, dtype=float)
    score = 0.0
    pos = neg = 0
    negate = 0
    mult = 1.0
    emotion_hits = {e: 0 for e in EMOTIONS}
    word_to_emotion = {}
    for emo, words in EMOTIONS.items():
        for w in words:
            word_to_emotion[w] = emo
    for tok in tokens:
        if tok in NEGATIONS or tok.endswith("n't"):
            negate = 2
            continue
        if tok in INTENSIFIERS:
            mult = INTENSIFIERS[tok]
            continue
        if tok in SENTIMENT:
            s = SENTIMENT[tok] * mult
            if negate:
                s = -s
            score += s
            if s > 0:
                pos += 1
            else:
                neg += 1
        if tok in word_to_emotion:
            emotion_hits[word_to_emotion[tok]] += 1
        if negate:
            negate -= 1
        mult = 1.0
    n = len(tokens)
    return np.array([
        score, score / n, pos, neg,
        emotion_hits["joy"], emotion_hits["anger"], emotion_hits["sadness"],
        emotion_hits["fear"], emotion_hits["surprise"], emotion_hits["disgust"],
        exclamation_count(text), 1.0 if has_caps_shout(text) else 0.0, n,
    ], dtype=float)


def stack_features(tfidf_matrix, texts):
    lex = np.vstack([lexicon_features(t) for t in texts])
    from scipy import sparse
    return sparse.hstack([tfidf_matrix, sparse.csr_matrix(lex)]).tocsr()
