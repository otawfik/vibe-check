"""End-to-end VibeChecker pipeline: text in, vibe report out."""
import numpy as np

from .data import generate_dataset
from .features import lexicon_features
from .lexicon import EMOTIONS, SENTIMENT
from .models import LexiconBaseline, VibeClassifier
from .preprocess import tokenize

# vibe -> base sentiment polarity used for the -100..100 meter
_VIBE_POLARITY = {"positive": 62, "hype": 88, "chill": 34,
                  "neutral": 0, "negative": -72}


class VibeChecker:
    def __init__(self, classifier=None, baseline=None):
        self.classifier = classifier or VibeClassifier()
        self.baseline = baseline or LexiconBaseline()

    @classmethod
    def trained_default(cls):
        """Train on the synthetic corpus (fast, reproducible)."""
        texts, labels = generate_dataset()
        clf = VibeClassifier().fit(texts, labels)
        return cls(classifier=clf)

    def fit(self, texts, labels):
        self.classifier.fit(texts, labels)
        return self

    def check(self, text: str) -> dict:
        if not text or not str(text).strip():
            return {"text": text, "vibe": "neutral", "vibe_score": 0,
                    "confidence": 0.0, "emotions": {"neutral": 1.0},
                    "top_emotion": "neutral", "keywords": [],
                    "baseline_vibe": "neutral", "agree": True}
        proba = self.classifier.predict_proba([text])[0]
        vibe = max(proba, key=proba.get)
        confidence = float(proba[vibe])
        baseline_vibe = self.baseline.predict_one(text)

        # emotion distribution from lexicon hits, smoothed
        feats = lexicon_features(text)
        emo_raw = {"joy": feats[4], "anger": feats[5], "sadness": feats[6],
                   "fear": feats[7], "surprise": feats[8], "disgust": feats[9]}
        total = sum(emo_raw.values())
        if total == 0:
            emotions = {"neutral": 0.7, **{k: 0.05 for k in emo_raw}}
        else:
            emotions = {k: float(v / total * 0.9) for k, v in emo_raw.items()}
            emotions["neutral"] = 0.1
        top_emotion = max(emotions, key=emotions.get)

        # vibe meter: classifier polarity nudged by raw lexicon sum
        lex_sum = float(feats[0])
        score = int(np.clip(_VIBE_POLARITY[vibe] + lex_sum * 4, -100, 100))

        tokens = tokenize(text)
        keywords = sorted({t for t in tokens if t in SENTIMENT},
                          key=lambda t: -abs(SENTIMENT[t]))[:5]
        return {
            "text": text, "vibe": vibe, "vibe_score": score,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
            "emotions": {k: round(float(v), 4) for k, v in emotions.items()},
            "top_emotion": top_emotion,
            "keywords": keywords,
            "baseline_vibe": baseline_vibe,
            "agree": baseline_vibe == vibe,
        }
