"""VibeChecker: two trained models placed on Russell's circumplex of affect.

Psychology's circumplex model (Russell, 1980) describes a feeling by two axes:
valence (unpleasant to pleasant) and arousal (calm to activated). Vibe Check
estimates both and names the quadrant:

  valence  P(positive) - P(negative) from the sentiment model, in [-1, 1]
  energy   the emotion model's probabilities weighted by where each emotion
           sits on the arousal axis, nudged by typing cues (!!, CAPS,
           elongated words) and a small list of calm / activated words, in [0, 1]

  pleasant + high energy = hype      unpleasant + high energy = heated
  pleasant + low energy  = chill     unpleasant + low energy  = down

The arousal placements and cue weights are hand-set, not learned; they're
constants here so you can see them. The word cues exist because TweetEval's
four emotions have no calm-pleasant class: without them, contentment reads
as high-energy joy.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .lexicon import ACTIVE, CALM
from .preprocess import exclamation_count, has_caps_shout, tokenize

AROUSAL = {"anger": 0.85, "joy": 0.70, "optimism": 0.35, "sadness": 0.20}
ENERGY_SPLIT = 0.5
NEUTRAL_BAND = 0.35
_ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")
_SHOUTED_WORD = re.compile(r"\b[A-Z]{3,}\b")


def surface_energy(text):
    """Energy nudge from how something is typed and which arousal words it uses.

    Returns (bump, cues). Typing cues: exclamation marks, a shouted post or a
    single shouted word, elongation. Word cues: CALM words pull energy down,
    ACTIVE words push it up, each capped so they nudge rather than decide.
    """
    cues = []
    bump = 0.0
    n_excl = exclamation_count(text)
    if n_excl:
        bump += 0.06 * min(n_excl, 3)
        cues.append("!" * min(n_excl, 3))
    if has_caps_shout(text):
        bump += 0.12
        cues.append("CAPS")
    elif _SHOUTED_WORD.search(text):
        bump += 0.08
        cues.append("CAPS word")
    if _ELONGATED.search(text):
        bump += 0.06
        cues.append("elongation")
    tokens = set(tokenize(text))
    calm, active = sorted(tokens & CALM), sorted(tokens & ACTIVE)
    if calm:
        bump -= min(0.08 * len(calm), 0.24)
        cues.append("calm: " + ", ".join(calm))
    if active:
        bump += min(0.06 * len(active), 0.18)
        cues.append("active: " + ", ".join(active))
    return bump, cues


def quadrant(valence, energy, sentiment_label):
    if sentiment_label == "neutral" and abs(valence) < NEUTRAL_BAND:
        return "neutral"
    if valence >= 0:
        return "hype" if energy >= ENERGY_SPLIT else "chill"
    return "heated" if energy >= ENERGY_SPLIT else "down"


def drivers(pipeline, text, positive="positive", negative="negative", k=5):
    """Words and bigrams pushing valence up or down, from the sentiment model's weights.

    Contribution = TF-IDF value x (weight for positive - weight for negative),
    over the word block only; character n-grams aren't readable as reasons.
    """
    union = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    words = dict(union.transformer_list)["words"]
    x = words.transform([text]).tocsr()
    classes = list(clf.classes_)
    w = clf.coef_[classes.index(positive)] - clf.coef_[classes.index(negative)]
    names = words.get_feature_names_out()
    contrib = [(names[j], float(v * w[j])) for j, v in zip(x.indices, x.data)
               if not set(names[j].split()) <= ENGLISH_STOP_WORDS]
    contrib.sort(key=lambda t: t[1])
    up = [(t, round(c, 3)) for t, c in reversed(contrib) if c > 0][:k]
    down = [(t, round(c, 3)) for t, c in contrib if c < 0][:k]
    return up, down


class VibeChecker:
    def __init__(self, sentiment, emotion):
        self.sentiment = sentiment
        self.emotion = emotion

    def check(self, text):
        text = str(text or "").strip()
        if not text:
            raise ValueError("text is required")
        s_proba = dict(zip(self.sentiment.classes_, self.sentiment.predict_proba([text])[0]))
        e_proba = dict(zip(self.emotion.classes_, self.emotion.predict_proba([text])[0]))
        sentiment_label = max(s_proba, key=s_proba.get)

        valence = float(s_proba["positive"] - s_proba["negative"])
        bump, cues = surface_energy(text)
        energy = float(np.clip(sum(AROUSAL[e] * p for e, p in e_proba.items()) + bump, 0, 1))
        up, down = drivers(self.sentiment, text)

        return {
            "text": text,
            "vibe": quadrant(valence, energy, sentiment_label),
            "valence": round(valence, 3),
            "energy": round(energy, 3),
            "meter": int(round(valence * 100)),
            "sentiment": {k: round(float(v), 3) for k, v in s_proba.items()},
            "emotion": {k: round(float(v), 3) for k, v in e_proba.items()},
            "top_emotion": max(e_proba, key=e_proba.get),
            "cues": cues,
            "drivers_up": up,
            "drivers_down": down,
        }
