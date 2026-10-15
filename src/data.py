"""Synthetic labeled dataset for vibe classification.

Real labeled social data needs licensing / privacy review, so the demo
ships a template-generated corpus: each vibe class has sentence frames
and slot fillers, sampled with a fixed seed so training is reproducible.
The point is demonstrating the full ML pipeline end to end, not claiming
state-of-the-art accuracy on wild data.
"""
import random

# vibe label -> (sentiment templates, emotion)
TEMPLATES = {
    "positive": [
        "i love this so much, best day ever",
        "this is amazing, i am so happy right now",
        "we won, lets go, incredible game",
        "feeling great today, everything is awesome",
        "lol this made my day, so fun",
        "proud of the team, fantastic work everyone",
        "this song is fire, on repeat all day",
        "best coffee ever, beautiful morning",
        "excited for tonight, gonna be lit",
        "thank you all, i feel so loved",
    ],
    "negative": [
        "i hate this, worst day ever",
        "this is terrible, i am so angry right now",
        "we lost again, horrible game, i am furious",
        "feeling sad today, everything sucks",
        "this is so boring and annoying, hate it",
        "i am tired of this broken trash app",
        "crying right now, this hurt so much",
        "awful food, disgusting, never again",
        "stressed and worried, this is a nightmare",
        "mid movie, boring, total flop",
    ],
    "neutral": [
        "meeting is at three tomorrow afternoon",
        "the train arrives in twenty minutes",
        "please send me the report when you can",
        "the store opens at nine in the morning",
        "i bought groceries and went home",
        "the weather report says rain on friday",
        "class is in room two hundred four",
        "the package will arrive next week",
        "we have a quiz on chapter five",
        "the library closes at ten tonight",
    ],
    "hype": [
        "lets gooooo this is insane, best night ever",
        "omg omg we did it, incredible, lets go",
        "this is fire fire fire, unreal energy tonight",
        "hype hype hype, the crowd is going crazy",
        "no way, shocked, this is absolutely incredible",
        "party time, celebrating all night, so excited",
        "the goat does it again, clutch win, amazing",
        "wow wow wow, unexpected, i am shaking",
        "lit lit lit, best party ever, love this",
        "yes yes yes, we won the whole thing",
    ],
    "chill": [
        "quiet morning, coffee and a good book, nice",
        "feeling calm today, slow walk by the river",
        "okay day, fine weather, just relaxing at home",
        "soft music, cozy room, feeling good and calm",
        "slow sunday, sweet and peaceful vibes",
        "enjoying the sunset, beautiful and quiet",
        "calm evening, tea and a movie, nice and cozy",
        "relaxed today, no stress, just fine",
        "gentle rain outside, cozy and happy inside",
        "peaceful walk in the park, feeling glad",
    ],
}

VIBE_LABELS = ["positive", "negative", "neutral", "hype", "chill"]

_FILLERS = ["", " honestly", " today", " right now", " for real", " tbh"]


def generate_dataset(n_per_class: int = 120, seed: int = 7):
    """Return (texts, vibe_labels) with light augmentation."""
    rng = random.Random(seed)
    texts, labels = [], []
    for label, frames in TEMPLATES.items():
        for _ in range(n_per_class):
            base = rng.choice(frames)
            text = base + rng.choice(_FILLERS)
            if rng.random() < 0.15:
                text = text + "!"
            if rng.random() < 0.1:
                text = text.upper() if label in ("hype", "negative") else text
            texts.append(text)
            labels.append(label)
    paired = list(zip(texts, labels))
    rng.shuffle(paired)
    return [p[0] for p in paired], [p[1] for p in paired]


def train_test_split(texts, labels, test_ratio=0.25, seed=7):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return ([texts[i] for i in tr], [texts[i] for i in te],
            [labels[i] for i in tr], [labels[i] for i in te])
