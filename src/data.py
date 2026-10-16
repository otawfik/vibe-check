"""Data: TweetEval for the real models, a tiny synthetic corpus for tests.

TweetEval (Barbieri et al., 2020) is the standard Twitter benchmark. Two of its
tasks are used here, each with official train / validation / test splits:

  sentiment  SemEval-2017 Task 4A   negative, neutral, positive    45,615 / 2,000 / 12,284
  emotion    SemEval-2018 Task 1    anger, joy, optimism, sadness   3,257 /   374 /  1,421

The tweets aren't redistributed here; load_tweeteval downloads them from the
TweetEval repository on first use and caches them.
"""
import os
import random
import urllib.request

TWEETEVAL_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/{name}"
LABELS = {
    "sentiment": ["negative", "neutral", "positive"],
    "emotion": ["anger", "joy", "optimism", "sadness"],
}
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "vibe-check")


def _fetch(task, name, cache_dir):
    path = os.path.join(cache_dir, f"{task}_{name}")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(TWEETEVAL_URL.format(task=task, name=name), path + ".part")
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_tweeteval(task, split, cache_dir=DEFAULT_CACHE):
    """(texts, labels) for one TweetEval task and split (train, val or test)."""
    if task not in LABELS:
        raise KeyError(f"unknown task: {task!r}")
    texts = _fetch(task, f"{split}_text.txt", cache_dir)
    ids = _fetch(task, f"{split}_labels.txt", cache_dir)
    labels = [LABELS[task][int(i)] for i in ids if i.strip()]
    if len(texts) != len(labels):
        raise ValueError(f"{task}/{split}: {len(texts)} texts but {len(labels)} labels")
    return texts, labels


# ---- tiny offline corpus for the test suite --------------------------------

_TOY = {
    "sentiment": {
        "positive": ["love this so much", "best day ever", "this is amazing", "so happy right now",
                     "great game tonight", "proud of the team", "beautiful morning"],
        "negative": ["hate this", "worst day ever", "this is terrible", "so angry right now",
                     "awful service", "broken again", "sad and tired"],
        "neutral": ["meeting at three", "train arrives at noon", "store opens at nine",
                    "report is due friday", "class in room four", "package ships monday", "bus at eight"],
    },
    "emotion": {
        "anger": ["so angry about this", "furious at the refs", "this is ridiculous", "i am mad"],
        "joy": ["so happy we won", "laughing all day", "best party ever", "this made my day"],
        "optimism": ["tomorrow will be better", "we will get there", "hopeful about next year", "keep going"],
        "sadness": ["i miss you so much", "crying all night", "feeling lonely", "heartbroken today"],
    },
}
_FILL = ["", " honestly", " today", " right now", " tbh", " fr"]


def toy(task, n_per_class=40, seed=7):
    """Small synthetic (texts, labels) for fast offline tests."""
    rng = random.Random(seed)
    out = []
    for label, frames in _TOY[task].items():
        for _ in range(n_per_class):
            text = rng.choice(frames) + rng.choice(_FILL)
            if rng.random() < 0.2:
                text += "!!"
            out.append((text, label))
    rng.shuffle(out)
    return [t for t, _ in out], [y for _, y in out]
