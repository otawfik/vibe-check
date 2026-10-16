"""Scoring with TweetEval's official metric per task.

Sentiment is scored by macro-averaged recall and emotion by macro-F1, as in
the TweetEval paper, so the numbers are directly comparable to its table.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

OFFICIAL = {"sentiment": "macro_recall", "emotion": "macro_f1"}


def score(task, y_true, y_pred, labels):
    out = {
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    out = {k: round(float(v), 4) for k, v in out.items()}
    out["official"] = out[OFFICIAL[task]]
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out
