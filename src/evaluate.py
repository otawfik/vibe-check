"""Evaluation helpers: accuracy and per-class report without extra deps."""
from collections import defaultdict


def accuracy(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def per_class_report(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    report = {}
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        report[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                       "f1": round(f1, 4), "support": sum(1 for a in y_true if a == lab)}
    return report
