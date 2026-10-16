"""Benchmark on TweetEval: an ablation per task, tuned on validation, scored once on test.

    python scripts/benchmark.py     # ~3-5 minutes; downloads TweetEval on first run

For every variant, C is picked on the official validation split by the task's
official metric; the test split is touched once, at the end. Writes
models/metrics.json and artifacts/vibe.joblib (the two full models).
"""
import json
import os
import sys
import time

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, evaluate, models  # noqa: E402

C_GRID = [0.25, 0.5, 1, 2, 4, 8]

# TweetEval paper / repo leaderboard (Barbieri et al., 2020), same metric and test set
PUBLISHED = {
    "sentiment": {"SVM": 62.9, "FastText": 62.9, "LSTM": 58.3, "RoBERTa-base": 71.3, "TimeLMs-2021": 73.7},
    "emotion": {"SVM": 64.7, "FastText": 65.2, "LSTM": 66.0, "RoBERTa-base": 76.1, "TimeLMs-2021": 80.2},
}


def run_task(task):
    labels = data.LABELS[task]
    x_tr, y_tr = data.load_tweeteval(task, "train")
    x_va, y_va = data.load_tweeteval(task, "val")
    x_te, y_te = data.load_tweeteval(task, "test")
    print(f"\n{task}: {len(x_tr)} train / {len(x_va)} val / {len(x_te)} test")

    rows = []
    base = models.LexiconBaseline(task).fit(x_tr, y_tr)
    rows.append({"variant": "lexicon-rules", "C": None,
                 **evaluate.score(task, y_te, base.predict(x_te), labels)})
    best_full = None
    for variant in models.VARIANTS:
        t0 = time.perf_counter()
        tuned = []
        for C in C_GRID:
            m = models.build(variant, C).fit(x_tr, y_tr)
            tuned.append((evaluate.score(task, y_va, m.predict(x_va), labels)["official"], C, m))
        val, C, m = max(tuned, key=lambda t: t[0])
        res = evaluate.score(task, y_te, m.predict(x_te), labels)
        rows.append({"variant": variant, "C": C, "val_official": val, **res})
        print(f"  {variant:<10} C={C:<5} val {val:.3f}  test {res['official']:.3f}  ({time.perf_counter() - t0:.0f}s)")
        if variant == models.FULL:
            best_full = m
    for r in rows:
        r["name"] = models.VARIANT_NAMES[r["variant"]]
    return {"labels": labels, "metric": evaluate.OFFICIAL[task], "n_train": len(x_tr), "n_val": len(x_va),
            "n_test": len(x_te), "rows": rows, "published": PUBLISHED[task]}, best_full


def main():
    out, fitted = {}, {}
    for task in ("sentiment", "emotion"):
        out[task], fitted[task] = run_task(task)
    out["dataset"] = "TweetEval (Barbieri et al., 2020), official train/val/test splits"
    with open(os.path.join(ROOT, "models", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    art = os.path.join(ROOT, "artifacts", "vibe.joblib")
    joblib.dump(fitted, art, compress=3)
    print(f"\nsaved models/metrics.json; artifact {os.path.getsize(art) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
