"""Flask demo: paste a post, see where it lands on the circumplex.

The benchmark tables come from models/metrics.json (committed). /api/check
loads the two trained models on first use: from VIBE_ARTIFACT or artifacts/ if
present, otherwise from the GitHub release (checksum-verified, cached in /tmp).
"""
import hashlib
import json
import os
import threading
import urllib.request

from flask import Flask, jsonify, render_template, request

from src.vibe import AROUSAL, VibeChecker

ROOT = os.path.dirname(os.path.abspath(__file__))
RELEASE_URL = "https://github.com/otawfik/vibe-check/releases/download/v1.0/vibe.joblib"
RELEASE_SHA256 = "d524ac24742080e5837a99d41342ef72736277560619d08072504ab27145e519"
MAX_CHARS = 1000

app = Flask(__name__)

with open(os.path.join(ROOT, "models", "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

EXAMPLES = [
    "LETS GOOOO we actually won the whole thing!!! best night ever",
    "slow sunday, coffee and a good book, honestly perfect",
    "they cancelled my flight AGAIN and nobody at the desk will help. unreal",
    "i miss the old days. everyone moved away and it's just quiet now",
    "the meeting has been moved to 3pm thursday",
    "not gonna lie this update is mid",
]

_checker = None
_lock = threading.Lock()


def _artifact_path():
    for candidate in (os.environ.get("VIBE_ARTIFACT"), os.path.join(ROOT, "artifacts", "vibe.joblib")):
        if candidate and os.path.exists(candidate):
            return candidate
    cached = "/tmp/vibe-check-v1.joblib"
    if not os.path.exists(cached):
        tmp = cached + ".part"
        urllib.request.urlretrieve(RELEASE_URL, tmp)
        with open(tmp, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != RELEASE_SHA256:
            os.remove(tmp)
            raise RuntimeError("model artifact checksum mismatch")
        os.replace(tmp, cached)
    return cached


def checker():
    global _checker
    with _lock:
        if _checker is None:
            import joblib
            fitted = joblib.load(_artifact_path())
            _checker = VibeChecker(fitted["sentiment"], fitted["emotion"])
    return _checker


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS, examples=EXAMPLES, arousal=AROUSAL)


@app.route("/api/check", methods=["POST"])
def api_check():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()[:MAX_CHARS]
    if not text:
        return jsonify({"error": "text is required"}), 400
    return jsonify(checker().check(text))


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True)
