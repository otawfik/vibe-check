"""Flask demo UI for Vibe Check."""
from flask import Flask, jsonify, render_template, request

from src.vibe import VibeChecker

app = Flask(__name__)
checker = VibeChecker.trained_default()

EXAMPLES = [
    "lets gooooo we won, this is incredible, best night ever!!",
    "quiet morning, coffee and a good book, feeling calm and cozy",
    "i hate this broken app, worst update ever, so angry",
    "meeting moved to three tomorrow, please bring the report",
    "feeling sad today, i miss everyone so much",
]


@app.route("/")
def index():
    return render_template("index.html", examples=EXAMPLES)


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or request.form.get("text", "")
    if not str(text).strip():
        return jsonify({"error": "text is required"}), 400
    return jsonify(checker.check(str(text)))


@app.route("/check", methods=["POST"])
def check_form():
    text = request.form.get("text", "")
    result = checker.check(text) if text.strip() else None
    return render_template("index.html", examples=EXAMPLES,
                           result=result, text=text)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
