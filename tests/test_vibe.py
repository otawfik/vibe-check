from src.vibe import VibeChecker


def test_check_positive():
    c = VibeChecker.trained_default()
    r = c.check("i love this, amazing day, so happy")
    assert r["vibe"] in ("positive", "hype", "chill")
    assert r["vibe_score"] > 0
    assert 0 <= r["confidence"] <= 1


def test_check_negative():
    c = VibeChecker.trained_default()
    r = c.check("i hate this terrible broken thing, so angry")
    assert r["vibe"] == "negative"
    assert r["vibe_score"] < 0


def test_check_empty():
    c = VibeChecker.trained_default()
    r = c.check("")
    assert r["vibe"] == "neutral"
    assert r["vibe_score"] == 0


def test_emotions_present():
    c = VibeChecker.trained_default()
    r = c.check("i am scared and worried about tomorrow")
    assert "fear" in r["emotions"]
    assert r["keywords"] is not None
