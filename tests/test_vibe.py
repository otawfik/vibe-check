import pytest

from src import vibe


def test_quadrants():
    assert vibe.quadrant(0.8, 0.9, "positive") == "hype"
    assert vibe.quadrant(0.8, 0.2, "positive") == "chill"
    assert vibe.quadrant(-0.8, 0.9, "negative") == "heated"
    assert vibe.quadrant(-0.8, 0.2, "negative") == "down"
    assert vibe.quadrant(0.1, 0.9, "neutral") == "neutral"
    assert vibe.quadrant(0.6, 0.9, "neutral") == "hype"


def test_surface_energy_typing_cues():
    bump, cues = vibe.surface_energy("LETS GOOOO WE WON!!!")
    assert bump > 0.3 and {"!!!", "CAPS", "elongation"} <= set(cues)
    assert vibe.surface_energy("see you at noon") == (0.0, [])


def test_single_shouted_word_counts():
    bump, cues = vibe.surface_energy("they cancelled it AGAIN")
    assert "CAPS word" in cues and bump > 0


def test_calm_words_pull_energy_down_capped():
    bump, cues = vibe.surface_energy("slow quiet cozy calm sunday with tea")
    assert bump == -0.24
    assert cues[0].startswith("calm: ")


def test_check_report(fitted):
    c = vibe.VibeChecker(fitted["sentiment"], fitted["emotion"])
    r = c.check("best day ever, so happy right now!!")
    assert r["vibe"] in {"hype", "chill"}
    assert -1 <= r["valence"] <= 1 and 0 <= r["energy"] <= 1
    assert abs(sum(r["sentiment"].values()) - 1) < 0.01
    assert abs(sum(r["emotion"].values()) - 1) < 0.01
    assert r["meter"] == round(r["valence"] * 100)


def test_negative_post_goes_left(fitted):
    c = vibe.VibeChecker(fitted["sentiment"], fitted["emotion"])
    assert c.check("worst day ever, this is terrible")["valence"] < 0


def test_drivers_point_the_right_way(fitted):
    up, down = vibe.drivers(fitted["sentiment"], "love this but awful service")
    assert any("love" in t for t, _ in up)
    assert any("awful" in t for t, _ in down)


def test_empty_text_rejected(fitted):
    with pytest.raises(ValueError):
        vibe.VibeChecker(fitted["sentiment"], fitted["emotion"]).check("   ")
