from src.preprocess import exclamation_count, has_caps_shout, normalize_repeats, tokenize


def test_tokenize_expands_emoji_and_slang():
    toks = tokenize("omg this is fire 🔥 fr")
    assert "fire" in toks
    assert "real" in toks  # fr -> for real


def test_tokenize_strips_urls_and_mentions():
    toks = tokenize("check https://x.com @friend great news")
    assert "great" in toks
    assert not any("http" in t for t in toks)


def test_normalize_repeats():
    assert normalize_repeats("soooo") == "soo"
    assert normalize_repeats("gooooo") == "goo"


def test_caps_and_exclamation():
    assert has_caps_shout("THIS IS INSANE WOW")
    assert not has_caps_shout("this is a calm sentence")
    assert exclamation_count("wow!! yes!") == 3


def test_empty():
    assert tokenize("") == []
    assert tokenize(None) == []
