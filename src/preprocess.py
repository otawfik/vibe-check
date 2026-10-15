"""Text preprocessing for short social posts.

Handles the stuff that breaks naive tokenizers on real posts: emoji,
slang, ALL CAPS shouting, repeated letters (soooo), hashtags, mentions,
and negation scope (``not good`` should not score as good).
"""
import re

from .lexicon import EMOJI, SLANG

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_TOKEN = re.compile(r"[a-z']+")


def expand_emoji(text: str) -> str:
    for emoji, words in EMOJI.items():
        text = text.replace(emoji, f" {words} ")
    return text


def normalize_repeats(word: str) -> str:
    """soooo -> soo, loooove -> loove (cap runs at 2)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def tokenize(text: str) -> list[str]:
    """Lowercase, expand emoji/slang, strip urls/mentions, keep hashtags."""
    if not text:
        return []
    text = expand_emoji(str(text))
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ")
    raw = _TOKEN.findall(text.lower().replace("’", "'"))
    tokens = []
    for tok in raw:
        tok = normalize_repeats(tok.strip("'"))
        if not tok:
            continue
        # slang expansion can be multi-word
        if tok in SLANG:
            tokens.extend(SLANG[tok].split())
        else:
            tokens.append(tok)
    return tokens


def has_caps_shout(text: str) -> bool:
    words = [w for w in str(text).split() if len(w) >= 3]
    if not words:
        return False
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words) >= 0.5


def exclamation_count(text: str) -> int:
    return str(text).count("!")
