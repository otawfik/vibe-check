"""Curated lexicons for sentiment, emotion, slang, and emoji.

Hand-built for short social text (posts, comments, messages). Scores are
small integers so the lexicon baseline stays interpretable: you can point
at exactly which word moved the needle.
"""

# word -> sentiment score in [-3, 3]
SENTIMENT = {
    # positive
    "love": 3, "loved": 3, "amazing": 3, "awesome": 3, "best": 3,
    "perfect": 3, "incredible": 3, "fantastic": 3, "brilliant": 3,
    "great": 2, "good": 2, "happy": 2, "excited": 2, "fun": 2,
    "cool": 2, "nice": 2, "win": 2, "winning": 2, "proud": 2,
    "glad": 2, "enjoy": 2, "enjoyed": 2, "beautiful": 2, "sweet": 2,
    "fire": 2, "lit": 2, "goat": 2, "clutch": 2, "hype": 1,
    "okay": 1, "ok": 1, "fine": 1, "thanks": 1, "thank": 1,
    "lol": 1, "haha": 1, "yes": 1,
    # negative
    "hate": -3, "hated": -3, "terrible": -3, "awful": -3, "worst": -3,
    "horrible": -3, "disgusting": -3, "furious": -3,
    "bad": -2, "sad": -2, "angry": -2, "annoying": -2, "boring": -2,
    "stupid": -2, "dumb": -2, "sucks": -2, "suck": -2, "fail": -2,
    "failed": -2, "lose": -2, "lost": -2, "pain": -2, "hurt": -2,
    "tired": -1, "meh": -1, "no": -1, "never": -1, "wrong": -1,
    "slow": -1, "broken": -2, "cry": -2, "crying": -2, "scared": -2,
    "afraid": -2, "worried": -2, "stress": -2, "stressed": -2,
    "sick": -1, "dead": -1, "mid": -2, "trash": -2, "flop": -2,
}

# emotion -> trigger words
EMOTIONS = {
    "joy": ["happy", "excited", "love", "amazing", "awesome", "fun",
            "great", "lol", "haha", "win", "winning", "celebrate",
            "party", "glad", "proud", "fire", "lit", "yes"],
    "anger": ["angry", "furious", "hate", "annoying", "stupid", "dumb",
              "ridiculous", "unbelievable", "mad", "rage", "trash"],
    "sadness": ["sad", "cry", "crying", "miss", "lonely", "hurt",
                "pain", "tired", "lost", "sorry", "heartbroken"],
    "fear": ["scared", "afraid", "worried", "nervous", "anxious",
             "stress", "stressed", "terrified", "panic"],
    "surprise": ["wow", "omg", "whoa", "unexpected", "shocked",
                 "surprise", "surprised", "incredible", "unbelievable"],
    "disgust": ["disgusting", "gross", "awful", "terrible", "nasty",
                "horrible", "sick"],
}

EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]

SLANG = {
    "fr": "for real", "ngl": "not gonna lie", "tbh": "to be honest",
    "imo": "in my opinion", "idk": "i do not know",
    "smh": "shaking my head", "lol": "laughing", "lmao": "laughing hard",
    "omg": "oh my god", "btw": "by the way", "fyi": "for your information",
    "goat": "greatest of all time", "mid": "mediocre", "cap": "lie",
    "nocap": "no lie", "lowkey": "somewhat", "highkey": "very",
    "vibe": "mood", "vibes": "mood", "yeet": "throw",
    "sus": "suspicious", "bet": "agreed", "fam": "friends",
    "bro": "friend", "dude": "friend",
}

EMOJI = {
    "😂": "laughing", "🤣": "laughing hard", "😍": "love",
    "🔥": "fire amazing", "💀": "dead laughing", "😢": "sad crying",
    "😭": "crying sad", "😡": "angry furious", "🤬": "furious angry",
    "😱": "scared shocked", "🤯": "shocked surprised", "🤮": "disgusting gross",
    "😴": "tired boring", "🥳": "celebrate party happy", "❤️": "love",
    "💔": "heartbroken sad", "👍": "good yes", "👎": "bad no",
    "🙏": "thank please", "✨": "beautiful amazing",
}

INTENSIFIERS = {"very": 1.5, "so": 1.4, "really": 1.5, "extremely": 2.0,
                "super": 1.6, "totally": 1.5, "absolutely": 1.8}

NEGATIONS = {"not", "no", "never", "dont", "can't", "wont", "isnt", "aint"}

# arousal cues: words that signal a calm or an activated state regardless of
# whether it's pleasant (the circumplex's vertical axis). Hand-picked, small,
# and applied as a capped nudge on top of the emotion model, not a classifier.
CALM = {"calm", "quiet", "slow", "cozy", "cosy", "relax", "relaxing", "relaxed", "peaceful", "chill",
        "chilling", "sleepy", "tea", "coffee", "book", "reading", "nap", "gentle", "soft", "lazy",
        "sunday", "breeze", "sunset", "unwind", "rest", "resting"}
ACTIVE = {"insane", "crazy", "unreal", "wild", "lets", "hype", "hyped", "screaming", "omg", "furious",
          "rage", "won", "win", "party", "pumped", "explode", "losing", "fuming", "again"}
