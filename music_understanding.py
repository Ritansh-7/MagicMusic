import re

INTENT_KEYWORDS = {
    "devotional": {"bhajan", "mata", "rani", "devi", "deva", "darshan", "bhakti", "jai", "om", "shiva", "ram", "krishna", "hanuman", "guru", "sai", "allah", "bulawa", "bulava", "aarti", "prabhu"},
    "romantic": {"love", "ishq", "pyaar", "pyar", "dil", "jaan", "saajan", "sanam", "mohabbat", "romantic", "tujhe", "tu", "mere", "saiyaara", "saathiya", "jaane", "jaanu"},
    "sad": {"sad", "dard", "udaas", "heartbreak", "alone", "broken", "rona", "tears", "judai", "viraha", "khamoshi", "tanha", "bewafa"},
    "motivational": {"zinda", "jeet", "sultan", "brother", "run", "power", "action", "bhaag", "milkha", "singham", "fighter", "zindagi", "kadam", "jeene", "jeena"},
    "party": {"party", "dance", "disco", "bhangra", "dhamaal", "dhamaka", "chaiyya", "ramta", "kajra", "lat", "lag", "gaye", "nach", "nachna"},
    "comedy": {"comedy", "funny", "hasna", "masti", "chatur", "naar", "phir", "hera", "pheri", "golmaal", "housefull"},
    "patriotic": {"desh", "bharat", "india", "sande", "aate", "hain", "vande", "mataram", "soldier", "army", "tiranga", "watan"},
}

MOOD_KEYWORDS = {
    "uplifting": {"jai", "bulawa", "zinda", "jeet", "hope", "asha", "ummeed"},
    "reverent": {"mata", "rani", "devi", "bhajan", "aarti", "prabhu", "om"},
    "tender": {"dil", "jaan", "saajan", "tujhe", "mere", "pyaar", "ishq"},
    "energetic": {"dance", "party", "disco", "bhangra", "dhamaal", "zinda", "run"},
    "melancholic": {"dard", "udaas", "judai", "viraha", "tanha", "bewafa", "rona"},
    "playful": {"chatur", "masti", "kajra", "lat", "gaye", "nach", "funny"},
}

GENRE_KEYWORDS = {
    "bhajan": {"bhajan", "mata", "rani", "devi", "aarti", "bhakti"},
    "qawwali": {"qawwali", "allah", "sufi", "ishq"},
    "ghazal": {"ghazal", "shayari"},
    "filmi": {"film", "soundtrack", "from"},
    "disco": {"disco", "party", "dance"},
    "folk": {"folk", "lok", "bhojpuri"},
}

THEME_KEYWORDS = {
    "god_goddess": {"mata", "rani", "devi", "deva", "shiva", "ram", "krishna", "hanuman", "allah", "prabhu", "guru", "sai"},
    "love": {"love", "ishq", "pyaar", "pyar", "dil", "jaan", "saajan", "sanam", "mohabbat", "tujhe"},
    "separation": {"judai", "viraha", "tanha", "bewafa", "khamoshi"},
    "nation": {"desh", "bharat", "india", "watan", "tiranga", "soldier"},
    "celebration": {"party", "dance", "nach", "shaadi", "wedding", "celebration"},
    "success": {"zinda", "jeet", "success", "fighter", "struggle"},
}

def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def _classify_dim(text, kw_map, default):
    toks = _tokens(text)
    scores = {k: len(v & toks) for k, v in kw_map.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return default, 0.3, scores
    conf = min(1.0, scores[best] / 2.0)
    return best, conf, scores

def classify(text):
    intent, iconf, iscores = _classify_dim(text, INTENT_KEYWORDS, "romantic")
    mood, mconf, mscores = _classify_dim(text, MOOD_KEYWORDS, "tender")
    genre, gconf, gscores = _classify_dim(text, GENRE_KEYWORDS, "filmi")
    theme, tconf, tscores = _classify_dim(text, THEME_KEYWORDS, "love")
    return intent, iconf, iscores, mood, mconf, mscores, genre, gconf, gscores, theme, tconf, tscores

def profile(item):
    text = " ".join(str(item.get(f, "")) for f in ("title", "artists", "album", "track_genre", "language"))
    intent, iconf, iscores, mood, mconf, mscores, genre, gconf, gscores, theme, tconf, tscores = classify(text)
    return {
        "primary_intent": intent,
        "intent_confidence": iconf,
        "mood": mood,
        "mood_confidence": mconf,
        "genre": genre,
        "genre_confidence": gconf,
        "theme": theme,
        "theme_confidence": tconf,
    }
