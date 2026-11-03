import re

INTENT_KEYWORDS = {
    "devotional": {"bhajan", "mata", "rani", "devi", "deva", "darshan", "bhakti", "jai", "om", "shiva", "ram", "krishna", "hanuman", "guru", "sai", "allah", "bulawa", "bulava"},
    "romantic": {"love", "ishq", "pyaar", "pyar", "dil", "jaan", "saajan", "sanam", "mohabbat", "romantic", "tujhe", "tu", "mere", "saiyaara", "saathiya"},
    "sad": {"sad", "dard", "udaas", "heartbreak", "alone", "broken", "rona", "tears", "judai", "viraha", "khamoshi"},
    "motivational": {"zinda", "jeet", "sultan", "brother", "run", "power", "action", "bhaag", "milkha", "singham", "fighter", "zindagi", "kadam"},
    "party": {"party", "dance", "disco", "bhangra", "dhamaal", "dhamaka", "chaiyya", "ramta", "kajra", "lat", "lag", "gaye"},
    "comedy": {"comedy", "funny", "hasna", "masti", "chatur", "naar", "phir", "hera", "pheri"},
    "patriotic": {"desh", "bharat", "india", "sande", "aate", "hain", "vande", "mataram", "soldier", "army"},
}

def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def classify(text):
    toks = _tokens(text)
    scores = {k: len(v & toks) for k, v in INTENT_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        best = "romantic"
    conf = min(1.0, scores[best] / 3.0) if scores[best] > 0 else 0.3
    return best, conf, scores

def profile(item):
    text = " ".join(str(item.get(f, "")) for f in ("title", "artists", "album", "track_genre", "language"))
    intent, conf, scores = classify(text)
    return {"primary_intent": intent, "confidence": conf, "scores": scores}
