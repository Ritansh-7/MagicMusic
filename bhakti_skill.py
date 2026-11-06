"""
Bhakti Recommendation Skill - Skill file for devotional/bhakti song recommendations.

Defines bhakti domain knowledge: keywords, artists, deities, mood profiles,
and math-based mood scoring. Used by bhakti_agent.py to provide fast,
accurate devotional recommendations without LLM latency for mood scoring.
"""

from __future__ import annotations

import re
from typing import Any

BHAKTI_SKILL = {
    "name": "bhakti_devotional",
    "version": "1.0.0",
    "description": "Devotional / bhakti music recommendation skill",
    "bhakti_tokens": {
        "bhajan", "aarti", "chalisa", "kirtan", "stuti", "mantra", "dhun",
        "stotram", "stotra", "namah", "jai", "om", "hare", "ram", "krishna",
        "shiva", "shree", "hanuman", "durga", "ganesha", "lakshmi",
        "saraswati", "vishnu", "narayan", "mahadev", "shankar", "bholenath",
        "sai", "bhakti", "bhakt", "prarthana", "vandana", "aaradhana",
        "namaskar", "namo", "ashtak", "sahastranam", "kavach",
        "pooja", "archana", "deepa", "deep", "diya", "tilak",
        "sindoor", "mangal", "prasad", "bhog", "bhandara", "langar",
        "gurdwara", "mandir", "temple", "ashram", "math", "peeth", "peetham",
        "satsang", "bhajan samrat", "gayatri", "japa", "yagna", "havan",
        "deepak", "ganga", "yamuna", "kailash", "trishul", "damru", "rudra",
        "shivay", "bhole", "gopal", "govind", "hare rama", "hare krishna",
        "shri ram", "sita ram", "radhe", "radha",
    },
    "bhakti_artists": {
        "anup jalota", "narendra chanchal", "lakha", "lakkha", "bhajan samrat",
        "mani har prasad", "pankaj udhas", "jagjit singh", "ghulam ali",
        "mehdi hassan", "lata mangeshkar", "kishore kumar", "mohammed rafi",
        "mukesh", "asha bhosle", "hemant kumar", "manna dey", "talat mahmood",
        "geeta dutt", "bhupinder singh", "chitra singh", "hariharan",
        "sadhna sargam", "manhar udhas", "sonu nigam", "kavita krishnamurthy",
        "nusrat fateh ali khan", "rahat fateh ali khan", "abida parveen",
        "maithili thakur", "manoj sharma", "asif jeriya", "ami joshi",
    },
    "deities": {
        "shiva", "shankar", "mahadev", "bholenath", "rudra", "kailash",
        "trishul", "damru", "ganga", "yamuna", "shivay", "bhole",
        "vishnu", "narayan", "hari", "krishna", "ram", "shree",
        "durga", "lakshmi", "saraswati", "parvati", "gouri", "shakti",
        "bhagwati", "jagdamba", "bhairava", "ganesha", "ganpati",
        "hanuman", "anjaney", "bajrang", "bajrangbali", "sai", "bhakt",
    },
    "mood_profiles": {
        "reverent": {"energy": 0.35, "valence": 0.60, "tempo": 80, "danceability": 0.40, "acousticness": 0.70},
        "uplifting": {"energy": 0.65, "valence": 0.80, "tempo": 100, "danceability": 0.55, "acousticness": 0.50},
        "tender": {"energy": 0.30, "valence": 0.55, "tempo": 70, "danceability": 0.35, "acousticness": 0.75},
        "melancholic": {"energy": 0.25, "valence": 0.30, "tempo": 65, "danceability": 0.25, "acousticness": 0.80},
        "energetic": {"energy": 0.80, "valence": 0.70, "tempo": 120, "danceability": 0.65, "acousticness": 0.40},
        "playful": {"energy": 0.60, "valence": 0.75, "tempo": 110, "danceability": 0.60, "acousticness": 0.55},
        "calm": {"energy": 0.25, "valence": 0.65, "tempo": 60, "danceability": 0.30, "acousticness": 0.85},
        "romantic": {"energy": 0.45, "valence": 0.65, "tempo": 85, "danceability": 0.45, "acousticness": 0.60},
        "sad": {"energy": 0.30, "valence": 0.25, "tempo": 70, "danceability": 0.30, "acousticness": 0.75},
        "unknown": {"energy": 0.50, "valence": 0.50, "tempo": 90, "danceability": 0.50, "acousticness": 0.50},
    },
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


# Genre/form words that describe the devotional FORM of the music. A
# Bollywood film song can mention a deity name ("bhagwan", "shiva", "ram")
# without being devotional music, so a single generic deity token must never
# by itself classify a seed as bhakti. These words, by contrast, describe
# the musical form (bhajan, chalisa, aarti, stotram, mantra, dhun, ...).
BHAKTI_GENRE_TOKENS: set[str] = {
    "bhajan", "aarti", "chalisa", "kirtan", "stuti", "mantra", "dhun",
    "stotram", "stotra", "namaskar", "namo", "ashtak", "sahstranam",
    "sahastranam", "kavach", "pooja", "archana", "deepa", "deep", "diya",
    "tilak", "sindoor", "mangal", "prasad", "bhog", "bhandara", "langar",
    "gurdwara", "mandir", "temple", "ashram", "math", "peeth", "peetham",
    "satsang", "gayatri", "japa", "yagna", "havan", "deepak", "ganga",
    "yamuna", "kailash", "trishul", "damru", "rudra", "shivay", "bhole",
    "gopal", "govind", "hare", "krishna", "ram", "shree", "sita", "radhe",
    "radha",
}


def is_strong_bhakti_seed(item: dict[str, Any]) -> bool:
    """Decide whether a SEED track is genuinely devotional music.

    ``is_bhakti_skill_match`` is intentionally lenient: it is used to surface
    devotional *candidates* from a broad pool, where missing a real bhajan
    is worse than a false positive.  A seed is different — classifying a
    secular film song as a bhakti seed wipes out the whole recommendation
    queue, because the bhakti gate then demands devotional-only candidates.

    A seed is only considered bhakti when it shows a strong form signal:
      1. a genre/form token in the title (bhajan, chalisa, aarti, ...), or
      2. two or more bhakti tokens in the title, or
      3. a genre token in the album plus at least one bhakti token in the
         title (covers albums like "Shiva Stotram" with varied track titles).

    A single generic deity token ("bhagwan", "shiva", "ram") in a film song
    title is explicitly NOT enough.
    """
    title_key = _compact_key(item.get("title", ""))
    album_key = _compact_key(item.get("album", ""))
    title_tokens = _tokens(item.get("title", ""))
    album_tokens = _tokens(item.get("album", ""))

    title_genre = title_tokens & BHAKTI_GENRE_TOKENS
    album_genre = album_tokens & BHAKTI_GENRE_TOKENS
    title_bhakti = title_tokens & BHAKTI_SKILL["bhakti_tokens"]

    if title_genre:
        return True
    if len(title_bhakti) >= 2:
        return True
    if album_genre and title_bhakti:
        return True
    return False


def is_bhakti_skill_match(item: dict[str, Any]) -> bool:
    text_parts = [
        str(item.get("title", "")),
        str(item.get("artists", "")),
        str(item.get("album", "")),
        str(item.get("language", "")),
        " ".join(str(item.get("intent_labels", []))),
        str(item.get("llm_intent", "")),
        str(item.get("llm_religion", "")),
        str(item.get("mood_label", "")),
    ]
    text = " ".join(text_parts).casefold()
    text_tokens = _tokens(text)
    title_key = _compact_key(item.get("title", ""))
    album_key = _compact_key(item.get("album", ""))
    bhakti_tokens = BHAKTI_SKILL["bhakti_tokens"]

    # 1. Title/album token match (strong signal - includes deity names)
    title_tokens = _tokens(item.get("title", ""))
    album_tokens = _tokens(item.get("album", ""))
    if bhakti_tokens & title_tokens:
        return True
    if bhakti_tokens & album_tokens:
        return True

    # 2. Multi-word bhakti phrases in title/album
    for tok in bhakti_tokens:
        if " " in tok:
            if tok in title_key or tok in album_key:
                return True

    # 3. Intent labels / LLM intent (strong signal)
    intent_text = " ".join(str(item.get("intent_labels", []))).casefold()
    intent_text += " " + str(item.get("llm_intent", "")).casefold()
    intent_tokens = _tokens(intent_text)
    if bhakti_tokens & intent_tokens:
        return True
    if "devotional" in intent_tokens:
        return True

    # 4. Artist match (supporting - requires title/album/intent confirmation)
    artists = str(item.get("artists", "")).casefold()
    bhakti_artists = BHAKTI_SKILL["bhakti_artists"]
    artist_match = any(artist in artists for artist in bhakti_artists)
    if artist_match:
        if bhakti_tokens & title_tokens:
            return True
        if bhakti_tokens & album_tokens:
            return True
        if bhakti_tokens & intent_tokens:
            return True

    # 5. LLM religion (supporting - requires title/album/intent confirmation)
    religion = str(item.get("llm_religion", "")).casefold()
    if religion in {"hindu", "sikh", "jain", "buddhist"}:
        if bhakti_tokens & title_tokens:
            return True
        if bhakti_tokens & album_tokens:
            return True
        if bhakti_tokens & intent_tokens:
            return True

    return False


def resolve_deity(item: dict[str, Any]) -> str | None:
    text_parts = [
        str(item.get("title", "")),
        str(item.get("artists", "")),
        str(item.get("album", "")),
        str(item.get("llm_religion", "")),
    ]
    text = " ".join(text_parts).casefold()
    text_tokens = _tokens(text)
    deities = BHAKTI_SKILL["deities"]
    for deity in sorted(deities, key=len, reverse=True):
        if deity in text_tokens or deity in text:
            return deity
    return None


def infer_bhakti_mood(item: dict[str, Any]) -> tuple[str, float]:
    mood_profiles = BHAKTI_SKILL["mood_profiles"]
    features = {}
    for feat in ("energy", "valence", "tempo", "danceability", "acousticness"):
        val = item.get(feat)
        if val is None:
            val = item.get(f"audio_{feat}")
        if val is not None:
            try:
                features[feat] = float(val)
            except (TypeError, ValueError):
                pass
    if features:
        best_mood = "unknown"
        best_score = -1.0
        for mood, profile in mood_profiles.items():
            score = _cosine_similarity(features, profile)
            if score > best_score:
                best_score = score
                best_mood = mood
        confidence = min(1.0, max(0.0, best_score))
        return best_mood, confidence
    text = " ".join(
        str(item.get(field, "")) for field in ("title", "mood_label", "intent_labels")
    ).casefold()
    text_tokens = _tokens(text)
    mood_keywords = {
        "reverent": {"reverent", "devotion", "worship", "prayer", "sacred", "holy"},
        "uplifting": {"uplifting", "joy", "celebration", "bliss", "ananda"},
        "tender": {"tender", "soft", "gentle", "loving", "sweet"},
        "melancholic": {"melancholic", "longing", "separation", "virah"},
        "energetic": {"energetic", "powerful", "loud", "bhajan samrat"},
        "playful": {"playful", "leela", "raslila", "mischief"},
        "calm": {"calm", "peace", "shanti", "meditation", "dhyana"},
        "sad": {"sad", "sorrow", "pain", "tears"},
    }
    for mood, keywords in mood_keywords.items():
        if keywords & text_tokens:
            return mood, 0.55
    return "unknown", 0.30


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in common_keys)
    norm_a = sum(a[k] ** 2 for k in common_keys) ** 0.5
    norm_b = sum(b[k] ** 2 for k in common_keys) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def bhakti_score(item: dict[str, Any], seed: dict[str, Any]) -> float:
    score = 0.0
    if is_bhakti_skill_match(item):
        score += 0.40
    seed_deity = resolve_deity(seed)
    item_deity = resolve_deity(item)
    if seed_deity and item_deity and seed_deity == item_deity:
        score += 0.30
    elif seed_deity and not item_deity:
        score += 0.10
    seed_mood, seed_conf = infer_bhakti_mood(seed)
    item_mood, item_conf = infer_bhakti_mood(item)
    if seed_mood != "unknown" and item_mood != "unknown":
        if seed_mood == item_mood:
            score += 0.30 * min(seed_conf, item_conf)
        else:
            compatible_moods = {
                "reverent": {"calm", "tender", "uplifting"},
                "calm": {"reverent", "tender", "meditative"},
                "uplifting": {"reverent", "playful", "energetic"},
                "tender": {"reverent", "calm", "romantic"},
            }
            if item_mood in compatible_moods.get(seed_mood, set()):
                score += 0.15
    return min(1.0, score)