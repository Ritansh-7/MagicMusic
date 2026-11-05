import re
from typing import Optional

try:
    import llm_classifier as _lc
except Exception:  # pragma: no cover
    _lc = None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _classify_dim_llm(text: str, dimension: str, allowed: tuple[str, ...]) -> tuple[str, float]:
    """Classify one dimension using the LLM only. No hardcoded keywords."""
    if _lc is None:
        return "unknown", 0.0
    try:
        result = _lc.classify_text(text)
        if not result:
            return "unknown", 0.0
        label = str(result.get(dimension, "unknown") or "unknown").strip().casefold()
        confidence = float(result.get("intent_confidence", 0.0) or 0.0)
        if label not in allowed:
            label = "unknown"
        return label, confidence
    except Exception:
        return "unknown", 0.0


def classify(text: str) -> tuple:
    """Classify text using the LLM only. No hardcoded keyword lists."""
    intent, iconf = _classify_dim_llm(text, "intent", _lc._INTENTS if _lc else ())
    mood, mconf = _classify_dim_llm(text, "mood", _lc._MOODS if _lc else ())
    genre, gconf = _classify_dim_llm(text, "genre", _lc._SUB_GENRES if _lc else ())
    theme, tconf = _classify_dim_llm(text, "theme", _lc._THEMES if _lc else ())
    religion, rconf = _classify_dim_llm(text, "religion", _lc._RELIGIONS if _lc else ())
    return (
        intent, iconf, {},
        mood, mconf, {},
        genre, gconf, {},
        theme, tconf, {},
        religion, rconf, {},
    )

def profile(item):
    try:
        import llm_classifier as _lc
        r = _lc.classify_item(item) if hasattr(_lc, "classify_item") else {}
        if r and r.get("primary_intent") not in (None, "unknown"):
            return r
    except Exception:
        pass
    return {"primary_intent": "unknown", "intent_confidence": 0.0, "mood": "unknown", "mood_confidence": 0.0, "genre": "unknown", "genre_confidence": 0.0, "theme": "unknown", "theme_confidence": 0.0, "religion": "unknown", "religion_confidence": 0.0, "sub_genre": "unknown", "intent_source": "none"}
    """Classify a track dict using the LLM as the primary source.

    Falls back to "unknown" with low confidence when the LLM is unavailable.
    No hardcoded keyword lists are used.
    """
    text = " ".join(str(item.get(f, "")) for f in ("title", "artists", "album", "track_genre", "language"))
    try:
        llm = _lc.classify_text(text) if _lc else {}
        if llm:
            return {
                "primary_intent": llm.get("intent", "unknown"),
                "intent_confidence": float(llm.get("intent_confidence", 0.0) or 0.0),
                "mood": llm.get("mood", "unknown"),
                "mood_confidence": float(llm.get("intent_confidence", 0.0) or 0.0),
                "genre": llm.get("sub_genre", "unknown"),
                "genre_confidence": float(llm.get("intent_confidence", 0.0) or 0.0),
                "theme": llm.get("theme", "unknown"),
                "theme_confidence": float(llm.get("intent_confidence", 0.0) or 0.0),
                "religion": llm.get("religion", "unknown"),
                "religion_confidence": float(llm.get("intent_confidence", 0.0) or 0.0),
                "sub_genre": llm.get("sub_genre", "unknown"),
                "intent_source": "llm",
            }
    except Exception:
        pass
    return {
        "primary_intent": "unknown",
        "intent_confidence": 0.0,
        "mood": "unknown",
        "mood_confidence": 0.0,
        "genre": "unknown",
        "genre_confidence": 0.0,
        "theme": "unknown",
        "theme_confidence": 0.0,
        "religion": "unknown",
        "religion_confidence": 0.0,
        "sub_genre": "unknown",
        "intent_source": "none",
    }
