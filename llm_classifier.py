"""Intent, mood, sub-genre and religion classification for tracks.

Uses an OpenAI-compatible chat-completions endpoint (Groq, OpenAI, Ollama,
Together, local vLLM, ...) instead of transformers/torch, so no multi-gigabyte
model download is required. Results are cached in an in-memory dict and a
SQLite file so each recording is classified once per deployment.

Configuration (environment variables or Streamlit secrets):
    LLM_API_KEY           API key (also tries OPENAI_API_KEY / GROQ_API_KEY)
    LLM_BASE_URL          chat-completions URL
                          (default https://api.openai.com/v1/chat/completions)
    LLM_MODEL             model name (default gpt-4o-mini)
    LLM_REQUEST_TIMEOUT   seconds per request (default 20)
    LLM_INTENT_CACHE_DB   SQLite cache path (default intent_cache.sqlite3)

Set LLM_API_KEY to "" to disable LLM classification; the keyword fallbacks in
music_understanding.py still apply, so the app keeps working without a key.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 20.0
DEFAULT_CACHE_DB = "intent_cache.sqlite3"

_INTENTS = (
    "devotional",
    "romantic",
    "sad",
    "party",
    "patriotic",
    "ghazal",
    "motivational",
    "energetic",
    "acoustic_indie",
    "calm",
    "life",
    "film",
    "unknown",
)
_SUB_GENRES = (
    "bhajan",
    "aarti",
    "chalisa",
    "kirtan",
    "mantra",
    "qawwali",
    "ghazal",
    "film",
    "pop",
    "rock",
    "indie",
    "folk",
    "unknown",
)
_RELIGIONS = ("hindu", "christian", "muslim", "sikh", "generic", "unknown")
_MOODS = (
    "reverent", "uplifting", "tender", "melancholic", "energetic", "playful",
    "calm", "romantic", "sad", "unknown",
)
_THEMES = (
    "hindu_devotional", "christian_devotional", "muslim_devotional",
    "sikh_devotional", "love", "separation", "nation", "celebration",
    "devotional", "life", "film", "unknown",
)

_SYSTEM_PROMPT = """You classify Indian (Hindi/Hinglish) and English songs into
intent, mood, sub-genre and religion. Return ONLY one JSON object:
{"intent": ..., "intent_confidence": 0..1, "sub_genre": ..., "religion": ...,
 "mood": ..., "theme": ...}

Allowed intent: devotional, romantic, sad, party, patriotic, ghazal,
motivational, energetic, acoustic_indie, calm, life, film, unknown
Allowed sub_genre: bhajan, aarti, chalisa, kirtan, mantra, qawwali, ghazal,
film, pop, rock, indie, folk, unknown
Allowed religion: hindu, christian, muslim, sikh, generic, unknown
Allowed mood: reverent, uplifting, tender, melancholic, energetic, playful,
calm, romantic, sad, unknown
Allowed theme: hindu_devotional, christian_devotional, muslim_devotional,
sikh_devotional, love, separation, nation, celebration, life, devotional,
film, unknown

Examples:
"Shree Hanuman Chalisa - Hariharan, Lalit Sen - Shree Hanuman Chalisa
(Hanuman Ashtak)" ->
{"intent": "devotional", "intent_confidence": 0.97, "sub_genre": "chalisa",
 "religion": "hindu", "mood": "reverent", "theme": "hindu_devotional"}
"Merawali Re - Bansi Tiwari - Chal Mann Vaishno Dham (Mata Bhajan)" ->
{"intent": "devotional", "intent_confidence": 0.95, "sub_genre": "bhajan",
 "religion": "hindu", "mood": "reverent", "theme": "hindu_devotional"}
"Humko Humare Baba Se - Sadhana Sargam - Devotional-09" ->
{"intent": "devotional", "intent_confidence": 0.92, "sub_genre": "bhajan",
 "religion": "hindu", "mood": "reverent", "theme": "hindu_devotional"}
"Raja - Raja Yeshu Raja - Kumar Sanu - Raja Raja Yeshu Raja(Christian
Devotional)" ->
{"intent": "devotional", "intent_confidence": 0.9, "sub_genre": "film",
 "religion": "christian", "mood": "uplifting", "theme": "christian_devotional"}
"Kuch Kuch Hota Hai - Jatin-Lalit, Udit Narayan, Alka Yagnik - Kuch Kuch
Hota Hai" ->
{"intent": "romantic", "intent_confidence": 0.9, "sub_genre": "film",
 "religion": "generic", "mood": "romantic", "theme": "love"}
"Tip Tip Barsa Paani - Alka Yagnik, Udit Narayan - Mohra" ->
{"intent": "romantic", "intent_confidence": 0.85, "sub_genre": "film",
 "religion": "generic", "mood": "playful", "theme": "love"}
"Jeena Yahan Marna Yahan - Mukesh - Anand" ->
{"intent": "life", "intent_confidence": 0.93, "sub_genre": "film",
 "religion": "generic", "mood": "melancholic", "theme": "life"}
"Hum Tere Pyaar Mein - Mahendra Kapoor" ->
{"intent": "romantic", "intent_confidence": 0.92, "sub_genre": "film",
 "religion": "generic", "mood": "romantic", "theme": "love"}
"Khat - 2025 Hindi romantic" ->
{"intent": "romantic", "intent_confidence": 0.9, "sub_genre": "film",
 "religion": "generic", "mood": "romantic", "theme": "love"}
"Hui Aankh Nam - Anuradha Paudwal - Saathi" ->
{"intent": "romantic", "intent_confidence": 0.85, "sub_genre": "film",
 "religion": "generic", "mood": "melancholic", "theme": "separation"}
"""

_lock = threading.RLock()
_memory_cache: dict[str, dict] = {}
_sqlite_conn: Optional[sqlite3.Connection] = None


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _cache_key(item: dict) -> str:
    return _compact_key(
        f"{item.get('title', '')} {item.get('artists', '')} {item.get('album', '')}"
    )

def cache_key(item: dict) -> str:
    """Public cache key for a track dict (title + artists + album)."""
    return _cache_key(item)


def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(DEFAULT_CACHE_DB, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_intent_cache ("
        "key TEXT PRIMARY KEY, payload TEXT, ts REAL)"
    )
    conn.commit()
    return conn


def _get_sqlite_conn() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = _connect_sqlite()
    return _sqlite_conn


def _cache_get(key: str) -> Optional[dict]:
    with _lock:
        if key in _memory_cache:
            return _memory_cache[key]
        try:
            row = _get_sqlite_conn().execute(
                "SELECT payload FROM llm_intent_cache WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if row:
            try:
                payload = json.loads(row[0])
                _memory_cache[key] = payload
                return payload
            except (ValueError, TypeError):
                return None
    return None


def _cache_put(key: str, payload: dict) -> None:
    with _lock:
        _memory_cache[key] = payload
        try:
            _get_sqlite_conn().execute(
                "INSERT OR REPLACE INTO llm_intent_cache (key, payload, ts) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            _get_sqlite_conn().commit()
        except sqlite3.Error:
            pass

def _streamlit_secret(key: str) -> Optional[str]:
    """Read a value from Streamlit secrets if available, without crashing
    when streamlit is not installed or the secrets file is missing."""
    try:
        import streamlit as st  # type: ignore
        secrets = getattr(st, "secrets", None)
        if secrets is not None and key in secrets:
            value = secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return None


def _llm_base_url() -> str:
    return (
        os.getenv("LLM_BASE_URL")
        or _streamlit_secret("LLM_BASE_URL")
        or DEFAULT_BASE_URL
    )


def _llm_api_key() -> Optional[str]:
    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()
    return _streamlit_secret("LLM_API_KEY")


def _llm_model() -> str:
    return (
        os.getenv("LLM_MODEL")
        or _streamlit_secret("LLM_MODEL")
        or DEFAULT_MODEL
    )


def _llm_timeout() -> float:
    raw = (
        os.getenv("LLM_REQUEST_TIMEOUT")
        or _streamlit_secret("LLM_REQUEST_TIMEOUT")
        or str(DEFAULT_TIMEOUT)
    )
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT


def _call_llm(text: str) -> dict:
    api_key = _llm_api_key()
    if not api_key:
        return {}
    try:
        response = requests.post(
            _llm_base_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _llm_model(),
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 250,
            },
            timeout=_llm_timeout(),
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return {}
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _clamp_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _valid(value: str, allowed: tuple[str, ...]) -> str:
    value = str(value or "").strip().casefold()
    return value if value in allowed else "unknown"


def classify_text(text: str, allow_network: bool = True) -> dict:
    """Classify one free-text track description. Returns {} when the LLM is
    unavailable or unconfigured (callers fall back to keyword rules)."""
    if not text or not text.strip():
        return {}
    key = _compact_key(text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if not allow_network:
        return {}
    payload = _call_llm(
        _SYSTEM_PROMPT.rsplit("Text:", 1)[0] + "\nText: " + text
    )
    if payload:
        intent = _valid(payload.get("intent"), _INTENTS)
        confidence = _clamp_confidence(payload.get("intent_confidence", 0.0))
        sub_genre = _valid(payload.get("sub_genre"), _SUB_GENRES)
        religion = _valid(payload.get("religion"), _RELIGIONS)
        mood = _valid(payload.get("mood"), _MOODS)
        theme = _valid(payload.get("theme"), _THEMES)
        payload = {
            "intent": intent,
            "intent_confidence": confidence,
            "sub_genre": sub_genre,
            "religion": religion,
            "mood": mood,
            "theme": theme,
        }
        _cache_put(key, payload)
    return payload or {}


def classify_item(item: dict, allow_network: bool = True) -> dict:
    """Classify a track dict. Returns a music_understanding-compatible profile
    (primary_intent, intent_confidence, sub_genre, religion, mood, ...), or {}
    when classification is unavailable."""
    text = " ".join(
        str(item.get(field, ""))
        for field in ("title", "artists", "album", "language")
    )
    llm = classify_text(text, allow_network=allow_network)
    if not llm:
        return {}
    return {
        "primary_intent": llm["intent"],
        "intent_confidence": llm["intent_confidence"],
        "sub_genre": llm["sub_genre"],
        "religion": llm["religion"],
        "mood": llm["mood"],
        "mood_confidence": llm["intent_confidence"],
        "genre": llm["sub_genre"],
        "genre_confidence": llm["intent_confidence"],
        "theme": llm["theme"],
        "theme_confidence": llm["intent_confidence"],
        "intent_source": "llm",
    }


def classify_items(items: list[dict], concurrency: int = 4) -> dict[str, dict]:
    """Classify a list of tracks, returning {cache_key: profile}. Tracks whose
    key is already cached are not re-classified. Without a configured API key
    every value is {} (no network call is made)."""
    keyed: dict[str, dict] = {}
    todo: list[dict] = []
    seen_keys: set[str] = set()
    for item in items:
        key = _cache_key(item)
        keyed.setdefault(key, {})
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _cache_get(key) is None:
            todo.append(item)
    if not todo or not _llm_api_key():
        return keyed
    concurrency = max(1, min(concurrency, len(todo)))

    def work(item: dict) -> tuple[str, dict]:
        return _cache_key(item), classify_item(item)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(work, item): item for item in todo}
        try:
            for future in as_completed(futures, timeout=_llm_timeout() * 3):
                key, profile = future.result()
                keyed[key] = profile
                if profile:
                    _cache_put(key, profile)
        except Exception:
            pass
    return keyed
