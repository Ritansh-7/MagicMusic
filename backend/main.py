from __future__ import annotations

import hashlib
import json
import os
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.cache import Cache
from backend.auth import issue_token, read_token
from jiosaavn_client import JioSaavnClient
from recommender import (
    infer_seed_themes,
    infer_track_language,
    is_weak_live_title,
    recommendation_title_key,
    seed_context_discovery_queries,
)
from user_store import UserStore

ROOT = Path(__file__).resolve().parents[1]


def load_local_secrets() -> None:
    """Use Streamlit secrets locally; hosting platforms should use env vars."""
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
        for key in ("DATABASE_URL", "REDIS_URL", "COZYBEATS_SESSION_SECRET", "TUNEMATCH_SESSION_SECRET"):
            value = values.get(key)
            if value and not os.getenv(key):
                os.environ[key] = str(value)
    except (OSError, tomllib.TOMLDecodeError):
        pass


load_local_secrets()


class EventIn(BaseModel):
    user_id: str = Field(default="anonymous", max_length=160)
    event_type: str = Field(min_length=2, max_length=40)
    track_id: str = Field(default="", max_length=300)
    payload: dict[str, Any] = Field(default_factory=dict)


class RecommendationIn(BaseModel):
    seed: dict[str, Any]
    user_id: str = Field(default="anonymous", max_length=160)
    limit: int = Field(default=24, ge=1, le=50)
    exclude_ids: list[str] = Field(default_factory=list)


class AuthIn(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=8, max_length=200)


def current_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required")
    user = read_token(authorization.split(" ", 1)[1].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Session expired; please login again")
    return user


def load_catalog() -> pd.DataFrame:
    path = ROOT / "spotify_tracks_data.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


catalog = load_catalog()
hybrid_engine = None
client = JioSaavnClient()
cache = Cache()
user_store = UserStore(database_url=os.getenv("DATABASE_URL") or None)
_recent_recs: dict[str, list[str]] = {}


def _get_hybrid_engine():
    global hybrid_engine
    if hybrid_engine is None and not catalog.empty:
        # Lazy import avoids loading the local embedding/model during API boot.
        from recommender import HybridRecommendationEngine, SpotifyRecommendationEngine
        hybrid_engine = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    return hybrid_engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="CozyBeats API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "COZYBEATS_ALLOWED_ORIGINS",
        "https://cozybeats.vercel.app,http://localhost:3000,http://localhost:8501",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _key(prefix: str, value: str) -> str:
    return f"cozybeats:{prefix}:{hashlib.sha256(value.encode()).hexdigest()}"


def _search(query: str, limit: int) -> list[dict]:
    return client.search_direct(query, limit=min(max(limit, 1), 40))


def _track_identity(item: dict[str, Any]) -> str:
    return str(item.get("song_id") or item.get("title", "")).casefold().strip()


def _is_repeat_filler(item: dict[str, Any], seed: dict[str, Any]) -> bool:
    if is_weak_live_title(item):
        return True
    seed_title = recommendation_title_key(seed.get("title", ""))
    item_title = recommendation_title_key(item.get("title", ""))
    if seed_title and item_title == seed_title:
        return True
    seed_lang = infer_track_language(seed) or str(seed.get("language", "")).casefold()
    item_lang = infer_track_language(item) or str(item.get("language", "")).casefold()
    if seed_lang and item_lang and seed_lang != item_lang:
        return True
    return False


def _catalog_fallback(seed: dict[str, Any], existing: list[dict], limit: int) -> list[dict]:
    """Keep the shelf full when a live provider returns too few playable records.

    This is deliberately a last-mile fallback: live JioSaavn records stay first,
    while the local catalog supplies same-language/genre-era context instead of
    showing an empty recommendation shelf.
    """
    if catalog.empty or len(existing) >= limit:
        return []
    blocked = {
        str(item.get("song_id") or item.get("track_id") or item.get("title", "")).casefold()
        for item in existing
    }
    seed_key = str(seed.get("song_id") or seed.get("track_id") or seed.get("title", "")).casefold()
    blocked.add(seed_key)
    seed_artist = str(seed.get("artists", "")).casefold()
    seed_genres = {token.strip() for token in str(seed.get("track_genre", "")).casefold().split(",") if token.strip()}
    rows: list[tuple[float, dict[str, Any]]] = []
    for _, row in catalog.iterrows():
        track_id = str(row.get("track_id", ""))
        if not track_id or track_id.casefold() in blocked:
            continue
        artists = str(row.get("artists", ""))
        genres = {token.strip() for token in str(row.get("track_genre", "")).casefold().split(",") if token.strip()}
        score = float(row.get("popularity", 0) or 0) / 1000.0
        if seed_artist and seed_artist in artists.casefold():
            score += 2.0
        score += 1.0 * len(seed_genres.intersection(genres))
        item = {
            "song_id": track_id,
            "title": str(row.get("track_name", "Untitled")),
            "artists": artists,
            "album": str(row.get("album_name", "")),
            "track_genre": str(row.get("track_genre", "")),
            "image_url": str(row.get("image_url", "")),
            "stream_url": "",
            "recommendation_reason": "Same genre/artist context from the local catalog",
        }
        rows.append((score, item))
    rows.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in rows[: max(0, limit - len(existing))]]


def _catalog_shelf_fallback(query: str, existing: list[dict], limit: int) -> list[dict]:
    """Fill category shelves from the local catalog when live search is sparse."""
    if catalog.empty or len(existing) >= limit:
        return []
    text = query.casefold()
    hindi_terms = {"hindi", "romantic", "classic", "bollywood", "desi", "punjabi"}
    english_terms = {"english", "global", "pop", "rock", "indie", "dance", "acoustic"}
    mood_terms = {token for token in ("romantic", "chill", "popular", "trending", "acoustic") if token in text}
    existing_ids = {str(item.get("song_id") or item.get("track_id") or item.get("title", "")).casefold() for item in existing}
    indian_markers = {"filmi", "bollywood", "desi", "punjabi", "hindi", "bhangra", "indian", "urdu"}
    rows: list[tuple[float, dict[str, Any]]] = []
    for _, row in catalog.iterrows():
        track_id = str(row.get("track_id", ""))
        if not track_id or track_id.casefold() in existing_ids:
            continue
        genre_text = str(row.get("track_genre", "")).casefold()
        tokens = {token.strip() for token in genre_text.split(",") if token.strip()}
        score = float(row.get("popularity", 0) or 0) / 1000.0
        if hindi_terms.intersection(tokens) or ("hindi" in text and "filmi" in tokens):
            score += 2.0
        if "english" in text and not tokens.intersection(indian_markers):
            score += 2.0
        if mood_terms.intersection(tokens):
            score += 1.0
        item = {
            "song_id": track_id,
            "title": str(row.get("track_name", "Untitled")),
            "artists": str(row.get("artists", "")),
            "album": str(row.get("album_name", "")),
            "track_genre": str(row.get("track_genre", "")),
            "image_url": str(row.get("image_url", "")),
            "stream_url": "",
            "recommendation_reason": "Curated from the CozyBeats catalog",
        }
        rows.append((score, item))
    rows.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in rows[: max(0, limit - len(existing))]]


def _recommend(request: RecommendationIn) -> list[dict]:
    seed = dict(request.seed)
    title = str(seed.get("title", "")).strip()
    artist = str(seed.get("artists", "")).split(",")[0].strip()
    if not title:
        raise ValueError("seed.title is required")
    seed.setdefault("intent_labels", infer_seed_themes(seed))
    language = infer_track_language(seed) or str(seed.get("language", "")).strip()
    queries = seed_context_discovery_queries(seed, seed.get("year"), language, [artist] if artist else [])
    queries = list(dict.fromkeys([f"{title} {artist}".strip(), title, *queries]))[:6]
    with ThreadPoolExecutor(max_workers=min(6, len(queries))) as pool:
        candidate_groups = list(pool.map(lambda q: _search(q, 24), queries))
    blocked = {
        str(value).casefold().strip()
        for value in request.exclude_ids
        if str(value).strip()
    }
    blocked.update(_recent_recs.get(request.user_id, []))
    blocked.add(_track_identity(seed))
    candidates: list[dict] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for item in group:
            identity = _track_identity(item)
            title_key = recommendation_title_key(item.get("title", ""))
            if not identity or identity in seen or identity in blocked or title_key in blocked:
                continue
            if _is_repeat_filler(item, seed):
                continue
            seen.add(identity)
            if title_key:
                seen.add(title_key)
            candidates.append(item)
    engine = _get_hybrid_engine()
    ranked_items: list[dict] = []
    if engine:
        try:
            ranked = engine.rank_jio_results(candidates, request.user_id, catalog, [], seed=seed)
            ranked_items = [dict(item) for item in ranked if not _is_repeat_filler(item, seed)]
        except Exception:
            ranked_items = []
    if not ranked_items:
        ranked_items = candidates
    limit = max(20, int(request.limit or 24))
    unique: list[dict] = []
    used: set[str] = set()
    for item in ranked_items:
        identity = _track_identity(item)
        title_key = recommendation_title_key(item.get("title", ""))
        keys = {identity, title_key}
        if not identity or keys.intersection(used):
            continue
        used.update(keys)
        unique.append(item)
        if len(unique) >= limit:
            break
    remembered = _recent_recs.setdefault(request.user_id, [])
    for item in unique:
        identity = _track_identity(item)
        if identity and identity not in remembered:
            remembered.append(identity)
    _recent_recs[request.user_id] = remembered[-80:]
    return unique


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "service": "cozybeats-api", "catalog_tracks": len(catalog)}


@app.post("/v1/auth/register")
async def register(request: AuthIn):
    ok, result = await run_in_threadpool(user_store.register, request.email, request.password)
    if not ok:
        raise HTTPException(status_code=409, detail=result)
    email = request.email.strip().lower()
    return {"token": issue_token(result, email), "user": {"id": result, "email": email}}


@app.post("/v1/auth/login")
async def login(request: AuthIn):
    ok, result = await run_in_threadpool(user_store.authenticate, request.email, request.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    email = request.email.strip().lower()
    return {"token": issue_token(result, email), "user": {"id": result, "email": email}}


@app.get("/v1/auth/me")
async def me(user: dict[str, str] = Depends(current_user)):
    return {"user": user}


@app.get("/v1/history")
async def history(user: dict[str, str] = Depends(current_user)):
    """Return the listener's recent playable tracks for every client layout."""
    events = await run_in_threadpool(user_store.user_events, user["user_id"])
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in reversed(events):
        if event.get("event_type") not in {"play", "play_start", "jio_play"}:
            continue
        try:
            payload = json.loads(event.get("metadata_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        item = dict(payload)
        item.setdefault("song_id", event.get("track_id", ""))
        item.setdefault("title", item.get("song") or "Recently played song")
        identity = str(item.get("song_id") or item.get("title", "")).casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        items.append(item)
        if len(items) >= 20:
            break
    return {"items": items}


@app.get("/v1/home")
async def home(user: dict[str, str] = Depends(current_user)):
    categories = {
        "Hindi picks": "hindi songs",
        "Bollywood romance": "hindi romantic songs",
        "English pop": "english pop songs",
        "English chill": "english love songs",
    }
    cache_key = _key("home", "catalog-v5-fast-home")
    cached = cache.get_json(cache_key)
    if cached is not None:
        return {"shelves": cached, "cached": True}
    with ThreadPoolExecutor(max_workers=len(categories)) as pool:
        groups = list(pool.map(lambda q: _search(q, 20), categories.values()))
    shelves = []
    for (title, search_query), items in zip(categories.items(), groups):
        items = [item for item in items if not is_weak_live_title(item)]
        items.extend(_catalog_shelf_fallback(search_query, items, 20))
        shelves.append({"title": title, "items": items[:20]})
    cache.set_json(cache_key, shelves, ttl=900)
    return {"shelves": shelves, "cached": False}


@app.get("/v1/search")
async def search(q: str = Query(min_length=1, max_length=160), limit: int = Query(default=20, ge=1, le=50)):
    key = _key("search", f"{q.casefold()}:{limit}")
    cached = cache.get_json(key)
    if cached is not None:
        return {"items": cached, "cached": True}
    items = await run_in_threadpool(_search, q, limit)
    cache.set_json(key, items, ttl=600)
    return {"items": items, "cached": False}


@app.post("/v1/recommendations")
async def recommendations(request: RecommendationIn, user: dict[str, str] = Depends(current_user)):
    request.user_id = user["user_id"]
    request.limit = max(request.limit, 20)
    title = str(request.seed.get("title", "")).strip()
    if not title:
        raise HTTPException(status_code=422, detail="seed.title is required")
    try:
        items = await run_in_threadpool(_recommend, request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"recommendation provider unavailable: {type(exc).__name__}") from exc
    return {"items": items, "cached": False}


@app.post("/v1/events", status_code=202)
async def write_event(event: EventIn, user: dict[str, str] = Depends(current_user)):
    event.user_id = user["user_id"]
    # Persist behavior in UserStore so Aiven MySQL is the shared source of truth.
    metadata = json.dumps(event.payload, default=str)
    session_id = str(event.payload.get("session_id", "web"))
    duration_ms = int(event.payload.get("duration_ms", 0) or 0)
    await run_in_threadpool(
        user_store.log_event,
        event.user_id,
        event.track_id,
        event.event_type,
        session_id,
        duration_ms,
        metadata,
    )
    return {"accepted": True}

