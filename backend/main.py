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
    limit: int = Field(default=20, ge=1, le=50)


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
    return client.search(query, limit=min(max(limit, 1), 50))


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
    queries = list(dict.fromkeys([f"{title} {artist}".strip(), title, f"{artist} songs".strip()]))
    with ThreadPoolExecutor(max_workers=min(3, len(queries))) as pool:
        candidate_groups = list(pool.map(lambda q: _search(q, 30), queries))
    candidates: list[dict] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for item in group:
            identity = str(item.get("song_id") or item.get("title", "")).casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            candidates.append(item)
    engine = _get_hybrid_engine()
    if engine and not catalog.empty:
        try:
            ranked = engine.rank_jio_results(candidates, request.user_id, catalog, [], seed=seed)
            ranked_items = [dict(item) for item in ranked[: request.limit]]
            return ranked_items + _catalog_fallback(seed, ranked_items, request.limit)
        except Exception:
            pass
    live_items = candidates[: request.limit]
    return live_items + _catalog_fallback(seed, live_items, request.limit)


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


@app.get("/v1/home")
async def home(user: dict[str, str] = Depends(current_user)):
    categories = {
        "Hindi picks": "hindi songs",
        "English / global": "english pop hits",
        "Romantic": "hindi romantic songs",
        "Classics": "hindi evergreen classics",
        "Chill": "chill acoustic songs",
        "Trending": "latest popular songs",
    }
    # Home shelves are curated by category, so share them across users. User-specific
    # ranking stays in /v1/recommendations; this prevents one provider fan-out per user.
    cache_key = _key("home", "catalog-v2")
    cached = cache.get_json(cache_key)
    if cached is not None:
        return {"shelves": cached, "cached": True}
    with ThreadPoolExecutor(max_workers=len(categories)) as pool:
        groups = list(pool.map(lambda q: _search(q, 12), categories.values()))
    shelves = []
    for (title, search_query), items in zip(categories.items(), groups):
        items = list(items)
        items.extend(_catalog_shelf_fallback(search_query, items, 16))
        shelves.append({"title": title, "items": items[:16]})
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
    request.limit = max(request.limit, 10)
    title = str(request.seed.get("title", "")).strip()
    if not title:
        raise HTTPException(status_code=422, detail="seed.title is required")
    key = _key("recommendations", f"{request.user_id}:{title}:{request.limit}")
    cached = cache.get_json(key)
    if cached is not None:
        return {"items": cached, "cached": True}
    try:
        items = await run_in_threadpool(_recommend, request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"recommendation provider unavailable: {type(exc).__name__}") from exc
    cache.set_json(key, items, ttl=180)
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

