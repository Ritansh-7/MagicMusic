from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.cache import Cache
from backend.storage import EventStore
from jiosaavn_client import JioSaavnClient

ROOT = Path(__file__).resolve().parents[1]


class EventIn(BaseModel):
    user_id: str = Field(default="anonymous", max_length=160)
    event_type: str = Field(min_length=2, max_length=40)
    track_id: str = Field(default="", max_length=300)
    payload: dict[str, Any] = Field(default_factory=dict)


class RecommendationIn(BaseModel):
    seed: dict[str, Any]
    user_id: str = Field(default="anonymous", max_length=160)
    limit: int = Field(default=20, ge=1, le=50)


def load_catalog() -> pd.DataFrame:
    path = ROOT / "spotify_tracks_data.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


catalog = load_catalog()
hybrid_engine = None
client = JioSaavnClient()
cache = Cache()
events = EventStore()


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
    allow_origins=os.getenv("COZYBEATS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _key(prefix: str, value: str) -> str:
    return f"cozybeats:{prefix}:{hashlib.sha256(value.encode()).hexdigest()}"


def _search(query: str, limit: int) -> list[dict]:
    return client.search(query, limit=min(max(limit, 1), 50))


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
            return [dict(item) for item in ranked[: request.limit]]
        except Exception:
            pass
    return candidates[: request.limit]


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "service": "cozybeats-api", "catalog_tracks": len(catalog)}


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
async def recommendations(request: RecommendationIn):
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
async def write_event(event: EventIn):
    await run_in_threadpool(events.write, event.user_id, event.event_type, event.track_id, event.payload)
    return {"accepted": True}

