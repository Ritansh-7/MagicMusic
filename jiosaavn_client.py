from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests
import re


@lru_cache(maxsize=512)
def _cached_search_page(base_url: str, query: str) -> tuple[dict[str, Any], ...]:
    """Cache one upstream page so reruns do not refetch the same query."""
    try:
        response = requests.get(
            f"{base_url}/result/",
            params={"query": query, "lyrics": "false"},
            timeout=3.5,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return ()
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = (
            payload.get("results")
            or payload.get("data")
            or payload.get("songs")
            or payload.get("items")
            or []
        )
    else:
        raw_items = []
    return tuple(
        JioSaavnClient._normalize(item)
        for item in raw_items
        if isinstance(item, dict)
    )


class JioSaavnClient:
    """Unofficial public JioSaavn metadata adapter.

    No JioSaavn account is required for public search. The API base can be
    overridden with JIOSAAVN_API_URL when running a local fork of the service.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv(
            "JIOSAAVN_API_URL", "https://saavnapi-nine.vercel.app"
        )).rstrip("/")

    @staticmethod
    def _artist_names(item: dict[str, Any]) -> str:
        singers = item.get("singers") or item.get("artist") or item.get("artists") or ""
        if isinstance(singers, list):
            return ", ".join(
                str(a.get("name", a)) if isinstance(a, dict) else str(a)
                for a in singers
            )
        return str(singers)

    @classmethod
    def _normalize(cls, item: dict[str, Any]) -> dict[str, Any]:
        images = item.get("image") or item.get("images") or item.get("image_url") or ""
        if isinstance(images, list):
            images = images[-1] if images else ""
        if isinstance(images, dict):
            images = images.get("url", "")
        stream_url = (
            item.get("media_url")
            or item.get("vlink")
            or item.get("media_preview_url")
            or item.get("url")
            or item.get("stream_url")
            or item.get("download_url")
            or ""
        )
        title = str(item.get("title") or item.get("song") or item.get("name") or "Unknown song").strip()
        artists = cls._artist_names(item).strip()
        album = str(item.get("album") or item.get("album_name") or "").strip()
        normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        normalized_artists = re.sub(r"[^a-z0-9]+", " ", artists.lower()).strip()
        normalized_album = re.sub(r"[^a-z0-9]+", " ", album.lower()).strip()
        raw_duration = item.get("duration_ms") or item.get("duration") or 0
        try:
            duration_ms = int(float(raw_duration))
        except (TypeError, ValueError):
            duration_ms = 0
        song_id = str(item.get("id") or item.get("songid") or item.get("song_id") or "")
        version_text = f"{title} {album}".lower()
        version_type = (
            "remix" if "remix" in version_text else
            "cover" if "cover" in version_text else
            "live" if "live" in version_text else
            "original"
        )
        return {
            "title": title,
            "artists": artists,
            "album": album,
            "language": str(item.get("language") or "").strip().lower(),
            "image_url": str(images or ""),
            "duration": item.get("duration") or item.get("duration_ms") or "",
            "year": item.get("year") or item.get("release_year") or "",
            "perma_url": item.get("perma_url") or item.get("song_url") or item.get("tiny_url") or "",
            # Prefer the CDN media URL: preview.saavncdn.com is frequently
            # rate-limited (429), while the AAC URL is the playable source.
            # Some responses expose only these fields instead of older `url`.
            "stream_url": stream_url,
            "stream_format": (
                "audio/mp4"
                if str(stream_url).lower().split("?", 1)[0].endswith(".mp4")
                else "audio/mpeg"
            ),
            "song_id": song_id,
            # Stable fields used by deduplication, quality checks and
            # recommendation explanations.
            "normalized_title": normalized_title,
            "normalized_artists": normalized_artists,
            "normalized_album": normalized_album,
            "duration_ms": duration_ms,
            "version_type": version_type,
            "quality_score": sum(bool(value) for value in (title, artists, album, images, stream_url)),
        }

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Search several high-signal expansions in parallel. The old sequential
        # loop could make one search wait for 18 upstream requests; cached pages
        # and ordered result merging preserve relevance while cutting latency.
        candidates = (
            query,
            f"{query} song",
            f"{query} music",
            f"{query} hindi",
            f"{query} soundtrack",
            f"{query} songs",
        )
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
            pages = executor.map(
                lambda candidate: self.search_direct(candidate, limit=min(limit, 12)),
                candidates,
            )
        first_wave: list[dict[str, Any]] = []
        for raw_items in pages:
            if not raw_items:
                continue
            for item in raw_items:
                key = str(item.get("song_id") or "") or (
                    f"{item.get('title', '')}|{item.get('artists', '')}".lower()
                )
                if key in seen:
                    continue
                seen.add(key)
                collected.append(item)
                first_wave.append(item)
                if len(collected) >= limit:
                    return collected[:limit]

        # Some public Saavn mirrors return only the exact title matches for a
        # title query. Expand using the artists/albums found in that first
        # wave, so a search for a classic song still gets a useful shelf and
        # not just two alternate releases of the same recording.
        discovery_queries: list[str] = []
        for item in first_wave[:8]:
            artists = [part.strip() for part in str(item.get("artists", "")).split(",") if part.strip()]
            discovery_queries.extend(f"{artist} songs" for artist in artists[:3])
            album = str(item.get("album", "")).strip()
            if album:
                discovery_queries.append(f"{album} songs")
        discovery_queries = list(dict.fromkeys(discovery_queries))[:10]
        if discovery_queries and len(collected) < limit:
            with ThreadPoolExecutor(max_workers=min(6, len(discovery_queries))) as executor:
                discovery_pages = executor.map(
                    lambda candidate: self.search_direct(candidate, limit=min(20, limit)),
                    discovery_queries,
                )
            for raw_items in discovery_pages:
                for item in raw_items or []:
                    key = str(item.get("song_id") or "") or (
                        f"{item.get('title', '')}|{item.get('artists', '')}".lower()
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(item)
                    if len(collected) >= limit:
                        return collected[:limit]
        return collected[:limit]

    def search_direct(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        """Fetch one upstream search page without query expansion."""
        query = (query or "").strip()
        if not query:
            return []
        return list(_cached_search_page(self.base_url, query)[:limit])
