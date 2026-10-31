from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests
import re


def _extract_raw_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        nested = (
            payload.get("results")
            or payload.get("data")
            or payload.get("songs")
            or payload.get("items")
            or []
        )
        return [item for item in nested if isinstance(item, dict)] if isinstance(nested, list) else []
    return []


def _fetch_endpoint(base_url: str, path: str, query: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{base_url}{path}",
            params={"query": query, "lyrics": "false"},
            timeout=3.5,
        )
        response.raise_for_status()
        return _extract_raw_items(response.json())
    except (requests.RequestException, ValueError):
        return []


@lru_cache(maxsize=512)
def _cached_search_page(base_url: str, query: str) -> tuple[dict[str, Any], ...]:
    """Cache merged /result/ and /song/ pages for one query."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        pages = list(executor.map(
            lambda path: _fetch_endpoint(base_url, path, query),
            ("/result/", "/song/"),
        ))
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_items in pages:
        for item in raw_items:
            key = str(item.get("id") or item.get("songid") or item.get("song_id") or "") or (
                f"{item.get('title') or item.get('song') or item.get('name')}|{item.get('singers') or item.get('primaryArtists') or ''}"
            ).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return tuple(
        JioSaavnClient._normalize(item)
        for item in merged
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
        # The current Saavn mirror often leaves ``singers`` empty and puts the
        # actual performer in ``primary_artists``.  Keep the fallback order
        # explicit so artist context is never silently lost to the composer.
        singers = (
            item.get("singers")
            or item.get("primary_artists")
            or item.get("artist")
            or item.get("artists")
            or item.get("primaryArtists")
            or ""
        )
        if isinstance(singers, list):
            return ", ".join(
                str(a.get("name", a)) if isinstance(a, dict) else str(a)
                for a in singers
            )
        return str(singers)

    @staticmethod
    def _recording_key(item: dict[str, Any]) -> str:
        """Return a stable key that ignores release/version suffixes."""
        title = str(item.get("title") or item.get("song") or item.get("name") or "")
        title = re.sub(
            r"\b(from|version|soundtrack|original motion picture soundtrack|"
            r"jhankar beats|remix|reprise|live|instrumental|karaoke|vocals only)\b.*$",
            " ",
            title.casefold(),
        )
        title = re.sub(r"[^a-z0-9]+", " ", title).strip()
        # Search shelves should show one card per recording. Different
        # releases sometimes disagree about singer metadata, so including the
        # artist in this key would let the same song appear twice.
        return title

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
        cast = str(
            item.get("starring")
            or item.get("actors")
            or item.get("cast")
            or ""
        ).strip()
        normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        normalized_artists = re.sub(r"[^a-z0-9]+", " ", artists.lower()).strip()
        normalized_album = re.sub(r"[^a-z0-9]+", " ", album.lower()).strip()
        raw_duration = item.get("duration_ms") or item.get("duration") or 0
        try:
            duration_value = float(raw_duration)
            # Saavn's ``duration`` is seconds; only fields explicitly named
            # duration_ms are already in milliseconds.
            duration_ms = int(duration_value if item.get("duration_ms") else duration_value * 1000)
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
            "starring": cast,
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

    @staticmethod
    def _recover_album_artists(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill missing artists only from an unambiguous album peer result."""
        album_artists: dict[str, str] = {}
        ambiguous_albums: set[str] = set()
        for item in items:
            album_key = re.sub(r"[^a-z0-9]+", " ", str(item.get("album", "")).casefold()).strip()
            artists = str(item.get("artists", "")).strip()
            if not album_key or not artists:
                continue
            existing = album_artists.get(album_key)
            if existing and existing.casefold() != artists.casefold():
                ambiguous_albums.add(album_key)
            else:
                album_artists[album_key] = artists
        for item in items:
            if str(item.get("artists", "")).strip():
                continue
            album_key = re.sub(r"[^a-z0-9]+", " ", str(item.get("album", "")).casefold()).strip()
            recovered = album_artists.get(album_key)
            if recovered and album_key not in ambiguous_albums:
                item["artists"] = recovered
                item["normalized_artists"] = re.sub(
                    r"[^a-z0-9]+", " ", recovered.casefold()
                ).strip()
        return items

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        normalized_query = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
        query_tokens = set(normalized_query.split())
        devotional_query = any(
            token in normalized_query
            for token in ("bajrang", "bajran", "bajrangbali", "hanuman", "anjaney")
        )
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        page_size = max(limit, 40)

        def add_items(raw_items: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
            for item in raw_items or []:
                key = self._recording_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append(item)

        def finalize() -> list[dict[str, Any]]:
            finalized = self._recover_album_artists(collected)
            if entity_search:
                matching = []
                for item in finalized:
                    fields = {
                        re.sub(r"[^a-z0-9]+", " ", str(item.get(field, "")).casefold()).strip()
                        for field in ("title", "artists", "album", "starring")
                    }
                    if any(
                        normalized_query in value or query_tokens.issubset(set(value.split()))
                        for value in fields
                    ):
                        matching.append(item)
                if matching:
                    finalized = matching
            return finalized[:limit]

        # One direct page first. Extra generic expansions (male/female/hindi)
        # cost tokens and mix unrelated songs into artist/film catalogues.
        first_wave = self.search_direct(query, limit=page_size)
        add_items(first_wave)

        def entity_hit(item: dict[str, Any], field: str) -> bool:
            value = re.sub(r"[^a-z0-9]+", " ", str(item.get(field, "")).casefold()).strip()
            if not value or not query_tokens:
                return False
            return normalized_query in value or query_tokens.issubset(set(value.split()))

        artist_hits = sum(1 for item in first_wave if entity_hit(item, "artists"))
        album_hits = sum(1 for item in first_wave if entity_hit(item, "album"))
        cast_hits = sum(1 for item in first_wave if entity_hit(item, "starring"))
        exact_title = any(
            re.sub(r"[^a-z0-9]+", " ", str(item.get("title", "")).casefold()).strip()
            == normalized_query
            for item in first_wave
        )
        entity_search = (artist_hits >= 1 or album_hits >= 1 or cast_hits >= 1) and not exact_title

        if devotional_query:
            extra_queries = (f"{query} bhajan", "Hanuman bhajan")
        elif entity_search:
            extra_queries = (f"{query} songs", f"{query} hits", f"{query} top songs")
        elif len(collected) < min(8, limit):
            extra_queries = ()
        else:
            extra_queries = ()
        if extra_queries and len(collected) < limit:
            with ThreadPoolExecutor(max_workers=min(4, len(extra_queries))) as executor:
                for raw_items in executor.map(
                    lambda candidate: self.search_direct(candidate, limit=page_size),
                    extra_queries,
                ):
                    add_items(raw_items)

        # Tiny upstream pages need a second wave from matched titles/albums
        # so an artist or film search can grow beyond 4–5 songs.
        if entity_search and len(collected) < limit:
            discovery_queries: list[str] = []
            for item in collected[:12]:
                title = str(item.get("title", "")).strip()
                artists = [
                    part.strip()
                    for part in str(item.get("artists", "")).split(",")
                    if part.strip()
                ]
                album = str(item.get("album", "")).strip()
                if title and artists:
                    discovery_queries.append(f"{title} {artists[0]}")
                discovery_queries.extend(f"{artist} songs" for artist in artists[:1])
                if album:
                    discovery_queries.append(album)
            discovery_queries = list(dict.fromkeys(discovery_queries))[:10]
            if discovery_queries:
                with ThreadPoolExecutor(max_workers=min(8, len(discovery_queries))) as executor:
                    for raw_items in executor.map(
                        lambda candidate: self.search_direct(candidate, limit=page_size),
                        discovery_queries,
                    ):
                        add_items(raw_items)
                        if len(collected) >= limit:
                            return finalize()
        return finalize()

    def search_direct(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        """Fetch one upstream search page without query expansion."""
        query = (query or "").strip()
        if not query:
            return []
        return list(_cached_search_page(self.base_url, query)[:limit])
