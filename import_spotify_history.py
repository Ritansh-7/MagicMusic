"""Convert Spotify privacy-export JSON files into a clean listening-history CSV.

Usage:
    python import_spotify_history.py "C:\\Users\\you\\Downloads\\SpotifyData" \
        --output spotify_history.csv

The importer supports both Extended Streaming History field names and the
older Streaming History format. It intentionally excludes IP addresses,
user-agent strings, country, and other technical/private fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return default


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "tracks", "history"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    return []


def import_history(source: str | Path, output: str | Path) -> pd.DataFrame:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Spotify export path does not exist: {source_path}")
    json_files = sorted(source_path.rglob("*.json")) if source_path.is_dir() else [source_path]
    if not json_files:
        raise ValueError(f"No JSON files found under: {source_path}")
    rows: list[dict[str, Any]] = []

    for path in json_files:
        for item in _records_from_json(path):
            track_name = _first(item, "track_name", "trackName")
            artist_name = _first(item, "artist_name", "artistName")
            # Ignore podcasts, videos, README files, and non-listening JSON.
            if not track_name or not artist_name:
                continue
            rows.append(
                {
                    "played_at": _first(item, "ts", "endTime"),
                    "track_name": str(track_name),
                    "artists": str(artist_name),
                    "album_name": str(_first(item, "album_name", "albumName", default="") or ""),
                    "ms_played": int(_first(item, "ms_played", "msPlayed", default=0) or 0),
                    "spotify_track_uri": _first(item, "spotify_track_uri", "spotifyTrackUri", default=""),
                    "skipped": _first(item, "skipped", default=False),
                    "reason_start": _first(item, "reason_start", default=""),
                    "reason_end": _first(item, "reason_end", default=""),
                    "platform": _first(item, "platform", default=""),
                }
            )

    columns = [
        "played_at", "track_name", "artists", "album_name", "ms_played",
        "spotify_track_uri", "skipped", "reason_start", "reason_end", "platform",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["played_at", "track_name", "artists", "ms_played"]
        ).sort_values("played_at", na_position="last").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Extracted Spotify data folder or a JSON file")
    parser.add_argument(
        "--output",
        default="spotify_history.csv",
        help="Output CSV path (default: spotify_history.csv)",
    )
    args = parser.parse_args()
    frame = import_history(args.source, args.output)
    print(f"Imported {len(frame):,} listening events")
    print(f"Saved clean history to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
