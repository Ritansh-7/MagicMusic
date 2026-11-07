from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/audio_features.sqlite3")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audio_feature_cache (
            track_id TEXT PRIMARY KEY,
            features TEXT NOT NULL,
            model_version TEXT NOT NULL DEFAULT 'v1',
            audio_hash TEXT,
            updated_at TEXT
        )"""
    )
    conn.commit()
    return conn


def get(track_id: str) -> dict | None:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT features FROM audio_feature_cache WHERE track_id=?", (str(track_id),)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None
    finally:
        conn.close()


def put(track_id: str, features: dict, model_version: str = "v1", audio_hash: str | None = None) -> None:
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO audio_feature_cache (track_id, features, model_version, audio_hash, updated_at) VALUES (?,?,?,?,datetime('now'))",
            (str(track_id), json.dumps(features), model_version, audio_hash),
        )
        conn.commit()
    finally:
        conn.close()
