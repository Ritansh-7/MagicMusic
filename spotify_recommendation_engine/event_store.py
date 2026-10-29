from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventStore:
    """Minimal local event store for explicit recommendation feedback."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("TUNEMATCH_DB_PATH", "tune_match_events.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )

    def log(self, user_id: str, track_id: str, event_type: str, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_events
                    (user_id, track_id, event_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(track_id),
                    str(event_type),
                    json.dumps(metadata or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def count_for_user(self, user_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM recommendation_events WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return int(row["count"])
