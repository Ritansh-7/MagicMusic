from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class EventStore:
    """SQLite dev store; DATABASE_URL can later point to a managed Postgres adapter."""

    def __init__(self) -> None:
        self.path = Path(os.getenv("COZYBEATS_EVENT_DB", "cozybeats_api.sqlite3"))
        self._lock = threading.RLock()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, event_type TEXT, "
                "track_id TEXT, payload TEXT, created_at TEXT NOT NULL)"
            )
            conn.commit()

    def write(self, user_id: str, event_type: str, track_id: str = "", payload: dict | None = None) -> None:
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO events(user_id,event_type,track_id,payload,created_at) VALUES(?,?,?,?,?)",
                (user_id, event_type, track_id, json.dumps(payload or {}), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

