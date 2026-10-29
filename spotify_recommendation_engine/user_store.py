from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class UserStore:
    """Local account and behavior store for the demo app.

    Passwords are salted PBKDF2 hashes; raw passwords are never stored.
    """

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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    track_id TEXT,
                    event_type TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
        return f"{salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            salt_hex, digest_hex = encoded.split("$", 1)
            candidate = UserStore._hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
            return hmac.compare_digest(candidate, digest_hex)
        except (ValueError, TypeError):
            return False

    def register(self, email: str, password: str) -> tuple[bool, str]:
        email = email.strip().lower()
        if "@" not in email or len(password) < 8:
            return False, "Use a valid email and a password with at least 8 characters."
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)",
                    (email, self._hash_password(password), datetime.now(timezone.utc).isoformat()),
                )
                return True, str(cursor.lastrowid)
        except sqlite3.IntegrityError:
            return False, "An account with this email already exists."

    def authenticate(self, email: str, password: str) -> tuple[bool, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        if not row or not self._verify_password(password, row["password_hash"]):
            return False, "Email or password is incorrect."
        return True, str(row["id"])

    def user_email(self, user_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT email FROM users WHERE id = ?",
                (str(user_id),),
            ).fetchone()
        return str(row["email"]) if row else ""

    def is_admin(self, user_id: str) -> bool:
        """Admin access is allowlisted by an environment variable, never by UI input."""
        admin_email = os.getenv("TUNEMATCH_ADMIN_EMAIL", "").strip().lower()
        return bool(admin_email and self.user_email(user_id).lower() == admin_email)

    def admin_users(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, email, created_at FROM users ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_events(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user_id, track_id, event_type, session_id, duration_ms,
                       metadata_json, created_at
                FROM behavior_events
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def log_event(
        self,
        user_id: str,
        track_id: str | None,
        event_type: str,
        session_id: str,
        duration_ms: int = 0,
        metadata_json: str = "{}",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO behavior_events
                    (user_id, track_id, event_type, session_id, duration_ms, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(user_id), track_id, event_type, session_id, int(duration_ms), metadata_json,
                 datetime.now(timezone.utc).isoformat()),
            )

    def liked_track_ids(self, user_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT track_id FROM behavior_events WHERE user_id = ? AND event_type = 'like' AND track_id IS NOT NULL",
                (str(user_id),),
            ).fetchall()
        return [str(row["track_id"]) for row in rows]

    def interaction_events(self):
        """Return anonymized interaction rows used by the implicit-feedback model."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user_id, track_id, event_type, session_id, duration_ms, created_at
                FROM behavior_events
                WHERE track_id IS NOT NULL
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self, user_id: str, session_id: str | None = None) -> dict[str, int]:
        session_clause = " AND session_id = ?" if session_id else ""
        parameters = (str(user_id), str(session_id)) if session_id else (str(user_id),)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS events,
                    COUNT(DISTINCT CASE WHEN event_type IN ('play_start', 'jio_play') THEN track_id END) AS tracks_played,
                    COUNT(CASE WHEN event_type IN ('like', 'jio_like') THEN 1 END) AS likes,
                    COUNT(CASE WHEN event_type IN ('skip', 'jio_skip') THEN 1 END) AS skips,
                    COALESCE(SUM(duration_ms), 0) AS listening_ms
                FROM behavior_events WHERE user_id = ?{session_clause}
                """,
                parameters,
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}
