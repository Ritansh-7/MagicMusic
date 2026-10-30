from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # PostgreSQL is only required when DATABASE_URL is configured.
    psycopg = None
    dict_row = None

try:
    import pymysql
except ImportError:  # MySQL is optional for local SQLite development.
    pymysql = None


class _MySQLConnection:
    """Small execute-compatible wrapper matching the SQLite/Postgres paths."""

    def __init__(self, database_url: str):
        parsed = urlparse(database_url)
        self.connection = pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=(parsed.path or "").lstrip("/"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            # Each user/event write is independent. Autocommit avoids holding
            # a transaction open across a remote Streamlit rerun.
            autocommit=True,
            # Do not let a sleeping/unreachable cloud database freeze a
            # Streamlit rerun indefinitely. Aiven connections normally finish
            # well below this; these are only a safety bound for cold starts.
            connect_timeout=5,
            read_timeout=6,
            write_timeout=6,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type:
                # The server can close the socket while a query is in flight.
                # Never mask the original database error with a second error
                # from rollback on that already-dead connection.
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            else:
                try:
                    self.connection.commit()
                except Exception:
                    pass
        finally:
            try:
                self.connection.close()
            except Exception:
                pass

    def execute(self, query: str, parameters=()):
        cursor = self.connection.cursor()
        cursor.execute(query, parameters)
        return cursor


class UserStore:
    """Local account and behavior store for the demo app.

    Passwords are salted PBKDF2 hashes; raw passwords are never stored.
    """

    def __init__(self, path: str | Path | None = None, database_url: str | None = None):
        self.database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        self.database_kind = self._database_kind(self.database_url)
        if self.database_kind == "postgres" and psycopg is None:
            raise RuntimeError("PostgreSQL DATABASE_URL is configured, but psycopg is not installed.")
        if self.database_kind == "mysql" and pymysql is None:
            raise RuntimeError("MySQL DATABASE_URL is configured, but pymysql is not installed.")
        self.path = Path(path or os.getenv("TUNEMATCH_DB_PATH", "tune_match_events.sqlite3"))
        if not self.database_url:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def uses_postgres(self) -> bool:
        return self.database_kind == "postgres"

    @property
    def uses_mysql(self) -> bool:
        return self.database_kind == "mysql"

    @staticmethod
    def _database_kind(database_url: str) -> str:
        scheme = urlparse(database_url).scheme.lower()
        if scheme.startswith("mysql"):
            return "mysql"
        if scheme.startswith("postgres") or scheme.startswith("postgresql"):
            return "postgres"
        return "sqlite"

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if (self.uses_postgres or self.uses_mysql) else query

    def _connect(self):
        if self.uses_postgres:
            return psycopg.connect(self.database_url, row_factory=dict_row)
        if self.uses_mysql:
            return _MySQLConnection(self.database_url)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            if self.uses_postgres:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS behavior_events (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id),
                        track_id TEXT,
                        event_type TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        duration_ms BIGINT DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                return
            if self.uses_mysql:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        email VARCHAR(320) UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS behavior_events (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        user_id BIGINT NOT NULL,
                        track_id TEXT,
                        event_type VARCHAR(64) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        duration_ms BIGINT DEFAULT 0,
                        metadata_json LONGTEXT NOT NULL,
                        created_at VARCHAR(64) NOT NULL,
                        INDEX idx_behavior_user (user_id),
                        INDEX idx_behavior_created (created_at),
                        CONSTRAINT fk_behavior_user FOREIGN KEY (user_id) REFERENCES users(id)
                    ) ENGINE=InnoDB
                    """
                )
                return
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
                query = "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)"
                if self.uses_postgres:
                    query += " RETURNING id"
                cursor = connection.execute(
                    self._sql(query),
                    (email, self._hash_password(password), datetime.now(timezone.utc).isoformat()),
                )
                if self.uses_postgres:
                    return True, str(cursor.fetchone()["id"])
                return True, str(cursor.lastrowid)
        except (
            sqlite3.IntegrityError,
            getattr(getattr(psycopg, "errors", None), "UniqueViolation", RuntimeError),
            getattr(getattr(pymysql, "err", None), "IntegrityError", RuntimeError),
        ):
            return False, "An account with this email already exists."

    def authenticate(self, email: str, password: str) -> tuple[bool, str]:
        with self._connect() as connection:
            row = connection.execute(
                self._sql("SELECT id, email, password_hash FROM users WHERE email = ?"),
                (email.strip().lower(),),
            ).fetchone()
        if not row or not self._verify_password(password, row["password_hash"]):
            return False, "Email or password is incorrect."
        return True, str(row["id"])

    def user_email(self, user_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                self._sql("SELECT email FROM users WHERE id = ?"),
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
                self._sql("""
                INSERT INTO behavior_events
                    (user_id, track_id, event_type, session_id, duration_ms, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """),
                (str(user_id), track_id, event_type, session_id, int(duration_ms), metadata_json,
                 datetime.now(timezone.utc).isoformat()),
            )

    def liked_track_ids(self, user_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                self._sql("SELECT DISTINCT track_id FROM behavior_events WHERE user_id = ? AND event_type = 'like' AND track_id IS NOT NULL"),
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
                self._sql(f"""
                SELECT
                    COUNT(*) AS events,
                    COUNT(DISTINCT CASE WHEN event_type IN ('play_start', 'jio_play') THEN track_id END) AS tracks_played,
                    COUNT(CASE WHEN event_type IN ('like', 'jio_like') THEN 1 END) AS likes,
                    COUNT(CASE WHEN event_type IN ('skip', 'jio_skip') THEN 1 END) AS skips,
                    COALESCE(SUM(duration_ms), 0) AS listening_ms
                FROM behavior_events WHERE user_id = ?{session_clause}
                """),
                parameters,
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}
