from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class Cache:
    """Redis-first cache with a bounded in-process fallback for Windows dev."""

    def __init__(self) -> None:
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.RLock()
        self._redis = None
        url = os.getenv("REDIS_URL", "").strip()
        if url:
            try:
                import redis
                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._redis.get(key) if self._redis else None
            if raw is None:
                with self._lock:
                    entry = self._memory.get(key)
                    if not entry:
                        return None
                    expires, raw = entry
                    if expires < time.time():
                        self._memory.pop(key, None)
                        return None
            return json.loads(raw)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: int = 300) -> None:
        raw = json.dumps(value, default=str)
        try:
            if self._redis:
                self._redis.setex(key, ttl, raw)
                return
        except Exception:
            pass
        with self._lock:
            if len(self._memory) > 2048:
                self._memory.pop(next(iter(self._memory)))
            self._memory[key] = (time.time() + ttl, raw)

