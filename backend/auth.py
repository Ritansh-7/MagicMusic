from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time


def _secret() -> bytes:
    value = os.getenv("COZYBEATS_SESSION_SECRET") or os.getenv("TUNEMATCH_SESSION_SECRET")
    return (value or "local-dev-change-this-secret").encode()


def issue_token(user_id: str, email: str, ttl: int = 60 * 60 * 24 * 14) -> str:
    expiry = int(time.time()) + ttl
    payload = f"{user_id}:{expiry}:{email}".encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    signature = hmac.new(_secret(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def read_token(token: str) -> dict[str, str] | None:
    try:
        encoded, signature = token.split(".", 1)
        expected = hmac.new(_secret(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        user_id, expiry, email = base64.urlsafe_b64decode(encoded + "===").decode().split(":", 2)
        if int(expiry) < int(time.time()):
            return None
        return {"user_id": user_id, "email": email}
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
