from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets as token_secrets
import time
from typing import Any
from urllib.parse import urlencode

import requests


AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_URL = "https://api.spotify.com/v1"


class SpotifyOAuth:
    """Small server-side Spotify OAuth/Web API client for Streamlit."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.redirect_uri = redirect_uri.strip()

    @classmethod
    def from_environment(cls) -> "SpotifyOAuth | None":
        """Read credentials from Streamlit secrets or environment variables."""
        try:
            import streamlit as st

            client_id = st.secrets.get("SPOTIFY_CLIENT_ID", "")
            client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", "")
            redirect_uri = st.secrets.get("SPOTIFY_REDIRECT_URI", "")
        except Exception:
            client_id = client_secret = redirect_uri = ""
        client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID", "")
        client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET", "")
        redirect_uri = redirect_uri or os.getenv(
            "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501"
        )
        if not client_id or not client_secret:
            return None
        return cls(client_id, client_secret, redirect_uri)

    @property
    def scopes(self) -> str:
        return " ".join(
            [
                "user-read-private",
                "user-top-read",
                "user-read-recently-played",
                "user-library-read",
            ]
        )

    def authorization_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "state": state,
            # Force a fresh consent screen while developing so newly-added
            # scopes are actually granted to the next access token.
            "show_dialog": "true",
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    def exchange_code(self, code: str) -> dict[str, Any]:
        credentials = f"{self.client_id}:{self.client_secret}".encode()
        basic = base64.b64encode(credentials).decode()
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            },
            headers={"Authorization": f"Basic {basic}"},
            timeout=20,
        )
        response.raise_for_status()
        token = response.json()
        token["expires_at"] = time.time() + int(token.get("expires_in", 3600)) - 60
        return token

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        credentials = f"{self.client_id}:{self.client_secret}".encode()
        basic = base64.b64encode(credentials).decode()
        response = requests.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            headers={"Authorization": f"Basic {basic}"},
            timeout=20,
        )
        response.raise_for_status()
        token = response.json()
        token.setdefault("refresh_token", refresh_token)
        token["expires_at"] = time.time() + int(token.get("expires_in", 3600)) - 60
        return token

    def get(self, access_token: str, path: str, **params: Any) -> dict[str, Any]:
        response = requests.get(
            f"{API_URL}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


def new_oauth_state(signing_secret: str) -> str:
    """Create a short-lived, signed OAuth state that survives a new session."""
    payload = f"{int(time.time())}.{token_secrets.token_urlsafe(24)}"
    signature = hmac.new(
        signing_secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}.{signature}"


def verify_oauth_state(state: str, signing_secret: str, max_age_seconds: int = 600) -> bool:
    """Validate callback state without relying only on Streamlit session state."""
    try:
        timestamp, nonce, signature = state.split(".", 2)
        payload = f"{timestamp}.{nonce}"
        expected = hmac.new(
            signing_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return (
            hmac.compare_digest(signature, expected)
            and 0 <= time.time() - int(timestamp) <= max_age_seconds
        )
    except (AttributeError, TypeError, ValueError):
        return False


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
