from __future__ import annotations

import json
import os

import requests


def spotify_audio_features(track_id: str, client_id: str | None = None, client_secret: str | None = None) -> dict[str, Any]:
    """Fetch audio features from Spotify Web API for a track ID."""
    client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return {}
    try:
        token_resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=20,
        )
        token_resp.raise_for_status()
        token = token_resp.json().get("access_token")
        if not token:
            return {}
        feat_resp = requests.get(
            f"https://api.spotify.com/v1/audio-features/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if not feat_resp.ok:
            return {}
        data = feat_resp.json() or {}
        return {
            "tempo": data.get("tempo", 0.0),
            "energy": data.get("energy", 0.0),
            "valence": data.get("valence", 0.0),
            "danceability": data.get("danceability", 0.0),
            "acousticness": data.get("acousticness", 0.0),
            "instrumentalness": data.get("instrumentalness", 0.0),
        }
    except Exception:
        return {}


def fetch_and_cache_spotify(track_id: str, spotify_track_id: str, client_id: str | None = None, client_secret: str | None = None) -> dict[str, Any]:
    existing = get_cached_features(track_id)
    if existing:
        return existing
    features = spotify_audio_features(spotify_track_id, client_id, client_secret)
    if features:
        cache_features(track_id, features)
    return features


from pathlib import Path
from typing import Any

CACHE_PATH = Path("audio_feature_cache.json")


def _load_cache() -> dict[str, dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def extract_features(audio_path: str) -> dict[str, Any]:
    """Extract audio features from an authorized local audio file using Librosa."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return {}

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception:
        return {}

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = [float(x) for x in np.mean(mfcc, axis=1)]
    except Exception:
        return {}

    return {
        "tempo": float(tempo) if tempo is not None else 0.0,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zcr,
        "mfcc": mfcc_mean,
    }


def get_cached_features(track_id: str) -> dict[str, Any] | None:
    cache = _load_cache()
    return cache.get(str(track_id))


def cache_features(track_id: str, features: dict[str, Any]) -> None:
    cache = _load_cache()
    cache[str(track_id)] = features
    _save_cache(cache)


def extract_and_cache(track_id: str, audio_path: str) -> dict[str, Any]:
    existing = get_cached_features(track_id)
    if existing:
        return existing
    features = extract_features(audio_path)
    if features:
        cache_features(track_id, features)
    return features
