from __future__ import annotations

from .librosa_analyzer import analyze as librosa_analyze
from .essentia_analyzer import analyze as essentia_analyze
from .music_prediction_models import predict
from .audio_feature_cache import get as cache_get, put as cache_put


def run(track_id: str, audio_path: str) -> dict:
    cached = cache_get(track_id)
    if cached:
        return cached
    features = {}
    features.update(librosa_analyze(audio_path) or {})
    features.update(essentia_analyze(audio_path) or {})
    features.update(predict(features) or {})
    if features:
        cache_put(track_id, features)
    return features
