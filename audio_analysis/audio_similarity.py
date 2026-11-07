from __future__ import annotations

import math


def vector(features: dict) -> list[float]:
    return [
        float(features.get("energy", 0.0) or 0.0),
        float(features.get("valence", 0.0) or 0.0),
        float(features.get("danceability", 0.0) or 0.0),
        float(features.get("tempo", 0.0) or 0.0) / 200.0,
        float(features.get("acousticness", 0.0) or 0.0),
    ]


def cosine(a: dict, b: dict) -> float:
    va, vb = vector(a), vector(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
