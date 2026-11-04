import math
from datetime import datetime

def session_profile(events: list[dict]) -> dict:
    if not events:
        return {}
    weighted = {}
    total_weight = 0.0
    now = datetime.utcnow().timestamp()
    for event in events:
        weight = float(event.get("weight", 1.0))
        age = max(0.0, now - float(event.get("timestamp", now)))
        recency = math.exp(-age / 3600.0)
        w = weight * recency
        profile = event.get("profile", {})
        for key, value in profile.items():
            weighted[key] = weighted.get(key, 0.0) + w
        total_weight += w
    if not total_weight:
        return {}
    return {key: value / total_weight for key, value in weighted.items()}