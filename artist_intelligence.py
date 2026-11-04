def artist_affinity(candidates: list[dict], seed_profile: dict) -> dict:
    scores = {}
    for item in candidates:
        artist = str(item.get("artists", "")).strip()
        if not artist:
            continue
        scores.setdefault(artist, 0.0)
        scores[artist] += 1.0
    total = sum(scores.values()) or 1.0
    return {artist: value / total for artist, value in scores.items()}