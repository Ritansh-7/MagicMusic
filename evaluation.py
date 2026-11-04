def precision_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / min(k, len(recommended))