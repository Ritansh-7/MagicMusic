from __future__ import annotations

import re
from difflib import SequenceMatcher


def title_query_match_score(title: str, query: str) -> float:
    """Score a title against a query, tolerating small transliteration variants."""
    normalized_title = re.sub(r"[^a-z0-9]+", " ", str(title).casefold()).strip()
    normalized_query = re.sub(r"[^a-z0-9]+", " ", str(query).casefold()).strip()
    title_tokens = normalized_title.split()
    query_tokens = normalized_query.split()
    if not title_tokens or not query_tokens:
        return 0.0
    if normalized_query == normalized_title:
        return 1.0
    if f" {normalized_query} " in f" {normalized_title} ":
        return 1.0
    if len(title_tokens) < len(query_tokens):
        return 0.0

    best_score = 0.0
    width = len(query_tokens)
    for start in range(len(title_tokens) - width + 1):
        similarities = [
            SequenceMatcher(None, query_token, title_token).ratio()
            for query_token, title_token in zip(
                query_tokens, title_tokens[start:start + width]
            )
        ]
        if min(similarities) >= 0.72:
            best_score = max(best_score, sum(similarities) / width)
    return best_score


def prioritize_title_matches(
    items: list[dict],
    query: str,
    minimum_score: float = 0.82,
) -> list[dict]:
    """Move strong song-title matches ahead of broader album/artist results."""
    scored = [
        (title_query_match_score(item.get("title", ""), query), index, item)
        for index, item in enumerate(items)
    ]
    matching = [
        (score, index, item)
        for score, index, item in scored
        if score >= minimum_score
    ]
    matching.sort(key=lambda result: (-result[0], result[1]))
    matched_ids = {id(item) for _, _, item in matching}
    remaining = [item for item in items if id(item) not in matched_ids]
    return [item for _, _, item in matching] + remaining
