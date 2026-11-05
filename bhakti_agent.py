"""
Bhakti Recommendation Agent - Fast devotional song recommendations.
Combines skill file + LLM + math-based mood scoring.
"""

from __future__ import annotations

import re
from typing import Any

from bhakti_skill import (
    bhakti_score,
    infer_bhakti_mood,
    is_bhakti_skill_match,
    resolve_deity,
)


class BhaktiAgent:
    def __init__(self, llm_classifier=None, llm_enabled: bool = True):
        self.llm = llm_classifier
        self.llm_enabled = llm_enabled
        self._llm_cache = {}

    def recommend(self, seed, candidates, top_n=30, min_score=0.30):
        if not candidates:
            return []
        seed_profile = self._classify_item(seed)
        scored = []
        for item in candidates:
            if self._same_recording(seed, item):
                continue
            # Devotional seeds surface devotional-only candidates: block Bollywood
            # romantic/party/energetic leaks and require a devotional signal.
            if self._seed_is_devotional(seed, seed_profile):
                if self._is_romantic_item(item):
                    continue
                if not self._is_devotional_item(item):
                    continue
            score = self._score_candidate(item, seed, seed_profile)
            if score >= min_score:
                item = dict(item)
                item["bhakti_score"] = round(score, 4)
                item["bhakti_reason"] = self._bhakti_reason(item, seed, score)
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        seen_keys = set()
        results = []
        for score, item in scored:
            key = self._recording_key(item)
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            results.append(item)
            if len(results) >= top_n:
                break
        return results

    def is_bhakti(self, item):
        if is_bhakti_skill_match(item):
            return True
        if self.llm_enabled and self.llm:
            profile = self._classify_item(item)
            if profile.get("primary_intent") == "devotional":
                return True
            if profile.get("religion") in {"hindu", "sikh", "jain", "buddhist"}:
                return True
        return False

    def get_mood(self, item):
        return infer_bhakti_mood(item)

    def get_deity(self, item):
        return resolve_deity(item)

    _ROMANTIC_TERMS = {
        "love", "pyaar", "pyar", "ishq", "mohabbat", "jaan", "sanam", "sajan",
        "saiyaara", "milan", "romantic", "yaar", "dilwale", "saajan", "sajna",
        "dil to", "pagal hai", "chura ke dil", "tumse milne", "mere khwabon",
        "ho gaya hai tujhko", "pehli pehli bar", "dekha hai pehli", "tujhe dekha to",
        "mera dil", "tera dil", "dil hai", "dil ne", "dil se", "dil ki",
    }

    def _is_romantic_item(self, item: dict) -> bool:
        text = " ".join(
            str(item.get(field, "")) for field in ("title", "album", "artists", "intent_labels")
        ).casefold()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        if self._ROMANTIC_TERMS & tokens:
            return True
        return any(term in text for term in self._ROMANTIC_TERMS if " " in term)

    def _seed_is_devotional(self, seed: dict, seed_profile: dict) -> bool:
        if seed_profile.get("primary_intent") == "devotional":
            return True
        return is_bhakti_skill_match(seed)

    def _is_devotional_item(self, item: dict) -> bool:
        if is_bhakti_skill_match(item):
            return True
        if self.llm_enabled and self.llm:
            profile = self._classify_item(item)
            if profile.get("primary_intent") == "devotional":
                return True
            if profile.get("religion") in {"hindu", "sikh", "jain", "buddhist"}:
                return True
        return False

    def _score_candidate(self, item, seed, seed_profile):
        score = bhakti_score(item, seed)
        if self.llm_enabled and self.llm:
            item_profile = self._classify_item(item)
            seed_intent = seed_profile.get("primary_intent", "unknown")
            item_intent = item_profile.get("primary_intent", "unknown")
            seed_religion = seed_profile.get("religion", "unknown")
            item_religion = item_profile.get("religion", "unknown")
            if seed_intent == "devotional" and item_intent == "devotional":
                score = min(1.0, score + 0.25)
            if (
                seed_religion
                and item_religion
                and seed_religion not in {"unknown", "generic"}
                and item_religion not in {"unknown", "generic"}
                and seed_religion == item_religion
            ):
                score = min(1.0, score + 0.15)
            if seed_intent == "devotional" and item_intent in {"romantic", "party", "energetic"}:
                return 0.0
        item_mood, item_conf = infer_bhakti_mood(item)
        seed_mood, seed_conf = infer_bhakti_mood(seed)
        if (
            item_mood != "unknown"
            and seed_mood != "unknown"
            and item_mood == seed_mood
            and item_conf > 0.4
            and seed_conf > 0.4
        ):
            score = min(1.0, score + 0.10)
        return min(1.0, score)

    def _bhakti_reason(self, item, seed, score):
        reasons = []
        if is_bhakti_skill_match(item):
            reasons.append("Bhakti skill match")
        seed_deity = resolve_deity(seed)
        item_deity = resolve_deity(item)
        if seed_deity and item_deity and seed_deity == item_deity:
            reasons.append("Same deity: " + seed_deity)
        item_mood, item_conf = infer_bhakti_mood(item)
        seed_mood, seed_conf = infer_bhakti_mood(seed)
        if item_mood != "unknown" and seed_mood != "unknown" and item_mood == seed_mood:
            reasons.append("Same mood: " + item_mood)
        if score >= 0.75:
            reasons.append("Strong devotional match")
        elif score >= 0.50:
            reasons.append("Good devotional match")
        else:
            reasons.append("Bhakti match")
        return "; ".join(reasons)

    def _classify_item(self, item):
        key = self._cache_key(item)
        if key in self._llm_cache:
            return self._llm_cache[key]
        profile = {
            "primary_intent": "unknown",
            "intent_confidence": 0.0,
            "religion": "unknown",
            "mood": "unknown",
        }
        if self.llm_enabled and self.llm:
            try:
                if hasattr(self.llm, "classify_item"):
                    result = self.llm.classify_item(item)
                elif hasattr(self.llm, "classify_text"):
                    text = " ".join(str(item.get(f, "")) for f in ("title", "artists", "album", "language"))
                    result = self.llm.classify_text(text)
                else:
                    result = {}
                if result:
                    profile = {
                        "primary_intent": str(result.get("intent", result.get("primary_intent", "unknown"))),
                        "intent_confidence": float(result.get("intent_confidence", 0.0) or 0.0),
                        "religion": str(result.get("religion", "unknown")),
                        "mood": str(result.get("mood", "unknown")),
                    }
            except Exception:
                pass
        self._llm_cache[key] = profile
        return profile

    @staticmethod
    def _cache_key(item):
        title = str(item.get("title", "")).strip().casefold()
        artists = str(item.get("artists", "")).strip().casefold()
        return title + "|" + artists

    @staticmethod
    def _same_recording(a, b):
        key_a = BhaktiAgent._cache_key(a)
        key_b = BhaktiAgent._cache_key(b)
        return key_a == key_b and bool(key_a)

    @staticmethod
    def _recording_key(item):
        isrc = str(item.get("isrc", "")).strip().casefold()
        if isrc:
            return "isrc:" + isrc
        title = re.sub(r"[^a-z0-9]+", " ", str(item.get("title", "")).casefold()).strip()
        artists = re.sub(r"[^a-z0-9]+", " ", str(item.get("artists", "")).casefold()).strip()
        return (title + "|" + artists) if title else artists


def quick_bhakti_recommend(seed, candidates, top_n=30, llm_classifier=None):
    agent = BhaktiAgent(llm_classifier=llm_classifier, llm_enabled=llm_classifier is not None)
    return agent.recommend(seed, candidates, top_n=top_n)


def is_devotional(item, llm_classifier=None):
    from bhakti_skill import is_bhakti_skill_match
    if is_bhakti_skill_match(item):
        return True
    if llm_classifier:
        try:
            result = llm_classifier.classify_item(item)
            if result and result.get("primary_intent") == "devotional":
                return True
        except Exception:
            pass
    return False
