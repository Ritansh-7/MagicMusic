def build_features(seed_profile: dict, candidate_profile: dict) -> dict:
    return {
        "intent_similarity": 1.0 if seed_profile.get("primary_intent") == candidate_profile.get("primary_intent") else 0.0,
        "theme_similarity": 1.0 if seed_profile.get("theme") == candidate_profile.get("theme") else 0.0,
        "language_similarity": 1.0 if seed_profile.get("language") == candidate_profile.get("language") else 0.0,
    }