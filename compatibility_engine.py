def compatibility(seed_profile: dict, candidate_profile: dict) -> float:
    weights = {
        "intent": 0.25,
        "theme": 0.20,
        "mood": 0.15,
        "genre": 0.10,
        "language": 0.15,
        "religion": 0.15,
    }
    score = 0.0
    for key, weight in weights.items():
        seed_val = seed_profile.get(key)
        cand_val = candidate_profile.get(key)
        if seed_val and cand_val and seed_val == cand_val:
            score += weight
        elif seed_val == "unknown" or cand_val == "unknown":
            score += weight * 0.25
    return round(score, 4)
EOF
python -m py_compile compatibility_engine.py; if ($?) { echo "compile_ok" }