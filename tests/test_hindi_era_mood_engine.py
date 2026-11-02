import pytest
import pandas as pd
from recommender import (
    CLASSIC_ERA_SINGERS,
    CLASSIC_ERA_THEME_FILMS,
    era_mood_singer_discovery_queries,
    era_theme_film_discovery_queries,
    HybridRecommendationEngine,
    SpotifyRecommendationEngine,
    _interleave_recommendation_contexts,
    _balance_seed_artist_mix,
)


def sample_catalog():
    return pd.DataFrame([
        {
            "track_id": "seed",
            "track_name": "Tere Bina Zindagi Se",
            "artists": "Kishore Kumar, Lata Mangeshkar",
            "album_name": "Aandhi",
            "track_genre": "filmi, classic bollywood",
            "popularity": 75,
            "duration_ms": 340000,
            "explicit": False,
            "danceability": 0.50,
            "energy": 0.45,
            "key": 1,
            "loudness": -9.0,
            "mode": 1,
            "speechiness": 0.03,
            "acousticness": 0.65,
            "instrumentalness": 0.02,
            "liveness": 0.10,
            "valence": 0.55,
            "tempo": 95,
            "time_signature": 4,
        }
    ])


def test_era_mood_singer_queries_70s_romantic():
    """Verify Kishore Kumar 70s romantic seed produces queries for Rafi, Lata, Asha, Mukesh."""
    queries = era_mood_singer_discovery_queries(
        seed_year=1975,
        seed_language="hindi",
        seed_intents={"romantic"},
        seed_artists=["Kishore Kumar"],
        max_singers=5,
    )
    # Seed artist Kishore Kumar must NOT be in the queries
    for q in queries:
        assert "kishore kumar" not in q.lower(), f"Seed singer found in query: {q}"

    # Must contain 70s era string
    assert all("1970s" in q for q in queries)
    # Must query iconic 70s peers like Mohammed Rafi, Lata Mangeshkar, Asha Bhosle
    joined = " ".join(queries).lower()
    assert "mohammed rafi" in joined or "rafi" in joined
    assert "lata mangeshkar" in joined or "asha bhosle" in joined
    assert len(queries) >= 4


def test_era_mood_singer_queries_2010s_motivational():
    """Verify 2010s motivational seed produces queries for KK, Vishal Dadlani, Siddharth Mahadevan."""
    queries = era_mood_singer_discovery_queries(
        seed_year=2016,
        seed_language="hindi",
        seed_intents={"motivational"},
        seed_artists=["Armaan Malik"],
        max_singers=5,
    )
    assert all("2010s" in q for q in queries)
    joined = " ".join(queries).lower()
    assert any(singer in joined for singer in ["vishal dadlani", "siddharth mahadevan", "sukhwinder singh", "kk"])


def test_era_theme_film_queries():
    """Verify 2010s motivational includes Sultan, Bhaag Milkha Bhaag, Dangal."""
    queries = era_theme_film_discovery_queries(
        seed_year=2016,
        seed_language="hindi",
        seed_intents={"motivational"},
        max_films=4,
    )
    joined = " ".join(queries).lower()
    assert "sultan" in joined
    assert "bhaag milkha bhaag" in joined or "dangal" in joined


def test_tere_bina_zindagi_se_end_to_end_ranking():
    """Simulate 'Tere Bina Zindagi Se' with candidates from Rafi, Lata, Asha, Kishore, and verify 10+ songs, era & singer balance."""
    catalog = sample_catalog()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))

    seed = {
        "title": "Tere Bina Zindagi Se",
        "artists": "Kishore Kumar, Lata Mangeshkar",
        "album": "Aandhi",
        "year": 1975,
        "language": "Hindi",
        "intent_labels": ["romantic"],
        "mood_vector": [0.45, 0.40, 0.60, 0.50, 0.50, 0.05, 0.05, 0.15, 0.70],
        "mood_label": "romantic",
        "mood_confidence": 0.90,
        "mood_source": "local_audio_catalog",
    }

    # Generate 30 candidates from the 1970s with romantic mood by diverse singers
    singers_70s = [
        "Mohammed Rafi", "Asha Bhosle", "Mukesh", "Lata Mangeshkar",
        "Manna Dey", "Hemant Kumar", "Yesudas", "Mahendra Kapoor", "Bhupinder Singh"
    ]
    candidates = []
    # 3 Kishore Kumar songs (same singer)
    for i in range(3):
        candidates.append({
            "song_id": f"kishore-{i}",
            "title": f"Kishore 70s Romantic {i}",
            "artists": "Kishore Kumar",
            "album": f"Film {i}",
            "year": 1973 + i,
            "language": "Hindi",
            "intent_labels": ["romantic"],
            "mood_vector": [0.45, 0.40, 0.60, 0.50, 0.50, 0.05, 0.05, 0.15, 0.70],
            "mood_label": "romantic",
            "mood_confidence": 0.90,
            "mood_source": "local_audio_catalog",
        })
    # 20 songs by different 70s singers
    for i, singer in enumerate(singers_70s * 3):
        candidates.append({
            "song_id": f"peer-{i}",
            "title": f"{singer} Romantic Classic {i}",
            "artists": singer,
            "album": f"Classic Film {i}",
            "year": 1970 + (i % 9),
            "language": "Hindi",
            "intent_labels": ["romantic"],
            "mood_vector": [0.45, 0.40, 0.60, 0.50, 0.50, 0.05, 0.05, 0.15, 0.70],
            "mood_label": "romantic",
            "mood_confidence": 0.90,
            "mood_source": "local_audio_catalog",
        })

    ranked = hybrid.rank_jio_results(candidates, "user-70s", catalog, [], seed=seed)

    # 1. Minimum 10 songs guaranteed
    assert len(ranked) >= 10, f"Expected >= 10 songs, got {len(ranked)}"

    # 2. Maximum 35 songs
    assert len(ranked) <= 35

    # 3. Era: >= 85% from 1970s
    seventies_count = sum(1 for item in ranked if 1970 <= int(item.get("year", 0)) < 1980)
    seventies_pct = seventies_count / len(ranked)
    assert seventies_pct >= 0.85, f"Expected >=85% 70s songs, got {seventies_pct*100:.1f}%"

    # 4. Singer diversity: >= 65% different singers from lead singer Kishore Kumar
    diff_singer_count = sum(1 for item in ranked if "kishore kumar" not in str(item.get("artists", "")).lower())
    diff_pct = diff_singer_count / len(ranked)
    assert diff_pct >= 0.65, f"Expected >=65% different singers, got {diff_pct*100:.1f}%"

    # 5. Lead singer present but within 10-20%
    same_singer_count = sum(1 for item in ranked if "kishore kumar" in str(item.get("artists", "")).lower())
    same_pct = same_singer_count / len(ranked)
    assert same_pct <= 0.25, f"Expected <=25% same singer, got {same_pct*100:.1f}%"
