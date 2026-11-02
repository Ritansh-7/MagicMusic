import pandas as pd
import pytest
from recommender import (
    SpotifyRecommendationEngine,
    HybridRecommendationEngine,
    _interleave_recommendation_contexts,
    _balance_seed_artist_mix,
)

def sample_catalog():
    return pd.DataFrame([
        {
            "track_id": "seed",
            "track_name": "Pehla Nasha",
            "artists": "Udit Narayan, Sadhana Sargam",
            "album_name": "Jo Jeeta Wohi Sikandar",
            "track_genre": "filmi, bollywood",
            "popularity": 75,
            "duration_ms": 290000,
            "explicit": False,
            "danceability": 0.55,
            "energy": 0.48,
            "key": 2,
            "loudness": -8.5,
            "mode": 1,
            "speechiness": 0.04,
            "acousticness": 0.45,
            "instrumentalness": 0.01,
            "liveness": 0.12,
            "valence": 0.62,
            "tempo": 105,
            "time_signature": 4,
        }
    ])


def test_era_distribution_90s_with_touches():
    """90s song should yield ~90% 90s, ~5% adjacent 2000s, ~5% adjacent 80s."""
    candidates = (
        [{"song_id": f"nineties-{i}", "year": 1990 + (i % 10), "artists": f"Singer {i}"} for i in range(25)]
        + [{"song_id": f"twenties-{i}", "year": 2000 + (i % 5), "artists": f"Singer {25+i}"} for i in range(5)]
        + [{"song_id": f"eighties-{i}", "year": 1980 + (i % 5), "artists": f"Singer {30+i}"} for i in range(5)]
    )

    ordered = _interleave_recommendation_contexts(candidates, seed_year=1995)
    first_twenty = ordered[:20]
    decades = [int(item["year"]) // 10 * 10 for item in first_twenty]

    # In 20 slots: 18 should be 1990s (90%), 1 2000s (5%), 1 1980s (5%)
    assert decades.count(1990) == 18, f"Expected 18 from 1990s, got {decades.count(1990)}"
    assert decades.count(2000) == 1, f"Expected 1 from 2000s, got {decades.count(2000)}"
    assert decades.count(1980) == 1, f"Expected 1 from 1980s, got {decades.count(1980)}"


def test_era_distribution_80s_with_touches():
    """80s song should yield ~90% 80s, ~5% 90s touch, ~5% 70s touch."""
    candidates = (
        [{"song_id": f"eighties-{i}", "year": 1980 + (i % 10), "artists": f"Singer {i}"} for i in range(25)]
        + [{"song_id": f"nineties-{i}", "year": 1990 + (i % 5), "artists": f"Singer {25+i}"} for i in range(5)]
        + [{"song_id": f"seventies-{i}", "year": 1970 + (i % 5), "artists": f"Singer {30+i}"} for i in range(5)]
    )

    ordered = _interleave_recommendation_contexts(candidates, seed_year=1984)
    first_twenty = ordered[:20]
    decades = [int(item["year"]) // 10 * 10 for item in first_twenty]

    assert decades.count(1980) == 18, f"Expected 18 from 1980s, got {decades.count(1980)}"
    assert decades.count(1990) == 1, f"Expected 1 from 1990s, got {decades.count(1990)}"
    assert decades.count(1970) == 1, f"Expected 1 from 1970s, got {decades.count(1970)}"


def test_singer_diversity_distribution():
    """Playing singer should be 10-20%, different singers should be 60-70%+."""
    seed = {"artists": "Arijit Singh", "title": "Tum Hi Ho"}
    candidates = (
        [{"song_id": f"arijit-{i}", "artists": "Arijit Singh", "title": f"Song {i}"} for i in range(15)]
        + [{"song_id": f"other-{i}", "artists": f"Different Singer {i}", "title": f"Other {i}"} for i in range(25)]
    )

    balanced = _balance_seed_artist_mix(candidates, seed)
    queue = balanced[:30]

    arijit_count = sum("arijit singh" in item["artists"].lower() for item in queue)
    different_singers_count = sum("arijit singh" not in item["artists"].lower() for item in queue)

    arijit_pct = arijit_count / len(queue)
    different_pct = different_singers_count / len(queue)

    # 10% to 20% same singer
    assert 0.10 <= arijit_pct <= 0.20, f"Expected 10-20% same singer, got {arijit_pct*100:.1f}%"
    # 60% to 70%+ different singers
    assert different_pct >= 0.60, f"Expected >=60% different singers, got {different_pct*100:.1f}%"


def test_mood_sentiment_primary_driver():
    """Matching mood & sentiment must rank higher, and conflicting mood is filtered out."""
    catalog = sample_catalog()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))

    # Romantic 90s seed
    romantic_vector = [0.45, 0.45, 0.65, 0.50, 0.50, 0.05, 0.05, 0.15, 0.75]
    seed = {
        "title": "Tujhe Dekha Toh",
        "artists": "Kumar Sanu, Lata Mangeshkar",
        "album": "DDLJ",
        "year": 1995,
        "language": "Hindi",
        "mood_vector": romantic_vector,
        "mood_label": "romantic",
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["romantic"],
    }

    # Candidate A: Same era 1995, different artist, same romantic mood
    # Candidate B: Same era 1995, same artist (Kumar Sanu), but energetic/fight intent & dissimilar mood
    candidates = [
        {
            "song_id": "other-romantic",
            "title": "Bahon Ke Darmiyan",
            "artists": "Hariharan, Alka Yagnik",
            "album": "Khamoshi",
            "year": 1996,
            "language": "Hindi",
            "mood_vector": romantic_vector,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "sanu-energetic-fight",
            "title": "Fight Dhamaka Song",
            "artists": "Kumar Sanu",
            "album": "Action Film",
            "year": 1995,
            "language": "Hindi",
            "mood_vector": [0.90, 0.85, 0.30, 0.70, 0.05, 0.01, 0.20, 0.30, 0.90],
            "mood_label": "energetic",
            "mood_confidence": 0.90,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["energetic"],
        },
    ]

    ranked = hybrid.rank_jio_results(candidates, "mood-test-user", catalog, [], seed=seed)

    # The romantic song is selected, and conflicting mood candidate is blocked
    assert len(ranked) >= 1
    assert ranked[0]["song_id"] == "other-romantic"
    assert ranked[0]["recommendation_features"]["mood"] > 0.95
    assert "sanu-energetic-fight" not in [x["song_id"] for x in ranked]
