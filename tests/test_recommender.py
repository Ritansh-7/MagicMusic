import json
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import (
    HybridRecommendationEngine,
    SpotifyRecommendationEngine,
    _canonical_recording_key,
    _infer_candidate_intents,
    _interleave_recommendation_contexts,
    recommendation_history_keys,
    recommendation_title_key,
)
from ranking_model import _feature_row


def sample_data():
    return pd.DataFrame([
        {"track_id":"a", "track_name":"Song A", "artists":"Artist One", "album_name":"Album",
         "track_genre":"pop, dance pop", "popularity":80, "duration_ms":200000, "explicit":False,
         "danceability":0.8, "energy":0.8, "key":1, "loudness":-5, "mode":1,
         "speechiness":0.05, "acousticness":0.1, "instrumentalness":0.0, "liveness":0.1,
         "valence":0.8, "tempo":120, "time_signature":4},
        {"track_id":"b", "track_name":"Song B", "artists":"Artist One", "album_name":"Album",
         "track_genre":"pop, dance pop", "popularity":70, "duration_ms":205000, "explicit":False,
         "danceability":0.78, "energy":0.82, "key":1, "loudness":-5.2, "mode":1,
         "speechiness":0.04, "acousticness":0.12, "instrumentalness":0.0, "liveness":0.12,
         "valence":0.78, "tempo":121, "time_signature":4},
        {"track_id":"c", "track_name":"Song C", "artists":"Other Artist", "album_name":"Album C",
         "track_genre":"ambient, sleep", "popularity":40, "duration_ms":240000, "explicit":False,
         "danceability":0.2, "energy":0.1, "key":4, "loudness":-18, "mode":0,
         "speechiness":0.03, "acousticness":0.9, "instrumentalness":0.7, "liveness":0.08,
         "valence":0.15, "tempo":70, "time_signature":4},
    ])


def test_recommendations_exclude_seed_and_return_results():
    engine = SpotifyRecommendationEngine(sample_data())
    result = engine.recommend("a", n=2)
    assert len(result) == 2
    assert "a" not in result["track_id"].tolist()


def test_search_finds_artist():
    engine = SpotifyRecommendationEngine(sample_data())
    result = engine.search("Artist One")
    assert len(result) == 2


def test_explicit_filter():
    data = sample_data()
    data.loc[1, "explicit"] = True
    engine = SpotifyRecommendationEngine(data)
    result = engine.recommend("a", allow_explicit=False)
    assert "b" not in result["track_id"].tolist()


def test_profile_recommendations_exclude_onboarding_tracks():
    engine = SpotifyRecommendationEngine(sample_data())
    profile = engine.build_user_profile(track_ids=["a"], genres=["pop"])
    result = engine.recommend_for_profile(profile, n=2, exclude_ids=["a"])
    assert "a" not in result["track_id"].tolist()
    assert len(result) == 2


def test_canonical_recording_key_collapses_release_versions():
    original = {"title": "Darkhaast", "artists": "Arijit Singh"}
    alternate = {"title": "Darkhaast (From 'Shivaay')", "artists": "Arijit Singh"}
    vocals = {"title": "Darkhaast - Arijit Singh Vocals Only", "artists": "Arijit Singh"}
    slowed = {"title": "Snowman (Slowed Down)", "artists": "Sia"}
    sped_up = {"title": "Snowman (Sped Up Version)", "artists": "Sia"}
    assert _canonical_recording_key(original) == _canonical_recording_key(alternate)
    assert _canonical_recording_key(original) == _canonical_recording_key(vocals)
    assert _canonical_recording_key(slowed) == _canonical_recording_key(sped_up)
    rihanna_stay = _canonical_recording_key({"title": "Stay", "artists": "Rihanna"})
    justin_stay = _canonical_recording_key({"title": "Stay", "artists": "Justin Bieber"})
    assert rihanna_stay != justin_stay


def test_recommendation_title_key_collapses_slowed_and_sped_up_releases():
    assert recommendation_title_key("Snowman (Slowed Down) (Slowed Down)") == "snowman"
    assert recommendation_title_key("Snowman (Sped Up Version)") == "snowman"
    assert recommendation_title_key("Song (Part 1)") != recommendation_title_key("Song (Part 2)")
    assert recommendation_title_key("Live and Let Die") == "live and let die"


def test_recommendation_history_matches_long_title_across_artist_credit_variants():
    original = recommendation_history_keys({
        "title": "Until I Found You",
        "artists": "Stephen Sanchez",
    })
    alternate = recommendation_history_keys({
        "title": "Until I Found You (Em Beihold Version)",
        "artists": "Stephen Sanchez, Em Beihold",
    })
    assert original.intersection(alternate)
    assert not recommendation_history_keys({
        "title": "Stay",
        "artists": "Rihanna",
    }).intersection(recommendation_history_keys({
        "title": "Stay",
        "artists": "Justin Bieber",
    }))


def test_seed_exclusion_does_not_exclude_other_tracks_from_same_soundtrack():
    seed = {
        "title": "Get Ready To Fight",
        "artists": "Siddharth Basrur, Benny Dayal",
        "album": "Baaghi",
    }
    same_song = {
        "title": "Get Ready To Fight (From Baaghi)",
        "artists": "Siddharth Basrur",
        "album": "Baaghi",
    }
    soundtrack_song = {
        "title": "Sab Tera",
        "artists": "Armaan Malik",
        "album": "Baaghi",
    }
    assert recommendation_history_keys(seed).intersection(
        recommendation_history_keys(same_song)
    )
    assert not recommendation_history_keys(seed).intersection(
        recommendation_history_keys(soundtrack_song)
    )


def test_intent_detection_separates_motivational_from_romantic_candidates():
    seed = {
        "title": "Get Ready To Fight",
        "intent_labels": ["energetic", "motivational"],
    }
    romantic_candidate = {
        "title": "Badtameez Dil",
        "intent_labels": ["romantic"],
    }
    motivational_candidate = {
        "title": "Zinda",
        "intent_labels": ["motivational", "energetic"],
    }
    seed_intents = _infer_candidate_intents(seed)
    assert seed_intents == {"energetic", "motivational"}
    assert not seed_intents.intersection(_infer_candidate_intents(romantic_candidate))
    assert seed_intents.intersection(_infer_candidate_intents(motivational_candidate))
    assert "motivational" in _infer_candidate_intents({"title": "Fight for victory"})


def test_ranker_keeps_same_soundtrack_but_rejects_conflicting_intent():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    seed = {
        "title": "Get Ready To Fight",
        "artists": "Artist One",
        "album": "Baaghi Soundtrack",
        "year": 2016,
        "language": "hindi",
        "mood_vector": mood,
        "mood_label": "energetic",
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["motivational", "energetic"],
    }
    candidates = [
        {
            "song_id": "same-seed",
            "title": "Get Ready To Fight (From Baaghi)",
            "artists": "Different Credit",
            "album": "Baaghi Soundtrack",
            "year": 2016,
            "language": "hindi",
            "mood_vector": mood,
            "mood_label": "energetic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["motivational", "energetic"],
        },
        {
            "song_id": "soundtrack-track",
            "title": "Baaghi Action Song",
            "artists": "Other Singer",
            "album": "Baaghi Soundtrack",
            "year": 2016,
            "language": "hindi",
            "mood_vector": mood,
            "mood_label": "energetic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["motivational", "energetic"],
        },
        {
            "song_id": "romantic-song",
            "title": "Badtameez Dil",
            "artists": "Artist One",
            "album": "Different Soundtrack",
            "year": 2016,
            "language": "hindi",
            "mood_vector": mood,
            "mood_label": "energetic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        },
    ]

    ranked = hybrid.rank_jio_results(
        candidates,
        "test-user",
        catalog,
        [],
        seed=seed,
    )

    assert [item["song_id"] for item in ranked] == ["soundtrack-track"]


def test_listener_mood_and_era_profiles_follow_recent_listening():
    engine = HybridRecommendationEngine(SpotifyRecommendationEngine(sample_data()))
    recent_mood = [1.0] + [0.0] * 8
    older_mood = [0.0, 1.0] + [0.0] * 7
    now = pd.Timestamp.now(tz="UTC")
    engine.fit([
        {
            "user_id": "listener",
            "track_id": "recent",
            "event_type": "jio_play",
            "created_at": now.isoformat(),
            "metadata_json": json.dumps({
                "artists": "Recent Artist",
                "mood_vector": recent_mood,
                "year": 2022,
            }),
        },
        {
            "user_id": "listener",
            "track_id": "older",
            "event_type": "jio_play",
            "created_at": (now - pd.Timedelta(days=365)).isoformat(),
            "metadata_json": json.dumps({
                "artists": "Older Artist",
                "mood_vector": older_mood,
                "year": 1996,
            }),
        },
    ])

    profile = engine.user_mood_profiles["listener"]
    assert profile[0] > profile[1]
    eras = engine.user_era_profiles["listener"]
    assert eras[2020] > eras[1990]
    ranked = engine.rank_jio_results(
        [{
            "song_id": "current-era-song",
            "title": "Current Era Song",
            "artists": "Anchor Artist",
            "album": "Another Album",
            "year": 2022,
            "language": "Hindi",
            "mood_vector": recent_mood,
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
        }],
        "listener",
        sample_data(),
        [],
        seed={
            "title": "Current seed",
            "artists": "Anchor Artist",
            "album": "Seed Album",
            "year": 2022,
            "language": "Hindi",
            "mood_vector": recent_mood,
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
        },
    )
    assert len(ranked) == 1
    assert ranked[0]["recommendation_features"]["listener_mood"] > 0.99
    assert ranked[0]["recommendation_features"]["listener_era"] > 0.7


def test_recommendations_cycle_through_era_taste_artist_and_soundtrack():
    candidates = [
        {"song_id": "filler"},
        {"song_id": "same-era", "year": 2016},
        {"song_id": "same-taste", "_listener_mood_match": 0.8},
        {"song_id": "same-artist", "_seed_artist_match": 1.0},
        {"song_id": "same-film", "_album_match": 1.0},
        {"song_id": "next-era", "year": 2019},
    ]

    ordered = _interleave_recommendation_contexts(candidates, 2015)

    assert [item["song_id"] for item in ordered[:4]] == [
        "same-era",
        "same-taste",
        "same-artist",
        "same-film",
    ]
    assert sorted(item["song_id"] for item in ordered) == sorted(
        item["song_id"] for item in candidates
    )


def test_logged_impression_schema_is_read_by_ranker():
    row = _feature_row({
        "score": 0.8,
        "rank": 2,
        "features": {"audio": 0.7, "metadata": 0.6, "collaborative": 0.4},
    })
    assert row[0] == 0.8
    assert row[3] == 0.7
    assert row[7] == 0.4
    assert row[-1] == 2 / 70
