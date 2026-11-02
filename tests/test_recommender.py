import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import (
    SpotifyRecommendationEngine,
    _canonical_recording_key,
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
