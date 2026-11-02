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
    infer_audio_intents,
    is_compilation_album,
    recommendation_source_work,
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


def test_source_work_and_compilation_detection():
    assert recommendation_source_work({
        "title": "Darkhaast (From 'Shivaay')",
    }) == "shivaay"
    assert recommendation_source_work({
        "title": "Song (From Aashiqui 2)",
    }) == "aashiqui 2"
    assert is_compilation_album("World Music Day - Best Of Bollywood Hits")
    assert is_compilation_album("Hindi Chill Mix")
    assert not is_compilation_album("Shivaay")


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


def test_playful_intent_uses_descriptive_metadata_and_lively_audio_features():
    described = {"title": "A song", "mood": "playful, comic and theatrical"}
    mellow = {"energy": 0.38, "valence": 0.54, "danceability": 0.46, "tempo": 109}

    assert _infer_candidate_intents(described) == {"playful"}
    assert infer_audio_intents(mellow) == set()


def test_playful_theatrical_seed_prefers_lively_audio_over_same_artist_ballad():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    seed_mood = [0.68, 0.625, 0.82, 0.68, 0.25, 0.15, 0.10, 0.20, 0.875]
    lively_mood = [0.59, 0.70, 0.768, 0.705, 0.839, 0.0, 0.217, 0.097, 0.90]
    mellow_mood = [0.38, 0.55, 0.541, 0.461, 0.872, 0.0, 0.031, 0.08, 0.82]
    seed = {
        "title": "Ek Chatur Naar Karke Sringar",
        "artists": "Kishore Kumar, Manna Dey, Mehmood",
        "album": "Padosan",
        "year": 1968,
        "language": "Hindi",
        "mood_vector": seed_mood,
        "mood_confidence": 0.72,
        "mood_source": "metadata_fallback",
    }
    lively = {
        "song_id": "lively-classic",
        "title": "Mere Samnewali Khidki Mein",
        "artists": "Kishore Kumar",
        "album": "Padosan",
        "year": 1968,
        "language": "Hindi",
        "energy": 0.591,
        "valence": 0.768,
        "danceability": 0.705,
        "tempo": 139.782,
        "mood_vector": lively_mood,
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
    }
    romantic_ballad = {
        "song_id": "mellow-romantic",
        "title": "O Mere Dil Ke Chain",
        "artists": "Kishore Kumar",
        "album": "Mere Jeevan Saathi",
        "year": 1972,
        "language": "Hindi",
        "energy": 0.377,
        "valence": 0.541,
        "danceability": 0.461,
        "tempo": 109.017,
        "mood_vector": mellow_mood,
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["romantic"],
    }

    assert "playful" in _infer_candidate_intents(seed)
    assert infer_audio_intents(lively) == {"playful"}
    ranked = hybrid.rank_jio_results(
        [romantic_ballad, lively],
        "playful-user",
        catalog,
        [],
        seed=seed,
    )

    assert [item["song_id"] for item in ranked] == ["lively-classic"]


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


def test_same_artist_without_mood_intent_or_soundtrack_is_not_enough():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    ranked = hybrid.rank_jio_results(
        [{
            "song_id": "artist-only",
            "title": "Unrelated Song",
            "artists": "Anchor Artist",
            "album": "World Music Day - Best Of Bollywood Hits",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["general"],
        }],
        "era-user",
        catalog,
        [],
        seed={
            "title": "Darkhaast (From 'Shivaay')",
            "artists": "Anchor Artist",
            "album": "World Music Day - Best Of Bollywood Hits",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["general"],
        },
    )

    assert ranked == []


def test_same_source_film_is_soundtrack_evidence_even_on_compilation_release():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    ranked = hybrid.rank_jio_results(
        [{
            "song_id": "same-film",
            "title": "Other Song (From 'Shivaay')",
            "artists": "Different Singer",
            "album": "World Music Day - Best Of Bollywood Hits",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["general"],
        }],
        "era-user",
        catalog,
        [],
        seed={
            "title": "Darkhaast (From 'Shivaay')",
            "artists": "Anchor Artist",
            "album": "World Music Day - Best Of Bollywood Hits",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["general"],
        },
    )

    assert [item["song_id"] for item in ranked] == ["same-film"]
    assert ranked[0]["recommendation_features"]["soundtrack"] == 1.0


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


def test_recommendations_mix_seed_and_nearby_decades_but_drop_distant_years():
    candidates = [
        *({"song_id": f"seventies-{index}", "year": 1970 + index % 10} for index in range(14)),
        *({"song_id": f"sixties-{index}", "year": 1960 + index} for index in range(2)),
        *({"song_id": f"eighties-{index}", "year": 1980 + index} for index in range(2)),
        {"song_id": "near-other-decade", "year": 1989},
        {"song_id": "near-second-decade", "year": 1990},
        {"song_id": "distant-year", "year": 1995},
        {"song_id": "unknown-year", "year": ""},
    ]

    ordered = _interleave_recommendation_contexts(candidates, 1975)
    decades = [int(item["year"]) // 10 * 10 for item in ordered[:10]]

    assert len(ordered) == 20
    assert decades.count(1970) == 7
    assert decades.count(1960) + decades.count(1980) == 2
    assert decades.count(1990) == 1
    assert "distant-year" not in {item["song_id"] for item in ordered}
    assert "unknown-year" not in {item["song_id"] for item in ordered}


def test_context_priority_within_same_decade_is_taste_then_artist_then_film():
    candidates = [
        {"song_id": "film", "year": 1974, "_album_match": 1.0},
        {"song_id": "artist", "year": 1976, "_seed_artist_match": 1.0},
        {"song_id": "taste", "year": 1978, "_listener_mood_match": 0.8},
    ]

    ordered = _interleave_recommendation_contexts(candidates, 1975)

    assert [item["song_id"] for item in ordered] == ["taste", "artist", "film"]


def test_context_priority_balances_era_and_mood_before_artist_or_soundtrack():
    candidates = [
        {
            "song_id": "near-era-mood-match",
            "year": 2014,
            "_era_match": 0.95,
            "_mood_match": 0.85,
        },
        {
            "song_id": "artist-soundtrack-only",
            "year": 2010,
            "_era_match": 0.97,
            "_mood_match": 0.10,
            "_seed_artist_match": 1.0,
            "_album_match": 1.0,
        },
        {
            "song_id": "farther-era-mood-match",
            "year": 2019,
            "_era_match": 0.70,
            "_mood_match": 0.75,
        },
    ]

    ordered = _interleave_recommendation_contexts(candidates, 2013)

    assert [item["song_id"] for item in ordered] == [
        "near-era-mood-match",
        "farther-era-mood-match",
        "artist-soundtrack-only",
    ]


def test_same_album_name_with_distant_year_is_not_treated_as_soundtrack():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    ranked = hybrid.rank_jio_results(
        [{
            "song_id": "same-name-different-year",
            "title": "Lootera",
            "artists": "R Nait, Afsana Khan",
            "album": "Lootera",
            "year": 2019,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        }],
        "era-user",
        catalog,
        [],
        seed={
            "title": "Sawaar Loon",
            "artists": "Monali Thakur",
            "album": "Lootera",
            "year": 2013,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        },
    )

    assert len(ranked) == 1
    assert ranked[0]["recommendation_features"]["soundtrack"] == 0.0
    assert ranked[0]["recommendation_features"]["era"] > 0.0


def test_ranker_excludes_blocked_title_and_distant_decades_for_each_seed():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    candidates = [
        {
            "song_id": "same-decade-early",
            "title": "Same Decade Early",
            "artists": "Anchor Artist",
            "album": "Another Album",
            "year": 1971,
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "near-decade",
            "title": "Nearby Decade",
            "artists": "Other Artist",
            "album": "Another Album",
            "year": 1985,
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "same-decade-late",
            "title": "Same Decade Late",
            "artists": "Anchor Artist",
            "album": "Another Album",
            "year": 1979,
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "blocked-song",
            "title": "Jo Bhi Kasmein (From Raaz)",
            "artists": "Anchor Artist",
            "album": "Raaz",
            "year": 1972,
        },
        {
            "song_id": "blocked-title-variant",
            "title": "Jo Bhi Kasmein - Raaz Soundtrack",
            "artists": "Anchor Artist",
            "album": "Raaz",
            "year": 1976,
        },
        {
            "song_id": "next-decade",
            "title": "Next Decade",
            "artists": "Anchor Artist",
            "album": "Another Album",
            "year": 1980,
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "unknown-year",
            "title": "Unknown Year",
            "artists": "Anchor Artist",
            "album": "Another Album",
        },
    ]

    seed = {
        "title": "Seed Song",
        "artists": "Anchor Artist",
        "album": "Seed Album",
        "year": 1975,
        "intent_labels": ["romantic"],
    }
    seventies = hybrid.rank_jio_results(
        candidates, "era-user", catalog, [], seed=seed,
    )
    eighties = hybrid.rank_jio_results(
        candidates,
        "era-user",
        catalog,
        [],
        seed={**seed, "year": 1985},
    )

    assert {item["song_id"] for item in seventies} == {
        "same-decade-early",
        "same-decade-late",
        "near-decade",
        "next-decade",
    }
    assert {item["song_id"] for item in eighties} == {
        "same-decade-early",
        "same-decade-late",
        "near-decade",
        "next-decade",
    }


def test_ranker_can_return_10_plus_era_close_shared_artist_songs():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    candidates = [
        {
            "song_id": f"track-{year}-{index}",
            "title": f"Artist Song {year} {index}",
            "artists": "Seed Singer",
            "album": f"Album {index}",
            "year": year,
            "language": "Hindi",
            "intent_labels": ["general"],
        }
        for year in [
            1971, 1972, 1973, 1974, 1976, 1977, 1979,
            1965, 1968, 1980, 1981, 1983, 1985, 1989,
        ]
        for index in range(1)
    ]

    ranked = hybrid.rank_jio_results(
        candidates,
        "era-user",
        catalog,
        [],
        seed={
            "title": "Seed Song",
            "artists": "Seed Singer",
            "album": "Seed Album",
            "year": 1975,
            "language": "Hindi",
            "intent_labels": ["romantic"],
        },
    )

    assert len(ranked) >= 10
    assert all(abs(int(item["year"]) - 1975) <= 15 for item in ranked)
    assert sum(int(item["year"]) // 10 == 197 for item in ranked[:10]) >= 7


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
