import json
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import (
    HybridRecommendationEngine,
    SpotifyRecommendationEngine,
    _canonical_recording_key,
    _era_artist_discovery_queries,
    _infer_candidate_intents,
    _interleave_recommendation_contexts,
    _mood_era_discovery_queries,
    infer_audio_intents,
    infer_metadata_mood_labels,
    infer_same_soundtrack_audio_intents,
    _soundtrack_peer_artists,
    is_compilation_album,
    recommendation_source_work,
    recommendation_seed_year,
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


def test_compilation_seed_year_comes_from_matching_film_tracks():
    seed = {
        "title": "Darkhaast (From 'Shivaay')",
        "album": "World Music Day - Best Of Bollywood Hits",
        "year": 2026,
    }
    candidates = [
        {
            "title": "Other Song (From 'Shivaay')",
            "album": "Shivaay",
            "year": 2016,
        },
        {
            "title": "Unrelated Song",
            "album": "Other Film",
            "year": 1980,
        },
    ]

    assert recommendation_seed_year(seed, candidates) == 2016
    assert recommendation_seed_year(seed, [candidates[1]]) == 0


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
    assert "devotional" in _infer_candidate_intents({"title": "Darshan Do Ghanshyam"})


def test_playful_intent_uses_descriptive_metadata_and_lively_audio_features():
    described = {"title": "A song", "mood": "playful, comic and theatrical"}
    mellow = {"energy": 0.38, "valence": 0.54, "danceability": 0.46, "tempo": 109}

    assert _infer_candidate_intents(described) == {"playful"}
    assert infer_audio_intents(mellow) == set()


def test_metadata_mood_does_not_infer_emotion_from_artist_name():
    assert infer_metadata_mood_labels({
        "title": "A song",
        "artists": "Love",
        "album": "A collection",
    }) == set()
    assert infer_metadata_mood_labels({
        "title": "Ishq",
        "artists": "A neutral artist",
    }) == {"romantic"}


def test_recommendation_reasons_do_not_claim_metadata_mood_is_verified_audio():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    seed = {
        "title": "Seed ballad",
        "artists": "Anchor Artist",
        "year": 1975,
        "mood_vector": mood,
        "mood_label": "romantic",
        "mood_confidence": 0.58,
        "mood_source": "metadata_fallback",
        "intent_labels": ["romantic"],
    }
    candidate = {
        "song_id": "metadata-mood-match",
        "title": "Another ballad",
        "artists": "Anchor Artist",
        "year": 1976,
        "mood_vector": mood,
        "mood_label": "romantic",
        "mood_confidence": 0.58,
        "mood_source": "metadata_fallback",
        "intent_labels": ["romantic"],
    }

    ranked = hybrid.rank_jio_results(
        [candidate], "metadata-mood-user", catalog, [], seed=seed,
    )

    assert ranked[0]["recommendation_reason"] == (
        "Same artist; metadata suggests a similar vibe"
    )


def test_sparse_seed_uses_same_soundtrack_audio_as_a_low_confidence_intent_hint():
    seed = {
        "title": "Babu Samjho Ishare",
        "album": "Chalti Ka Naam Gaadi",
        "intent_labels": ["general"],
    }
    soundtrack_peer = {
        "title": "Ek Ladki Bheegi Bhagi Si",
        "album": "Chalti Ka Naam Gaadi (Original Motion Picture Soundtrack)",
        "mood_source": "local_audio_catalog",
        "energy": 0.455,
        "valence": 0.892,
        "danceability": 0.563,
        "tempo": 120.784,
    }
    unrelated_track = {
        **soundtrack_peer,
        "album": "Different Film",
        "energy": 0.90,
        "valence": 0.95,
        "danceability": 0.80,
    }
    mellow_soundtrack_peer = {
        **soundtrack_peer,
        "energy": 0.30,
        "valence": 0.45,
        "danceability": 0.40,
    }

    assert infer_same_soundtrack_audio_intents(seed, [soundtrack_peer]) == {"playful"}
    assert infer_same_soundtrack_audio_intents(seed, [unrelated_track]) == set()
    assert infer_same_soundtrack_audio_intents(
        seed,
        [soundtrack_peer, mellow_soundtrack_peer],
    ) == set()


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


def test_compilation_seed_uses_same_film_year_to_find_other_same_era_songs():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    seed = {
        "title": "Darkhaast (From 'Shivaay')",
        "artists": "Mithoon, Arijit Singh",
        "album": "World Music Day - Best Of Bollywood Hits",
        "year": 2026,
        "language": "Hindi",
        "intent_labels": ["general"],
    }
    candidates = [{
        "song_id": "same-film-year-anchor",
        "title": "Other Song (From 'Shivaay')",
        "artists": "Different Singer",
        "album": "Shivaay",
        "year": 2016,
        "language": "Hindi",
        "intent_labels": ["general"],
    }] + [
        {
            "song_id": f"same-era-{index}",
            "title": f"Era Song {index}",
            "artists": f"Era Singer {index}",
            "album": f"Different Film {index}",
            "year": 2014 + index % 5,
            "language": "Hindi",
            "intent_labels": ["romantic" if index % 2 else "energetic"],
        }
        for index in range(1, 13)
    ]

    ranked = hybrid.rank_jio_results(
        candidates, "shivaay-radio-user", catalog, [], seed=seed,
    )

    assert len(ranked) >= 10
    assert all(abs(int(item["year"]) - 2016) <= 15 for item in ranked)
    assert all(item["recommendation_features"]["language"] == 1.0 for item in ranked)
    assert sum(
        item["recommendation_features"]["soundtrack"] == 1.0
        for item in ranked[:10]
    ) <= 1


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


def test_era_discovery_queries_include_seed_artists_and_only_nearby_decades():
    queries = _era_artist_discovery_queries(
        1958,
        ["Kishore Kumar", "Manna Dey", "Kishore Kumar"],
    )

    assert "Kishore Kumar 1956 songs" in queries
    assert "Kishore Kumar 1957 songs" in queries
    assert "Kishore Kumar 1958 songs" in queries
    assert "Kishore Kumar 1960 songs" in queries
    assert "Kishore Kumar 1960s songs" in queries
    assert "Manna Dey 1956 songs" in queries
    assert len(queries) == len(set(queries))
    assert all("1940s" not in query for query in queries)
    assert all("1950s" not in query for query in queries)
    assert all("1970s" not in query for query in queries)
    assert all("1980s" not in query for query in queries)
    assert _era_artist_discovery_queries(1975, ["Kishore Kumar"]) == [
        "Kishore Kumar 1973 songs",
        "Kishore Kumar 1974 songs",
        "Kishore Kumar 1975 songs",
        "Kishore Kumar 1976 songs",
        "Kishore Kumar 1977 songs",
        "Kishore Kumar 1960s songs",
        "Kishore Kumar 1980s songs",
    ]


def test_mood_era_discovery_includes_same_decade_mood_queries():
    assert _mood_era_discovery_queries(
        2016, "Hindi", {"romantic"},
    ) == ["2010s hindi romantic songs"]
    assert _mood_era_discovery_queries(
        2016, "Hindi", {"romantic", "sad"},
    ) == ["2010s hindi romantic songs", "2010s hindi sad songs"]
    assert _mood_era_discovery_queries(
        2016, "", {"romantic"},
    ) == ["2010s romantic songs"]


def test_soundtrack_peer_artists_expand_era_search_without_repeating_seed_singers():
    seed = {
        "title": "Babu Samjho Ishare",
        "artists": "Kishore Kumar, Manna Dey",
        "album": "Chalti Ka Naam Gaadi",
    }
    same_work = {
        "title": "Haal Kaisa Hai Janab Ka (From 'Chalti Ka Naam Gaadi')",
        "artists": "Asha Bhosle, Kishore Kumar",
        "album": "Drive",
    }
    different_work = {
        "title": "Another Song",
        "artists": "Lata Mangeshkar",
        "album": "Different Film",
    }

    assert _soundtrack_peer_artists(seed, [same_work, different_work]) == [
        "Asha Bhosle"
    ]


def test_era_radio_returns_songs_from_other_films_when_soundtrack_has_few_tracks():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    seed = {
        "title": "Babu Samjho Ishare",
        "artists": "Kishore Kumar, Manna Dey",
        "album": "Chalti Ka Naam Gaadi",
        "year": 1958,
        "language": "Hindi",
        "mood_vector": [0.50, 0.50, 0.50, 0.50, 0.40, 0.15, 0.10, 0.20, 0.80],
        "mood_confidence": 0.22,
        "mood_source": "metadata_fallback",
        "intent_labels": ["general"],
    }
    candidates = [
        {
            "song_id": "same-film",
            "title": "Ek Ladki Bheegi Bhagi Si",
            "artists": "Kishore Kumar",
            "album": "Chalti Ka Naam Gaadi",
            "year": 1958,
            "language": "Hindi",
        },
        {
            "song_id": "same-film-reissue",
            "title": "Another Soundtrack Song (From 'Chalti Ka Naam Gaadi')",
            "artists": "Kishore Kumar",
            "album": "Drive",
            "year": 1958,
            "language": "Hindi",
        },
        *[
            {
                "song_id": f"other-film-{index}",
                "title": f"Era Match Song {index}",
                "artists": "Kishore Kumar" if index % 2 else "Manna Dey",
                "album": f"Different Film {index}",
                "year": 1958 + index,
                "language": "Hindi",
            }
            for index in range(1, 13)
        ],
    ]

    ranked = hybrid.rank_jio_results(
        candidates,
        "old-film-radio-user",
        catalog,
        [],
        seed=seed,
    )

    assert len(ranked) >= 10
    assert len({item["album"] for item in ranked[:10]}) >= 8
    assert sum(item["album"] == "Chalti Ka Naam Gaadi" for item in ranked[:10]) <= 1
    assert sum(
        item["album"] == "Chalti Ka Naam Gaadi"
        or recommendation_source_work(item) == "chalti ka naam gaadi"
        for item in ranked[:10]
    ) <= 1


def test_same_era_mood_radio_prefers_other_films_over_soundtrack_flood():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    seed = {
        "title": "Seed Ballad",
        "artists": "Seed Singer",
        "album": "Seed Film",
        "year": 2016,
        "language": "Hindi",
        "mood_vector": mood,
        "mood_label": "romantic",
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["romantic"],
    }
    candidates = [
        {
            "song_id": f"same-film-{index}",
            "title": f"Seed Film Song {index}",
            "artists": f"Seed Singer {index}",
            "album": "Seed Film",
            "year": 2016,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        }
        for index in range(1, 4)
    ] + [
        {
            "song_id": f"other-film-{index}",
            "title": f"Era Romance {index}",
            "artists": f"Different Singer {index}",
            "album": f"Different Film {index}",
            "year": 2014 + index % 5,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        }
        for index in range(1, 13)
    ]

    ranked = hybrid.rank_jio_results(
        candidates, "same-era-romance-user", catalog, [], seed=seed,
    )

    assert len(ranked) >= 10
    assert sum(item["album"] == "Seed Film" for item in ranked[:10]) <= 1
    assert len({item["artists"] for item in ranked[:10]}) >= 8
    assert all(item["recommendation_features"]["mood"] > 0.99 for item in ranked[:10])


def test_context_priority_within_same_decade_preserves_ranked_diversity():
    candidates = [
        {"song_id": "film", "year": 1974, "_album_match": 1.0},
        {"song_id": "artist", "year": 1976, "_seed_artist_match": 1.0},
        {"song_id": "taste", "year": 1978, "_listener_mood_match": 0.8},
    ]

    ordered = _interleave_recommendation_contexts(candidates, 1975)

    assert [item["song_id"] for item in ordered] == ["taste", "film", "artist"]


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


def test_ranker_returns_20_contextual_candidates_when_live_years_are_missing():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    seed = {
        "title": "Seed ballad",
        "artists": "Anchor Artist",
        "album": "Seed film",
        "year": 1975,
        "language": "Hindi",
        "mood_vector": mood,
        "mood_label": "romantic",
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["romantic"],
    }
    candidates = [
        {
            "song_id": f"same-artist-no-year-{index}",
            "title": f"Romantic tune {index}",
            "artists": "Anchor Artist",
            "album": f"Film {index}",
            "language": "Hindi",
            "mood_vector": mood,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        }
        for index in range(20)
    ] + [
        {
            "song_id": "distant-year-other-artist",
            "title": "Wrong artist",
            "artists": "Other Artist",
            "album": "Film B",
            "year": 2020,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        },
    ]

    ranked = hybrid.rank_jio_results(candidates, "missing-year-user", catalog, [], seed=seed)

    assert len(ranked) >= 20
    assert all(item["song_id"].startswith("same-artist-no-year-") for item in ranked)


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


def test_radio_rotates_singers_while_preserving_seed_mood_and_era():
    catalog = sample_data()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))
    mood = [0.8, 0.65, 0.7, 0.75, 0.2, 0.1, 0.08, 0.2, 0.8]
    artists = ["Mukesh"] * 6 + [
        "Lata Mangeshkar",
        "Kishore Kumar",
        "Manna Dey",
        "Mohammed Rafi",
        "Asha Bhosle",
        "Hemlata",
    ]
    candidates = [
        {
            "song_id": f"era-track-{index}",
            "title": f"Romantic tune {index}",
            "artists": artist,
            "album": f"Film {index}",
            "year": 1971 + index % 8,
            "language": "Hindi",
            "mood_vector": mood,
            "mood_label": "romantic",
            "mood_confidence": 0.95,
            "mood_source": "local_audio_catalog",
            "intent_labels": ["romantic"],
        }
        for index, artist in enumerate(artists)
    ]
    seed = {
        "title": "Seed ballad",
        "artists": "Mukesh",
        "album": "Seed film",
        "year": 1975,
        "language": "Hindi",
        "mood_vector": mood,
        "mood_label": "romantic",
        "mood_confidence": 0.95,
        "mood_source": "local_audio_catalog",
        "intent_labels": ["romantic"],
    }

    ranked = hybrid.rank_jio_results(
        candidates, "era-mood-user", catalog, [], seed=seed,
    )
    top_eight_artists = [item["artists"] for item in ranked[:8]]

    assert len(set(top_eight_artists)) >= 5
    assert top_eight_artists.count("Mukesh") <= 2
    assert all(abs(int(item["year"]) - 1975) <= 15 for item in ranked[:8])
    assert all(item["recommendation_features"]["mood"] > 0.99 for item in ranked[:8])


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
