import pytest
import pandas as pd
from recommender import (
    SpotifyRecommendationEngine,
    HybridRecommendationEngine,
    resolve_true_release_year,
    double_verify_candidate,
    is_compilation_album,
)


def sample_catalog():
    return pd.DataFrame([
        {
            "track_id": "seed",
            "track_name": "Thoda Thoda Pyaar",
            "artists": "Stebin Ben, Nilesh Ahuja",
            "album_name": "Valentines Special",
            "track_genre": "filmi, romantic",
            "popularity": 75,
            "duration_ms": 240000,
            "explicit": False,
            "danceability": 0.55,
            "energy": 0.50,
            "key": 1,
            "loudness": -8.0,
            "mode": 1,
            "speechiness": 0.03,
            "acousticness": 0.60,
            "instrumentalness": 0.01,
            "liveness": 0.10,
            "valence": 0.65,
            "tempo": 95,
            "time_signature": 4,
        }
    ])


def test_is_compilation_album_detection():
    """Verify compilation albums are correctly caught and unmasked."""
    assert is_compilation_album("Valentines Special")
    assert is_compilation_album("Feel Good Valentines")
    assert is_compilation_album("House Party Hits")
    assert is_compilation_album("Ek Taraf Uska Ghar (Ghazal Lofi)")
    assert is_compilation_album("Best Of Arijit Singh - Collection Of Romantic Songs")
    assert is_compilation_album("Geet Ghazal Collection")
    assert not is_compilation_album("Saiyaara")
    assert not is_compilation_album("Mismatched: Season 2")


def test_resolve_true_release_year():
    """Verify original film release year is resolved from annotations or works."""
    assert resolve_true_release_year({"title": "Tumhe Jo Maine Dekha (From 'Main Hoon Na')", "year": 2025}) == 2004
    assert resolve_true_release_year({"title": "Ye Ishq Hai (From 'Jab We Met')", "year": 2025}) == 2007
    assert resolve_true_release_year({"title": "Chori Kiya Re Jiya (From 'Dabangg')", "year": 2025}) == 2010
    assert resolve_true_release_year({"title": "Tere Naina (From 'Chandni Chowk To China')", "year": 2025}) == 2009
    assert resolve_true_release_year({"title": "Barbaad (From 'Saiyaara')", "year": 2026}) == 2026
    assert resolve_true_release_year({"title": "Thoda Thoda Pyaar", "year": 2026}) == 2021
    # Vintage Ghazal artist peak era
    assert resolve_true_release_year({"title": "Ek Taraf Uska Ghar (Ghazal Lofi)", "artists": "Pankaj Udhas", "year": 2024}) == 1990
    assert resolve_true_release_year({"title": "Aaj Phir Unka Samana", "artists": "Jagjit Singh", "year": 2020}) == 1990


def test_double_verify_blocks_devotional_and_vintage_and_party_and_old_films():
    """Double verification must block bhajans, vintage ghazals, party bangers, and 2000s Bollywood compilations for modern romantic seeds."""
    seed = {
        "title": "Thoda Thoda Pyaar",
        "artists": "Stebin Ben, Nilesh Ahuja, Kumaar",
        "album": "Valentines Special",
        "year": 2026,
        "language": "Hindi",
        "intent_labels": ["romantic", "acoustic_indie"],
    }

    # 1. Devotional song (Namo Namo)
    devotional_song = {
        "title": "Namo Namo",
        "artists": "Amit Trivedi",
        "album": "Kedarnath",
        "year": 2018,
        "language": "Hindi",
        "intent_labels": ["devotional"],
    }
    is_valid, reason = double_verify_candidate(seed, devotional_song, seed_year=2021)
    assert not is_valid, "Devotional song should be blocked for romantic seed"
    assert "Devotional" in reason or "devotional" in reason

    # 2. Vintage Ghazal artist (Pankaj Udhas) disguised as 2024 lofi
    pankaj_udhas = {
        "title": "Ek Taraf Uska Ghar (Ghazal Lofi)",
        "artists": "Pankaj Udhas, Sachin Gupta",
        "album": "Ek Taraf Uska Ghar (Ghazal Lofi)",
        "year": 2024,
        "language": "Hindi",
    }
    is_valid, reason = double_verify_candidate(seed, pankaj_udhas, seed_year=2021)
    assert not is_valid, "Pankaj Udhas should be blocked for modern 2020s romantic seed"
    assert "Vintage" in reason or "vintage" in reason or "era" in reason

    # 3. Party club banger (Kala Chashma) disguised as 2026 House Party Hits
    kala_chashma = {
        "title": "Kala Chashma",
        "artists": "Kumaar, Badshah, Neha Kakkar",
        "album": "House Party Hits",
        "year": 2026,
        "language": "Hindi",
        "intent_labels": ["energetic"],
    }
    is_valid, reason = double_verify_candidate(seed, kala_chashma, seed_year=2021)
    assert not is_valid, "Kala Chashma should be blocked for cozy romantic seed"
    assert "Party" in reason or "party" in reason

    # 4. Old film song (Main Hoon Na 2004) disguised as 2025 Feel Good Valentines
    main_hoon_na = {
        "title": "Tumhe Jo Maine Dekha (From 'Main Hoon Na')",
        "artists": "Abhijeet, Shreya Ghoshal",
        "album": "Feel Good Valentines",
        "year": 2025,
        "language": "Hindi",
    }
    is_valid, reason = double_verify_candidate(seed, main_hoon_na, seed_year=2021)
    assert not is_valid, "Main Hoon Na (2004) must be blocked due to true era distance (> 15 yrs) from 2021"
    assert "era" in reason.lower()

    # 5. Matching modern romantic pop songs (Mahiye Jinna Sohna, Sahiba, Barbaad)
    mahiye = {
        "title": "Mahiye Jinna Sohna",
        "artists": "Darshan Raval",
        "album": "Mahiye Jinna Sohna",
        "year": 2023,
        "language": "Hindi",
        "intent_labels": ["romantic"],
    }
    is_valid, reason = double_verify_candidate(seed, mahiye, seed_year=2021)
    assert is_valid, f"Mahiye Jinna Sohna should pass double verification: {reason}"

    sahiba = {
        "title": "Sahiba",
        "artists": "Jasleen Royal, Stebin Ben",
        "album": "Sahiba",
        "year": 2024,
        "language": "Hindi",
        "intent_labels": ["romantic"],
    }
    is_valid, reason = double_verify_candidate(seed, sahiba, seed_year=2021)
    assert is_valid, f"Sahiba should pass double verification: {reason}"


def test_end_to_end_thoda_thoda_pyaar_ranking():
    """Verify rank_jio_results strictly recommends genuine same-era romantic songs and excludes bhajans, vintage ghazals, party bangers, and old 2004 compilations."""
    catalog = sample_catalog()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))

    seed = {
        "title": "Thoda Thoda Pyaar",
        "artists": "Stebin Ben, Nilesh Ahuja, Kumaar",
        "album": "Valentines Special",
        "year": 2026,
        "language": "Hindi",
        "intent_labels": ["romantic"],
        "mood_label": "romantic",
        "mood_confidence": 0.85,
        "mood_vector": [0.55, 0.50, 0.65, 0.60, 0.03, 0.01, 0.10, 0.65, 0.55],
    }

    candidates = [
        # Good candidates (same era 2020s, same romantic taste)
        {
            "song_id": "barbaad",
            "title": "Barbaad (From 'Saiyaara')",
            "artists": "Jubin Nautiyal, The Rish",
            "album": "Pyaar, Forever and Ever",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "mahiye",
            "title": "Mahiye Jinna Sohna",
            "artists": "Darshan Raval",
            "album": "Mahiye Jinna Sohna",
            "year": 2023,
            "language": "Hindi",
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "sahiba",
            "title": "Sahiba",
            "artists": "Jasleen Royal, Stebin Ben",
            "album": "Sahiba",
            "year": 2024,
            "language": "Hindi",
            "intent_labels": ["romantic"],
        },
        {
            "song_id": "aise_kyun",
            "title": "Aise Kyun (Ghazal Version)",
            "artists": "Anurag Saikia, Rekha Bhardwaj",
            "album": "Mismatched: Season 2",
            "year": 2022,
            "language": "Hindi",
            "intent_labels": ["romantic", "acoustic_indie"],
        },
        {
            "song_id": "nafrat",
            "title": "Nafrat",
            "artists": "Darshan Raval",
            "album": "I Loved You",
            "year": 2025,
            "language": "Hindi",
            "intent_labels": ["romantic"],
        },
        # Bad candidates that must be rejected
        {
            "song_id": "pankaj_udhas",
            "title": "Ek Taraf Uska Ghar (Ghazal Lofi)",
            "artists": "Pankaj Udhas, Sachin Gupta",
            "album": "Ek Taraf Uska Ghar (Ghazal Lofi)",
            "year": 2024,
            "language": "Hindi",
        },
        {
            "song_id": "namo_namo",
            "title": "Namo Namo",
            "artists": "Amit Trivedi",
            "album": "Kedarnath",
            "year": 2018,
            "language": "Hindi",
            "intent_labels": ["devotional"],
        },
        {
            "song_id": "kala_chashma",
            "title": "Kala Chashma",
            "artists": "Kumaar, Badshah, Neha Kakkar",
            "album": "House Party Hits",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["energetic"],
        },
        {
            "song_id": "main_hoon_na",
            "title": "Tumhe Jo Maine Dekha (From 'Main Hoon Na')",
            "artists": "Abhijeet, Shreya Ghoshal",
            "album": "Feel Good Valentines",
            "year": 2025,
            "language": "Hindi",
        },
    ]

    ranked = hybrid.rank_jio_results(candidates, "user-thoda", catalog, [], seed=seed)
    recommended_ids = {item["song_id"] for item in ranked}

    # Verify good candidates are recommended
    assert "barbaad" in recommended_ids
    assert "mahiye" in recommended_ids
    assert "sahiba" in recommended_ids

    # Verify bad candidates are strictly excluded
    assert "pankaj_udhas" not in recommended_ids, "Pankaj Udhas vintage ghazal was recommended!"
    assert "namo_namo" not in recommended_ids, "Devotional Namo Namo was recommended!"
    assert "kala_chashma" not in recommended_ids, "Party banger Kala Chashma was recommended!"
    assert "main_hoon_na" not in recommended_ids, "2004 Main Hoon Na was recommended for 2021 pop seed!"


def test_de_mauka_zindagi_excludes_english_pop_devotional_and_rap():
    """Verify De Mauka Zindagi (Aspirants) rejects Sabrina Carpenter, Neem Karoli Baba, and MTV Hustle rap, and prevents duplicate Ludo titles."""
    catalog = sample_catalog()
    hybrid = HybridRecommendationEngine(SpotifyRecommendationEngine(catalog))

    seed = {
        "title": "De Mauka Zindagi",
        "artists": "Nilotpal Bora, Juno",
        "album": "Aspirants: Season 1 (Music from the TVF Original Series)",
        "year": 2026,
        "language": "Hindi",
        "intent_labels": ["motivational", "acoustic_indie"],
    }

    candidates = [
        # Good matching indie tracks from same series and genre
        {
            "song_id": "dhaaga",
            "title": "Dhaaga",
            "artists": "Nilotpal Bora",
            "album": "Aspirants: Season 1 (Music from the TVF Original Series)",
            "year": 2021,
            "language": "Hindi",
            "intent_labels": ["motivational", "acoustic_indie"],
        },
        {
            "song_id": "beparwah",
            "title": "Beparwah",
            "artists": "Laksya Bhatnagar",
            "album": "Aspirants: Season 1 (Music from the TVF Original Series)",
            "year": 2021,
            "language": "Hindi",
            "intent_labels": ["motivational", "acoustic_indie"],
        },
        {
            "song_id": "kho_gaye",
            "title": "Kho Gaye Hum Kahan",
            "artists": "Jasleen Royal, Prateek Kuhad",
            "album": "Baar Baar Dekho",
            "year": 2016,
            "language": "Hindi",
            "intent_labels": ["acoustic_indie"],
        },
        # 1. English Pop track by Sabrina Carpenter (Name collision on 'Juno')
        {
            "song_id": "juno_sabrina",
            "title": "Juno",
            "artists": "Sabrina Carpenter",
            "album": "Girls Trip",
            "year": 2026,
            "language": "English",
            "intent_labels": ["pop"],
        },
        # 2. Spiritual / Devotional chant to Neem Karoli Baba
        {
            "song_id": "neem_karoli",
            "title": "Neem Karoli Baba",
            "artists": "Sumeet Kr Tyagi, Nilotpal Bora",
            "album": "Neem Karoli Baba",
            "year": 2024,
            "language": "Hindi",
        },
        # 3. Aggressive Rap from MTV Hustle
        {
            "song_id": "ludo_hustle",
            "title": "Ludo",
            "artists": "XEEMO, Aditya Pushkarna",
            "album": "Hustle 5 - Episode 2",
            "year": 2026,
            "language": "Hindi",
            "intent_labels": ["rap_hiphop"],
        },
        # 4. Duplicate title Ludo by Tony Kakkar
        {
            "song_id": "ludo_tony",
            "title": "Ludo",
            "artists": "Tony Kakkar",
            "album": "Ludo",
            "year": 2025,
            "language": "Hindi",
            "intent_labels": ["energetic"],
        },
    ]

    ranked = hybrid.rank_jio_results(candidates, "user-aspirants", catalog, [], seed=seed)
    recommended_ids = {item["song_id"] for item in ranked}

    # Verify matching indie tracks are recommended
    assert "dhaaga" in recommended_ids
    assert "beparwah" in recommended_ids

    # Verify bad candidates are strictly excluded
    assert "juno_sabrina" not in recommended_ids, "Sabrina Carpenter English pop was recommended for Hindi indie seed!"
    assert "neem_karoli" not in recommended_ids, "Devotional Neem Karoli Baba was recommended for secular indie seed!"
    assert "ludo_hustle" not in recommended_ids, "MTV Hustle rap was recommended for gentle indie seed!"
    assert "ludo_tony" not in recommended_ids, "Party track was recommended for gentle indie seed!"

    # Verify duplicate titles cannot occupy the queue
    titles = [item.get("title", "").strip().lower() for item in ranked]
    assert len(titles) == len(set(titles)), f"Duplicate titles found in queue: {titles}"


def test_universal_language_isolation():
    """Verify English pop seeds strictly reject Hindi songs, and Hindi seeds strictly reject English songs."""
    from recommender import infer_track_language

    assert infer_track_language({"title": "De Mauka Zindagi", "artists": "Nilotpal Bora, Juno"}) == "hindi"
    assert infer_track_language({"title": "Juno", "artists": "Sabrina Carpenter"}) == "english"
    assert infer_track_language({"title": "Until I Found You", "artists": "Stephen Sanchez"}) == "english"

    english_seed = {
        "title": "Until I Found You",
        "artists": "Stephen Sanchez",
        "album": "Easy On My Eyes",
        "year": 2022,
        "language": "English",
        "intent_labels": ["romantic"],
    }
    hindi_candidate = {
        "title": "Tere Fitoor",
        "artists": "Arijit Singh",
        "album": "Genius",
        "year": 2018,
        "language": "Hindi",
        "intent_labels": ["romantic"],
    }
    is_valid, reason = double_verify_candidate(english_seed, hindi_candidate, seed_year=2022)
    assert not is_valid, "Hindi song should be blocked for English seed"
    assert "Language conflict" in reason
