from jiosaavn_client import JioSaavnClient


def test_normalize_uses_primary_artists_when_singers_is_empty():
    item = JioSaavnClient._normalize(
        {
            "id": "song-1",
            "song": "Believer",
            "singers": "",
            "primary_artists": "Imagine Dragons",
            "album": "Evolve",
            "duration": "204",
            "media_url": "https://aac.saavncdn.com/example.mp4",
        }
    )

    assert item["artists"] == "Imagine Dragons"
    assert item["duration_ms"] == 204000
    assert item["stream_url"].endswith("example.mp4")


def test_normalize_preserves_explicit_duration_milliseconds():
    item = JioSaavnClient._normalize(
        {
            "song": "Short Song",
            "primary_artists": "Artist",
            "duration_ms": 123456,
            "url": "https://example.com/song.mp3",
        }
    )

    assert item["duration_ms"] == 123456
