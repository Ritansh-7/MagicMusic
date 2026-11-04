def retrieve_metadata(song: dict) -> dict:
    return {
        "title": song.get("title"),
        "artists": song.get("artists"),
        "album": song.get("album"),
        "year": song.get("year"),
        "language": song.get("language"),
        "genre": song.get("track_genre") or song.get("genre"),
    }