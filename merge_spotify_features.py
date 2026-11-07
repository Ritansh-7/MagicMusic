from __future__ import annotations

import pandas as pd

CATALOG = "spotify_tracks_data.csv"
SPOTIFY = "spotify_recommendation_engine/SpotifySongs.csv"


def merge():
    cat = pd.read_csv(CATALOG)
    sp = pd.read_csv(SPOTIFY)
    sp = sp.rename(columns={"SongName": "track_name", "ArtistName": "artists"})
    cat["_key"] = (cat["track_name"].str.lower().str.strip() + "|" + cat["artists"].str.lower().str.strip())
    sp["_key"] = (sp["track_name"].str.lower().str.strip() + "|" + sp["artists"].str.lower().str.strip())
    merged = cat.merge(sp, on="_key", how="left", suffixes=("", "_sp"))
    for col in ["Danceability", "Energy", "Valence", "Tempo", "Acousticness", "Instrumentalness", "Speechiness", "Liveness"]:
        if col in merged.columns:
            merged[col.lower()] = merged[col].fillna(merged.get(col.lower()))
    merged.to_csv("spotify_tracks_data.csv", index=False)
    print("merged rows", len(merged), "matched", merged["Danceability"].notna().sum())


if __name__ == "__main__":
    merge()
