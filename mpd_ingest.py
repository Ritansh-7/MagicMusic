from __future__ import annotations

import glob
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "spotify_recommendation_engine/data"
OUT_DIR = "data_lake/silver"
MANIFEST = "data_lake/manifest.csv"


def _checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(max_slices: int | None = None):
    os.makedirs(OUT_DIR + "/mpd_playlists", exist_ok=True)
    os.makedirs(OUT_DIR + "/mpd_playlist_tracks", exist_ok=True)
    os.makedirs(OUT_DIR + "/mpd_tracks", exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if max_slices:
        files = files[:max_slices]

    manifest_rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        pl_rows, pt_rows, tr_rows = [], [], []
        seen = set()
        uniq = []
        for pl in obj.get("playlists", []):
            pl_rows.append({
                "playlist_id": pl["pid"],
                "playlist_name": pl.get("name", ""),
                "num_tracks": pl.get("num_tracks", 0),
                "num_artists": pl.get("num_artists", 0),
                "num_albums": pl.get("num_albums", 0),
                "num_followers": pl.get("num_followers", 0),
                "modified_at": pl.get("modified_at", 0),
                "num_edits": pl.get("num_edits", 0),
            })
            for t in pl.get("tracks", []):
                uri = t.get("track_uri", "")
                if not uri:
                    continue
                pt_rows.append({
                    "playlist_id": pl["pid"],
                    "position": t.get("pos", 0),
                    "spotify_track_uri": uri,
                    "track_name": t.get("track_name", ""),
                    "artist_name": t.get("artist_name", ""),
                    "album_name": t.get("album_name", ""),
                    "duration_ms": t.get("duration_ms", 0),
                })
                if uri not in seen:
                    seen.add(uri)
                    uniq.append({
                        "spotify_track_uri": uri,
                        "track_name": t.get("track_name", ""),
                        "artist_name": t.get("artist_name", ""),
                        "album_name": t.get("album_name", ""),
                        "duration_ms": t.get("duration_ms", 0),
                    })
        slice_name = os.path.basename(path)
        pq.write_table(pa.Table.from_pylist(pl_rows), f"{OUT_DIR}/mpd_playlists/{slice_name}.parquet")
        pq.write_table(pa.Table.from_pylist(pt_rows), f"{OUT_DIR}/mpd_playlist_tracks/{slice_name}.parquet")
        pq.write_table(pa.Table.from_pylist(uniq), f"{OUT_DIR}/mpd_tracks/{slice_name}.parquet")
        manifest_rows.append((slice_name, _checksum(path), len(pl_rows), len(pt_rows), len(uniq), "done"))
        # release memory
        del obj, pl_rows, pt_rows, tr_rows, uniq, seen

    # write manifest
    import csv
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slice_name", "checksum", "playlists", "tracks", "unique_tracks", "status"])
        w.writerows(manifest_rows)
    print("processed", len(manifest_rows), "slices")


if __name__ == "__main__":
    ingest()
