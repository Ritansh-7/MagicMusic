from __future__ import annotations

import glob
import json
import re
from collections import Counter

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

data_dir = "spotify_recommendation_engine/data"
processed = 0
playlists = 0
occurrences = 0
unique_tracks = set()
dev_title_occ = 0
unique_dev_title = set()
unique_dev_meta = set()
playlists_with_dev = 0
title_counter = Counter()
artist_counter = Counter()

for path in sorted(glob.glob(data_dir + "/*.json")):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    processed += 1
    for pl in obj.get("playlists", []):
        playlists += 1
        has_dev = False
        for t in pl.get("tracks", []):
            occurrences += 1
            uri = t.get("track_uri", "")
            unique_tracks.add(uri)
            title = t.get("track_name", "")
            artist = t.get("artist_name", "")
            album = t.get("album_name", "")
            if DEVANAGARI.search(title):
                dev_title_occ += 1
                unique_dev_title.add(uri)
                has_dev = True
                title_counter[title] += 1
                artist_counter[artist] += 1
            if DEVANAGARI.search(title + " " + artist + " " + album):
                unique_dev_meta.add(uri)
        if has_dev:
            playlists_with_dev += 1

print("========== DATASET SUMMARY ==========")
print(f"JSON files processed: {processed:,}")
print(f"Total playlists: {playlists:,}")
print(f"Total track occurrences: {occurrences:,}")
print(f"Unique track IDs: {len(unique_tracks):,}")
print("\n========== DEVANAGARI ANALYSIS ==========")
print("Track-title occurrences containing Devanagari:", f"{dev_title_occ:,}")
print("Unique tracks with Devanagari titles:", f"{len(unique_dev_title):,}")
print("Unique tracks with Devanagari anywhere in metadata:", f"{len(unique_dev_meta):,}")
print("Playlists containing at least one Devanagari title:", f"{playlists_with_dev:,}")
if occurrences:
    print(f"Devanagari title occurrence percentage: {(dev_title_occ/occurrences)*100:.5f}%")
print("\n========== SAMPLE TRACKS ==========")
for title, count in title_counter.most_common(30):
    print(f"{count:>7,} | {title}")
print("\n========== SAMPLE ARTISTS ==========")
for artist, count in artist_counter.most_common(30):
    print(f"{count:>7,} | {artist}")
print("\nNOTE: Devanagari candidates are not all necessarily Hindi.")
print("Romanized Hindi titles are not detected by this script.")
