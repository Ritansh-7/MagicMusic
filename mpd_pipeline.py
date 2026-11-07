from __future__ import annotations

import json
import glob
import os
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def iter_playlists(data_dir: str):
    for path in sorted(glob.glob(os.path.join(data_dir, "mpd.slice.*.json"))):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        for pl in obj.get("playlists", []):
            yield pl


def build_cooccurrence(data_dir: str, out_csv: str = "mpd_cooccurrence.csv", max_playlists: int | None = None):
    from collections import Counter, defaultdict
    counts = Counter()
    n = 0
    for pl in iter_playlists(data_dir):
        tracks = [t.get("track_uri") for t in pl.get("tracks", []) if t.get("track_uri")]
        for a in tracks:
            for b in tracks:
                if a != b:
                    counts[(a, b)] += 1
        n += 1
        if max_playlists and n >= max_playlists:
            break
    rows = [(a, b, c) for (a, b), c in counts.items()]
    if pd is not None:
        df = pd.DataFrame(rows, columns=["track_a", "track_b", "count"])
        df.to_csv(out_csv, index=False)
    else:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("track_a,track_b,count\n")
            for a, b, c in rows:
                f.write(f"{a},{b},{c}\n")
    return n
