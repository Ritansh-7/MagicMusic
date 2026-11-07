from __future__ import annotations

import glob
import json
import os

from gensim.models import Word2Vec

DATA_DIR = "spotify_recommendation_engine/data"
MODEL_PATH = "mpd_item2vec.model"
MAX_SLICES = 50


def sentences():
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:MAX_SLICES]:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        for pl in obj.get("playlists", []):
            tracks = [t.get("track_uri") for t in pl.get("tracks", []) if t.get("track_uri")]
            if len(tracks) >= 2:
                yield tracks


def train():
    corpus = list(sentences())
    model = Word2Vec(corpus, vector_size=64, window=5, min_count=5, workers=4, epochs=3)
    model.save(MODEL_PATH)
    return model


if __name__ == "__main__":
    m = train()
    print("saved", MODEL_PATH, "vocab", len(m.wv))
