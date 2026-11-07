from __future__ import annotations

import numpy as np
import pandas as pd
import faiss

from metadata_embeddings import load_or_compute


def build_index(data_path: str = "spotify_tracks_data.csv", out_path: str = "indexes/metadata_index.faiss"):
    import os
    os.makedirs("indexes", exist_ok=True)
    df = pd.read_csv(data_path)
    emb = load_or_compute(df)
    arr = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, out_path)
    print("built", out_path, "vectors", arr.shape)


if __name__ == "__main__":
    build_index()
