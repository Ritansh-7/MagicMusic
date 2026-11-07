from __future__ import annotations

import numpy as np
import pandas as pd
import faiss

from metadata_embeddings import load_or_compute


class CandidateRetriever:
    def __init__(self, data_path: str = "spotify_tracks_data.csv", index_path: str = "indexes/metadata_index.faiss"):
        self.data = pd.read_csv(data_path)
        self.index = faiss.read_index(index_path)
        self.emb = np.asarray(load_or_compute(self.data), dtype="float32")
        faiss.normalize_L2(self.emb)

    def semantic(self, seed_idx: int, k: int = 20):
        q = self.emb[seed_idx:seed_idx + 1]
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, k + 1)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != seed_idx][:k]
