import os
import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_PATH = "metadata_embeddings.npy"

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def build_track_text(row):
    parts = [
        str(row.get("track_name", "")),
        str(row.get("artists", "")),
        str(row.get("album_name", "")),
        str(row.get("track_genre", "")),
    ]
    return " | ".join(p for p in parts if p and p != "nan")

def compute_embeddings(df):
    model = get_model()
    texts = [build_track_text(r) for _, r in df.iterrows()]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

def load_or_compute(df):
    if os.path.exists(CACHE_PATH):
        emb = np.load(CACHE_PATH)
        if emb.shape[0] == len(df):
            return emb
    emb = compute_embeddings(df)
    np.save(CACHE_PATH, emb)
    return emb