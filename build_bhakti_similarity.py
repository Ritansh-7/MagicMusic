from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

CSV = "bhakti_jiosaavn.csv"


def build():
    df = pd.read_csv(CSV)
    text = (df["title"].fillna("") + " " + df["artists"].fillna("") + " " + df["album"].fillna("")).astype(str)
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(text)
    svd = TruncatedSVD(n_components=min(64, X.shape[1] - 1), random_state=42)
    emb = svd.fit_transform(X)
    sim = cosine_similarity(emb)
    out = pd.DataFrame(sim, index=df["title"], columns=df["title"])
    out.to_csv("bhakti_similarity.csv")
    np.save("bhakti_svd.npy", emb)
    print("built bhakti similarity", sim.shape)


if __name__ == "__main__":
    build()