from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

CSV = "spotify_recommendation_engine/SpotifySongs.csv"


def build():
    df = pd.read_csv(CSV)
    feats = ["Danceability", "Energy", "Valence", "Tempo", "Acousticness", "Instrumentalness", "Speechiness", "Liveness"]
    X = df[feats].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    svd = TruncatedSVD(n_components=min(8, Xs.shape[1]), random_state=42)
    emb = svd.fit_transform(Xs)
    sim = cosine_similarity(emb)
    out = pd.DataFrame(sim, index=df["SongName"], columns=df["SongName"])
    out.to_csv("hindi_similarity.csv")
    np.save("hindi_svd.npy", emb)
    print("built similarity", sim.shape)


if __name__ == "__main__":
    build()
