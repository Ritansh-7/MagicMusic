from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

# Placeholder: build ALS from a user-item interactions CSV
# columns: user_id, track_id, weight


def train(interactions_path: str = "interactions.csv", factors: int = 32):
    df = pd.read_csv(interactions_path)
    users = df["user_id"].astype("category")
    items = df["track_id"].astype("category")
    from scipy.sparse import csr_matrix
    mat = csr_matrix((df["weight"].astype(float), (users.cat.codes, items.cat.codes)))
    model = AlternatingLeastSquares(factors=factors, iterations=10)
    model.fit(mat)
    return model, users.cat.categories, items.cat.categories
