import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

DB_PATH = "tune_match_events.sqlite3"

def load_events():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT user_id, track_id, event_type FROM behavior_events", conn)
    conn.close()
    return df

def build_user_item_matrix(df):
    weight_map = {
        "jio_like": 2.0,
        "jio_play": 1.0,
        "like": 2.0,
        "play_start": 1.0,
        "listen_progress": 0.5,
        "jio_skip": -1.0,
        "skip": -1.0,
    }
    df = df[df["event_type"].isin(weight_map)].copy()
    df = df[df["track_id"].notna()]
    df["w"] = df["event_type"].map(weight_map)
    agg = df.groupby(["user_id", "track_id"])["w"].sum().reset_index()
    users = agg["user_id"].unique()
    items = agg["track_id"].unique()
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {t: i for i, t in enumerate(items)}
    mat = np.zeros((len(users), len(items)))
    for _, r in agg.iterrows():
        mat[u_idx[r["user_id"]], i_idx[r["track_id"]]] = r["w"]
    return mat, users, items, u_idx, i_idx

def train(n_components=8):
    df = load_events()
    mat, users, items, u_idx, i_idx = build_user_item_matrix(df)
    if mat.shape[1] < 2 or mat.shape[0] < 2:
        return None
    k = min(n_components, mat.shape[1] - 1, mat.shape[0] - 1)
    if k < 1:
        return None
    svd = TruncatedSVD(n_components=k, random_state=42)
    user_emb = svd.fit_transform(mat)
    item_emb = svd.components_.T
    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "users": users,
        "items": items,
        "u_idx": u_idx,
        "i_idx": i_idx,
    }

def item_similarity(model, track_id):
    if model is None or track_id not in model["i_idx"]:
        return {}
    idx = model["i_idx"][track_id]
    vec = model["item_emb"][idx]
    norms = np.linalg.norm(model["item_emb"], axis=1)
    denom = norms[idx] * norms
    denom[denom == 0] = 1e-9
    sim = (model["item_emb"] @ vec) / denom
    return {model["items"][i]: float(sim[i]) for i in range(len(model["items"]))}