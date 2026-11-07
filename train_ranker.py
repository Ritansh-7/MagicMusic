from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def train_ranker(train_path: str = "ranking_train.csv"):
    df = pd.read_csv(train_path)
    group = df.groupby("query_id").size().tolist()
    features = [c for c in df.columns if c not in ("query_id", "label")]
    dtrain = lgb.Dataset(df[features], label=df["label"], group=group)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10]}
    model = lgb.train(params, dtrain, num_boost_round=100)
    model.save_model("ranking_model.txt")
    return model
