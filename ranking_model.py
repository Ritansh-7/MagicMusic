"""Optional learned ranking layer for TuneMatch.

The recommender must remain useful before enough feedback exists.  This module
therefore trains only from recommendation-impression events and falls back to
the deterministic hybrid score until there is enough labelled data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


RANK_FEATURES = (
    "base_score", "context", "mood", "intent", "metadata", "era",
    "language", "transition", "short_term", "long_term", "audio_confidence",
    "popularity", "rank_position",
)


def _feature_row(item: dict) -> list[float]:
    features = item.get("recommendation_features") or {}
    return [
        float(item.get("recommendation_score", features.get("base_score", 0.0)) or 0.0),
        float(features.get("context", 0.0) or 0.0),
        float(features.get("mood", 0.0) or 0.0),
        float(features.get("intent", 0.0) or 0.0),
        float(features.get("metadata", 0.0) or 0.0),
        float(features.get("era", 0.0) or 0.0),
        float(features.get("language", 0.0) or 0.0),
        float(features.get("transition", 0.0) or 0.0),
        float(features.get("short_term", 0.0) or 0.0),
        float(features.get("long_term", 0.0) or 0.0),
        float(features.get("audio_confidence", 0.0) or 0.0),
        float(item.get("popularity", 0.0) or 0.0) / 100.0,
        float(item.get("recommendation_rank", 0.0) or 0.0) / 70.0,
    ]


class LearnedRanker:
    """LambdaMART when available, guarded by a small-data fallback.

    Labels are generated from later implicit feedback after an impression:
    like=2, play=1, skip=0.  No label means a neutral impression and is not
    used as a positive example.  This is deliberately conservative to avoid
    training on unobserved negatives.
    """

    min_examples = 30
    min_positive = 5

    def __init__(self) -> None:
        self.model = None
        self.backend = "hybrid_fallback"
        self.examples = 0

    @staticmethod
    def _event_rows(events: pd.DataFrame) -> Iterable[tuple[list[float], float, str, int]]:
        if events.empty:
            return
        events = events.copy()
        if "created_at" in events:
            events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce", utc=True)
        query_id = 0
        for index, row in events.iterrows():
            if row.get("event_type") != "recommendation_impression":
                continue
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
            except (TypeError, ValueError):
                continue
            items = metadata.get("items", []) if isinstance(metadata, dict) else []
            if not isinstance(items, list):
                continue
            later = events.iloc[index + 1:] if isinstance(index, (int, np.integer)) else events.loc[index:]
            later = later[later.get("user_id", pd.Series(dtype=str)).astype(str) == str(row.get("user_id"))]
            outcomes: dict[str, float] = {}
            for _, event in later.iterrows():
                track_id = str(event.get("track_id") or "")
                if not track_id:
                    continue
                event_type = str(event.get("event_type") or "")
                value = 2.0 if event_type in {"like", "jio_like"} else 1.0 if event_type in {"play_start", "jio_play"} else 0.0
                outcomes[track_id] = max(outcomes.get(track_id, 0.0), value)
            for item in items:
                if not isinstance(item, dict):
                    continue
                track_id = str(item.get("track_id") or "")
                # Every exposed item becomes a zero/positive pair. This makes
                # the training data a real ranking query instead of learning
                # only from tracks that were eventually clicked.
                yield _feature_row(item), outcomes.get(track_id, 0.0), str(row.get("user_id")), query_id
            query_id += 1

    def fit(self, events: pd.DataFrame | list[dict]) -> "LearnedRanker":
        self.model = None
        self.backend = "hybrid_fallback"
        frame = events if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
        rows = list(self._event_rows(frame))
        self.examples = len(rows)
        positives = sum(label > 0 for _, label, _, _ in rows)
        if len(rows) < self.min_examples or positives < self.min_positive:
            return self
        x = np.asarray([features for features, _, _, _ in rows], dtype=float)
        y = np.asarray([label for _, label, _, _ in rows], dtype=float)
        groups = [sum(1 for _, _, _, group_id in rows if group_id == query_id) for query_id in sorted({group_id for _, _, _, group_id in rows})]
        try:
            from lightgbm import LGBMRanker

            self.model = LGBMRanker(
                objective="lambdarank", n_estimators=80, learning_rate=0.05,
                num_leaves=15, min_child_samples=8, verbosity=-1, random_state=42,
            )
            # Each observed interaction is a compact one-item query.  The
            # fallback still protects the app when LightGBM has no wheel.
            self.model.fit(x, y, group=groups)
            self.backend = "lightgbm_lambdamart"
        except (ImportError, ModuleNotFoundError, ValueError, RuntimeError):
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor

                self.model = HistGradientBoostingRegressor(
                    max_iter=80, learning_rate=0.05, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=42,
                )
                self.model.fit(x, y)
                self.backend = "hist_gradient_boosting_fallback"
            except (ImportError, ValueError, RuntimeError):
                self.model = None
        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        if not self.model or not items:
            return np.zeros(len(items), dtype=float)
        try:
            values = np.asarray(self.model.predict(np.asarray([_feature_row(item) for item in items])), dtype=float)
            if values.size and values.max() > values.min():
                values = (values - values.min()) / (values.max() - values.min())
            return np.clip(values, 0.0, 1.0)
        except (ValueError, RuntimeError, AttributeError):
            return np.zeros(len(items), dtype=float)

