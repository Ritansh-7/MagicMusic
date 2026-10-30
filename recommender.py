from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity


AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "loudness", "duration_ms", "key", "mode", "time_signature"
]


class SpotifyRecommendationEngine:
    """
    Content-based recommender for a track catalog.

    Similarity is built from:
      - genre text (TF-IDF)
      - artist text (TF-IDF, lower weight)
      - scaled audio features (weighted numeric vectors)

    Final ranking combines content similarity with popularity.
    No user listening history is assumed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        genre_weight: float = 1.0,
        artist_weight: float = 0.35,
        audio_weight: float = 0.8,
        popularity_weight: float = 0.15,
    ):
        self.data = data.copy()
        self.genre_weight = genre_weight
        self.artist_weight = artist_weight
        self.audio_weight = audio_weight
        self.popularity_weight = popularity_weight
        self._prepare_data()
        self._build_features()

    def _prepare_data(self):
        required = {"track_id", "track_name", "artists", "track_genre", "popularity"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        self.data = self.data.drop_duplicates(subset="track_id").reset_index(drop=True)

        for col in ["track_name", "artists", "album_name", "track_genre"]:
            if col not in self.data:
                self.data[col] = ""
            self.data[col] = self.data[col].fillna("").astype(str).str.strip()

        for col in AUDIO_FEATURES + ["popularity"]:
            if col in self.data:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                self.data[col] = self.data[col].fillna(self.data[col].median())

        if "explicit" not in self.data:
            self.data["explicit"] = False
        self.data["explicit"] = self.data["explicit"].fillna(False).astype(bool)
        self.data["popularity_norm"] = self.data["popularity"].clip(0, 100) / 100.0

        # Multi-genre values such as "filmi, modern bollywood" become text
        # that TF-IDF can compare. Keep the original display column unchanged.
        self.data["_genre_text"] = (
            self.data["track_genre"].str.lower()
            .str.replace(",", " ", regex=False)
            .str.replace("-", " ", regex=False)
        )
        self.data["_artist_text"] = (
            self.data["artists"].str.lower()
            .str.replace(",", " ", regex=False)
            .str.replace("&", " and ", regex=False)
        )

    def _build_features(self):
        self.genre_vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
        )
        genre_matrix = self.genre_vectorizer.fit_transform(self.data["_genre_text"])

        self.artist_vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
        )
        artist_matrix = self.artist_vectorizer.fit_transform(self.data["_artist_text"])

        numeric_cols = [c for c in AUDIO_FEATURES if c in self.data.columns]
        self.numeric_cols = numeric_cols
        self.scaler = MinMaxScaler()
        numeric_matrix = self.scaler.fit_transform(self.data[numeric_cols].astype(float))
        numeric_matrix = csr_matrix(numeric_matrix)

        # Block weights control how strongly each feature family contributes.
        self.feature_matrix = hstack([
            genre_matrix * self.genre_weight,
            artist_matrix * self.artist_weight,
            numeric_matrix * self.audio_weight,
        ]).tocsr()
        self.feature_matrix = normalize(self.feature_matrix)

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search track name, artist, album, and genre using case-insensitive text."""
        query = (query or "").strip()
        if not query:
            return self.data.head(0).drop(columns=["_genre_text", "_artist_text"], errors="ignore")
        fields = ["track_name", "artists", "album_name", "track_genre"]
        mask = pd.Series(False, index=self.data.index)
        for col in fields:
            mask |= self.data[col].str.contains(query, case=False, regex=False, na=False)
        result = self.data.loc[mask].sort_values(
            ["popularity", "track_name"], ascending=[False, True]
        ).head(limit)
        return self._public_columns(result)

    def recommend(
        self,
        track_id: str,
        n: int = 10,
        same_genre: bool = False,
        allow_explicit: bool = True,
        popularity_weight: float | None = None,
    ) -> pd.DataFrame:
        """Recommend similar tracks, excluding the seed track itself."""
        matches = self.data.index[self.data["track_id"].astype(str) == str(track_id)].tolist()
        if not matches:
            raise ValueError(f"Unknown track_id: {track_id}")
        seed_idx = matches[0]
        seed_genres = set(
            x.strip().lower() for x in self.data.loc[seed_idx, "track_genre"].split(",") if x.strip()
        )

        similarities = cosine_similarity(
            self.feature_matrix[seed_idx], self.feature_matrix
        ).ravel()

        result = self.data.copy()
        result["similarity"] = similarities
        result = result[result["track_id"].astype(str) != str(track_id)]

        if same_genre and seed_genres:
            result = result[result["track_genre"].apply(
                lambda value: bool(seed_genres.intersection(
                    x.strip().lower() for x in str(value).split(",") if x.strip()
                ))
            )]
        if not allow_explicit:
            result = result[~result["explicit"]]

        pw = self.popularity_weight if popularity_weight is None else float(popularity_weight)
        pw = min(max(pw, 0.0), 0.8)
        similarity_weight = 1.0 - pw
        result["recommendation_score"] = (
            similarity_weight * result["similarity"]
            + pw * result["popularity_norm"]
        )
        result = result.sort_values(
            ["recommendation_score", "similarity", "popularity"],
            ascending=[False, False, False]
        ).head(max(1, int(n)))

        return self._public_columns(result)

    def build_user_profile(
        self,
        track_ids: list[str] | None = None,
        genres: list[str] | None = None,
    ) -> csr_matrix:
        """Build a cold-start-aware listener profile in the catalog feature space.

        The profile blends selected tracks with onboarding genre choices.  This is
        deliberately kept in the same vector space as tracks so it can be used for
        retrieval without inventing collaborative signals the dataset does not have.
        """
        vectors = []
        if track_ids:
            for track_id in track_ids:
                matches = self.data.index[self.data["track_id"].astype(str) == str(track_id)]
                if len(matches):
                    vectors.append(self.feature_matrix[int(matches[0])])

        if genres:
            genre_mask = self.data["track_genre"].apply(
                lambda value: any(
                    wanted.lower() in str(value).lower() for wanted in genres if wanted
                )
            )
            genre_indices = np.flatnonzero(genre_mask.to_numpy())
            if len(genre_indices):
                # Onboarding is useful for cold start, but should yield to behavior
                # once the listener has selected enough tracks.
                vectors.append(self.feature_matrix[genre_indices].mean(axis=0))

        if not vectors:
            return csr_matrix(self.feature_matrix.mean(axis=0))
        profile = vectors[0]
        for vector in vectors[1:]:
            profile = profile + vector
        return normalize(csr_matrix(profile))

    def recommend_for_profile(
        self,
        profile: csr_matrix,
        n: int = 10,
        exclude_ids: list[str] | None = None,
        allow_explicit: bool = True,
        popularity_weight: float = 0.12,
        diversity: float = 0.18,
    ) -> pd.DataFrame:
        """Rank catalog items for a listener profile with light artist diversity."""
        similarities = cosine_similarity(profile, self.feature_matrix).ravel()
        result = self.data.copy()
        result["profile_similarity"] = similarities
        excluded = {str(item) for item in (exclude_ids or [])}
        result = result[~result["track_id"].astype(str).isin(excluded)]
        if not allow_explicit:
            result = result[~result["explicit"]]
        pw = min(max(float(popularity_weight), 0.0), 0.8)
        result["recommendation_score"] = (
            (1.0 - pw) * result["profile_similarity"]
            + pw * result["popularity_norm"]
        )
        result = result.sort_values(
            ["recommendation_score", "profile_similarity", "popularity"],
            ascending=[False, False, False],
        )

        # Greedy maximal-marginal-relevance-style reranking avoids a shelf filled
        # by one artist while retaining relevance as the dominant signal.
        selected = []
        artist_counts: dict[str, int] = {}
        for idx, row in result.iterrows():
            artist = str(row["artists"])
            penalty = diversity * artist_counts.get(artist, 0)
            adjusted = float(row["recommendation_score"]) - penalty
            selected.append((adjusted, idx))
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            if len(selected) >= max(1, int(n)) * 4:
                break
        selected.sort(key=lambda item: item[0], reverse=True)
        chosen_indices = [idx for _, idx in selected[:max(1, int(n))]]
        output = result.loc[chosen_indices].copy()
        output["recommendation_score"] = output["recommendation_score"].round(6)
        return self._public_columns(output.reset_index(drop=True))

    def shelves(
        self,
        profile: csr_matrix,
        n: int = 8,
        allow_explicit: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Create Spotify-like recommendation shelves from one listener profile."""
        base = self.recommend_for_profile(profile, n=n, allow_explicit=allow_explicit)
        high_energy = self.data[self.data["energy"] >= 0.72].copy()
        hidden_gems = self.data[self.data["popularity"] <= 45].copy()
        for frame in (high_energy, hidden_gems):
            frame["profile_similarity"] = cosine_similarity(
                profile, self.feature_matrix[frame.index]
            ).ravel()
            frame["recommendation_score"] = frame["profile_similarity"]
        high_energy = high_energy.sort_values(
            ["recommendation_score", "energy"], ascending=[False, False]
        ).head(n)
        hidden_gems = hidden_gems.sort_values(
            ["recommendation_score", "popularity"], ascending=[False, True]
        ).head(n)
        return {
            "Because you like your picks": base,
            "High-energy discoveries": self._public_columns(high_energy),
            "Hidden gems": self._public_columns(hidden_gems),
        }

    def recommend_by_features(
        self,
        genre: str | None = None,
        energy: float | None = None,
        danceability: float | None = None,
        valence: float | None = None,
        n: int = 10,
        allow_explicit: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend using a preference vector rather than a seed track.
        Inputs energy/danceability/valence use Spotify-style 0..1 scales.
        """
        pool = self.data.copy()
        if genre and genre != "Any":
            pool = pool[pool["track_genre"].str.contains(genre, case=False, regex=False)]
        if not allow_explicit:
            pool = pool[~pool["explicit"]]
        if pool.empty:
            return self._public_columns(pool)

        score = np.zeros(len(pool), dtype=float)
        used = 0
        for col, target in [
            ("energy", energy), ("danceability", danceability), ("valence", valence)
        ]:
            if target is not None and col in pool:
                score += 1.0 - np.abs(pool[col].to_numpy(dtype=float) - float(target))
                used += 1
        if used:
            score /= used
            pool["preference_match"] = score
        else:
            pool["preference_match"] = 0.0
        pool["recommendation_score"] = (
            0.75 * pool["preference_match"] + 0.25 * pool["popularity_norm"]
        )
        pool = pool.sort_values(
            ["recommendation_score", "popularity"], ascending=[False, False]
        ).head(max(1, int(n)))
        return self._public_columns(pool)

    @staticmethod
    def _public_columns(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(
            columns=["_genre_text", "_artist_text"], errors="ignore"
        ).reset_index(drop=True)

    def catalog_stats(self) -> dict:
        return {
            "tracks": int(len(self.data)),
            "artists": int(self.data["artists"].nunique()),
            "genres": int(self.data["track_genre"].nunique()),
            "mean_popularity": round(float(self.data["popularity"].mean()), 2),
            "explicit_tracks": int(self.data["explicit"].sum()),
        }


class HybridRecommendationEngine:
    """Hybrid ranker for external catalog results such as JioMusic.

    It combines:
      - content similarity: preferred artists/titles from the listener profile
      - collaborative filtering: implicit play/like/skip behavior
      - matrix factorization: TruncatedSVD embeddings of the user-item matrix
      - popularity: a small cold-start tie-breaker

    The model intentionally works with external song IDs, so JioMusic results
    can be ranked without pretending that their IDs are Spotify catalog IDs.
    """

    EVENT_WEIGHTS = {
        "play_start": 1.0,
        "jio_play": 1.0,
        "like": 3.0,
        "jio_like": 3.0,
        "skip": -1.0,
    }

    def __init__(self, content_engine: SpotifyRecommendationEngine):
        self.content_engine = content_engine
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.user_factors = np.empty((0, 0))
        self.item_factors = np.empty((0, 0))
        self.item_popularity: dict[str, float] = {}
        self.transition_scores: dict[str, dict[str, float]] = {}
        self.user_recent_items: dict[str, list[str]] = {}

    def fit(self, events: list[dict] | pd.DataFrame) -> "HybridRecommendationEngine":
        frame = pd.DataFrame(events)
        if frame.empty or not {"user_id", "track_id", "event_type"}.issubset(frame.columns):
            return self
        frame = frame.dropna(subset=["user_id", "track_id"]).copy()
        frame["user_id"] = frame["user_id"].astype(str)
        frame["track_id"] = frame["track_id"].astype(str)
        frame["weight"] = frame["event_type"].map(self.EVENT_WEIGHTS).fillna(0.0)
        frame = frame[frame["weight"] != 0]
        if frame.empty:
            return self
        if "created_at" in frame:
            frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
        else:
            frame["created_at"] = pd.NaT
        frame = frame.sort_values(["user_id", "session_id", "created_at"] if "session_id" in frame else ["user_id", "created_at"])

        # Recent listening context: used for freshness and repeat avoidance.
        self.user_recent_items = {
            str(user_id): group["track_id"].astype(str).tolist()[-30:]
            for user_id, group in frame.groupby("user_id")
        }

        # Sequential transition model: what tracks followed a track in a
        # session? This is a lightweight next-track signal.
        self.transition_scores = {}
        if "session_id" in frame:
            play_events = frame[frame["event_type"].isin(["jio_play", "play_start"])]
            for (_, _), group in play_events.groupby(["user_id", "session_id"]):
                sequence = group["track_id"].astype(str).tolist()
                for previous, following in zip(sequence, sequence[1:]):
                    if previous == following:
                        continue
                    next_items = self.transition_scores.setdefault(previous, {})
                    next_items[following] = next_items.get(following, 0.0) + 1.0
        interactions = frame.pivot_table(
            index="user_id", columns="track_id", values="weight", aggfunc="sum", fill_value=0.0
        )
        if interactions.empty:
            return self
        self.user_index = {str(value): index for index, value in enumerate(interactions.index)}
        self.item_index = {str(value): index for index, value in enumerate(interactions.columns)}
        matrix = interactions.to_numpy(dtype=float)
        # Negative skips are useful as penalties but must not make the SVD
        # unstable. Shift each user's row only for factorization; ranking uses
        # the learned latent preference direction.
        matrix = matrix - matrix.min(axis=1, keepdims=True)
        if min(matrix.shape) < 2:
            self.user_factors = matrix
            self.item_factors = np.eye(matrix.shape[1], dtype=float)
            return self
        components = max(1, min(32, min(matrix.shape) - 1))
        model = TruncatedSVD(n_components=components, random_state=42)
        self.user_factors = model.fit_transform(matrix)
        self.item_factors = model.components_.T
        totals = frame.groupby("track_id")["weight"].sum()
        if len(totals):
            low, high = float(totals.min()), float(totals.max())
            span = high - low or 1.0
            self.item_popularity = {str(k): (float(v) - low) / span for k, v in totals.items()}
        return self

    def collaborative_scores(self, user_id: str, item_ids: list[str]) -> np.ndarray:
        if str(user_id) not in self.user_index or self.user_factors.size == 0:
            return np.zeros(len(item_ids), dtype=float)
        user_vector = self.user_factors[self.user_index[str(user_id)]]
        raw = []
        for item_id in item_ids:
            index = self.item_index.get(str(item_id))
            raw.append(float(np.dot(user_vector, self.item_factors[index])) if index is not None else 0.0)
        scores = np.asarray(raw, dtype=float)
        if scores.size and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def rank_jio_results(
        self,
        results: list[dict],
        user_id: str,
        catalog: pd.DataFrame,
        preferred_track_ids: list[str],
        seed: dict | None = None,
    ) -> list[dict]:
        if not results:
            return []
        preferred = catalog[catalog["track_id"].astype(str).isin({str(x) for x in preferred_track_ids})]
        preferred_artists = [str(x).lower() for x in preferred.get("artists", pd.Series(dtype=str)).tolist()]
        preferred_titles = [str(x).lower() for x in preferred.get("track_name", pd.Series(dtype=str)).tolist()]
        item_ids = [str(item.get("song_id") or item.get("title", "")) for item in results]
        collaborative = self.collaborative_scores(user_id, item_ids)
        seed_id = str((seed or {}).get("song_id") or "")
        transition_map = self.transition_scores.get(seed_id, {})
        transition_values = np.asarray(
            [transition_map.get(item_id, 0.0) for item_id in item_ids], dtype=float
        )
        if transition_values.size and transition_values.max() > 0:
            transition_values = transition_values / transition_values.max()
        recent_items = set(self.user_recent_items.get(str(user_id), [])[-15:])

        content = []
        seed_artists = [
            value.strip().lower()
            for value in str((seed or {}).get("artists", "")).split(",")
            if value.strip()
        ]
        seed_album = str((seed or {}).get("album", "")).strip().lower()
        try:
            seed_year = int(str((seed or {}).get("year", ""))[:4])
        except ValueError:
            seed_year = 0
        for item in results:
            artist = str(item.get("artists", "")).lower()
            title = str(item.get("title", "")).lower()
            language = str(item.get("language", "")).lower()
            seed_language = str((seed or {}).get("language", "")).lower()
            artist_match = sum(1 for value in preferred_artists if value and value in artist)
            title_match = sum(1 for value in preferred_titles if value and value == title)
            profile_content = min(1.0, artist_match * 0.25 + title_match * 0.75)
            seed_artist_match = max(
                (1.0 if value in artist else 0.0 for value in seed_artists),
                default=0.0,
            )
            album_match = 1.0 if seed_album and seed_album == str(item.get("album", "")).strip().lower() else 0.0
            try:
                item_year = int(str(item.get("year", ""))[:4])
            except ValueError:
                item_year = 0
            era_match = (
                max(0.0, 1.0 - abs(seed_year - item_year) / 30.0)
                if seed_year and item_year else 0.0
            )
            language_match = 1.0 if seed_language and language and seed_language == language else 0.0
            seed_cluster = (seed or {}).get("cluster_id")
            item_cluster = item.get("cluster_id")
            cluster_match = (
                1.0
                if seed_cluster is not None and item_cluster is not None and str(seed_cluster) == str(item_cluster)
                else 0.0
            )
            # Metadata-level semantics: singer, album, era and language are
            # stronger evidence than title overlap alone for external songs.
            seed_content = (
                0.40 * seed_artist_match
                + 0.17 * album_match
                + 0.16 * era_match
                + 0.10 * language_match
                + 0.17 * cluster_match
            )
            content.append(max(profile_content, seed_content))
            if seed_artist_match:
                item["recommendation_reason"] = "Because you played this singer"
            elif album_match:
                item["recommendation_reason"] = "Similar album or soundtrack"
            elif language_match and era_match >= 0.65:
                item["recommendation_reason"] = "Similar language and era"
            elif cluster_match:
                item["recommendation_reason"] = "Similar mood and sound cluster"
            elif era_match >= 0.65:
                item["recommendation_reason"] = "Similar era and listening style"
            elif artist_match:
                item["recommendation_reason"] = "Because it matches your taste"
            else:
                item["recommendation_reason"] = "Fits your current listening context"
        content = np.asarray(content, dtype=float)
        popularity = np.asarray([
            self.item_popularity.get(item_id, 0.0) for item_id in item_ids
        ], dtype=float)
        # Content dominates cold-start. Once enough events exist, the
        # transition and collaborative signals influence the next-track order.
        hybrid = (
            0.50 * content
            + 0.25 * collaborative
            + 0.15 * transition_values
            + 0.10 * popularity
        )

        # Sequence-aware reranking: avoid immediate repeats and prevent one
        # artist/album from occupying the whole Up Next queue.
        ranked = sorted(zip(hybrid, results), key=lambda pair: pair[0], reverse=True)
        selected: list[dict] = []
        artist_counts: dict[str, int] = {}
        album_counts: dict[str, int] = {}
        remaining = ranked.copy()
        while remaining:
            best_index = 0
            best_score = float("-inf")
            for index, (base_score, item) in enumerate(remaining):
                item_id = str(item.get("song_id") or item.get("title", ""))
                artist_key = str(item.get("artists", "")).lower()
                album_key = str(item.get("album", "")).lower()
                adjusted = float(base_score)
                # Prevent the same singer from appearing repeatedly. The
                # first appearance is free; subsequent appearances are
                # increasingly expensive, while still allowing a second song
                # when the candidate pool is small.
                adjusted -= 0.22 * artist_counts.get(artist_key, 0)
                adjusted -= 0.05 * album_counts.get(album_key, 0)
                if item_id in recent_items:
                    adjusted -= 0.12
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = index
            _, chosen = remaining.pop(best_index)
            selected.append(chosen)
            artist_key = str(chosen.get("artists", "")).lower()
            album_key = str(chosen.get("album", "")).lower()
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            album_counts[album_key] = album_counts.get(album_key, 0) + 1
        return selected
