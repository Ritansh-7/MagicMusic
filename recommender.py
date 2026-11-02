from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd

HYBRID_MODEL_VERSION = "hybrid-v28-mood-artist-diversity"

EXCLUDED_RECOMMENDATION_TITLES = {"jo bhi kasmein"}
MAX_RECOMMENDATION_YEAR_DISTANCE = 15
MAX_ARTIST_REPEATS_IN_QUEUE = 2


def recommendation_source_work(item: dict) -> str:
    """Return a film/work named in a provider title annotation, if present."""
    title = str(item.get("title", ""))
    match = re.search(
        r"\bfrom\s+['\"“]?(.+?)['\"”]?(?:\)|\]|\s*[-–—]\s*|$)",
        title,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^a-z0-9]+", " ", match.group(1).casefold()).strip() if match else ""


def _artist_credit_keys(value: object) -> set[str]:
    if isinstance(value, list):
        names = [
            str(entry.get("name", "") if isinstance(entry, dict) else entry)
            for entry in value
        ]
    else:
        names = str(value or "").split(",")
    return {
        normalized
        for name in names
        if (normalized := re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip())
        and normalized not in {"unknown", "n a", "na"}
    }


def is_compilation_album(value: object) -> bool:
    """Identify album labels that describe a playlist/compilation, not a work."""
    album = str(value or "").casefold()
    return bool(re.search(
        r"\b(best\s+of|greatest\s+hits|hits|mix|collection|essentials|"
        r"world\s+music\s+day|top\s+(?:songs|hits)|playlist|radio)\b",
        album,
    ))


def _release_year(value: object) -> int:
    try:
        year = int(str(value or "").strip()[:4])
    except (TypeError, ValueError):
        return 0
    return year if 1900 <= year <= 2100 else 0


def _era_artist_discovery_queries(seed_year: object, artists: list[str]) -> list[str]:
    year = _release_year(seed_year)
    if not year:
        return []

    min_year = year - MAX_RECOMMENDATION_YEAR_DISTANCE
    max_year = year + MAX_RECOMMENDATION_YEAR_DISTANCE
    seed_decade = year // 10 * 10
    decades = [
        decade
        for decade in range(min_year // 10 * 10, max_year // 10 * 10 + 1, 10)
        if decade != seed_decade and min_year <= decade and decade + 9 <= max_year
    ]
    queries: list[str] = []
    seen_artists: set[str] = set()
    for artist in artists:
        artist = str(artist or "").strip()
        artist_key = re.sub(r"[^a-z0-9]+", " ", artist.casefold()).strip()
        if not artist_key or artist_key in seen_artists:
            continue
        seen_artists.add(artist_key)
        query_start = max(min_year, year - 2)
        query_end = min(max_year, year + 2)
        queries.extend(
            f"{artist} {query_year} songs"
            for query_year in range(query_start, query_end + 1)
        )
        queries.extend(f"{artist} {decade}s songs" for decade in decades)
        if len(seen_artists) >= 2:
            break
    return queries


def _recommendation_work_key(item: dict) -> str:
    work = recommendation_source_work(item)
    album = work or str(item.get("album", "")).casefold()
    if not album or is_compilation_album(album):
        return ""
    album = re.sub(
        r"\b(?:original\s+motion\s+picture\s+)?soundtrack\b",
        "",
        album,
    )
    return re.sub(r"[^a-z0-9]+", " ", album).strip()


def _soundtrack_peer_artists(seed: dict, candidates: list[dict]) -> list[str]:
    seed_work = _recommendation_work_key(seed)
    if not seed_work:
        return []
    seed_artists = {
        re.sub(r"[^a-z0-9]+", " ", artist.casefold()).strip()
        for artist in str(seed.get("artists", "")).split(",")
    }
    artists: dict[str, str] = {}
    for candidate in candidates:
        if _recommendation_work_key(candidate) != seed_work:
            continue
        for artist in str(candidate.get("artists", "")).split(","):
            artist = artist.strip()
            artist_key = re.sub(r"[^a-z0-9]+", " ", artist.casefold()).strip()
            if artist_key and artist_key not in seed_artists:
                artists.setdefault(artist_key, artist)
    return list(artists.values())[:1]


def _interleave_recommendation_contexts(
    items: list[dict],
    seed_year: int,
    include_unknown_year: bool = False,
) -> list[dict]:
    """Favor the seed decade, then mix in close decades by mood relevance."""
    seed_decade = seed_year // 10 * 10 if seed_year else 0
    buckets: dict[int, list[tuple[int, dict]]] = {0: [], 1: [], 2: []}
    unknown_year_items: list[dict] = []

    def context_order(
        entry: tuple[int, dict],
    ) -> tuple[float, float, int]:
        index, item = entry
        taste = max(
            float(item.get("_listener_mood_match", 0.0) or 0.0),
            float(item.get("_audio_match", 0.0) or 0.0),
            float(item.get("_mood_match", 0.0) or 0.0),
            float(item.get("_intent_match", 0.0) or 0.0),
            float(item.get("_short_term", 0.0) or 0.0),
            float(item.get("_medium_term", 0.0) or 0.0),
            float(item.get("_long_term", 0.0) or 0.0),
        )
        era_mood = (
            0.55 * float(item.get("_era_match", 0.0) or 0.0)
            + 0.45 * taste
        )
        return (-era_mood, -taste, index)

    if not seed_decade:
        return items
    for index, item in enumerate(items):
        year = _release_year(item.get("year"))
        if not year:
            if include_unknown_year:
                unknown_year_items.append(item)
            continue
        if abs(year - seed_year) > MAX_RECOMMENDATION_YEAR_DISTANCE:
            continue
        decade_delta = abs(year // 10 * 10 - seed_decade)
        bucket = 0 if decade_delta == 0 else 1 if decade_delta == 10 else 2
        buckets[bucket].append((index, item))
    for bucket_number, bucket in buckets.items():
        remaining = sorted(bucket, key=context_order)
        ordered_bucket: list[tuple[int, dict]] = []
        artist_counts: dict[str, int] = {}
        while remaining:
            repeat_counts = [
                max(
                    (artist_counts.get(artist, 0) for artist in _artist_credit_keys(item.get("artists", ""))),
                    default=0,
                )
                for _, item in remaining
            ]
            least_repeated = min(repeat_counts)
            eligible = [
                entry
                for entry, repeats in zip(remaining, repeat_counts)
                if repeats == least_repeated
            ]
            chosen = min(eligible, key=context_order)
            remaining.remove(chosen)
            ordered_bucket.append(chosen)
            for artist in _artist_credit_keys(chosen[1].get("artists", "")):
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
        buckets[bucket_number] = ordered_bucket

    schedule = (0, 0, 0, 0, 0, 0, 0, 1, 1, 2)
    ordered: list[dict] = []
    schedule_index = 0
    while any(buckets.values()):
        preferred_bucket = schedule[schedule_index % len(schedule)]
        schedule_index += 1
        available = [bucket for bucket, entries in buckets.items() if entries]
        bucket = (
            preferred_bucket
            if buckets[preferred_bucket]
            else min(available, key=lambda value: (abs(value - preferred_bucket), value))
        )
        _, item = buckets[bucket].pop(0)
        ordered.append(item)
    return ordered + unknown_year_items


def _prioritize_distinct_soundtracks(items: list[dict]) -> list[dict]:
    unique: list[dict] = []
    repeats: list[dict] = []
    seen_work_keys: set[str] = set()
    for item in items:
        work_key = _recommendation_work_key(item)
        if work_key and work_key in seen_work_keys:
            repeats.append(item)
        else:
            unique.append(item)
            if work_key:
                seen_work_keys.add(work_key)
    return unique + repeats


def _metadata_feature_tokens(item: dict) -> dict[str, float]:
    """Create a generic sparse metadata vector for diversity scoring."""
    text = " ".join(
        str(item.get(field, "")).lower()
        for field in ("title", "album", "artists", "language")
    )
    counts: dict[str, float] = {}
    for token in re.findall(r"[a-z0-9]+", text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _semantic_metadata_tokens(item: dict) -> dict[str, float]:
    """Build relevance tokens without repeating artist or compilation matches."""
    title = recommendation_title_key(item.get("title", ""))
    text = " ".join(
        (title, str(item.get("track_genre", "")), str(item.get("genre", "")))
    ).lower()
    counts: dict[str, float] = {}
    for token in re.findall(r"[a-z0-9]+", text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left).intersection(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _candidate_similarity(left: dict, right: dict) -> float:
    """Cosine-like similarity used only for MMR diversity reranking."""
    text_score = _sparse_cosine(
        left.get("_feature_tokens", {}), right.get("_feature_tokens", {})
    )
    left_mood = np.asarray(left.get("mood_vector", []), dtype=float)
    right_mood = np.asarray(right.get("mood_vector", []), dtype=float)
    mood_score = 0.0
    if left_mood.size and left_mood.size == right_mood.size:
        denominator = float(np.linalg.norm(left_mood) * np.linalg.norm(right_mood))
        mood_score = float(np.dot(left_mood, right_mood) / denominator) if denominator else 0.0
    left_year = int(left.get("year", 0) or 0)
    right_year = int(right.get("year", 0) or 0)
    era_score = float(np.exp(-((left_year - right_year) / 18.0) ** 2)) if left_year and right_year else 0.0
    return 0.55 * text_score + 0.30 * max(0.0, mood_score) + 0.15 * era_score


def _canonical_recording_key(item: dict) -> str:
    """Collapse alternate releases of one recording for a recommendation queue."""
    # Only an actual ISRC is a recording identifier. Provider song IDs often
    # represent releases, so using them here would miss duplicate versions.
    isrc = str(item.get("isrc") or "").strip().casefold()
    if isrc:
        return f"isrc:{isrc}"
    title = recommendation_title_key(item.get("title", ""))
    artists = re.sub(r"[^a-z0-9]+", " ", str(item.get("artists", "")).casefold()).strip()
    return f"{title}|{artists}" if title else artists


def recommendation_title_key(value: str) -> str:
    """Normalize release annotations so alternate versions do not repeat in a queue."""
    title = str(value).casefold()
    def strip_release_annotation(match: re.Match[str]) -> str:
        annotation = match.group(0)
        if re.search(
            r"\b(from|slowed|sped|nightcore|remix|reprise|live|instrumental|"
            r"karaoke|radio edit|vocals?\s+only|version|remaster|feat\.?)\b",
            annotation,
        ):
            return " "
        return annotation

    title = re.sub(r"\([^)]*\)|\[[^]]*\]", strip_release_annotation, title)
    title = re.sub(
        r"\s*[-–—]\s*[^()]*\b(vocals?\s+only|instrumental|karaoke|radio edit)\b.*$",
        " ",
        title,
    )
    title = re.sub(
        r"(?:\s+[-–—]\s*|\s+)(slowed(?:\s+down)?|sped(?:\s+up)?|nightcore|"
        r"remix|reprise|live|instrumental|karaoke|radio edit|"
        r"vocals?\s+only|version)\b.*$",
        " ",
        title,
    )
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def _is_excluded_recommendation_title(value: str) -> bool:
    title = recommendation_title_key(value)
    return any(
        title == excluded or title.startswith(f"{excluded} ")
        for excluded in EXCLUDED_RECOMMENDATION_TITLES
    )


def recommendation_history_keys(item: dict) -> set[str]:
    """Build stable session exclusions, tolerating alternate artist credits."""
    recording_key = _canonical_recording_key(item)
    keys = {recording_key} if recording_key else set()
    title_key = recommendation_title_key(item.get("title", ""))
    if len(title_key.split()) >= 3:
        keys.add(f"title:{title_key}")
    return keys


INTENT_KEYWORDS = {
    "romantic": {"love", "romantic", "ishq", "mohabbat", "pyaar", "pyar", "dil", "jaan", "milan", "sanam", "aashiqui", "prem", "saiyaara"},
    "playful": {"playful", "comic", "comedy", "humorous", "funny", "lighthearted", "light hearted", "mischief", "masti", "nautanki", "chatur", "nok jhok", "banter", "witty", "theatrical", "expressive", "call and response", "call response", "jugalbandi"},
    "ghazal": {"ghazal", "ghazals", "mehfil", "rekhta", "shayari", "ghulam ali", "chupke chupke classics"},
    "devotional": {"bhajan", "chalisa", "gurbani", "hanuman", "bajrang", "shiva", "shivay", "mahadev", "krishna", "radhe", "ghanshyam", "aarti", "qawwali devotional"},
    "motivational": {"motivation", "motivational", "fight", "fighting", "battle", "victory", "fighter", "warrior", "rise", "zinda", "jeet", "inspire", "goal", "champion", "unstoppable"},
    "energetic": {"fight", "fighting", "anthem", "rock", "dance", "party", "bhangra", "dhamaka", "dhamaal", "energetic", "power", "action", "high energy"},
    "acoustic_indie": {"acoustic", "unplugged", "indie", "folk", "coke studio", "singer songwriter"},
    "rap_hiphop": {"rap", "hip hop", "hiphop", "freestyle", "gully", "mixtape"},
    "calm": {"lofi", "sleep", "calm", "peace", "soothing", "sukoon", "instrumental"},
}


def infer_audio_intents(item: dict) -> set[str]:
    """Infer broad musical feel from audio features when those are available."""
    try:
        energy = float(item.get("energy", 0.0) or 0.0)
        valence = float(item.get("valence", 0.0) or 0.0)
        danceability = float(item.get("danceability", 0.0) or 0.0)
        tempo = float(item.get("tempo", 0.0) or 0.0)
    except (TypeError, ValueError):
        return set()
    if (
        energy >= 0.40
        and valence >= 0.72
        and danceability >= 0.50
        and 90 <= tempo <= 160
    ):
        return {"playful"}
    return set()


def infer_same_soundtrack_audio_intents(seed: dict, candidates: list[dict]) -> set[str]:
    """Use matching soundtrack peers as a low-confidence intent hint for sparse seeds."""
    if infer_audio_intents(seed):
        return set()

    seed_work = _recommendation_work_key(seed)
    if not seed_work:
        return set()

    peer_intents = []
    for candidate in candidates:
        if _recommendation_work_key(candidate) != seed_work:
            continue
        if str(candidate.get("mood_source", "")) != "local_audio_catalog":
            continue
        if not any(candidate.get(key) is not None for key in ("energy", "valence", "danceability", "tempo")):
            continue
        peer_intents.append(infer_audio_intents(candidate))
    if not peer_intents:
        return set()

    intent_counts: dict[str, int] = {}
    for intents in peer_intents:
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    minimum_votes = len(peer_intents) // 2 + 1
    return {
        intent for intent, count in intent_counts.items()
        if count >= minimum_votes
    }


def _infer_candidate_intents(item: dict) -> set[str]:
    """Infer musical intent from the track's own metadata, never its artist."""
    supplied_labels = item.get("intent_labels")
    if isinstance(supplied_labels, (list, tuple, set)):
        labels = {
            str(label).strip().casefold()
            for label in supplied_labels
            if str(label).strip().casefold() not in {"", "general", "neutral"}
        }
        if labels:
            return labels | infer_audio_intents(item)
    text = " ".join(
        str(item.get(field, ""))
        for field in (
            "title", "album", "track_genre", "genre", "language", "mood",
            "description", "tags", "intent_labels",
        )
    ).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    labels = {
        label for label, terms in INTENT_KEYWORDS.items()
        if any(term in text if " " in term else term in tokens for term in terms)
    }
    return labels | infer_audio_intents(item)


def infer_metadata_mood_labels(item: dict) -> set[str]:
    """Infer low-confidence mood labels without using artist names as evidence."""
    text = " ".join(
        str(item.get(field, ""))
        for field in (
            "title", "album", "language", "mood", "description", "tags",
            "track_genre", "genre",
        )
    ).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    groups = {
        "playful": {
            "playful", "comic", "comedy", "humorous", "funny", "lighthearted",
            "light hearted", "mischief", "masti", "nautanki", "chatur",
            "nok jhok", "banter", "witty", "theatrical", "expressive",
            "call and response", "call response", "jugalbandi",
        },
        "energetic": {
            "fight", "anthem", "rock", "zinda", "sultan", "power", "action",
            "dance", "party", "bhangra", "bang", "fire", "dhamaka", "warrior",
            "bhaag", "milkha", "singham", "chaiyya", "ramta", "appadipodu",
            "mappillai",
        },
        "romantic": {
            "love", "romantic", "ishq", "mohabbat", "pyaar", "dil", "jaan",
            "saiyaara",
        },
        "sad": {"sad", "heartbreak", "alone", "broken", "dard", "udaas"},
        "instrumental": {
            "theme", "score", "background", "instrumental", "bgm", "soundtrack",
        },
        "acoustic_indie": {
            "acoustic", "unplugged", "indie", "folk", "singer songwriter",
            "singer-songwriter", "coke studio", "lofi", "slowed", "reverb",
        },
        "ghazal": {
            "ghazal", "ghazals", "shayar", "shayari", "mehfil", "rekhta",
            "ghulam ali", "ghulam ali classics", "ustad bade ghulam ali",
            "ghulam ali khan", "chupke chupke classics",
        },
        "rap_hiphop": {
            "rap", "rapper", "hip hop", "hiphop", "gully", "freestyle",
            "mixtape",
        },
    }
    return {
        label
        for label, words in groups.items()
        if any(word in text if " " in word else word in tokens for word in words)
    }
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from ranking_model import LearnedRanker


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
        "jio_skip": -1.0,
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
        self.user_skipped_items: dict[str, set[str]] = {}
        self.short_artist_preferences: dict[str, dict[str, float]] = {}
        self.medium_artist_preferences: dict[str, dict[str, float]] = {}
        self.long_artist_preferences: dict[str, dict[str, float]] = {}
        self.user_mood_profiles: dict[str, np.ndarray] = {}
        self.user_era_profiles: dict[str, dict[int, float]] = {}
        self.bandit_counts: dict[str, float] = {}
        self.bandit_rewards: dict[str, float] = {}
        self.learned_ranker = LearnedRanker()

    def fit(self, events: list[dict] | pd.DataFrame) -> "HybridRecommendationEngine":
        frame = pd.DataFrame(events)
        self.user_index = {}
        self.item_index = {}
        self.user_factors = np.empty((0, 0))
        self.item_factors = np.empty((0, 0))
        self.item_popularity = {}
        self.transition_scores = {}
        self.user_recent_items = {}
        self.user_skipped_items = {}
        self.short_artist_preferences = {}
        self.medium_artist_preferences = {}
        self.long_artist_preferences = {}
        self.user_mood_profiles = {}
        self.user_era_profiles = {}
        self.learned_ranker.fit(frame)
        self.bandit_counts = {}
        self.bandit_rewards = {}
        if not frame.empty and {"track_id", "event_type"}.issubset(frame.columns):
            # Contextual-bandit feedback: observed actions only. Impressions
            # provide exposure counts; play/like/skip provide rewards.
            for row in frame.to_dict("records"):
                track_id = str(row.get("track_id") or "")
                event = str(row.get("event_type") or "")
                if not track_id:
                    continue
                if event == "recommendation_impression":
                    try:
                        payload = json.loads(row.get("metadata_json") or "{}")
                        for item in payload.get("items", []):
                            exposed_id = str(item.get("track_id") or "")
                            if exposed_id:
                                self.bandit_counts[exposed_id] = self.bandit_counts.get(exposed_id, 0.0) + 1.0
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                reward = {"jio_play": 1.0, "play_start": 1.0, "jio_like": 2.0, "like": 2.0, "jio_skip": -1.0, "skip": -1.0}.get(event)
                if reward is not None:
                    self.bandit_counts[track_id] = self.bandit_counts.get(track_id, 0.0) + 1.0
                    self.bandit_rewards[track_id] = self.bandit_rewards.get(track_id, 0.0) + reward
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
        # Two preference horizons: recent intent should react quickly, while
        # the long horizon keeps stable taste from disappearing after one skip.
        self.short_artist_preferences = {}
        self.medium_artist_preferences = {}
        self.long_artist_preferences = {}
        mood_sums: dict[str, np.ndarray] = {}
        mood_weights: dict[str, float] = {}
        era_weights: dict[str, dict[int, float]] = {}
        now = pd.Timestamp.now(tz="UTC")
        for row in frame.itertuples(index=False):
            metadata = getattr(row, "metadata_json", "") or ""
            try:
                payload = json.loads(metadata) if isinstance(metadata, str) else {}
            except (TypeError, ValueError):
                payload = {}
            artists = [
                part.strip().casefold()
                for part in str(payload.get("artists", "")).split(",")
                if part.strip()
            ]
            try:
                created_at = getattr(row, "created_at")
                age_days = (
                    max(0.0, (now - created_at).total_seconds() / 86400.0)
                    if pd.notna(created_at) else 30.0
                )
            except (AttributeError, TypeError, ValueError):
                age_days = 30.0
            weight = float(getattr(row, "weight", 0.0) or 0.0)
            short_weight = weight * float(np.exp(-age_days / 14.0))
            long_weight = weight * float(np.exp(-age_days / 180.0))
            user = str(getattr(row, "user_id", ""))
            if artists:
                short_map = self.short_artist_preferences.setdefault(user, {})
                medium_map = self.medium_artist_preferences.setdefault(user, {})
                long_map = self.long_artist_preferences.setdefault(user, {})
                for artist in artists:
                    short_map[artist] = short_map.get(artist, 0.0) + short_weight
                    medium_weight = weight * float(np.exp(-age_days / 30.0))
                    medium_map[artist] = medium_map.get(artist, 0.0) + medium_weight
                    long_map[artist] = long_map.get(artist, 0.0) + long_weight
            if weight <= 0:
                continue
            try:
                mood = np.asarray(payload.get("mood_vector", []), dtype=float)
            except (TypeError, ValueError):
                mood = np.asarray([])
            if mood.ndim == 1 and mood.size == 9 and np.all(np.isfinite(mood)):
                mood_sums[user] = mood_sums.get(user, np.zeros_like(mood)) + mood * short_weight
                mood_weights[user] = mood_weights.get(user, 0.0) + short_weight
            try:
                year = int(str(payload.get("year", ""))[:4])
            except (TypeError, ValueError):
                year = 0
            if 1900 <= year <= 2100 and long_weight > 0:
                decade = year // 10 * 10
                user_eras = era_weights.setdefault(user, {})
                user_eras[decade] = user_eras.get(decade, 0.0) + long_weight
        self.user_mood_profiles = {
            user: mood_sums[user] / mood_weights[user]
            for user in mood_sums
            if mood_weights.get(user, 0.0) > 0
        }
        self.user_era_profiles = era_weights
        frame = frame.sort_values(["user_id", "session_id", "created_at"] if "session_id" in frame else ["user_id", "created_at"])

        # Recent listening context: used for freshness and repeat avoidance.
        played_events = frame[frame["event_type"].isin(["jio_play", "play_start"])]
        self.user_recent_items = {
            str(user_id): group["track_id"].astype(str).tolist()[-12:]
            for user_id, group in played_events.groupby("user_id")
        }
        self.user_skipped_items = {
            str(user_id): set(group.loc[group["event_type"].isin(["skip", "jio_skip"]), "track_id"].astype(str))
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
        # Factorize only positive implicit confidence. Shifting a row that
        # contains a skip changes untouched zeroes into fake positive plays.
        # Explicit skips remain a separate penalty in collaborative_scores.
        matrix = np.clip(interactions.to_numpy(dtype=float), 0.0, None)
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
        skipped = self.user_skipped_items.get(str(user_id), set())
        if skipped:
            scores = np.asarray([
                max(0.0, float(score) - 0.35) if str(item_id) in skipped else float(score)
                for item_id, score in zip(item_ids, scores)
            ])
        return scores

    def rank_jio_results(
        self,
        results: list[dict],
        user_id: str,
        catalog: pd.DataFrame,
        preferred_track_ids: list[str],
        seed: dict | None = None,
        relaxed_context: bool = False,
    ) -> list[dict]:
        if not results:
            return []
        # Keep the local feature namespace initialized even when a live item
        # has incomplete metadata. This prevents stale deployments from
        # raising an UnboundLocalError before the candidate loop starts.
        feature_tokens: dict[str, float] = {}
        # Normalize duplicate API versions before scoring so one song cannot
        # occupy multiple positions in the ranked queue.
        deduped: list[dict] = []
        seen_result_keys: set[str] = set()
        for item in results:
            result_key = _canonical_recording_key(item)
            if not result_key:
                result_key = re.sub(
                    r"[^a-z0-9]+", " ",
                    f"{item.get('title', '')} {item.get('artists', '')}".lower(),
                ).strip()
            if result_key and result_key not in seen_result_keys:
                seen_result_keys.add(result_key)
                deduped.append(item)
        results = deduped
        seed_year = _release_year((seed or {}).get("year"))
        if is_compilation_album((seed or {}).get("album", "")):
            seed_year = 0
        if seed:
            seed_identity = recommendation_history_keys(seed)
            results = [
                item for item in results
                if not seed_identity.intersection(recommendation_history_keys(item))
            ]
            results = [
                item for item in results
                if not _is_excluded_recommendation_title(item.get("title", ""))
            ]
            if seed_year:
                compatible: list[dict] = []
                for item in results:
                    item_year = _release_year(item.get("year"))
                    if not item_year:
                        compatible.append(item)
                    elif abs(item_year - seed_year) <= MAX_RECOMMENDATION_YEAR_DISTANCE:
                        compatible.append(item)
                results = compatible
        if not results:
            return []
        preferred = catalog[catalog["track_id"].astype(str).isin({str(x) for x in preferred_track_ids})]
        preferred_artists = [str(x).lower() for x in preferred.get("artists", pd.Series(dtype=str)).tolist()]
        preferred_titles = [str(x).lower() for x in preferred.get("track_name", pd.Series(dtype=str)).tolist()]
        item_ids = [str(item.get("song_id") or item.get("title", "")) for item in results]
        collaborative = self.collaborative_scores(user_id, item_ids)
        for item, value in zip(results, collaborative):
            item["_collaborative_score"] = float(value)
        seed_id = str((seed or {}).get("song_id") or "")
        transition_map = self.transition_scores.get(seed_id, {})
        transition_values = np.asarray(
            [transition_map.get(item_id, 0.0) for item_id in item_ids], dtype=float
        )
        if transition_values.size and transition_values.max() > 0:
            transition_values = transition_values / transition_values.max()
        # Recent items are a hard freshness constraint for a recommendation
        # queue. Repeated songs are annoying even when their collaborative
        # score is high; the current seed is already excluded by the caller.
        # Keep freshness as a soft signal for sparse candidate pools. A hard
        # exclusion of every recently played track can remove the entire
        # relevant set for users with short histories.
        recent_items = set(self.user_recent_items.get(str(user_id), [])[-12:])
        has_seed = bool(seed and any((seed.get(field) or "") for field in ("title", "artists", "album")))
        seed_text = " ".join(
            str((seed or {}).get(field, ""))
            for field in ("title", "album", "artists", "language")
        ).lower()
        style_groups = {
            "high_energy": {"fight", "anthem", "rock", "zinda", "sultan", "brother", "run", "power", "action", "dance", "party", "bhangra", "bang", "fire", "dhamaka", "dhamaal", "warrior", "sher", "bhaag", "milkha", "singham", "chaiyya", "ramta", "appadipodu", "mappillai"},
            "romantic": {"love", "romantic", "ishq", "mohabbat", "pyaar", "dil", "jaan", "saiyaara"},
            "sad": {"sad", "heartbreak", "alone", "broken", "dard", "udaas"},
            "devotional": {"bhajan", "chalisa", "gurbani", "hanuman", "shiva", "ram"},
            "acoustic_indie": {"lofi", "acoustic", "unplugged", "instrumental", "slowed", "reverb", "indie", "folk", "coke studio", "singer songwriter"},
            "rap_hiphop": {"rap", "rapper", "hip hop", "hiphop", "gully", "freestyle", "mixtape"},
        }
        seed_tokens = set(re.findall(r"[a-z0-9]+", seed_text))
        seed_styles = {
            group for group, words in style_groups.items()
            if any((word in seed_text if " " in word else word in seed_tokens) for word in words)
        }
        seed_styles = seed_styles or {"neutral"}
        seed_mood = np.asarray((seed or {}).get("mood_vector", []), dtype=float)
        listener_mood = self.user_mood_profiles.get(str(user_id), np.asarray([]))
        listener_eras = self.user_era_profiles.get(str(user_id), {})
        seed_feature_tokens = _semantic_metadata_tokens(seed or {})
        seed_confidence = float((seed or {}).get("mood_confidence", 0.0) or 0.0)
        seed_intents = _infer_candidate_intents(seed or {})

        def mood_similarity(item: dict) -> float:
            item_mood = np.asarray(item.get("mood_vector", []), dtype=float)
            if seed_mood.size == 0 or item_mood.size == 0 or seed_mood.size != item_mood.size:
                return 0.0
            seed_confidence = float((seed or {}).get("mood_confidence", 0.0) or 0.0)
            item_confidence = float(item.get("mood_confidence", 0.0) or 0.0)
            seed_label = str((seed or {}).get("mood_label", "")).strip().lower()
            item_label = str(item.get("mood_label", "")).strip().lower()
            # Default/neutral metadata vectors are not evidence of similarity.
            # Require a confident shared label for fallback-vs-fallback, while
            # local audio vectors can compare directly.
            if seed_confidence < 0.75 or item_confidence < 0.75:
                if seed_confidence < 0.5 or item_confidence < 0.5:
                    return 0.0
                seed_labels = {part.strip() for part in seed_label.split(",") if part.strip() and part.strip() != "neutral"}
                item_labels = {part.strip() for part in item_label.split(",") if part.strip() and part.strip() != "neutral"}
                if not seed_labels.intersection(item_labels):
                    return 0.0
            denominator = float(np.linalg.norm(seed_mood) * np.linalg.norm(item_mood))
            return max(0.0, float(np.dot(seed_mood, item_mood) / denominator)) if denominator else 0.0

        def listener_mood_similarity(item: dict) -> float:
            item_mood = np.asarray(item.get("mood_vector", []), dtype=float)
            if (
                listener_mood.size == 0
                or item_mood.size != listener_mood.size
                or str(item.get("mood_source", "")) != "local_audio_catalog"
            ):
                return 0.0
            denominator = float(np.linalg.norm(listener_mood) * np.linalg.norm(item_mood))
            return (
                max(0.0, float(np.dot(listener_mood, item_mood) / denominator))
                if denominator else 0.0
            )

        def listener_era_similarity(year: int) -> float:
            total_weight = sum(listener_eras.values())
            if not year or total_weight <= 0:
                return 0.0
            return sum(
                weight * float(np.exp(-((year - (decade + 5)) / 18.0) ** 2))
                for decade, weight in listener_eras.items()
            ) / total_weight

        content = []
        seed_artists = [
            value.strip().lower()
            for value in str((seed or {}).get("artists", "")).split(",")
            if value.strip()
        ]
        seed_album = str((seed or {}).get("album", "")).strip().lower()
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
            item_year = _release_year(item.get("year"))
            era_delta = abs(seed_year - item_year) if seed_year and item_year else 0
            seed_source_work = recommendation_source_work(seed or {})
            item_source_work = recommendation_source_work(item)
            if seed_source_work or item_source_work:
                album_match = float(
                    bool(seed_source_work)
                    and seed_source_work == item_source_work
                )
            else:
                album_name = str(item.get("album", "")).strip()
                album_match = float(
                    bool(seed_album)
                    and seed_album == album_name.lower()
                    and not is_compilation_album(seed_album)
                    and not is_compilation_album(album_name)
                    and (not seed_year or not item_year or era_delta <= 3)
                )
            era_match = (
                float(np.exp(-((seed_year - item_year) / 18.0) ** 2))
                if seed_year and item_year else 0.0
            )
            same_era = bool(era_match >= 0.70)
            language_match = 1.0 if seed_language and language and seed_language == language else 0.0
            language_conflict = bool(seed_language and language and seed_language != language)
            seed_cluster = (seed or {}).get("cluster_id")
            item_cluster = item.get("cluster_id")
            cluster_match = (
                1.0
                if seed_cluster is not None and item_cluster is not None and str(seed_cluster) == str(item_cluster)
                else 0.0
            )
            # Metadata-level semantics: singer, album, era and language are
            # stronger evidence than title overlap alone for external songs.
            item_text = " ".join(
                str(item.get(field, ""))
                for field in ("title", "album", "artists", "language")
            ).lower()
            item_styles = {
                group for group, words in style_groups.items()
                if any((word in item_text if " " in word else word in set(re.findall(r"[a-z0-9]+", item_text))) for word in words)
            } or {"neutral"}
            style_match = 1.0 if (
                seed_styles != {"neutral"}
                and item_styles != {"neutral"}
                and seed_styles.intersection(item_styles)
            ) else 0.0
            mood_match = mood_similarity(item)
            listener_mood_match = listener_mood_similarity(item)
            listener_era_match = listener_era_similarity(item_year)
            # Cosine similarity is audio evidence only when both sides have
            # real catalog features. Metadata fallback vectors must not
            # masquerade as audio.
            audio_match = mood_match if (
                str((seed or {}).get("mood_source", "")) == "local_audio_catalog"
                and str(item.get("mood_source", "")) == "local_audio_catalog"
            ) else 0.0
            metadata_similarity = _sparse_cosine(
                seed_feature_tokens, _semantic_metadata_tokens(item)
            )
            item_intents = _infer_candidate_intents(item)
            intent_match = (
                len(seed_intents.intersection(item_intents)) / max(len(seed_intents), 1)
                if seed_intents else 0.0
            )
            intent_conflict = bool(seed_intents and item_intents and not seed_intents.intersection(item_intents))
            metadata_score = (
                0.35 * intent_match
                + 0.20 * era_match
                + 0.15 * language_match
                + 0.15 * style_match
                + 0.15 * metadata_similarity
            )
            # Current-track context is deliberately stronger than history.
            context_score = (
                0.15 * seed_artist_match
                + 0.12 * album_match
                + 0.08 * era_match
                + 0.08 * language_match
                + 0.12 * cluster_match
                + 0.25 * mood_match
                + 0.10 * metadata_similarity
                + 0.10 * intent_match
            )
            seed_content = (
                0.40 * seed_artist_match
                + 0.17 * album_match
                + 0.16 * era_match
                + 0.10 * language_match
                + 0.17 * cluster_match
            )
            content.append(max(profile_content, seed_content))
            item["_context_score"] = context_score
            item["_profile_score"] = profile_content
            item["_seed_artist_match"] = seed_artist_match
            item["_album_match"] = album_match
            item["_language_match"] = language_match
            item["_language_conflict"] = language_conflict
            item["_era_match"] = era_match
            item["_cluster_match"] = cluster_match
            item["_style_match"] = style_match
            item["_mood_match"] = mood_match
            item["_listener_mood_match"] = listener_mood_match
            item["_listener_era_match"] = listener_era_match
            item["_audio_match"] = audio_match
            item["_metadata_score"] = metadata_score
            item["_feature_tokens"] = _metadata_feature_tokens(item)
            item["_metadata_similarity"] = metadata_similarity
            item["_intent_match"] = intent_match
            item["_intent_conflict"] = intent_conflict
            item["_era_delta"] = era_delta
            item["_same_era"] = same_era
            audio_verified = audio_match >= 0.62
            if album_match and audio_verified and mood_match >= 0.62:
                item["recommendation_reason"] = "Same soundtrack and similar audio profile"
            elif album_match:
                item["recommendation_reason"] = "Same film or soundtrack"
            elif audio_verified and mood_match >= 0.78 and same_era:
                item["recommendation_reason"] = "Similar mood, era and sound"
            elif audio_verified and mood_match >= 0.78 and era_delta:
                item["recommendation_reason"] = "Similar mood with a different era"
            elif audio_verified and mood_match >= 0.78:
                item["recommendation_reason"] = "Very similar mood and sound"
            elif seed_artist_match and audio_verified and mood_match >= 0.62:
                item["recommendation_reason"] = "Same artist and similar audio profile"
            elif seed_artist_match and mood_match >= 0.62:
                item["recommendation_reason"] = "Same artist; metadata suggests a similar vibe"
            elif mood_match >= 0.62:
                item["recommendation_reason"] = "Metadata suggests a similar vibe; audio not verified"
            elif metadata_similarity >= 0.45:
                item["recommendation_reason"] = "Similar title, album and artist context"
            elif intent_match >= 0.5:
                item["recommendation_reason"] = "Metadata suggests a matching listening vibe"
            elif style_match and language_match:
                item["recommendation_reason"] = "Similar energy and language"
            elif style_match:
                item["recommendation_reason"] = "Similar energy and style"
            elif language_match and era_match >= 0.65:
                item["recommendation_reason"] = "Similar language and era"
            elif cluster_match:
                item["recommendation_reason"] = "Similar mood and sound cluster"
            elif era_match >= 0.65:
                item["recommendation_reason"] = "Similar era and listening style"
            elif artist_match and context_score >= 0.35:
                item["recommendation_reason"] = "Your taste, matched to this session"
            else:
                item["recommendation_reason"] = "Fits your current listening context"
        # Long-term and short-term artist embeddings are represented as
        # recency-weighted preference vectors. They are cheap, interpretable,
        # and remain useful when live IDs do not match the local catalog.
        short_artist = self.short_artist_preferences.get(str(user_id), {})
        medium_artist = self.medium_artist_preferences.get(str(user_id), {})
        long_artist = self.long_artist_preferences.get(str(user_id), {})
        short_max = max([abs(value) for value in short_artist.values()] or [1.0])
        medium_max = max([abs(value) for value in medium_artist.values()] or [1.0])
        long_max = max([abs(value) for value in long_artist.values()] or [1.0])
        for item in results:
            artists = [part.strip().casefold() for part in str(item.get("artists", "")).split(",") if part.strip()]
            short_score = min(1.0, max([short_artist.get(artist, 0.0) for artist in artists] or [0.0]) / short_max)
            medium_score = min(1.0, max([medium_artist.get(artist, 0.0) for artist in artists] or [0.0]) / medium_max)
            long_score = min(1.0, max([long_artist.get(artist, 0.0) for artist in artists] or [0.0]) / long_max)
            item["_short_term"] = max(0.0, short_score)
            item["_medium_term"] = max(0.0, medium_score)
            item["_long_term"] = max(0.0, long_score)
            item["recommendation_features"] = {
                "context": float(item.get("_context_score", 0.0)),
                "collaborative": float(item.get("_collaborative_score", 0.0)),
                "mood": float(item.get("_mood_match", 0.0)),
                "audio": float(item.get("_audio_match", 0.0)),
                "intent": float(item.get("_intent_match", 0.0)),
                "metadata": float(item.get("_metadata_similarity", 0.0)),
                "era": float(item.get("_era_match", 0.0)),
                "language": float(item.get("_language_match", 0.0)),
                "transition": float(transition_map.get(str(item.get("song_id") or item.get("title", "")), 0.0)),
                "short_term": item["_short_term"],
                "medium_term": item["_medium_term"],
                "long_term": item["_long_term"],
                "audio_confidence": float(item.get("mood_confidence", 0.0) or 0.0),
                "listener_mood": listener_mood_match,
                "listener_era": listener_era_match,
            }
        content = np.asarray(content, dtype=float)
        popularity = np.asarray([
            self.item_popularity.get(item_id, 0.0) for item_id in item_ids
        ], dtype=float)
        total_bandit = max(1.0, sum(self.bandit_counts.values()))
        bandit_values = np.asarray([
            (self.bandit_rewards.get(item_id, 0.0) / max(1.0, self.bandit_counts.get(item_id, 0.0)))
            + 0.08 * np.sqrt(np.log1p(total_bandit) / (1.0 + self.bandit_counts.get(item_id, 0.0)))
            for item_id in item_ids
        ], dtype=float)
        if bandit_values.size and bandit_values.max() > bandit_values.min():
            bandit_values = (bandit_values - bandit_values.min()) / (bandit_values.max() - bandit_values.min())
        # Content dominates cold-start. Once enough events exist, the
        # transition and collaborative signals influence the next-track order.
        # Context-first ranking: the current track defines the mood, while
        # history is injected only as a controlled secondary signal.
        context_values = np.asarray([float(item.get("_context_score", 0.0)) for item in results])
        artist_anchor_values = np.asarray([float(item.get("_seed_artist_match", 0.0)) for item in results])
        era_values = np.asarray([float(item.get("_era_match", 0.0)) for item in results])
        mood_values = np.asarray([float(item.get("_mood_match", 0.0)) for item in results])
        intent_values = np.asarray([float(item.get("_intent_match", 0.0)) for item in results])
        profile_values = np.asarray([float(item.get("_profile_score", 0.0)) for item in results])
        audio_values = np.asarray([float(item.get("_audio_match", 0.0)) for item in results])
        metadata_values = np.asarray([float(item.get("_metadata_score", 0.0)) for item in results])
        short_values = np.asarray([item.get("_short_term", 0.0) for item in results])
        medium_values = np.asarray([item.get("_medium_term", 0.0) for item in results])
        long_values = np.asarray([item.get("_long_term", 0.0) for item in results])
        if has_seed:
            # Audio and mood are the same cosine signal when both items have
            # verified features. Use metadata mood only as a fallback, and do
            # not renormalize by positive-score count: a legitimate zero is
            # evidence of dissimilarity, not missing data.
            mood_fallback_values = np.where(audio_values > 0.0, 0.0, mood_values)
            hybrid = (
                0.28 * audio_values
                + 0.14 * metadata_values
                + 0.05 * collaborative
                + 0.04 * (0.60 * short_values + 0.40 * medium_values)
                + 0.03 * np.asarray([item.get("_listener_mood_match", 0.0) for item in results])
                + 0.03 * np.asarray([item.get("_listener_era_match", 0.0) for item in results])
                + 0.07 * transition_values
                + 0.16 * context_values
                + 0.06 * artist_anchor_values
                + 0.08 * era_values
                + 0.04 * mood_fallback_values
                + 0.03 * intent_values
                + 0.03 * bandit_values
                + 0.01 * popularity
            )
        else:
            # Search results have no current-track anchor; preserve the
            # original taste/collaborative search ranking in that view.
            hybrid = (
                0.45 * content + 0.22 * collaborative + 0.12 * transition_values
                + 0.08 * popularity
                + 0.08 * short_values
                + 0.05 * long_values
            )

        # A learned layer is intentionally a small final correction. It only
        # activates after impression/play labels pass its data threshold.
        for item, score in zip(results, hybrid):
            item["recommendation_score"] = float(score)
        learned = self.learned_ranker.predict(results)
        if learned.size and np.any(learned):
            hybrid = 0.88 * hybrid + 0.12 * learned
        if hybrid.size and hybrid.max() > hybrid.min():
            hybrid = (hybrid - hybrid.min()) / (hybrid.max() - hybrid.min())

        # Sequence-aware reranking: avoid immediate repeats and prevent one
        # artist/album from occupying the whole Up Next queue.
        ranked = sorted(zip(hybrid, results), key=lambda pair: pair[0], reverse=True)
        selected: list[dict] = []
        artist_counts: dict[str, int] = {}
        album_counts: dict[str, int] = {}
        remaining = ranked.copy()
        while remaining:
            eligible: list[tuple[int, float, float, float]] = []
            for index, (base_score, item) in enumerate(remaining):
                item_id = str(item.get("song_id") or item.get("title", ""))
                artist_keys = _artist_credit_keys(item.get("artists", ""))
                album_key = _recommendation_work_key(item) or str(item.get("album", "")).lower()
                title_key = re.sub(r"[^a-z0-9]+", " ", str(item.get("title", "")).lower()).strip()
                # Do not surface recently played songs or weak fallback items.
                # A user-history recommendation is allowed only when it also
                # has meaningful session/context evidence.
                recently_played = item_id in recent_items or title_key in recent_items
                # Search is an explicit retrieval request: a user must still
                # be able to find a song they played recently. Freshness is a
                # recommendation-queue rule only.
                if not has_seed:
                    recently_played = False
                # Strict musical relevance gate. Artist, era or popularity
                # alone can never pass; a candidate needs verified audio,
                # matching intent, same album/cluster, or strong metadata
                # evidence. The queue may legitimately be shorter.
                candidate_year = _release_year(item.get("year"))
                year_compatible = (
                    not candidate_year
                    or 0 < int(item.get("_era_delta", 0) or 0) <= MAX_RECOMMENDATION_YEAR_DISTANCE
                )
                strong_context = (
                    float(item.get("_audio_match", 0.0)) >= 0.62
                    or float(item.get("_intent_match", 0.0)) >= 0.5
                    or float(item.get("_album_match", 0.0)) >= 1.0
                    or (
                        float(item.get("_seed_artist_match", 0.0)) >= 1.0
                        and year_compatible
                        and not bool(item.get("_intent_conflict", False))
                    )
                    or (
                        float(item.get("_metadata_score", 0.0)) >= 0.55
                        and float(item.get("_mood_match", 0.0)) >= 0.60
                    )
                    or (
                        float(item.get("_seed_artist_match", 0.0))
                        and float(item.get("_mood_match", 0.0)) >= 0.62
                    )
                )
                # Live providers often omit year, language and audio features.
                # In that case a direct shared-artist signal is still valid
                # evidence, but only for the current seed's artist—not as a
                # generic popularity fallback. A shared metadata token plus a
                # compatible era/language bridge is the second safe fallback.
                seed_artist_keys = {
                    re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
                    for value in seed_artists if value.strip()
                }
                item_artist_keys = {
                    re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
                    for value in str(item.get("artists", "")).split(",") if value.strip()
                }
                direct_artist_evidence = bool(seed_artist_keys.intersection(item_artist_keys))
                direct_link = (
                    direct_artist_evidence
                    or float(item.get("_album_match", 0.0)) >= 1.0
                )
                sparse_metadata_bridge = (
                    float(item.get("_metadata_similarity", 0.0)) >= 0.22
                    and (
                        float(item.get("_language_match", 0.0)) >= 1.0
                        or float(item.get("_era_match", 0.0)) >= 0.45
                        or float(item.get("_style_match", 0.0)) >= 1.0
                    )
                )
                low_confidence_seed = float((seed or {}).get("mood_confidence", 0.0) or 0.0) < 0.5
                mood_evidence = (
                    float(item.get("_audio_match", 0.0)) >= 0.50
                    or float(item.get("_mood_match", 0.0)) >= 0.58
                )
                if low_confidence_seed and (
                    (direct_artist_evidence and mood_evidence)
                    or sparse_metadata_bridge
                ):
                    strong_context = True
                # Do not recommend a known different-language track merely
                # because both titles contain an energy keyword. Cross-
                # language discovery remains possible through same-artist or
                # same-album evidence, but not as an accidental fallback.
                same_soundtrack = float(item.get("_album_match", 0.0)) >= 1.0
                safe_link = same_soundtrack or float(item.get("_cluster_match", 0.0)) >= 1.0
                if has_seed and bool(item.get("_language_conflict", False)) and not safe_link:
                    continue
                if bool(item.get("_intent_conflict", False)) and not same_soundtrack:
                    continue
                candidate_intents = item_intents
                specialized_intents = {"devotional", "ghazal", "motivational", "acoustic_indie", "energetic", "playful", "rap_hiphop"}
                seed_specialized = seed_intents.intersection(specialized_intents)
                candidate_specialized = candidate_intents.intersection(specialized_intents)
                if candidate_specialized and not seed_specialized.intersection(candidate_specialized) and not same_soundtrack:
                    continue
                # A confident seed intent is a hard semantic contract. A
                # candidate with no evidence for that intent is not allowed
                # merely because it is popular or from a similar era. Same
                # artist/album/cluster is the only safe exception.
                if (
                    has_seed
                    and seed_intents
                    and not candidate_intents
                    and not (
                        same_soundtrack
                        or (
                            float(item.get("_audio_match", 0.0)) >= 0.78
                            and float(item.get("_mood_match", 0.0)) >= 0.70
                        )
                        or (
                            float(item.get("_seed_artist_match", 0.0)) >= 1.0
                            and year_compatible
                        )
                    )
                ):
                    continue
                if has_seed and not strong_context:
                    if not relaxed_context:
                        continue
                    # Sparse live metadata (for example a single English pop
                    # track with no album catalogue) still needs a useful
                    # radio. In the relaxed pass, require at least one
                    # language, era, style, or metadata bridge instead of
                    # accepting arbitrary popularity.
                    relaxed_bridge = (
                        float(item.get("_audio_match", 0.0)) >= 0.50
                        or float(item.get("_mood_match", 0.0)) >= 0.58
                        or float(item.get("_intent_match", 0.0)) >= 0.5
                        or same_soundtrack
                    )
                    if not relaxed_bridge:
                        continue
                base_relevance = float(np.clip(base_score, 0.0, 1.0))
                if recently_played:
                    base_relevance *= 0.88
                adjusted = base_relevance
                redundancy = 0.0
                # Maximal Marginal Relevance: preserve relevance while
                # continuously penalising candidates too similar to songs
                # already selected. No version/artist/album names are special-
                # cased; the metadata feature space determines the penalty.
                if selected:
                    redundancy = max(_candidate_similarity(item, chosen) for chosen in selected)
                    adjusted = adjusted - 0.08 * redundancy
                # Prevent the same singer from appearing repeatedly. The
                # first appearance is free; subsequent appearances are
                # increasingly expensive, while still allowing a second song
                # when the candidate pool is small.
                repeat_count = max(
                    (artist_counts.get(artist, 0) for artist in artist_keys),
                    default=0,
                )
                adjusted -= 0.30 * repeat_count
                # Album/soundtrack continuity is useful for the first result,
                # but repeated items from the same album must make room for
                # equally relevant songs from the seed's era and intent.
                adjusted -= 0.25 * album_counts.get(album_key, 0)
                if selected and artist_keys.intersection(
                    _artist_credit_keys(selected[-1].get("artists", ""))
                ):
                    adjusted -= 0.10
                eligible.append((index, adjusted, base_relevance, redundancy))
            # If every remaining candidate was rejected by the freshness or
            # relevance gate, finish cleanly instead of leaking weak fallbacks.
            if not eligible:
                break
            less_repeated_artists = [
                candidate
                for candidate in eligible
                if all(
                    artist_counts.get(artist, 0) < MAX_ARTIST_REPEATS_IN_QUEUE
                    for artist in _artist_credit_keys(
                        remaining[candidate[0]][1].get("artists", "")
                    )
                )
            ]
            if less_repeated_artists:
                eligible = less_repeated_artists
            distinct_soundtracks = [
                candidate
                for candidate in eligible
                if not (
                    (work_key := _recommendation_work_key(remaining[candidate[0]][1]))
                    and album_counts.get(work_key, 0)
                )
            ]
            if distinct_soundtracks:
                eligible = distinct_soundtracks
            best = max(eligible, key=lambda pair: pair[1])
            best_index = best[0]
            _, chosen = remaining.pop(best_index)
            chosen["base_relevance_score"] = round(float(best[2]), 6)
            chosen["redundancy_penalty"] = round(float(best[3]), 6)
            chosen["final_rank_score"] = round(float(max(0.0, best[1])), 6)
            # recommendation_score remains the relevance score used for
            # impression labels; diversity should not turn it into zero.
            chosen["recommendation_score"] = chosen["base_relevance_score"]
            chosen["recommendation_features"] = {
                "base_relevance": chosen["base_relevance_score"],
                "redundancy_penalty": chosen["redundancy_penalty"],
                "final_rank_score": chosen["final_rank_score"],
                "context": round(float(chosen.get("_context_score", 0.0)), 4),
                "current_artist": round(float(chosen.get("_seed_artist_match", 0.0)), 4),
                "soundtrack": round(float(chosen.get("_album_match", 0.0)), 4),
                "current_era": round(float(chosen.get("_era_match", 0.0)), 4),
                "cluster": round(float(chosen.get("_cluster_match", 0.0)), 4),
                "collaborative": round(float(chosen.get("_collaborative_score", 0.0)), 4),
                "mood": round(float(chosen.get("_mood_match", 0.0)), 4),
                "audio": round(float(chosen.get("_audio_match", 0.0)), 4),
                "intent": round(float(chosen.get("_intent_match", 0.0)), 4),
                "metadata": round(float(chosen.get("_metadata_score", 0.0)), 4),
                "era": round(float(chosen.get("_era_match", 0.0)), 4),
                "language": round(float(chosen.get("_language_match", 0.0)), 4),
                "transition": round(float(transition_map.get(str(chosen.get("song_id") or chosen.get("title", "")), 0.0)), 4),
                "bandit": round(float(bandit_values[item_ids.index(str(chosen.get("song_id") or chosen.get("title", "")))]), 4),
                "short_term": round(float(chosen.get("_short_term", 0.0)), 4),
                "medium_term": round(float(chosen.get("_medium_term", 0.0)), 4),
                "long_term": round(float(chosen.get("_long_term", 0.0)), 4),
                "audio_confidence": round(float(chosen.get("mood_confidence", 0.0) or 0.0), 4),
                "listener_mood": round(float(chosen.get("_listener_mood_match", 0.0)), 4),
                "listener_era": round(float(chosen.get("_listener_era_match", 0.0)), 4),
            }
            chosen["recommendation_model_version"] = HYBRID_MODEL_VERSION
            selected.append(chosen)
            album_key = _recommendation_work_key(chosen) or str(chosen.get("album", "")).lower()
            for artist in _artist_credit_keys(chosen.get("artists", "")):
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            album_counts[album_key] = album_counts.get(album_key, 0) + 1

        # Keep year closeness and mood ahead of artist/soundtrack context.
        selected = _interleave_recommendation_contexts(
            selected,
            seed_year,
            include_unknown_year=True,
        )
        selected = _prioritize_distinct_soundtracks(selected)
        for item in selected:
            item.pop("_context_score", None)
            item.pop("_profile_score", None)
            for key in ("_seed_artist_match", "_album_match", "_language_match", "_language_conflict", "_era_match", "_cluster_match", "_style_match", "_mood_match", "_listener_mood_match", "_listener_era_match", "_audio_match", "_metadata_score", "_collaborative_score", "_feature_tokens", "_metadata_similarity", "_intent_match", "_intent_conflict", "_era_delta", "_same_era", "_short_term", "_medium_term", "_long_term"):
                item.pop(key, None)
        return selected
