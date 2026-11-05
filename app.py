from pathlib import Path
import os
import time
import uuid
import re
import json
import hashlib
import hmac
from difflib import SequenceMatcher
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from jiosaavn_client import JioSaavnClient
from recommender import (
    HYBRID_MODEL_VERSION,
    HybridRecommendationEngine,
    SpotifyRecommendationEngine,
    _era_artist_discovery_queries,
    _mood_era_discovery_queries,
    era_mood_singer_discovery_queries,
    era_theme_film_discovery_queries,
    _soundtrack_peer_artists,
    infer_audio_intents,
    infer_metadata_mood_labels,
    infer_same_soundtrack_audio_intents,
    is_compilation_album,
    recommendation_seed_year as _recommendation_seed_year,
    recommendation_source_work as _recommendation_source_work,
    recommendation_history_keys as _recommendation_history_keys,
    recommendation_title_key as _recommendation_title_key,
    infer_track_language as _infer_track_language,
)
import llm_classifier as _llc
from search_ranking import (
    include_all_results,
    prioritize_title_matches,
    title_query_match_score,
)
from spotify_api import SpotifyOAuth, new_oauth_state, verify_oauth_state
from user_store import UserStore
from bhakti_agent import BhaktiAgent, quick_bhakti_recommend, is_devotional

ERA_COMPATIBILITY_WINDOW_YEARS = 30

st.set_page_config(page_title="CozyBeats | NextGen Recommender", page_icon="🎧", layout="wide")

# A small bidirectional Streamlit component is used for the player because
# st.audio does not expose the browser's `ended` event to Python. The event
# lets the server advance the already-ranked queue without requiring a click.
COZYBEATS_AUDIO = st.components.v2.component(
    "cozybeats_audio_player",
    html="""
<div class="tm-audio-shell">
  <div class="tm-audio-label" id="tm-audio-label"></div>
  <audio id="tm-audio" controls preload="metadata"></audio>
</div>
""",
    css="""
.tm-audio-shell { padding: .2rem 0; }
.tm-audio-label { color: #b6d3d1; font-size: .82rem; margin-bottom: .3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
audio { width: 100%; min-height: 42px; }
""",
    js="""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const audio = parentElement.querySelector('#tm-audio');
  const label = parentElement.querySelector('#tm-audio-label');
  if (!audio || !data) return;

  const src = data.url || '';
  const changed = audio.dataset.tmSrc !== src;
  audio.loop = Boolean(data.loop);
  label.textContent = data.title || 'CozyBeats player';

  if (changed) {
    const shouldPlay = Boolean(data.autoplay);
    audio.pause();
    audio.src = src;
    audio.dataset.tmSrc = src;
    audio.load();
    if (shouldPlay && src) {
      const playPromise = audio.play();
      if (playPromise && playPromise.catch) playPromise.catch(() => {});
    }
  }

  audio.onended = () => {
    if (!audio.loop) setTriggerValue('ended', { track_id: data.track_id });
  };
  return () => { audio.onended = null; };
}
""",
)

st.html("""
<style>
:root {
  color-scheme: dark;
  --tm-ink: #f5f7ff;
  --tm-muted: #a5b0c5;
  --tm-line: rgba(174, 193, 230, .13);
  --tm-surface: rgba(20, 25, 40, .78);
  --tm-cyan: #7de8ff;
  --tm-violet: #b89aff;
  --tm-pink: #ff76c8;
}
.stApp {
  color: var(--tm-ink);
  background:
    radial-gradient(ellipse at 82% 4%, rgba(119, 68, 196, .2), transparent 34rem),
    radial-gradient(ellipse at 8% 23%, rgba(35, 155, 180, .14), transparent 34rem),
    linear-gradient(150deg, #090b12 0%, #101321 52%, #11101b 100%);
}
.block-container { max-width: 1320px; padding: 1.15rem 2rem 3rem; }
header[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
header[data-testid="stHeader"] { height: 2.5rem; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(19, 22, 38, .98), rgba(11, 14, 23, .98));
  border-right: 1px solid var(--tm-line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--tm-muted); }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(28, 35, 54, .9), rgba(17, 21, 34, .9));
  border: 1px solid var(--tm-line); padding: 14px; border-radius: 18px;
}
[data-testid="stMetricLabel"] { color: var(--tm-muted); }
[data-testid="stMetricValue"] { color: #e9faff; }
.stButton > button, .stLinkButton > a {
  min-height: 2.65rem; border-radius: 999px; border: 1px solid var(--tm-line);
  background: linear-gradient(135deg, rgba(47, 57, 82, .9), rgba(27, 31, 48, .92));
  color: var(--tm-ink); font-weight: 650;
  transition: border-color .16s ease, transform .16s ease, box-shadow .16s ease;
}
.stButton > button:hover, .stLinkButton > a:hover {
  border-color: rgba(125, 232, 255, .65); color: white; transform: translateY(-2px);
  box-shadow: 0 9px 24px rgba(0, 0, 0, .24);
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--tm-line); border-radius: 20px;
  background: linear-gradient(145deg, rgba(25, 30, 46, .84), rgba(15, 19, 31, .88));
  box-shadow: 0 10px 28px rgba(0, 0, 0, .12);
  transition: border-color .18s ease, transform .18s ease, background .18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(184, 154, 255, .38);
  transform: translateY(-2px);
}
[data-testid="stImage"] img {
  border-radius: 15px; object-fit: cover; box-shadow: 0 12px 26px rgba(0, 0, 0, .2);
}
.stTextInput input, .stMultiSelect div[data-baseweb="select"] { border-radius: 14px; }
[data-testid="stTextInputRootElement"] {
  min-height: 3.1rem; border: 1px solid rgba(125, 232, 255, .22);
  border-radius: 16px; background: rgba(12, 16, 27, .88);
}
[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--tm-cyan); box-shadow: 0 0 0 3px rgba(125, 232, 255, .12);
}
[data-testid="stForm"] {
  padding: .7rem; border: 1px solid var(--tm-line); border-radius: 22px;
  background: rgba(17, 21, 34, .72); box-shadow: 0 15px 40px rgba(0, 0, 0, .16);
}
[data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--tm-line); }
[data-baseweb="tab"] { border-radius: 12px 12px 0 0; color: var(--tm-muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--tm-cyan); }
[data-testid="stExpander"] {
  border: 1px solid var(--tm-line); border-radius: 18px; background: rgba(16, 20, 32, .62);
}
audio { width: 100%; min-height: 46px; accent-color: var(--tm-cyan); }
.shelf-title { color: var(--tm-ink); font-size: 1.55rem; font-weight: 820; margin: 1.55rem 0 .3rem; letter-spacing: -.035em; }
.shelf-subtitle { color: var(--tm-muted); margin-bottom: .9rem; font-size: .92rem; }
.welcome-title { color: var(--tm-ink); font-size: clamp(1.65rem, 3vw, 2.25rem); font-weight: 850; letter-spacing: -.045em; margin: 1.35rem 0 .18rem; }
.welcome-copy { color: var(--tm-muted); margin-bottom: 1rem; }
.hero-panel {
  position: relative; overflow: hidden; padding: clamp(1.4rem, 4vw, 2.5rem);
  border-radius: 28px; border: 1px solid rgba(184, 154, 255, .2);
  background:
    radial-gradient(ellipse at 90% 15%, rgba(255, 118, 200, .22), transparent 24rem),
    radial-gradient(ellipse at 62% 100%, rgba(76, 222, 255, .16), transparent 28rem),
    linear-gradient(118deg, rgba(27, 31, 52, .98), rgba(19, 25, 42, .9) 54%, rgba(29, 22, 43, .94));
  box-shadow: 0 24px 70px rgba(0, 0, 0, .25);
}
.hero-title {
  font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: .98; font-weight: 900;
  letter-spacing: -.065em; background: linear-gradient(100deg, #fff 5%, #a4f1ff 49%, #e0b2ff 88%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-copy { color: #c0c9db; font-size: clamp(.98rem, 1.4vw, 1.12rem); max-width: 720px; margin-top: .8rem; }
.hero-chip {
  display: inline-flex; margin: 1.15rem .5rem 0 0; padding: .4rem .8rem;
  border-radius: 999px; color: #e8fbff; background: rgba(125, 232, 255, .08);
  border: 1px solid rgba(125, 232, 255, .2); font-size: .8rem; font-weight: 650;
}
.hero-chip:nth-child(4) { background: rgba(255, 118, 200, .08); border-color: rgba(255, 118, 200, .2); }
.hero-chip:nth-child(5) { background: rgba(184, 154, 255, .09); border-color: rgba(184, 154, 255, .22); }
.recommendation-banner {
  padding: 1.25rem; border-radius: 22px; border: 1px solid rgba(184, 154, 255, .28);
  background: linear-gradient(112deg, rgba(40, 31, 63, .95), rgba(19, 43, 51, .9));
  margin: .55rem 0 1rem; box-shadow: 0 16px 38px rgba(0, 0, 0, .2);
}
.recommendation-kicker { color: var(--tm-pink); font-size: .74rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
.recommendation-title { color: var(--tm-ink); font-size: 1.45rem; font-weight: 850; margin-top: .25rem; letter-spacing: -.025em; }
.recommendation-meta { color: #c0c9db; font-size: .9rem; }
.recommendation-reason { color: #d5c7ff; font-size: .84rem; margin-top: .2rem; }
.feedback-card { padding: .85rem; border: 1px solid var(--tm-line); border-radius: 18px; background: rgba(22, 27, 42, .88); }
.feedback-title { color: var(--tm-cyan); font-size: .74rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
@media (max-width: 760px) {
  .block-container { padding: .8rem .7rem 2.3rem; }
  .hero-panel { border-radius: 21px; }
  .shelf-title { font-size: 1.32rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
}
</style>
""")

st.markdown('''<div class="hero-panel"><div class="hero-title">CozyBeats</div><div class="hero-copy">Your next favorite is one play away. Discover a mix that moves with you.</div><span class="hero-chip">✦ Made for your mood</span><span class="hero-chip">◉ Your sound, evolving</span><span class="hero-chip">⚡ Live discovery</span></div>''', unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "spotify_tracks_data.csv"
EVENT_DB_PATH = Path(__file__).parent / "tune_match_events.sqlite3"
HISTORY_PATH = Path(__file__).parent / "spotify_history.csv"
AUTH_SESSION_PARAM = "tm_session"
AUTH_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60


def _auth_session_secret() -> bytes:
    """Use a stable signing key so a browser refresh can restore login safely."""
    configured = os.getenv("TUNEMATCH_SESSION_SECRET", "").strip()
    if not configured:
        try:
            configured = str(st.secrets.get("TUNEMATCH_SESSION_SECRET", "")).strip()
        except (FileNotFoundError, KeyError, AttributeError):
            configured = ""
    configured = configured or f"tunematch-local:{EVENT_DB_PATH.resolve()}"
    return configured.encode("utf-8")


def _make_auth_session_ticket(user_id: str) -> str:
    issued_at = int(time.time())
    payload = f"{user_id}.{issued_at}"
    signature = hmac.new(_auth_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def _read_auth_session_ticket() -> tuple[str, int] | None:
    token = str(st.query_params.get(AUTH_SESSION_PARAM, "") or "")
    parts = token.split(".")
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        return None
    user_id, issued_text, signature = parts
    payload = f"{user_id}.{issued_text}"
    expected = hmac.new(_auth_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    try:
        issued_at = int(issued_text)
    except ValueError:
        return None
    if not hmac.compare_digest(signature, expected):
        return None
    if time.time() - issued_at > AUTH_SESSION_TTL_SECONDS:
        return None
    return user_id, issued_at


def configured_database_url() -> str:
    """Use persistent cloud Postgres when configured, otherwise local SQLite."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url.strip().rstrip("\\").strip('"').strip("'")
    try:
        return str(st.secrets.get("DATABASE_URL", "")).strip().rstrip("\\").strip('"').strip("'")
    except (FileNotFoundError, KeyError, AttributeError):
        return ""

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data
def load_imported_history(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def build_engine(data):
    return SpotifyRecommendationEngine(data)


MOOD_FEATURES = (
    "energy", "tempo", "valence", "danceability", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness",
)
MOOD_VECTOR_FEATURES = (
    "energy", "tempo", "valence", "danceability", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness",
)

INTENT_TERMS = {
    "romantic": {"love", "romantic", "ishq", "mohabbat", "pyaar", "pyar", "dil", "jaan", "sanam", "sajan", "saiyaara", "milan"},
    "playful": {"playful", "comic", "comedy", "humorous", "funny", "lighthearted", "light hearted", "mischief", "masti", "nautanki", "chatur", "nok jhok", "banter", "witty", "theatrical", "expressive", "call and response", "call response", "jugalbandi"},
    "ghazal": {"ghazal", "ghazals", "shayar", "shayari", "mehfil", "rekhta", "ghulam ali", "ghulam ali classics", "ustad bade ghulam ali", "ghulam ali khan", "chupke chupke classics"},
    "motivational": {"motivation", "motivational", "fight", "fighting", "battle", "zinda", "jeet", "victory", "fighter", "warrior", "rise", "goal", "inspire", "champion", "unstoppable"},
    "energetic": {"fight", "anthem", "rock", "power", "action", "dance", "party", "bhangra", "bang", "fire", "dhamaka", "dhamaal", "high", "energy"},
    "rap_hiphop": {"rap", "rapper", "hip hop", "hiphop", "gully", "desi hip hop", "freestyle", "mixtape"},
    "sad": {"sad", "heartbreak", "alone", "broken", "dard", "udaas", "tanha", "judai"},
    "devotional": {"bhajan", "chalisa", "gurbani", "hanuman", "bajrang", "bajrangbali", "bajranbali", "anjaney", "aarti", "kirtan", "mantra", "om namah shivay", "krishna", "ghanshyam", "allah", "khuda", "dhun", "narayan", "hari", "shree", "jai", "namah", "ram", "stotram", "stotra", "hare", "durga", "ganesha", "lakshmi", "saraswati", "vishnu", "mahadev", "shankar", "bholenath", "sai", "bhakti", "bhakt"},
    "calm": {"acoustic", "unplugged", "lofi", "sleep", "calm", "peace", "sukoon", "soothing", "instrumental"},
    "acoustic_indie": {"acoustic", "unplugged", "indie", "folk", "singer songwriter", "singer-songwriter", "coke studio", "soul"},
}


@st.cache_resource
def build_catalog_clusters(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, MiniBatchKMeans]:
    """Learn fast mood/style clusters from the local audio-feature catalog."""
    frame = data.copy()
    feature_frame = pd.DataFrame({
        column: pd.to_numeric(frame.get(column, 0), errors="coerce")
        for column in MOOD_FEATURES
    }).fillna(0.0)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(feature_frame)
    cluster_count = max(2, min(24, max(2, len(frame) // 150)))
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=512,
        n_init=3,
        random_state=42,
    )
    frame["cluster_id"] = model.fit_predict(matrix)
    return frame, scaler, model


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _visible_recording_key(item: dict) -> str:
    """Collapse alternate releases of one recording for search shelves."""
    title = _recommendation_title_key(item.get("title", ""))
    return title


def _mood_vector(values: dict) -> list[float]:
    """Convert mixed live/catalog features into a cheap comparable vector."""
    vector = []
    for feature in MOOD_VECTOR_FEATURES:
        value = pd.to_numeric(values.get(feature, 0.0), errors="coerce")
        value = float(value) if pd.notna(value) else 0.0
        if feature == "tempo":
            value /= 200.0
        elif feature == "loudness":
            value = (value + 60.0) / 60.0
        elif feature == "popularity":
            value /= 100.0
        vector.append(max(0.0, min(1.0, value)))
    return vector


def infer_metadata_mood(item: dict) -> dict:
    """Fast mood fallback for live songs without a local audio match.

    This intentionally avoids downloading every live preview. It gives the
    ranker a low-cost, low-confidence vector until an audio-feature match is
    available from the local catalog.
    """
    vector = {
        "energy": 0.50, "tempo": 100.0, "valence": 0.50, "danceability": 0.50,
        "acousticness": 0.40, "instrumentalness": 0.15, "speechiness": 0.10,
        "liveness": 0.20, "loudness": -12.0, "popularity": 35.0,
    }
    matched = infer_metadata_mood_labels(item)
    if "playful" in matched:
        vector.update(energy=0.68, tempo=125.0, valence=0.82, danceability=0.68, acousticness=0.25, loudness=-7.5)
    elif "energetic" in matched:
        vector.update(energy=0.85, tempo=135.0, valence=0.72, danceability=0.72, acousticness=0.18, loudness=-6.0)
    elif "sad" in matched:
        vector.update(energy=0.30, tempo=72.0, valence=0.20, danceability=0.25, acousticness=0.58, loudness=-15.0)
    elif "romantic" in matched:
        vector.update(energy=0.45, tempo=88.0, valence=0.62, danceability=0.42, acousticness=0.45, loudness=-11.0)
    if "instrumental" in matched:
        vector.update(instrumentalness=0.88, speechiness=0.02, acousticness=0.52)
    if "acoustic_indie" in matched:
        vector.update(energy=0.35, tempo=82.0, acousticness=0.78, instrumentalness=0.25, loudness=-14.0)
    if "ghazal" in matched:
        vector.update(energy=0.28, tempo=72.0, valence=0.38, danceability=0.25, acousticness=0.76, instrumentalness=0.12, loudness=-15.0)
    if "rap_hiphop" in matched:
        vector.update(energy=0.78, tempo=112.0, valence=0.58, danceability=0.72, acousticness=0.18, speechiness=0.28, loudness=-7.0)
    labels = sorted(matched) or ["neutral"]
    return {
        "mood_vector": _mood_vector(vector),
        "mood_label": ", ".join(labels),
        "mood_confidence": 0.58 if matched else 0.22,
        "mood_source": "metadata_fallback",
    }


def infer_music_intents(item: dict) -> list[str]:
    """Infer intent independently from a track's own metadata."""
    text = " ".join(
        str(item.get(field, ""))
        for field in (
            "title", "album", "track_genre", "genre", "language", "mood",
            "description", "tags", "intent_labels",
        )
    ).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    labels = []
    for label, terms in INTENT_TERMS.items():
        if any(term in text if " " in term else term in tokens for term in terms):
            labels.append(label)
    labels = set(labels) | infer_audio_intents(item)
    return sorted(labels) or ["general"]


@st.cache_resource
def build_live_catalog_lookup(data: pd.DataFrame) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for row in data.itertuples(index=False):
        title = _compact_key(getattr(row, "track_name", ""))
        artists = str(getattr(row, "artists", "")).split(",")[0]
        artist = _compact_key(artists)
        if not title or not artist:
            continue
        genre_text = str(getattr(row, "track_genre", "")).casefold()
        language_hint = next(
            (language for language in ("hindi", "tamil", "telugu", "punjabi", "bengali", "marathi", "malayalam", "kannada")
             if language in genre_text),
            "",
        )
        lookup[(title, artist)] = {
            "cluster_id": getattr(row, "cluster_id", None),
            "feature_source": "local_audio_catalog",
            "catalog_artists": str(getattr(row, "artists", "")),
            "catalog_language_hint": language_hint,
            "track_genre": str(getattr(row, "track_genre", "")),
            **{
                column: float(getattr(row, column, 0) or 0)
                for column in MOOD_FEATURES
            },
        }
        lookup[(title, artist)]["mood_vector"] = _mood_vector(lookup[(title, artist)])
        lookup[(title, artist)]["mood_label"] = "audio-features"
        lookup[(title, artist)]["mood_confidence"] = 0.95
        lookup[(title, artist)]["mood_source"] = "local_audio_catalog"
    return lookup


@st.cache_resource
def build_live_title_lookup(data: pd.DataFrame) -> list[dict]:
    """Index catalog titles for artist recovery when live metadata is sparse."""
    lookup = []
    for row in data.itertuples(index=False):
        title = _compact_key(getattr(row, "track_name", ""))
        artists = str(getattr(row, "artists", "")).strip()
        if title and artists:
            lookup.append({"title": title, "artists": artists})
    return lookup


def enrich_live_items(items: list[dict]) -> list[dict]:
    """Attach local audio features while preserving fuzzy-match provenance."""
    # Classify every live item once via the LLM (cached, no network without a
    # key) so intent/religion labels are accurate for downstream ranking and
    # the strict intent gate in rank_jio_results.
    try:
        llm_profiles = _llc.classify_items(items)
    except AttributeError:
        # Fallback: classify items one by one if batch method is unavailable
        llm_profiles = {}
        for item in items:
            key = re.sub(
                r"[^a-z0-9]+",
                " ",
                f"{item.get('title', '')} {item.get('artists', '')} {item.get('album', '')}",
            ).casefold().strip()
            profile = _llc.classify_item(item)
            if profile:
                llm_profiles[key] = profile
    for item in items:
        title = _compact_key(item.get("title", ""))
        artist_names = str(item.get("artists", "")).split(",")
        match = None
        for artist_name in artist_names:
            match = live_catalog_lookup.get((title, _compact_key(artist_name)))
            if match:
                break
        # Some JioSaavn search rows omit singers entirely. Recover the artist
        # from the local catalog using a generic title similarity, not a
        # song/version-specific rule. This handles suffixes such as alternate
        # versions while preserving the upstream title and artwork.
        title_match = None
        if not match and title:
            best_score = 0.0
            for candidate in live_title_lookup:
                candidate_title = candidate["title"]
                score = SequenceMatcher(None, title, candidate_title).ratio()
                if title in candidate_title or candidate_title in title:
                    score = max(score, 0.82)
                if score > best_score:
                    best_score = score
                    title_match = candidate
            if title_match and best_score >= 0.72:
                if not str(item.get("artists", "")).strip():
                    item["artists"] = title_match["artists"]
                match = live_catalog_lookup.get(
                    (title_match["title"], _compact_key(title_match["artists"].split(",")[0]))
                )
                if match:
                    item["feature_source"] = "local_audio_catalog_title_match"
        match_is_fuzzy = bool(match and title_match)
        if match:
            item.update({key: value for key, value in match.items() if key != "catalog_artists"})
            if match_is_fuzzy:
                item["feature_source"] = "local_audio_catalog_title_match"
                item["mood_source"] = "local_audio_catalog_title_match"
            if not str(item.get("artists", "")).strip():
                item["artists"] = match.get("catalog_artists", "")
            if not str(item.get("language", "")).strip() and match.get("catalog_language_hint"):
                item["language"] = match["catalog_language_hint"]
        else:
            item.setdefault("feature_source", "live_metadata_fallback")
            item.update({key: value for key, value in infer_metadata_mood(item).items() if key not in item})
        item["intent_labels"] = infer_music_intents(item)
        _cache_key_val = re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{item.get('title', '')} {item.get('artists', '')} {item.get('album', '')}",
        ).casefold().strip()
        llm_p = llm_profiles.get(_cache_key_val, {}) or {}
        if llm_p:
            item["llm_intent"] = llm_p.get("primary_intent")
            item["llm_religion"] = llm_p.get("religion")
            item["llm_sub_genre"] = llm_p.get("sub_genre")
            llm_intent = llm_p.get("primary_intent")
            llm_conf = float(llm_p.get("intent_confidence", 0.0) or 0.0)
            if (
                llm_intent
                and llm_intent not in {"unknown", "general", "neutral"}
                and llm_conf >= 0.60
                and llm_intent not in item["intent_labels"]
            ):
                item["intent_labels"] = sorted(
                    set(item["intent_labels"]) | {llm_intent}
                )
    return items


@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def fetch_dashboard_posters(
    items: tuple[tuple[str, str, str], ...],
    base_url: str,
) -> dict[str, dict]:
    """Resolve artwork for the visible local shelf cards from the live provider."""
    client = JioSaavnClient(base_url)

    def resolve(record: tuple[str, str, str]) -> tuple[str, dict]:
        track_id, title, artist = record
        try:
            candidates = client.search_direct(f"{title} {artist}".strip(), limit=5)
        except Exception:
            candidates = []
        title_key = _compact_key(title)
        artist_key = _compact_key(artist.split(",")[0])
        chosen = next(
            (
                item for item in candidates
                if _compact_key(item.get("title", "")) == title_key
                and artist_key
                and artist_key in _compact_key(item.get("artists", ""))
            ),
            next((item for item in candidates if _compact_key(item.get("title", "")) == title_key), None),
        )
        return track_id, chosen or {}

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(items)))) as executor:
        return dict(executor.map(resolve, items))


def play_dashboard_live_item(item: dict) -> None:
    """Start a dashboard pick; recommendations are prepared after audio renders."""
    track_id = str(item.get("song_id") or item.get("title", ""))
    st.session_state["jio_now_playing"] = item
    st.session_state["scroll_to_player"] = True
    st.session_state["jio_queue"] = []
    st.session_state.pop("jio_queue_seed_id", None)
    st.session_state.pop("jio_queue_attempted_id", None)
    st.session_state.pop("jio_queue_refill_attempted_id", None)
    st.session_state["jio_current_track_id"] = track_id
    st.session_state["jio_started_at"] = time.time()
    st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
    defer_player_event("jio_play", track_id, metadata=item)
    st.rerun()


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def fetch_english_dashboard_tracks(base_url: str) -> pd.DataFrame:
    """Build a cached English-only live shelf; never mix it with Hindi rows."""
    queries = (
        "english pop songs", "english rock classics", "english chill songs",
        "english love songs", "english sad songs", "english dance hits",
        "english indie songs", "english acoustic songs", "english 2000s hits",
        "english 2010s hits", "english 90s hits", "english party songs",
        "english feel good songs", "english heartbreak songs", "english soft songs",
        "english alternative songs", "english rnb songs", "english jazz songs",
        "english country songs", "english classics", "english trending songs",
        "english female singers", "english male singers", "english movie songs",
        "english radio hits", "english summer songs", "english night songs",
        "english workout songs", "english calm songs", "english retro songs",
        "english top hits", "english new music", "english slow songs",
        "english romantic hits", "english viral songs", "english piano songs",
        "english road trip songs", "english morning songs", "english golden hits",
        "english timeless songs", "english soundtrack songs", "english acoustic pop",
        "english rock ballads", "english edm hits", "english singer songwriter",
        "english disco hits", "english folk songs", "english soul music",
        "english blues songs", "english alternative rock",
    )
    blocked_terms = re.compile(
        r"hindi|bollywood|filmi|indian|bhangra|punjabi|desi|sufi|tamil|telugu|"
        r"bengali|gujarati|marathi|bhojpuri|assamese|odia|kannada|malayalam|"
        r"urdu|nepali|rajasthani|sanskrit|qawwali|ghazal|bhajan",
        re.IGNORECASE,
    )
    client = JioSaavnClient(base_url)

    with ThreadPoolExecutor(max_workers=10) as executor:
        pages = executor.map(lambda query: client.search_direct(query, limit=10), queries)
        rows: list[dict] = []
        seen: set[str] = set()
        for page in pages:
            for item in page:
                if str(item.get("language", "")).casefold() != "english":
                    continue
                identity = f"{item.get('title', '')}|{item.get('artists', '')}".casefold()
                searchable = " ".join(
                    str(item.get(field, ""))
                    for field in ("title", "artists", "album")
                )
                if not identity or identity in seen or blocked_terms.search(searchable):
                    continue
                seen.add(identity)
                rows.append(item)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["track_id"] = frame["song_id"].astype(str)
    frame["track_name"] = frame["title"].astype(str)
    frame["album_name"] = frame["album"].astype(str)
    return frame.head(1000).reset_index(drop=True)


@st.cache_resource
def build_hybrid_engine(_content_engine, model_version: str):
    """Keep one fitted model service per app process instead of rebuilding it on every rerun."""
    return HybridRecommendationEngine(_content_engine)


@st.cache_resource(show_spinner=False)
def build_user_store(path: str, database_url: str) -> UserStore:
    """Initialize the account store once per Streamlit process.

    Streamlit reruns the script after every widget interaction. Creating the
    remote MySQL tables on every rerun made login appear slow, especially on a
    cloud database with a cold connection.
    """
    return UserStore(path, database_url=database_url)


raw_data = load_data(DATA_PATH)
data, mood_scaler, mood_cluster_model = build_catalog_clusters(raw_data)
live_catalog_lookup = build_live_catalog_lookup(data)
live_title_lookup = build_live_title_lookup(data)
imported_history = load_imported_history(HISTORY_PATH)
engine = build_engine(data)
stats = engine.catalog_stats()
database_url = configured_database_url()
if "cloud_db_unavailable" not in st.session_state:
    st.session_state["cloud_db_unavailable"] = False
try:
    if database_url and st.session_state["cloud_db_unavailable"]:
        raise RuntimeError("cloud database was unavailable earlier in this session")
    user_store = build_user_store(str(EVENT_DB_PATH), database_url)
except Exception:
    # Keep the UI usable while a cloud database is paused, misconfigured, or
    # temporarily unreachable. Once DATABASE_URL is corrected and the app is
    # rebooted, UserStore automatically returns to the configured MySQL DB.
    if database_url:
        st.session_state["cloud_db_unavailable"] = True
    user_store = build_user_store(str(EVENT_DB_PATH), "")
    if database_url:
        st.warning(
            "Cloud database is unavailable right now. CozyBeats is using temporary local storage; "
            "fix DATABASE_URL and reboot to persist users and activity."
        )
jio_client = JioSaavnClient()
hybrid_engine = build_hybrid_engine(engine, HYBRID_MODEL_VERSION)
st.session_state.setdefault("local_session_id", uuid.uuid4().hex)
if "recommendation_variant" not in st.session_state:
    variant_key = str(st.session_state.get("local_user_id") or st.session_state["local_session_id"])
    bucket = int(hashlib.sha256(variant_key.encode("utf-8")).hexdigest()[:8], 16) % 2
    st.session_state["recommendation_variant"] = "diverse" if bucket else "control"
st.session_state.setdefault("jio_search_cache_v4", {})
st.session_state.pop("jio_search_cache_v3", None)
st.session_state.pop("jio_search_cache_v2", None)
# Invalidate shelves created before the full artist/film catalogue policy.
if st.session_state.get("jio_search_cache_policy") != "filtered-recordings-v6-title-priority":
    st.session_state.pop("jio_search_cache_v4", None)
    st.session_state["jio_search_cache_v4"] = {}
    st.session_state["jio_search_cache_policy"] = "filtered-recordings-v6-title-priority"
st.session_state.setdefault("jio_loop", False)
st.session_state.setdefault("scroll_to_player", False)
st.session_state.setdefault("jio_queue_refill_attempted_id", "")
# Never keep a queue generated by an older relevance policy. This is
# especially important after deploying a stricter specialized-intent gate:
# Streamlit preserves session_state across code updates/reconnects.
if st.session_state.get("jio_queue_model_version") != HYBRID_MODEL_VERSION:
    st.session_state.pop("jio_queue", None)
    st.session_state.pop("jio_queue_seed_id", None)
    st.session_state.pop("jio_queue_attempted_id", None)
    st.session_state["jio_queue_model_version"] = HYBRID_MODEL_VERSION
# Migrate the old queue behavior: older sessions stored search results here.
if "jio_queue_index" in st.session_state:
    st.session_state.pop("jio_queue", None)
    st.session_state.pop("jio_queue_index", None)
recommendation_history = st.session_state.setdefault("jio_recommendation_history", set())
if not isinstance(recommendation_history, set):
    recommendation_history = set(recommendation_history)
    st.session_state["jio_recommendation_history"] = recommendation_history
for queued_item in st.session_state.get("jio_queue", []):
    recommendation_history.update(_recommendation_history_keys(queued_item))


def refresh_hybrid_model(force: bool = False) -> None:
    """Refit only when the event store has changed; keeps UI reruns fast."""
    # A normal widget rerun cannot change the event table. Only reload remote
    # interactions after this session writes an event; admin analytics has its
    # own 10-second live refresh path.
    if not force and not st.session_state.get("events_dirty", True):
        return
    events = user_store.interaction_events()
    latest_event = events[-1].get("created_at") if events else ""
    signature = (len(events), str(latest_event))
    if force or st.session_state.get("hybrid_model_signature") != signature:
        hybrid_engine.fit(events)
        st.session_state["hybrid_model_signature"] = signature
        st.session_state["events_dirty"] = False


refresh_hybrid_model()


def log_behavior(
    event_type: str,
    track_id: str | None = None,
    duration_ms: int = 0,
    metadata: dict | None = None,
) -> None:
    user_id = st.session_state.get("local_user_id")
    if not user_id:
        return

    # Analytics must never take down the player or prevent a user from
    # signing in. A sleeping cloud MySQL instance can time out while writing;
    # pause writes briefly instead of blocking every Streamlit rerun.
    retry_after = float(st.session_state.get("event_write_retry_after", 0.0))
    if retry_after > time.time():
        return
    try:
        user_store.log_event(
            user_id=user_id,
            track_id=track_id,
            event_type=event_type,
            session_id=st.session_state["local_session_id"],
            duration_ms=duration_ms,
            metadata_json=json.dumps(metadata or {}),
        )
    except Exception:
        st.session_state["event_write_retry_after"] = time.time() + 30
        st.session_state["event_write_warning"] = True
        return
    st.session_state["events_dirty"] = True
    st.session_state["summary_dirty"] = True
    st.session_state["user_events_dirty"] = True


def defer_player_event(
    event_type: str,
    track_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    pending = st.session_state.setdefault("pending_player_events", [])
    pending.append({
        "event_type": event_type,
        "track_id": track_id,
        "metadata": metadata,
    })


def flush_player_events() -> None:
    pending = st.session_state.pop("pending_player_events", [])
    for event in pending:
        log_behavior(
            event["event_type"],
            event.get("track_id"),
            metadata=event.get("metadata"),
        )


def cached_user_summary(user_id: str, session_id: str | None = None) -> dict[str, int]:
    """Avoid a remote aggregate query on every Streamlit rerun."""
    cache = st.session_state.setdefault("user_summary_cache", {})
    cache_key = f"{user_id}:{session_id or 'all'}"
    if cache_key in cache and not st.session_state.get("summary_dirty"):
        return cache[cache_key]
    summary = user_store.summary(user_id, session_id=session_id)
    cache[cache_key] = summary
    st.session_state["summary_dirty"] = False
    return summary


def cached_user_events(user_id: str) -> list[dict]:
    """Cache private analytics rows until this listener writes another event."""
    cache = st.session_state.setdefault("user_events_cache", {})
    cache_key = str(user_id)
    if cache_key in cache and not st.session_state.get("user_events_dirty"):
        return cache[cache_key]
    events = user_store.user_events(user_id)
    cache[cache_key] = events
    st.session_state["user_events_dirty"] = False
    return events


def account_listening_time() -> None:
    """Persist elapsed player time between Streamlit reruns."""
    user_id = st.session_state.get("local_user_id")
    started_at = st.session_state.get("jio_started_at")
    if not user_id or not started_at:
        return
    now = time.time()
    last_accounted = float(st.session_state.get("jio_last_accounted_at", started_at))
    delta_ms = int(max(0.0, now - last_accounted) * 1000)
    if delta_ms >= 1000:
        log_behavior(
            "listen_progress",
            st.session_state.get("jio_current_track_id"),
            duration_ms=delta_ms,
        )
        st.session_state["jio_last_accounted_at"] = now


def spotify_access_token(client: SpotifyOAuth | None) -> str | None:
    """Return a valid session token, refreshing it when necessary."""
    token = st.session_state.get("spotify_token")
    if not client or not token:
        return None
    import time

    if float(token.get("expires_at", 0)) <= time.time():
        try:
            token = client.refresh_token(token["refresh_token"])
            st.session_state["spotify_token"] = token
        except Exception as exc:
            st.session_state.pop("spotify_token", None)
            st.sidebar.error(f"Spotify session expired: {exc}")
            return None
    return str(token["access_token"])


def sync_spotify_taste(client: SpotifyOAuth, access_token: str) -> dict:
    """Fetch the minimum user signals needed for personalization."""
    return {
        "profile": client.get(access_token, "me"),
        "top_tracks": client.get(access_token, "me/top/tracks", limit=50, time_range="medium_term").get("items", []),
        "recent_tracks": client.get(access_token, "me/player/recently-played", limit=50).get("items", []),
        "saved_tracks": client.get(access_token, "me/tracks", limit=50).get("items", []),
    }


def local_ids_for_spotify_taste(taste: dict, catalog: pd.DataFrame) -> list[str]:
    """Map Spotify track signals onto the local feature catalog when possible."""
    by_id = {str(row.track_id): str(row.track_id) for row in catalog.itertuples()}
    by_name = {
        (str(row.track_name).strip().lower(), str(row.artists).split(",")[0].strip().lower()): str(row.track_id)
        for row in catalog.itertuples()
    }
    found: list[str] = []
    for collection in (taste.get("top_tracks", []), taste.get("recent_tracks", []), taste.get("saved_tracks", [])):
        for entry in collection:
            track = entry.get("track", entry)
            track_id = str(track.get("id", ""))
            artists = track.get("artists", [])
            artist = str(artists[0].get("name", "")).strip().lower() if artists else ""
            key = (str(track.get("name", "")).strip().lower(), artist)
            match = by_id.get(track_id) or by_name.get(key)
            if match and match not in found:
                found.append(match)
    return found


def local_ids_for_imported_history(history: pd.DataFrame, catalog: pd.DataFrame) -> list[str]:
    """Map the strongest imported listening signals to local catalog tracks."""
    if history.empty or not {"track_name", "artists"}.issubset(history.columns):
        return []
    by_name = {
        (str(row.track_name).strip().lower(), str(row.artists).split(",")[0].strip().lower()): str(row.track_id)
        for row in catalog.itertuples()
    }
    ranked = history.copy()
    played = ranked["ms_played"] if "ms_played" in ranked else pd.Series(0, index=ranked.index)
    ranked["ms_played"] = pd.to_numeric(played, errors="coerce").fillna(0)
    ranked["artist_key"] = ranked["artists"].astype(str).str.split(",").str[0].str.strip().str.lower()
    ranked["track_key"] = ranked["track_name"].astype(str).str.strip().str.lower()
    ranked = ranked.groupby(["track_key", "artist_key"], as_index=False)["ms_played"].sum()
    ranked = ranked.sort_values("ms_played", ascending=False).head(100)
    return [
        by_name[(row.track_key, row.artist_key)]
        for row in ranked.itertuples()
        if (row.track_key, row.artist_key) in by_name
    ]


def personalize_jio_results(
    results: list[dict],
    catalog: pd.DataFrame,
    preferred_track_ids: list[str],
) -> list[dict]:
    """Rank JioSaavn search results using this listener's local taste profile.

    The catalog is only used as a private preference signal. Nothing from the
    local catalog is rendered as a result, so every visible song still comes
    from JioSaavn.
    """
    if not results or not preferred_track_ids:
        return results

    preferred = catalog[catalog["track_id"].astype(str).isin({str(x) for x in preferred_track_ids})]
    if preferred.empty:
        return results

    artist_preferences = [
        str(value).strip().lower()
        for value in preferred["artists"].dropna().tolist()
        if str(value).strip()
    ]
    track_preferences = [
        str(value).strip().lower()
        for value in preferred["track_name"].dropna().tolist()
        if str(value).strip()
    ]

    def compact(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def score(item: dict) -> tuple[int, int]:
        item_artist = compact(item.get("artists", ""))
        item_title = compact(item.get("title", ""))
        artist_score = sum(1 for artist in artist_preferences if compact(artist) in item_artist)
        title_score = sum(1 for title in track_preferences if compact(title) == item_title)
        return (artist_score * 3 + title_score * 5, 0)

    ranked = sorted(enumerate(results), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
    return [item for _, item in ranked]


def prioritize_search_results(results: list[dict], query: str) -> list[dict]:
    """Restore exact search intent after taste reranking.

    The recommendation ranker is intentionally taste-aware, but a search page
    must still put the requested title/album first. Language-suffixed alternate
    versions (for example ``- Tamil``) are kept, but ranked below the base
    result unless the user searched for that variant explicitly.
    """
    all_results = list(results)
    query_key = _compact_key(query)
    if not query_key:
        return results
    query_tokens = set(query_key.split())
    generic_terms = {
        "song", "songs", "music", "playlist", "playlists", "best", "top", "hits",
        "hindi", "bollywood", "romantic", "motivational", "sad", "energetic",
        "ghazal", "devotional", "acoustic", "indie", "film", "films", "soundtrack",
    }
    language_terms = {"hindi", "tamil", "telugu", "punjabi", "bengali", "marathi", "malayalam", "kannada"}
    query_languages = query_tokens.intersection(language_terms)
    meaningful_query_text = " ".join(
        token for token in query_key.split() if token not in generic_terms
    ) or query_key

    def score(item: dict) -> tuple[float, int]:
        title = _compact_key(item.get("title", ""))
        album = _compact_key(item.get("album", ""))
        artists = _compact_key(item.get("artists", ""))
        starring = _compact_key(item.get("starring", ""))
        title_tokens = set(title.split())
        # Providers commonly append recording variants to the title. Treat
        # those as exact title hits for ordering, while preserving the full
        # display title so Male/Female/Duet versions remain visible.
        base_title = re.sub(
            r"\b(from|version|soundtrack|jhankar beats|remix|reprise|live|"
            r"instrumental|karaoke|male|female|duet|vocals only)\b.*$",
            " ",
            title,
        ).strip()
        title_score = title_query_match_score(title, meaningful_query_text)
        exact_title = float(
            title == query_key
            or base_title == query_key
            or title_score >= 0.92
        )
        title_match = float(title_score >= 0.82)
        album_exact = float(album == query_key)
        album_match = float(query_key in album or query_tokens.issubset(set(album.split())))
        entity_title_match = float(
            query_key in title and any(marker in title for marker in ("from", "title track", "soundtrack"))
        )
        artist_match = float(query_key in artists)
        cast_match = float(
            query_key in starring
            or (meaningful_query and meaningful_query.issubset(set(starring.split())))
        )
        language = str(item.get("language", "")).lower()
        variant_penalty = 0.0
        if not query_languages and any(term in title_tokens for term in language_terms):
            variant_penalty = 2.0
        if query_languages and language and not query_languages.intersection({language}):
            variant_penalty += 1.0
        return (
            5.0 * exact_title
            + 4.0 * album_exact
            + 3.0 * entity_title_match
            + 1.5 * title_match
            + 1.5 * album_match
            + 1.0 * artist_match
            + 1.0 * cast_match
            - variant_penalty,
            0,
        )
    def match_kind(item: dict) -> str:
        title = _compact_key(item.get("title", ""))
        album = _compact_key(item.get("album", ""))
        artists = _compact_key(item.get("artists", ""))
        starring = _compact_key(item.get("starring", ""))
        meaningful = query_tokens - generic_terms
        title_hit = title_query_match_score(title, meaningful_query_text) >= 0.82
        artist_hit = bool(query_key in artists or (meaningful and meaningful.issubset(set(artists.split()))))
        album_hit = bool(query_key in album or (meaningful and meaningful.issubset(set(album.split()))))
        cast_hit = bool(query_key in starring or (meaningful and meaningful.issubset(set(starring.split()))))
        if title_hit:
            return "title"
        if artist_hit:
            return "artist"
        if album_hit:
            return "album"
        if cast_hit:
            return "cast"
        return ""

    kinds = [match_kind(item) for item in results]
    meaningful_query = query_tokens - generic_terms
    query_intents = {
        label for label, terms in INTENT_TERMS.items()
        if any((term in query_key if " " in term else term in query_tokens) for term in terms)
    }
    specialized_query_intents = query_intents.intersection(
        {"devotional", "ghazal", "motivational", "romantic", "sad", "acoustic_indie", "energetic", "rap_hiphop"}
    )
    if specialized_query_intents:
        intent_matches = [
            item for item in results
            if specialized_query_intents.intersection(set(item.get("intent_labels", [])))
        ]
        if intent_matches:
            results = intent_matches
            kinds = [match_kind(item) for item in results]
    devotional_search = any(token in query_key for token in ("bajrang", "bajran", "hanuman", "anjaney"))
    if devotional_search:
        devotional_terms = ("bajrang", "hanuman", "anjaney", "bhajan", "chalisa", "aarti", "kirtan", "mantra", "devotional")
        devotional_matches = [
            item for item in results
            if any(term in _compact_key(" ".join(str(item.get(field, "")) for field in ("title", "album", "artists"))) for term in devotional_terms)
        ]
        if devotional_matches:
            results = devotional_matches
            kinds = [match_kind(item) for item in results]
    # A clear title query must not leak unrelated artist/album results. Artist
    # and album queries remain broad within their matched entity.
    exact_title_query = any(
        _compact_key(item.get("title", "")) == query_key
        for item in results
    )
    artist_entity_matches = [
        item for item in results
        if query_key in _compact_key(item.get("artists", ""))
        or (
            meaningful_query
            and meaningful_query.issubset(
                set(_compact_key(item.get("artists", "")).split())
            )
        )
    ]
    album_entity_matches = [
        item for item in results
        if query_key in _compact_key(item.get("album", ""))
        or (
            meaningful_query
            and meaningful_query.issubset(set(_compact_key(item.get("album", "")).split()))
        )
    ]
    cast_entity_matches = [
        item for item in results
        if query_key in _compact_key(item.get("starring", ""))
        or (
            meaningful_query
            and meaningful_query.issubset(set(_compact_key(item.get("starring", "")).split()))
        )
    ]
    entity_catalog_search = bool(
        meaningful_query
        and not exact_title_query
        and (artist_entity_matches or album_entity_matches or cast_entity_matches)
    )
    if meaningful_query and artist_entity_matches and not exact_title_query:
        # An artist query is an entity search, not a title-token search.
        # Keep the artist's actual catalogue first; generic provider rows are
        # only allowed in the related fallback section.
        filtered = artist_entity_matches
    elif meaningful_query and album_entity_matches and not exact_title_query:
        filtered = album_entity_matches
    elif meaningful_query and cast_entity_matches and not exact_title_query:
        filtered = cast_entity_matches
    elif meaningful_query and "title" in kinds:
        filtered = [item for item, kind in zip(results, kinds) if kind == "title"]
    elif meaningful_query and "artist" in kinds:
        filtered = [item for item, kind in zip(results, kinds) if kind == "artist"]
    elif meaningful_query and ("artist" in kinds or "album" in kinds):
        filtered = [item for item, kind in zip(results, kinds) if kind in {"artist", "album"}]
    else:
        filtered = list(results)
    primary_ids = {id(item) for item in filtered}
    # Keep the exact entity matches first, but do not leave a normal song
    # search with a tiny shelf when the provider returned enough scored
    # related rows. Artist/film catalogues already show every matched song.
    if (
        meaningful_query
        and not entity_catalog_search
        and len(filtered) < 20
        and len(results) > len(filtered)
    ):
        ordered_all = [
            item for _, item in sorted(
                enumerate(results),
                key=lambda pair: (score(pair[1]), -pair[0]),
                reverse=True,
            )
        ]
        filtered_ids = {id(item) for item in filtered}
        for item in ordered_all:
            if id(item) in filtered_ids:
                continue
            item.setdefault("search_match_type", "scored_related_result")
            filtered.append(item)
            filtered_ids.add(id(item))
            if len(filtered) >= 20:
                break
    ordered_primary = [
        item for _, item in sorted(
            enumerate([item for item in filtered if id(item) in primary_ids]),
            key=lambda pair: (score(pair[1]), -pair[0]),
            reverse=True,
        )
    ]
    ordered_related = [
        item for _, item in sorted(
            enumerate([item for item in filtered if id(item) not in primary_ids]),
            key=lambda pair: (score(pair[1]), -pair[0]),
            reverse=True,
        )
    ]
    ordered = ordered_primary + ordered_related
    title_matches = [
        item for item in results
        if title_query_match_score(item.get("title", ""), meaningful_query_text) >= 0.82
    ]
    ordered = prioritize_title_matches(
        [*title_matches, *ordered],
        meaningful_query_text,
    )
    unique: list[dict] = []
    seen: set[str] = set()
    for item in ordered:
        key = _visible_recording_key(item)
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return include_all_results(unique, all_results, _visible_recording_key)


def build_jio_up_next(
    seed: dict,
    search_results: list[dict],
    client: JioSaavnClient,
    hybrid: HybridRecommendationEngine,
    user_id: str,
    catalog: pd.DataFrame,
    preferred_track_ids: list[str],
    limit: int = 35,
    exclude_items: list[dict] | None = None,
) -> list[dict]:
    """Build a separate recommendation queue; never reuse search results."""
    # Ensure the playing seed has the same mood representation as candidates,
    # including when it came from the player refresh path rather than search.
    seed = enrich_live_items([seed])[0]
    title = str(seed.get("title", "")).strip()
    artists = str(seed.get("artists", "")).strip()
    album = str(seed.get("album", "")).strip()

    def valid_artist(value: object) -> bool:
        normalized = str(value or "").strip().casefold()
        return bool(normalized and normalized not in {"unknown", "n/a", "na", "-"})

    # Two-step identity check for sparse live metadata:
    # 1) compare title/album similarity against the seed;
    # 2) let the strongest repeated artist vote become the resolved artist.
    # This is query-local consensus, so no singer name is hardcoded.
    seed_title_key = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
    seed_album_key = re.sub(r"[^a-z0-9]+", " ", album.casefold()).strip()
    artist_votes: dict[str, float] = {}
    artist_display: dict[str, str] = {}
    for candidate in [seed, *search_results]:
        candidate_title = str(candidate.get("title", "")).strip()
        candidate_artists = [
            value.strip() for value in str(candidate.get("artists", "")).split(",")
            if valid_artist(value)
        ]
        if not candidate_artists:
            continue
        candidate_title_key = re.sub(r"[^a-z0-9]+", " ", candidate_title.casefold()).strip()
        candidate_album_key = re.sub(
            r"[^a-z0-9]+", " ", str(candidate.get("album", "")).casefold()
        ).strip()
        title_similarity = SequenceMatcher(None, seed_title_key, candidate_title_key).ratio()
        album_similarity = SequenceMatcher(None, seed_album_key, candidate_album_key).ratio() if seed_album_key and candidate_album_key else 0.0
        evidence = max(title_similarity, 0.65 * album_similarity)
        if candidate is seed:
            evidence = 1.0
        if evidence < 0.45:
            continue
        for artist in candidate_artists:
            key = re.sub(r"\s+", " ", artist.casefold()).strip()
            artist_votes[key] = artist_votes.get(key, 0.0) + evidence
            artist_display.setdefault(key, artist)

    existing_artists = [name.strip() for name in artists.split(",") if valid_artist(name)]
    if artist_votes:
        consensus_key, consensus_score = max(artist_votes.items(), key=lambda pair: pair[1])
        total_votes = sum(artist_votes.values())
        # Require a meaningful share of the evidence; otherwise preserve all
        # upstream artists instead of forcing an uncertain identity.
        if not existing_artists and (consensus_score / max(total_votes, 1e-9) >= 0.35):
            existing_artists = [artist_display[consensus_key]]
            seed["artists"] = existing_artists[0]
        elif existing_artists and consensus_score > 1.0 and any(
            consensus_key == re.sub(r"\s+", " ", artist.casefold()).strip()
            for artist in existing_artists
        ):
            # Put the consensus artist first without discarding collaborators.
            existing_artists = [artist_display[consensus_key]] + [
                artist for artist in existing_artists
                if artist.casefold() != artist_display[consensus_key].casefold()
            ]
            seed["artists"] = ", ".join(existing_artists)
    generic_credit_stopwords = {
        "juno", "music", "records", "record", "studio", "studios", "company",
        "production", "productions", "audio", "series", "cast", "sound", "soundtrack",
        "theme", "original", "various", "artists", "artist", "entertainment"
    }
    filtered_artists = [
        name for name in existing_artists
        if name.casefold() not in generic_credit_stopwords
    ]
    artist_names = filtered_artists or existing_artists
    seed_language = _infer_track_language(seed) or str(seed.get("language", "")).strip().lower()
    if seed_language:
        seed["language"] = seed_language

    related_queries = []
    clean_title = _recommendation_title_key(title)
    if title:
        related_queries.append(title)
        if clean_title and clean_title != title.casefold():
            related_queries.append(clean_title)
            if artist_names:
                related_queries.append(f"{clean_title} {artist_names[0]}")
    source_work = _recommendation_source_work(seed)
    if source_work:
        related_queries.append(f"{source_work} soundtrack songs")
        related_queries.append(f"{source_work} songs")
    if album and not is_compilation_album(album):
        related_queries.append(f"{album} songs")
    for artist in artist_names[:2]:
        if seed_language:
            related_queries.append(f"{artist} {seed_language} songs")
        else:
            related_queries.append(f"{artist} songs")

    if seed_language:
        related_queries.extend([
            f"{seed_language} acoustic pop songs",
            f"{seed_language} indie songs",
            f"{seed_language} chill songs",
            f"feel good {seed_language} songs",
            f"{seed_language} ballads",
            f"{seed_language} classics",
        ])
    else:
        related_queries.extend(["acoustic pop songs", "indie songs", "chill songs", "classic songs"])
    # A sparse provider result must still produce a radio. This is especially
    # common for older international tracks whose artist catalogue is tiny.
    if len(search_results) < 5:
        # Discover fallback queries from the local audio model, not from a
        # hand-written genre list. This keeps sparse live searches honest.
        genre_catalog = catalog
        seed_cluster = seed.get("cluster_id")
        if seed_cluster is not None and "cluster_id" in catalog.columns:
            same_cluster = catalog[catalog["cluster_id"] == seed_cluster]
            if not same_cluster.empty:
                genre_catalog = same_cluster
        genres = genre_catalog.get("track_genre", pd.Series(dtype=str)).dropna().astype(str)
        if genres.size:
            for genre in genres.value_counts().head(3).index:
                related_queries.append(f"{genre.split(',')[0].strip()} songs")
    seed_year = _recommendation_seed_year(seed, search_results)
    # Recover a missing live release year from same-artist/album search rows.
    # This is data-driven and avoids singer-specific era rules.
    if not seed_year and not is_compilation_album(album):
        seed_artist_keys = {_compact_key(value) for value in artists.split(",") if _compact_key(value)}
        year_votes: list[int] = []
        for candidate in search_results:
            candidate_artists = {_compact_key(value) for value in str(candidate.get("artists", "")).split(",") if _compact_key(value)}
            candidate_album = _compact_key(candidate.get("album", ""))
            if seed_artist_keys.intersection(candidate_artists) or (
                album and candidate_album == _compact_key(album)
            ):
                try:
                    candidate_year = int(str(candidate.get("year", ""))[:4])
                    if 1900 <= candidate_year <= 2100:
                        year_votes.append(candidate_year)
                except (TypeError, ValueError):
                    pass
        if year_votes:
            seed_year = int(np.median(year_votes))
            seed["year"] = seed_year
    refresh_round = int(st.session_state.get("recommendation_refresh_round", 0) or 0)
    decade_query = f"{(seed_year // 10) * 10}s" if seed_year else ""
    language_query = seed_language or "pop"
    title_query_base = clean_title or title
    title_variants = [
        [
            f"songs like {title_query_base}",
            f"{artist_names[0]} hits" if artist_names else "",
        ],
        [
            f"similar to {title_query_base}",
            f"{decade_query} {language_query} hits" if decade_query else f"{language_query} indie pop songs",
        ],
        [
            f"{title_query_base} similar songs",
            f"{language_query} dance hits" if seed_language == "english" else f"{language_query} acoustic pop songs",
        ],
        [
            f"radio for {title_query_base}",
            f"{artist_names[0]} essentials" if artist_names else f"{language_query} feel good songs",
        ],
    ]
    seed_intents_set = set(seed.get("intent_labels", []))
    era_singer_queries = era_mood_singer_discovery_queries(
        seed_year,
        seed_language,
        seed_intents_set,
        artist_names,
        max_singers=5,
    )
    era_film_queries = era_theme_film_discovery_queries(
        seed_year,
        seed_language,
        seed_intents_set,
        max_films=4,
    )
    prioritized_queries = (
        related_queries[:3]
        + title_variants[refresh_round % len(title_variants)]
        + _mood_era_discovery_queries(
            seed_year,
            seed_language,
            seed_intents_set,
        )
        + era_singer_queries
        + era_film_queries
        + _era_artist_discovery_queries(seed_year, artist_names)
        + related_queries[3:]
    )
    # Put broad same-era/mood discovery before artist-only expansions so the
    # live queue can find comparable songs from other performers and films.

    collected: list[dict] = []
    provider_failures = 0
    ranking_error = ""
    ranking_error_detail = ""
    seen: set[str] = set()
    # Search results are valid radio candidates (especially same-album songs).
    # Exclude only the current seed and explicit old-queue items; excluding the
    # entire search shelf was the reason small searches produced empty queues.
    search_ids: set[str] = set()
    seen_recommendation_keys = st.session_state.setdefault(
        "jio_recommendation_history", set()
    )
    if not isinstance(seen_recommendation_keys, set):
        seen_recommendation_keys = set(seen_recommendation_keys)
        st.session_state["jio_recommendation_history"] = seen_recommendation_keys
    for previous_item in [seed, *(exclude_items or [])]:
        seen_recommendation_keys.update(_recommendation_history_keys(previous_item))
    search_titles = {_recommendation_title_key(seed.get("title", ""))}
    for item in [seed, *(exclude_items or [])]:
        search_ids.add(str(item.get("song_id") or item.get("title", "")))
    def collect_items(items: list[dict], source: str = "metadata_discovery") -> None:
        for item in items:
            key = str(item.get("song_id") or item.get("title", ""))
            title_key = _recommendation_title_key(item.get("title", ""))
            identity_keys = _recommendation_history_keys(item)
            if (
                not key
                or not title_key
                or not identity_keys
                or key in seen
                or identity_keys.intersection(seen_recommendation_keys)
                or identity_keys.intersection(seen)
                or key in search_ids
                or title_key in search_titles
            ):
                continue
            seen.add(key)
            seen.update(identity_keys)
            item.setdefault("candidate_source", source)
            collected.append(item)

    def collect_queries(queries: list[str], page_limit: int = 12, max_queries: int = 16, source: str = "metadata_discovery") -> None:
        # One bounded concurrent wave is faster than sequential batches. The
        # upstream client is cached, so repeated songs do not cause repeated
        # network work across Streamlit reruns.
        queries = list(dict.fromkeys(query for query in queries if query))[:max_queries]
        if not queries:
            return
        nonlocal provider_failures
        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as executor:
            futures = [executor.submit(client.search_direct, value, page_limit) for value in queries]
            for future in futures:
                try:
                    collect_items(future.result(), source)
                except Exception:
                    provider_failures += 1

    # The already-visible query matches are valid recommendation candidates.
    # They must enter the same collection/ranking pipeline as API discovery
    # results, otherwise a short search can produce an empty Up Next shelf.
    collect_items(
        [dict(item) for item in search_results if isinstance(item, dict)],
        source="seed_search_results",
    )
    collect_queries(prioritized_queries, page_limit=30, max_queries=20, source="seed_title_album_artist")
    soundtrack_peer_queries = _era_artist_discovery_queries(
        seed_year,
        _soundtrack_peer_artists(seed, collected),
    )
    collect_queries(
        soundtrack_peer_queries,
        page_limit=30,
        max_queries=6,
        source="soundtrack_peer_era",
    )
    # Artist radio needs the provider's expanded catalogue endpoint. The
    # narrow direct query often returns only one or two Coldplay recordings;
    # this extra artist-only fetch stays inside the recommendation pipeline,
    # where the ranker still enforces the seed's language, mood and era.
    for artist in artist_names[:2]:
        try:
            collect_items(
                client.search(artist, limit=40),
                source="seed_artist_catalogue",
            )
        except Exception:
            provider_failures += 1
    inferred_seed_year = _recommendation_seed_year(seed, collected)
    if inferred_seed_year and not seed_year:
        seed_year = inferred_seed_year
        seed["year"] = seed_year
        collect_queries(
            _mood_era_discovery_queries(
                seed_year,
                seed_language,
                set(seed.get("intent_labels", [])),
            )
            + _era_artist_discovery_queries(seed_year, artist_names),
            page_limit=30,
            max_queries=8,
            source="seed_inferred_era_mood",
        )
    enriched_collected = enrich_live_items(collected)
    soundtrack_intents = infer_same_soundtrack_audio_intents(
        seed,
        enriched_collected,
    )
    if soundtrack_intents:
        seed["intent_labels"] = sorted(
            (set(seed.get("intent_labels", [])) - {"general", "neutral"})
            | soundtrack_intents
        )
        seed["intent_source"] = "same_soundtrack_audio_proxy"
    # Bhakti agent: use dedicated devotional recommender when seed is bhakti/devotional
    bhakti_agent = BhaktiAgent(llm_classifier=_llc if _llc else None, llm_enabled=_llc is not None)
    is_bhakti_seed = bhakti_agent.is_bhakti(seed)
    try:
        if is_bhakti_seed:
            # Use dedicated BhaktiAgent for devotional seeds - ensures devotional
            # recommendations instead of romantic/popular fallbacks.
            ranked = bhakti_agent.recommend(
                seed,
                enriched_collected,
                top_n=35,
                min_score=0.30,
            )
        else:
            ranked = hybrid.rank_jio_results(
                enriched_collected,
                user_id,
                catalog,
                preferred_track_ids,
                seed=seed,
            )
    except Exception as exc:
        ranking_error = type(exc).__name__
        ranking_error_detail = str(exc)
        ranked = []
    # A strict pass is preferred, but sparse live metadata should not leave
    # the player with an empty queue when a language/era/style bridge exists.
    if len(ranked) < 15 and enriched_collected:
        try:
            if is_bhakti_seed:
                # Relaxed bhakti pass: lower min_score to surface more devotional candidates
                relaxed_ranked = bhakti_agent.recommend(
                    seed,
                    enriched_collected,
                    top_n=35,
                    min_score=0.15,
                )
            else:
                relaxed_ranked = hybrid.rank_jio_results(
                    enriched_collected,
                    user_id,
                    catalog,
                    preferred_track_ids,
                    seed=seed,
                    relaxed_context=True,
                )
            if len(relaxed_ranked) > len(ranked):
                ranked = relaxed_ranked
        except Exception:
            pass

    # Guarantee queue contains 10-35 songs matching seed's mood + era with diverse singers:
    # If the initial queries yield < 15 relevant songs, fire an intensive expansion wave
    # targeting other singers from that exact era doing the same mood + iconic era films.
    if len(ranked) < 15:
        era_term = f"{(seed_year // 10) * 10}s" if seed_year else ""
        lang_term = seed_language or "hindi"
        seed_mood_labels = [l for l in seed.get("intent_labels", []) if l not in {"general", "neutral"}]
        mood_str = seed_mood_labels[0] if seed_mood_labels else (
            "romantic" if "romantic" in str(seed.get("mood_label", ""))
            else "sad" if "sad" in str(seed.get("mood_label", ""))
            else "energetic" if "energetic" in str(seed.get("mood_label", ""))
            else "dance" if lang_term == "english"
            else "hits"
        )
        expansion_queries = []
        # Era-aware different singer queries (Rafi, Lata, Asha, etc. for 70s romantic)
        expansion_queries.extend(
            era_mood_singer_discovery_queries(
                seed_year,
                seed_language,
                seed_intents_set,
                artist_names,
                max_singers=8,
            )
        )
        # Iconic era film soundtrack queries (e.g. Sultan/Bhaag Milkha Bhaag for 2010s motivational)
        expansion_queries.extend(
            era_theme_film_discovery_queries(
                seed_year,
                seed_language,
                seed_intents_set,
                max_films=6,
            )
        )
        if era_term:
            expansion_queries.extend([
                f"{era_term} {lang_term} {mood_str} songs",
                f"{era_term} {lang_term} {mood_str} hits",
                f"{era_term} {lang_term} superhits",
                f"best of {era_term} {lang_term}",
                f"{era_term} {lang_term} top songs",
                f"{era_term} {lang_term} trending songs",
            ])
            if lang_term == "english":
                expansion_queries.extend([
                    f"{era_term} dance hits",
                    f"{era_term} pop hits",
                    f"{era_term} club hits",
                    f"{era_term} party hits",
                ])
        else:
            expansion_queries.extend([
                f"{lang_term} {mood_str} hits",
                f"{lang_term} {mood_str} songs",
                f"top {lang_term} {mood_str} songs",
                f"{lang_term} superhits",
            ])
        collect_queries(expansion_queries, page_limit=40, max_queries=16, source="same_era_mood_artist_expansion")
        enriched_collected = enrich_live_items(collected)
        try:
            expanded_ranked = hybrid.rank_jio_results(
                enriched_collected,
                user_id,
                catalog,
                preferred_track_ids,
                seed=seed,
                relaxed_context=True,
            )
            if is_bhakti_seed:
                expanded_ranked = [
                    item for item in expanded_ranked if bhakti_agent.is_bhakti(item)
                ]
            if len(expanded_ranked) > len(ranked):
                ranked = expanded_ranked
        except Exception:
            pass

    # Safety floor: never let queue drop below 10 songs under any circumstance
    if len(ranked) < 10:
        safety_queries = []
        era_term = f"{(seed_year // 10) * 10}s" if seed_year else ""
        lang_term = seed_language or "hindi"
        if era_term:
            safety_queries.extend([
                f"{era_term} {lang_term} hits",
                f"{era_term} {lang_term} songs",
                f"best of {era_term} songs",
            ])
        safety_queries.extend([
            f"{lang_term} classics" if era_term and int(str(seed_year)[:4]) < 2000 else f"{lang_term} top hits",
            f"popular {lang_term} songs",
        ])
        collect_queries(safety_queries, page_limit=40, max_queries=6, source="safety_floor_expansion")
        enriched_collected = enrich_live_items(collected)
        try:
            safety_ranked = hybrid.rank_jio_results(
                enriched_collected,
                user_id,
                catalog,
                preferred_track_ids,
                seed=seed,
                relaxed_context=True,
            )
            if is_bhakti_seed:
                safety_ranked = [
                    item for item in safety_ranked if bhakti_agent.is_bhakti(item)
                ]
            if len(safety_ranked) > len(ranked):
                ranked = safety_ranked
        except Exception:
            pass
    st.session_state["jio_queue_status"] = {
        "seed": title,
        "candidates_retrieved": len(collected),
        "relevant_candidates": len(ranked),
        "provider_query_failures": provider_failures,
        "ranking_error": ranking_error,
        "ranking_error_detail": ranking_error_detail,
        "candidate_sources": sorted({str(item.get("candidate_source", "unknown")) for item in collected}),
    }

    if not ranked:
        # Never bypass the ranker with an era/artist-only fallback. That path
        # produced queues with zero diagnostics and unrelated recommendations.
        # The caller can show a shorter/empty radio instead.
        return []
    unique: list[dict] = []
    used_recordings: set[str] = set()
    for item in ranked:
        identity_keys = _recommendation_history_keys(item)
        if identity_keys and not identity_keys.intersection(used_recordings):
            used_recordings.update(identity_keys)
            unique.append(item)
    queue = unique[:limit]
    seen_recommendation_keys.update(
        identity_key
        for item in queue
        for identity_key in _recommendation_history_keys(item)
    )
    for position, item in enumerate(queue, start=1):
        item["recommendation_rank"] = position
        item.setdefault("recommendation_source", item.get("candidate_source", "current_track_radio"))
    # One compact impression event per generated queue gives us exposure data
    # without inserting one database row per visible recommendation.
    if queue and st.session_state.get("local_user_id"):
        impression_key = "|".join([
            HYBRID_MODEL_VERSION,
            str(seed.get("song_id") or seed.get("title", "")),
            str(st.session_state.get("local_session_id", "")),
            "|".join(str(item.get("song_id") or item.get("title", "")) for item in queue[:25]),
        ])
        seen_impressions = st.session_state.setdefault("recommendation_impressions", set())
        if impression_key not in seen_impressions:
            seen_impressions.add(impression_key)
            log_behavior(
                "recommendation_impression",
                metadata={
                    "model_version": HYBRID_MODEL_VERSION,
                    "experiment_variant": st.session_state.get("recommendation_variant", "control"),
                    "seed_track_id": str(seed.get("song_id") or seed.get("title", "")),
                    "seed_title": seed.get("title", ""),
                    "seed_artists": seed.get("artists", ""),
                    "seed_album": seed.get("album", ""),
                    "seed_year": seed.get("year", ""),
                    "seed_mood": seed.get("mood_label") or seed.get("mood", ""),
                    "items": [
                        {
                            "track_id": str(item.get("song_id") or item.get("title", "")),
                            "title": item.get("title", ""),
                            "artists": item.get("artists", ""),
                            "album": item.get("album", ""),
                            "year": item.get("year", ""),
                            "rank": item.get("recommendation_rank"),
                            "score": item.get("recommendation_score", 0.0),
                            "reason": item.get("recommendation_reason", ""),
                            "source": item.get("recommendation_source", "current_track_radio"),
                            "features": item.get("recommendation_features", {}),
                        }
                        for item in queue[:25]
                    ],
                },
            )
    return queue


spotify_client = SpotifyOAuth.from_environment()
spotify_token = None
spotify_taste = st.session_state.get("spotify_taste")
with st.sidebar:
    st.header("Your CozyBeats account")
    local_user_id = st.session_state.get("local_user_id")
    if not local_user_id:
        restored_session = _read_auth_session_ticket()
        if restored_session:
            local_user_id, issued_at = restored_session
            st.session_state["local_user_id"] = local_user_id
            st.session_state.setdefault("local_session_started_at", issued_at)
            st.session_state["local_is_admin"] = user_store.is_admin(local_user_id)
    if local_user_id:
        summary = cached_user_summary(local_user_id)
        st.success("Signed in")
        if st.session_state.get("auth_storage_warning"):
            st.warning(st.session_state.pop("auth_storage_warning"))
        st.caption(f"{summary['tracks_played']} tracks played · {summary['likes']} likes")
        if st.session_state.get("local_session_started_at"):
            minutes = (time.time() - st.session_state["local_session_started_at"]) / 60
            st.caption(f"Current session: {minutes:.1f} min")
        if st.button("Log out of CozyBeats", width="stretch"):
            account_listening_time()
            started = st.session_state.pop("local_session_started_at", time.time())
            log_behavior("session_end", duration_ms=int((time.time() - started) * 1000))
            st.session_state.pop("local_user_id", None)
            st.session_state.pop("local_is_admin", None)
            st.session_state.pop("user_summary_cache", None)
            st.session_state.pop("user_events_cache", None)
            st.query_params.pop(AUTH_SESSION_PARAM, None)
            st.rerun()
    else:
        st.subheader("Welcome to CozyBeats")
        auth_choice = st.radio(
            "Choose an option to continue",
            ["I am new — create account", "I already have an account — log in"],
            index=None,
            key="auth_choice",
        )
        if auth_choice is None:
            st.info("First choose whether you are a new or existing user.")
            st.stop()
        is_new_user = auth_choice.startswith("I am new")
        auth_mode = "Create account" if is_new_user else "Log in"
        with st.form("local_auth", border=False):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(auth_mode, width="stretch")
        if submitted:
            try:
                if is_new_user:
                    ok, message = user_store.register(email, password)
                else:
                    ok, message = user_store.authenticate(email, password)
            except Exception:
                # A remote MySQL timeout must not break the auth screen and
                # must not silently create a user in ephemeral local storage.
                # The account is only considered created when MySQL confirms it.
                ok, message = False, (
                    "Cloud database timed out. No account was created; please try again after "
                    "checking that the Aiven MySQL service is running."
                )
            if ok:
                st.session_state["local_user_id"] = message
                bucket = int(hashlib.sha256(str(message).encode("utf-8")).hexdigest()[:8], 16) % 2
                st.session_state["recommendation_variant"] = "diverse" if bucket else "control"
                st.session_state["local_session_started_at"] = time.time()
                st.session_state["local_is_admin"] = user_store.is_admin(message)
                st.query_params[AUTH_SESSION_PARAM] = _make_auth_session_ticket(message)
                log_behavior("session_start")
                st.rerun()
            else:
                st.error(message)
    if not st.session_state.get("local_user_id"):
        st.info("Create a CozyBeats account to save likes, plays, skips, and session analytics.")
        st.stop()

account_listening_time()


@st.fragment(run_every="10s")
def render_admin_dashboard() -> None:
    """Admin-only observability view; normal listeners never see this data."""
    st.header("CozyBeats admin analytics")
    st.caption("Operational and data-science metrics from local implicit-feedback events · auto-refreshes every 10 seconds")
    events = pd.DataFrame(user_store.admin_events())
    users = pd.DataFrame(user_store.admin_users())
    status_a, status_b, status_c, status_d = st.columns(4)
    status_a.metric("Database", user_store.database_kind.upper())
    status_b.metric("Catalog tracks", f"{len(data):,}")
    status_c.metric("Mood clusters", int(mood_cluster_model.n_clusters))
    status_d.metric("ML features", len(MOOD_FEATURES))
    if events.empty:
        st.info("No listener events have been collected yet.")
        return

    events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce", utc=True)
    events["duration_min"] = pd.to_numeric(events["duration_ms"], errors="coerce").fillna(0) / 60000
    plays = events[events["event_type"].isin(["jio_play", "play_start"])]
    likes = events[events["event_type"].isin(["jio_like", "like"])]
    skips = events[events["event_type"].isin(["jio_skip", "skip"])]
    a, b, c, d = st.columns(4)
    a.metric("Registered users", int(len(users)))
    b.metric("Total events", int(len(events)))
    c.metric("Play starts", int(len(plays)))
    d.metric("Listening hours", f"{events['duration_min'].sum() / 60:.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Event mix")
        st.bar_chart(events["event_type"].value_counts().rename("events"))
    with right:
        st.subheader("User activity")
        by_user = events.groupby("user_id").agg(
            events=("event_type", "size"),
            plays=("event_type", lambda values: values.isin(["jio_play", "play_start"]).sum()),
            likes=("event_type", lambda values: values.isin(["jio_like", "like"]).sum()),
            skips=("event_type", lambda values: values.isin(["jio_skip", "skip"]).sum()),
        ).sort_values("events", ascending=False)
        st.bar_chart(by_user[["plays", "likes", "skips"]])

    daily = events.dropna(subset=["created_at"]).copy()
    if not daily.empty:
        daily["date"] = daily["created_at"].dt.date.astype(str)
        listening = daily.groupby("date")["duration_min"].sum().rename("minutes")
        st.subheader("Listening time over time")
        st.line_chart(listening)

    # Metadata quality and catalog observability.
    metadata_rows = []
    for raw in events.get("metadata_json", pd.Series(dtype=str)).fillna("{}"): 
        try:
            metadata_rows.append(json.loads(raw))
        except (TypeError, ValueError):
            metadata_rows.append({})
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        exposed_ids: set[str] = set()
        ndcg_values: list[float] = []
        precision_values: list[float] = []
        for index, row in enumerate(metadata_rows):
            if index >= len(events) or events.iloc[index]["event_type"] != "recommendation_impression":
                continue
            if not isinstance(row, dict):
                continue
            user_id = events.iloc[index]["user_id"]
            user_played = {
                str(value) for value in plays.loc[plays["user_id"] == user_id, "track_id"].dropna().tolist()
            }
            ranked_items = row.get("items", [])
            relevance = [
                int(str(item.get("track_id")) in user_played)
                for item in ranked_items[:10]
                if isinstance(item, dict) and item.get("track_id")
            ]
            if relevance:
                precision_values.append(sum(relevance) / len(relevance))
                dcg = sum((2 ** value - 1) / np.log2(position + 2) for position, value in enumerate(relevance))
                ideal = sorted(relevance, reverse=True)
                idcg = sum((2 ** value - 1) / np.log2(position + 2) for position, value in enumerate(ideal))
                ndcg_values.append(dcg / idcg if idcg else 0.0)
            for item in ranked_items:
                if isinstance(item, dict) and item.get("track_id"):
                    exposed_ids.add(str(item["track_id"]))
        played_ids = {
            str(value) for value in plays.get("track_id", pd.Series(dtype=str)).dropna().tolist()
        }
        recommended_plays = len(exposed_ids.intersection(played_ids))
        if exposed_ids:
            st.subheader("Recommendation evaluation")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Tracks exposed", f"{len(exposed_ids):,}")
            e2.metric("Exposed tracks played", f"{recommended_plays:,}")
            e3.metric("Exposure play-through", f"{recommended_plays / len(exposed_ids) * 100:.1f}%")
            e4.metric("Precision@10", f"{np.mean(precision_values) * 100:.1f}%" if precision_values else "—")
            e5.metric("NDCG@10", f"{np.mean(ndcg_values):.3f}" if ndcg_values else "—")
        # Explainability audit: show the queue generated for the latest play,
        # rather than leaving an older seed visible when the new queue is not
        # ready yet.
        latest_impression = None
        latest_play = plays.iloc[-1] if not plays.empty else None
        latest_play_index = plays.index[-1] if not plays.empty else None
        latest_play_metadata = (
            metadata_rows[latest_play_index]
            if latest_play_index is not None and latest_play_index < len(metadata_rows)
            else {}
        )
        if latest_play is not None:
            latest_play_track_id = str(latest_play.get("track_id") or "")
            for event_index in range(len(metadata_rows) - 1, -1, -1):
                event = events.iloc[event_index]
                event_metadata = metadata_rows[event_index]
                if (
                    isinstance(event_metadata, dict)
                    and event_metadata.get("items")
                    and event["event_type"] == "recommendation_impression"
                    and str(event_metadata.get("seed_track_id") or "") == latest_play_track_id
                    and str(event["user_id"]) == str(latest_play["user_id"])
                    and str(event["session_id"]) == str(latest_play["session_id"])
                    and event["created_at"] >= latest_play["created_at"]
                ):
                    latest_impression = (event, event_metadata)
                    break
        else:
            for event_index in range(len(metadata_rows) - 1, -1, -1):
                event_metadata = metadata_rows[event_index]
                if (
                    isinstance(event_metadata, dict)
                    and event_metadata.get("items")
                    and events.iloc[event_index]["event_type"] == "recommendation_impression"
                ):
                    latest_impression = (events.iloc[event_index], event_metadata)
                    break
        diagnostic_rows = []
        if latest_impression:
            impression_event, impression_metadata = latest_impression
            for item in impression_metadata.get("items", [])[:10]:
                if not isinstance(item, dict):
                    continue
                features = item.get("features") or {}
                diagnostic_rows.append({
                    "track": item.get("title", item.get("track_id", "")),
                    "artist": item.get("artists", ""),
                    "year": item.get("year", ""),
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "reason": item.get("reason", ""),
                    "base_relevance": features.get("base_relevance", item.get("score", 0.0)),
                    "final_rank_score": features.get("final_rank_score", item.get("score", 0.0)),
                    "redundancy_penalty": features.get("redundancy_penalty", 0.0),
                    "audio": features.get("audio", 0.0),
                    "mood": features.get("mood", 0.0),
                    "intent": features.get("intent", 0.0),
                    "metadata": features.get("metadata", 0.0),
                    "era": features.get("era", 0.0),
                    "collaborative": features.get("collaborative", 0.0),
                    "transition": features.get("transition", 0.0),
                    "source": item.get("source", ""),
                })
        if diagnostic_rows:
            st.subheader("Latest recommendation diagnostics")
            seed_details = " · ".join(
                str(value).strip()
                for value in (
                    impression_metadata.get("seed_title"),
                    impression_metadata.get("seed_artists"),
                    impression_metadata.get("seed_album"),
                    impression_metadata.get("seed_year"),
                )
                if str(value or "").strip()
            )
            st.caption(
                f"Latest logged queue · seed: {seed_details or 'unknown'} · "
                f"model: {impression_metadata.get('model_version', 'unknown')} · "
                f"generated: {impression_event['created_at']} · "
                "updates when the current song gets a newly generated queue. "
                "Metadata-only mood is not audio proof."
            )
            diagnostics = pd.DataFrame(diagnostic_rows)
            rank_scores = diagnostics[["track", "rank", "final_rank_score"]].copy()
            rank_scores["rank"] = pd.to_numeric(rank_scores["rank"], errors="coerce")
            rank_scores["final_rank_score"] = pd.to_numeric(
                rank_scores["final_rank_score"], errors="coerce"
            )
            rank_scores = rank_scores.dropna(
                subset=["rank", "final_rank_score"]
            ).sort_values("rank")
            if not rank_scores.empty:
                st.markdown("**Recommendation rank scores**")
                st.bar_chart(
                    rank_scores.set_index("track")[["final_rank_score"]],
                    horizontal=True,
                    height=max(240, len(rank_scores) * 36),
                )
            st.dataframe(diagnostics, width="stretch", hide_index=True)
        elif latest_play is not None:
            latest_play_title = (
                latest_play_metadata.get("title")
                or latest_play_metadata.get("track_name")
                or latest_play.get("track_id")
                or "Unknown song"
            )
            st.subheader("Latest recommendation diagnostics")
            st.caption(
                f"No recommendation queue has been generated for the latest played song "
                f"({latest_play_title}) yet."
            )
        source_counts = metadata.get("feature_source", pd.Series(dtype=str)).fillna("unknown").value_counts()
        matched = int(source_counts.get("local_audio_catalog", 0))
        fallback = int(source_counts.get("live_metadata_fallback", 0))
        st.subheader("Live feature enrichment")
        q1, q2, q3 = st.columns(3)
        q1.metric("Live items enriched", matched)
        q2.metric("Metadata fallback", fallback)
        q3.metric("Feature match rate", f"{matched / max(1, matched + fallback) * 100:.1f}%")
        st.bar_chart(source_counts.rename("events"))

        st.subheader("Most played artists")
        play_mask = events["event_type"].isin(["jio_play", "play_start"]).to_numpy()

        def metadata_artist_names(value: object) -> list[str]:
            if isinstance(value, list):
                return [
                    str(entry.get("name", entry)).strip()
                    if isinstance(entry, dict) else str(entry).strip()
                    for entry in value
                    if str(entry).strip()
                ]
            return [part.strip() for part in str(value or "").split(",") if part.strip()]

        artist_values: list[str] = []
        for index, row in enumerate(metadata.to_dict("records")):
            if index >= len(play_mask) or not play_mask[index]:
                continue
            raw_artists = row.get("artists") or row.get("artist") or row.get("singers")
            if not raw_artists and row.get("title"):
                event_title = _compact_key(row.get("title", ""))
                best_artist = ""
                best_score = 0.0
                for candidate in live_title_lookup:
                    candidate_title = candidate["title"]
                    score = SequenceMatcher(None, event_title, candidate_title).ratio()
                    if event_title in candidate_title or candidate_title in event_title:
                        score = max(score, 0.82)
                    if score > best_score:
                        best_score = score
                        best_artist = candidate["artists"]
                if best_score >= 0.72:
                    raw_artists = best_artist
            artist_values.extend(metadata_artist_names(raw_artists))

        # Count only actual play events, and merge harmless casing/whitespace
        # differences so one singer cannot be split into multiple rows.
        canonical_artists: dict[str, str] = {}
        for artist in artist_values:
            key = re.sub(r"\s+", " ", artist).strip().casefold()
            canonical_artists.setdefault(key, artist)
        artist_counts = pd.Series(
            [canonical_artists[re.sub(r"\s+", " ", artist).strip().casefold()] for artist in artist_values]
        ).value_counts().rename("plays")
        # Do not truncate this to the top 15: the admin view should expose
        # every artist present in the collected event metadata.
        if not artist_counts.empty:
            artist_table = artist_counts.to_frame(name="plays")
            artist_table.index.name = "artist"
            artist_table = artist_table.reset_index()
            st.dataframe(
                artist_table,
                width="stretch",
                hide_index=True,
            )
            # Use an explicit index for the chart. Passing a dataframe whose
            # index and column both carry the artist label can make pandas'
            # internal reset_index raise a duplicate-column ValueError.
            st.bar_chart(artist_table.set_index("artist")["plays"], horizontal=True)

        # Live audit trail: show the actual recent play metadata rather than
        # only aggregate artist counts. This makes the current singer/song
        # visible as soon as the next 10-second fragment refresh runs.
        play_metadata = metadata.loc[play_mask].copy()
        play_events = events.loc[
            events["event_type"].isin(["jio_play", "play_start"])
        ].copy()
        if not play_metadata.empty:
            user_names = {str(row["id"]): row["email"] for row in users.to_dict("records")}
            recent_rows = []
            for event, meta in zip(play_events.tail(15).iloc[::-1].to_dict("records"), play_metadata.tail(15).iloc[::-1].to_dict("records")):
                recent_rows.append({
                    "time": event.get("created_at"),
                    "user": user_names.get(str(event.get("user_id")), f"user {event.get('user_id')}"),
                    "song": meta.get("title") or meta.get("track_name") or event.get("track_id"),
                    "artist": meta.get("artists") or meta.get("artist") or "Unknown",
                    "album": meta.get("album") or "",
                })
            st.subheader("Live recent plays")
            st.dataframe(pd.DataFrame(recent_rows), width="stretch", hide_index=True)

        quality_columns = [column for column in ("title", "artists", "album", "stream_url", "image_url") if column in metadata]
        if quality_columns:
            quality = pd.DataFrame({column: metadata[column].fillna("").astype(str).str.strip().ne("") for column in quality_columns})
            quality = (quality.mean() * 100).round(1).rename("completeness_percent")
            st.subheader("External music data quality")
            st.dataframe(quality.to_frame(), width="stretch")

    # Collaborative-filtering health: the interaction matrix dimensions and
    # density tell us whether MF has enough signal or is still cold-start.
    interactions = events[events["track_id"].notna()]
    matrix_users = interactions["user_id"].nunique()
    matrix_items = interactions["track_id"].nunique()
    matrix_cells = max(1, matrix_users * matrix_items)
    st.subheader("Recommendation model health")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users in interaction matrix", int(matrix_users))
    m2.metric("Unique interacted songs", int(matrix_items))
    m3.metric("Matrix density", f"{len(interactions) / matrix_cells * 100:.2f}%")
    m4.metric("Like rate", f"{len(likes) / max(1, len(plays)) * 100:.1f}%")
    st.caption("Low matrix density means the system should rely more on content, mood and cold-start signals than collaborative filtering.")

    model_table = pd.DataFrame({
        "signal": ["audio similarity", "metadata/mood", "collaborative/MF", "recent + long-term taste", "session transition", "context + popularity", "diversity"],
        "weight_or_rule": ["35%", "15%", "15%", "20%", "10%", "3% + 2%", "hard freshness + artist/album spacing"],
    })
    with st.expander("Live model configuration", expanded=False):
        st.dataframe(model_table, width="stretch", hide_index=True)
        st.caption("MiniBatchKMeans clusters the local audio catalog; matched live songs inherit those features. Unmatched songs use metadata fallback.")
        st.bar_chart(data["cluster_id"].value_counts().sort_index().rename("catalog_tracks"))


if st.session_state.get("local_is_admin", False):
    with st.sidebar:
        workspace_view = st.radio("Workspace", ["CozyBeats", "Admin dashboard"], key="workspace_view")
    if workspace_view == "Admin dashboard":
        render_admin_dashboard()
        st.stop()

st.markdown(
    "<div class='welcome-title'>Good evening, listener</div>"
    "<div class='welcome-copy'>Press play, follow a feeling, and let your next favorite find you.</div>",
    unsafe_allow_html=True,
)

st.session_state.setdefault("active_search_query", "")
with st.form("global_search_form", border=False):
    search_input, search_action = st.columns([6, 1], vertical_alignment="bottom")
    with search_input:
        typed_search_query = st.text_input(
            "Search songs, artists, albums, or genres",
            placeholder="What do you want to listen to?",
            key="global_search_input",
        )
    with search_action:
        submitted_search = st.form_submit_button(
            "Search",
            icon=":material/search:",
            width="stretch",
        )
if submitted_search:
    st.session_state["active_search_query"] = typed_search_query.strip()
search_query = st.session_state.get("active_search_query", "")

# Personalization is learned from plays, likes, imported history, and the
# live recommendation feedback pipeline; no extra taste form is needed here.
onboarding_genres: list[str] = []
onboarding_tracks: list[str] = []

live_track_ids = local_ids_for_spotify_taste(spotify_taste or {}, data)
imported_track_ids = local_ids_for_imported_history(imported_history, data)
preferred_track_ids = (
    user_store.liked_track_ids(local_user_id)
    + live_track_ids
    + imported_track_ids
)


def _dashboard_language_frames(profile: object, excluded_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Build broad, personalized local shelves once per profile state."""
    ranked = engine.recommend_for_profile(
        profile,
        n=min(len(data), 1400),
        exclude_ids=excluded_ids,
        allow_explicit=False,
        popularity_weight=0.10,
        diversity=0.12,
    )
    genre_text = data["track_genre"].fillna("").astype(str).str.casefold()
    hindi_mask = genre_text.str.contains(
        r"bollywood|filmi|hindi|indian|bhangra|punjabi|desi pop|sufi|tollywood|bhojpuri|marathi",
        regex=True,
    )
    # The local dataset has sparse language labels. Treat the remaining
    # catalogue as English/global so the dashboard can still provide a full
    # 100+ track shelf without pretending every row has verified language.
    english_mask = ~hindi_mask
    ranked_ids = set(ranked["track_id"].astype(str)) if not ranked.empty else set()

    def shelf(mask: pd.Series) -> pd.DataFrame:
        selected = ranked[ranked["track_id"].astype(str).isin(set(data.loc[mask, "track_id"].astype(str)))]
        selected = selected.drop_duplicates(subset=["track_name", "artists"])
        if len(selected) < 100:
            fallback = data.loc[mask].copy()
            fallback = fallback[~fallback["track_id"].astype(str).isin(ranked_ids)]
            fallback = fallback.sort_values(["popularity", "track_name"], ascending=[False, True])
            selected = pd.concat([selected, fallback], ignore_index=True)
        return selected.drop_duplicates(subset=["track_name", "artists"]).head(1000).reset_index(drop=True)

    english_live = fetch_english_dashboard_tracks(jio_client.base_url)
    english_shelf = english_live if len(english_live) >= 100 else shelf(english_mask)
    return {"Hindi": shelf(hindi_mask), "English / global": english_shelf}


def _recommendation_feedback(track_id: str, section: str, rating: str) -> None:
    log_behavior(
        "recommendation_feedback",
        str(track_id),
        metadata={"rating": rating, "section": section},
    )
    st.toast("Got it — your dashboard will learn from this")


def _set_dashboard_browse_page(widget_key: str, page: int) -> None:
    st.session_state[widget_key] = page


def render_dashboard_shelf(language: str, tracks: pd.DataFrame) -> None:
    page_key = f"dashboard_page_{language}"
    browse_key = f"dashboard_browse_{language}"
    page_size = 12
    page_count = max(1, int(np.ceil(len(tracks) / page_size)))
    if browse_key not in st.session_state:
        initial_page = int(st.session_state.get(page_key, 0)) + 1
        st.session_state[browse_key] = min(max(initial_page, 1), page_count)
    else:
        st.session_state[browse_key] = min(
            max(int(st.session_state[browse_key]), 1),
            page_count,
        )
    page = int(st.session_state[browse_key]) - 1
    st.session_state[page_key] = page
    start = page * page_size
    visible = tracks.iloc[start:start + page_size]
    st.markdown(f"<div class='shelf-title'>{language} picks</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='shelf-subtitle'>100+ personalized picks · showing {start + 1}–{start + len(visible)} of {len(tracks)}</div>",
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Jump to page",
        list(range(1, page_count + 1)),
        index=page,
        key=browse_key,
    )
    poster_records = tuple(
        (
            str(row.get("track_id", "")),
            str(row.get("track_name", "")),
            str(row.get("artists", "")),
        )
        for _, row in visible.iterrows()
        if not str(row.get("image_url", ""))
    )
    posters = fetch_dashboard_posters(poster_records, jio_client.base_url)
    for row_start in range(0, len(visible), 4):
        shelf_cols = st.columns(4)
        for column_index, (_, row) in enumerate(visible.iloc[row_start:row_start + 4].iterrows()):
            with shelf_cols[column_index], st.container(border=True):
                title = str(row.get("track_name", "Unknown song"))
                artist = str(row.get("artists", "Unknown artist"))
                album = str(row.get("album_name", ""))
                artwork = str(row.get("image_url", "")) or posters.get(str(row.get("track_id", "")), {}).get("image_url", "")
                if artwork:
                    st.image(artwork, width="stretch")
                st.caption(f"{start + row_start + column_index + 1:02d}")
                st.markdown(f"**{title}**")
                st.caption(f"{artist}{f' · {album}' if album else ''}")
                feedback_key = str(row.get("track_id", title))
                live_item = row.to_dict() if row.get("song_id") else posters.get(str(row.get("track_id", "")), {})
                play_col, like_col = st.columns(2)
                if play_col.button("▶ Play", key=f"dashboard_play_{language}_{feedback_key}", width="stretch"):
                    if live_item.get("stream_url") or live_item.get("perma_url"):
                        play_dashboard_live_item(live_item)
                    else:
                        st.warning("Live preview unavailable for this pick.")
                if like_col.button("♡ Like", key=f"dashboard_like_{language}_{feedback_key}", width="stretch"):
                    like_id = str(live_item.get("song_id") or row.get("track_id", title))
                    log_behavior("jio_like", like_id, metadata=live_item or {"title": title, "artists": artist})
                    refresh_hybrid_model(force=True)
                    st.toast("Added to your taste profile")
                yes_col, no_col = st.columns(2)
                if yes_col.button("Yes", key=f"dash_yes_{language}_{feedback_key}", width="stretch"):
                    _recommendation_feedback(feedback_key, language, "good")
                if no_col.button("No", key=f"dash_no_{language}_{feedback_key}", width="stretch"):
                    _recommendation_feedback(feedback_key, language, "bad")
    prev_col, page_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        st.button(
            "← Previous",
            key=f"dash_prev_{language}",
            disabled=page <= 0,
            width="stretch",
            on_click=_set_dashboard_browse_page,
            args=(browse_key, page),
        )
    with page_col:
        st.caption(f"Page {page + 1} of {page_count}", text_alignment="center")
    with next_col:
        st.button(
            "Next →",
            key=f"dash_next_{language}",
            disabled=page >= page_count - 1,
            width="stretch",
            on_click=_set_dashboard_browse_page,
            args=(browse_key, page + 2),
        )


dashboard_profile = engine.build_user_profile(
    track_ids=preferred_track_ids,
    genres=onboarding_genres,
)
dashboard_signature = "|".join([
    ",".join(sorted(map(str, preferred_track_ids))),
    ",".join(sorted(map(str, onboarding_genres))),
])
if st.session_state.get("dashboard_signature") != dashboard_signature:
    st.session_state["dashboard_shelves"] = _dashboard_language_frames(
        dashboard_profile,
        preferred_track_ids,
    )
    st.session_state["dashboard_signature"] = dashboard_signature
dashboard_shelves = st.session_state["dashboard_shelves"]
with st.container(border=True):
    st.markdown("### Your music dashboard")
    total_dashboard_tracks = len(dashboard_shelves["Hindi"]) + len(dashboard_shelves["English / global"])
    count_a, count_b, count_c = st.columns(3)
    count_a.metric("Dashboard library", f"{total_dashboard_tracks:,} tracks")
    count_b.metric("Hindi shelf", f"{len(dashboard_shelves['Hindi']):,} picks")
    count_c.metric("English / global shelf", f"{len(dashboard_shelves['English / global']):,} picks")
    st.caption("Two mood-aware shelves with 1000+ tracks across the dashboard. Use Jump to page to browse the full library.")
    hindi_tab, english_tab = st.tabs(["Hindi", "English / global"])
    with hindi_tab:
        render_dashboard_shelf("Hindi", dashboard_shelves["Hindi"])
    with english_tab:
        render_dashboard_shelf("English / global", dashboard_shelves["English / global"])


def play_next_from_queue(mark_previous_as_skip: bool = True) -> bool:
    """Promote the next already-ranked item without waiting on discovery."""
    queue = list(st.session_state.get("jio_queue", []))
    if not queue:
        return False
    current_id = st.session_state.get("jio_current_track_id")
    if current_id and mark_previous_as_skip:
        defer_player_event("jio_skip", str(current_id))
    next_item = queue.pop(0)
    st.session_state["jio_queue"] = queue
    st.session_state["jio_now_playing"] = next_item
    st.session_state["jio_current_track_id"] = str(
        next_item.get("song_id") or next_item.get("title", "")
    )
    st.session_state["jio_started_at"] = time.time()
    st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
    defer_player_event(
        "jio_play",
        str(next_item.get("song_id") or next_item.get("title", "")),
        metadata=next_item,
    )
    next_id = str(next_item.get("song_id") or next_item.get("title", ""))
    st.session_state["jio_queue_attempted_id"] = next_id
    st.session_state["jio_queue_refill_attempted_id"] = ""
    return True

if not search_query:
    st.info("Search a song, artist, album, or genre to see matching music and play it.")
else:
    st.markdown('<div class="shelf-title">Search results</div>', unsafe_allow_html=True)
    st.markdown('<div class="shelf-subtitle">Songs · all live results</div>', unsafe_allow_html=True)
    try:
        search_cache = st.session_state["jio_search_cache_v4"]
        search_cache_key = search_query.strip().lower()
        jio_results = search_cache.get(search_cache_key)
        search_catalog_limit = 200
        if jio_results is None:
            jio_results = jio_client.search(search_query, limit=search_catalog_limit)
            if not jio_results:
                local_guess = engine.search(search_query, limit=1)
                if not local_guess.empty:
                    guess = local_guess.iloc[0]
                    fallback_query = f"{guess['track_name']} {guess.get('album_name', '')}".strip()
                    jio_results = jio_client.search(fallback_query, limit=search_catalog_limit)
            raw_jio_results = list(jio_results or [])
            # If the provider returned only broad token matches, recover an exact
            # local title/artist and use that identity to query JioSaavn. This is
            # what makes a bare title search recover the matching local
            # title/artist instead of stopping at unrelated token matches.
            query_key = _compact_key(search_query)
            provider_has_exact_title = any(
                _compact_key(item.get("title", "")) == query_key
                for item in raw_jio_results
            )
            if query_key and not provider_has_exact_title:
                local_exact = data[
                    data["track_name"].astype(str).map(_compact_key) == query_key
                ]
                if not local_exact.empty:
                    guess = local_exact.iloc[0]
                    artist = str(guess.get("artists", "")).split(",")[0].strip()
                    identity_query = f"{guess['track_name']} {artist}".strip()
                    if identity_query:
                        identity_results = jio_client.search(identity_query, limit=search_catalog_limit)
                        raw_jio_results = identity_results + raw_jio_results
            query_matches = prioritize_search_results(
                enrich_live_items(raw_jio_results),
                search_query,
            )
            query_intents = set(infer_music_intents({"title": search_query})) - {"general"}
            for item in query_matches:
                item["query_intents"] = sorted(query_intents)
            jio_results = query_matches
            if jio_results:
                search_cache[search_cache_key] = jio_results
        if jio_results:
            st.caption(f"{len(jio_results)} songs")
            for row_start in range(0, len(jio_results), 4):
                jio_cols = st.columns(4)
                for column_index, item in enumerate(jio_results[row_start:row_start + 4]):
                    i = row_start + column_index + 1
                    with jio_cols[column_index], st.container(border=True):
                        image_url = item.get("image_url", "")
                        song_id = item.get("song_id", "")
                        title = item.get("title", "Unknown song")
                        artists = item.get("artists", "Unknown artist")
                        album = item.get("album", "")
                        perma_url = item.get("perma_url", "")
                        if image_url:
                            st.image(image_url, width="stretch")
                        st.caption(f"Search result {i}")
                        st.markdown(f"**{title}**")
                        era = str(item.get("year", "")).strip()
                        details = f"Singer: {artists}"
                        if album:
                            details += f" · {album}"
                        if era:
                            details += f" · {era}"
                        st.caption(details)
                        if st.button("▶ Play", key=f"jio_play_{i}_{song_id or title}", width="stretch"):
                            # Set the selected song first; recommendation
                            # fetching must never make the player disappear.
                            st.session_state["jio_now_playing"] = item
                            st.session_state["scroll_to_player"] = True
                            st.session_state["jio_queue"] = []
                            st.session_state.pop("jio_queue_seed_id", None)
                            st.session_state.pop("jio_queue_attempted_id", None)
                            st.session_state.pop("jio_queue_refill_attempted_id", None)
                            st.session_state["jio_current_track_id"] = str(
                                item.get("song_id") or item.get("title", "")
                            )
                            st.session_state["jio_started_at"] = time.time()
                            st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                            defer_player_event(
                                "jio_play",
                                str(item.get("song_id") or item.get("title", "")),
                                metadata=item,
                            )
                            st.rerun()
                        if st.button("♡ Like", key=f"jio_like_{i}_{song_id or title}", width="stretch"):
                            log_behavior(
                                "jio_like",
                                str(item.get("song_id") or item.get("title", "")),
                                metadata=item,
                            )
                            # A like changes ranking, not the user's whole
                            # queue. Preserve the existing queue and rerank it
                            # in memory so the action stays instant and does
                            # not shrink 55 recommendations to a few API rows.
                            refresh_hybrid_model(force=True)
                            existing_queue = list(st.session_state.get("jio_queue", []))
                            if existing_queue:
                                st.session_state["jio_queue"] = hybrid_engine.rank_jio_results(
                                    existing_queue,
                                    local_user_id,
                                    data,
                                    preferred_track_ids,
                                    seed=item,
                                )[:70]
                            else:
                                st.session_state["jio_queue"] = build_jio_up_next(
                                    item,
                                    jio_results,
                                    jio_client,
                                    hybrid_engine,
                                    local_user_id,
                                    data,
                                    preferred_track_ids,
                                    exclude_items=[item],
                                )
                            st.toast("Added to your taste profile")
                        if perma_url:
                            st.link_button("Open song", perma_url, width="stretch")
        else:
            st.caption("No songs found for this search.")
    except Exception as exc:
        st.error(f"Music search failed: {type(exc).__name__}: {exc}")

jio_now = st.session_state.get("jio_now_playing")
if jio_now and (
    not jio_now.get("stream_url")
    or "preview.saavncdn.com" in str(jio_now.get("stream_url", ""))
) and jio_now.get("title"):
    # Upgrade a player item created before the API started exposing media_url.
    try:
        refreshed = jio_client.search(str(jio_now["title"]), limit=5)
        same_song = next(
            (item for item in refreshed if item.get("song_id") == jio_now.get("song_id")),
            next((item for item in refreshed if item.get("title") == jio_now.get("title")), None),
        )
        if same_song and same_song.get("stream_url"):
            jio_now = same_song
            st.session_state["jio_now_playing"] = same_song
    except Exception:
        pass
if jio_now:
    current_queue_seed = str(jio_now.get("song_id") or jio_now.get("title", ""))
    current_queue = list(st.session_state.get("jio_queue", []))
    st.html("<div id='tune-match-player-anchor'></div>")
    st.markdown('<div class="shelf-title">CozyBeats player</div>', unsafe_allow_html=True)
    now_cols = st.columns([1, 3, 1, 1])
    with now_cols[0]:
        if jio_now.get("image_url"):
            st.image(jio_now["image_url"], width="stretch", alt=jio_now.get("title", "CozyBeats song"))
    with now_cols[1]:
        st.markdown(f"**{jio_now.get('title', 'Unknown song')}**")
        now_details = f"Singer: {jio_now.get('artists', '')}"
        if jio_now.get("album"):
            now_details += f" · {jio_now.get('album')}"
        if jio_now.get("year"):
            now_details += f" · {jio_now.get('year')}"
        st.caption(now_details)
        current_id = str(jio_now.get("song_id") or jio_now.get("title", ""))
        # Component-v2 IDs cannot contain its internal `__` delimiter. Live
        # providers occasionally return titles/IDs containing underscores.
        audio_component_key = re.sub(r"[^A-Za-z0-9]+", "-", current_id).strip("-") or "track"
        stream_url = jio_now.get("stream_url", "")
        perma_url = jio_now.get("perma_url", "")
        if stream_url:
            audio_result = COZYBEATS_AUDIO(
                data={
                    "url": stream_url,
                    "title": jio_now.get("title", "CozyBeats player"),
                    "track_id": current_id,
                    "autoplay": True,
                    "loop": bool(st.session_state.get("jio_loop", False)),
                },
                key=f"cb-audio-{audio_component_key}",
                on_ended_change=lambda: None,
            )
            audio_ended = bool(
                getattr(audio_result, "ended", None)
                and not st.session_state.get("jio_loop", False)
            )
        elif perma_url:
            st.warning("Audio preview unavailable for this song.")
            st.link_button("Open song", perma_url, width="stretch")
        else:
            audio_ended = False
        flush_player_events()
        if audio_ended and play_next_from_queue(mark_previous_as_skip=False):
            st.rerun()
        existing_queue = list(st.session_state.get("jio_queue", []))
        queue_seed_changed = (
            st.session_state.get("jio_queue_seed_id") != current_id
        )
        if queue_seed_changed or (
            len(existing_queue) <= 8
            and st.session_state.get("jio_queue_refill_attempted_id") != current_id
        ):
            st.session_state["jio_queue_attempted_id"] = current_id
            st.session_state["jio_queue_refill_attempted_id"] = current_id
            try:
                refresh_hybrid_model()
                cached_results = st.session_state.get("jio_search_cache_v4", {}).get(
                    search_query.strip().lower(), []
                )
                fresh_queue = build_jio_up_next(
                    jio_now,
                    [*existing_queue, *cached_results],
                    jio_client,
                    hybrid_engine,
                    local_user_id,
                    data,
                    preferred_track_ids,
                    exclude_items=[jio_now] if queue_seed_changed else [jio_now, *existing_queue],
                )
                if queue_seed_changed:
                    st.session_state["jio_queue"] = fresh_queue[:30]
                else:
                    merged_queue = []
                    seen_recordings: set[str] = set()
                    for item in [*existing_queue, *fresh_queue]:
                        identity_keys = _recommendation_history_keys(item)
                        if identity_keys and not identity_keys.intersection(seen_recordings):
                            seen_recordings.update(identity_keys)
                            merged_queue.append(item)
                    st.session_state["jio_queue"] = merged_queue[:30]
                st.session_state["jio_queue_seed_id"] = current_id
            except Exception:
                st.session_state["jio_queue"] = [] if queue_seed_changed else existing_queue
        liked_ids = set(user_store.liked_track_ids(local_user_id))
        if current_id in liked_ids:
            st.success("♥ Liked")
        elif st.button("♡ Like this song", key=f"player_like_{current_id}", width="stretch"):
            st.session_state.pop("jio_queue_seed_id", None)
            st.session_state.pop("jio_queue_attempted_id", None)
            log_behavior("jio_like", current_id, metadata=jio_now)
            refresh_hybrid_model(force=True)
            existing_queue = list(st.session_state.get("jio_queue", []))
            if existing_queue:
                # Re-rank the already fetched queue immediately. This keeps
                # the full recommendation shelf visible after a live like.
                try:
                    st.session_state["jio_queue"] = hybrid_engine.rank_jio_results(
                        existing_queue,
                        local_user_id,
                        data,
                        preferred_track_ids,
                        seed=jio_now,
                    )[:70]
                except Exception:
                    # A like must never crash the player. Rebuild from the
                    # current seed if a malformed queue item reaches ranking.
                    try:
                        st.session_state["jio_queue"] = build_jio_up_next(
                            jio_now,
                            st.session_state.get("jio_search_cache_v4", {}).get(search_query.strip().lower(), []),
                            jio_client,
                            hybrid_engine,
                            local_user_id,
                            data,
                            preferred_track_ids,
                            exclude_items=[jio_now],
                        )
                    except Exception:
                        st.session_state["jio_queue"] = existing_queue
                st.session_state["jio_queue_seed_id"] = current_id
            else:
                cached_results = st.session_state.get("jio_search_cache_v4", {}).get(search_query.strip().lower(), [])
                try:
                    st.session_state["jio_queue"] = build_jio_up_next(
                        jio_now,
                        cached_results,
                        jio_client,
                        hybrid_engine,
                        local_user_id,
                        data,
                        preferred_track_ids,
                        exclude_items=[jio_now],
                    )
                    st.session_state["jio_queue_seed_id"] = current_id
                except Exception:
                    st.session_state["jio_queue"] = []
            st.toast("Added to your likes and taste profile")
            st.rerun()
    with now_cols[2]:
        if st.button("⏭ Next", key="jio_next", width="stretch"):
            if play_next_from_queue():
                st.rerun()
            else:
                st.info("No personalized recommendation is ready yet. Play another search result first.")
    with now_cols[3]:
        st.toggle("Loop", key="jio_loop")
    raw_up_next = st.session_state.get("jio_queue", [])
    up_next = []
    seen_up_next_recordings: set[str] = set()
    for item in raw_up_next:
        identity_keys = _recommendation_history_keys(item)
        if identity_keys and not identity_keys.intersection(seen_up_next_recordings):
            seen_up_next_recordings.update(identity_keys)
            up_next.append(item)
    st.session_state["jio_queue"] = up_next
    if up_next:
        feedback_left, feedback_right = st.columns([3, 1], vertical_alignment="center")
        with feedback_right:
            st.markdown(
                "<div class='feedback-card'><div class='feedback-title'>Quick vibe check</div>"
                "<div style='color:#c7d4e3;font-size:.86rem;margin:.25rem 0 .55rem'>Is this mix hitting?</div></div>",
                unsafe_allow_html=True,
            )
            yes_col, no_col = st.columns(2)
            feedback_seed = str(jio_now.get("song_id") or jio_now.get("title", ""))
            if yes_col.button("Yes", key=f"feedback_yes_{feedback_seed}", width="stretch"):
                log_behavior("recommendation_feedback", feedback_seed, metadata={"rating": "good", "queue_size": len(up_next)})
                st.toast("Love that — keeping this vibe")
            if no_col.button("No", key=f"feedback_no_{feedback_seed}", width="stretch"):
                log_behavior("recommendation_feedback", feedback_seed, metadata={"rating": "bad", "queue_size": len(up_next)})
                st.toast("Got it — I’ll learn from this")
    if up_next:
        st.markdown(f"<div class='shelf-subtitle'>Up next for you · {len(up_next)} personalized tracks</div>", unsafe_allow_html=True)
        featured = up_next[0]
        banner_left, banner_mid, banner_right = st.columns([1, 3, 1], vertical_alignment="center")
        with banner_left:
            if featured.get("image_url"):
                st.image(featured["image_url"], width="stretch", alt=f"Artwork for {featured.get('title', 'recommended song')}")
        with banner_mid:
            st.markdown("<div class='recommendation-kicker'>01 · next in your mix</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='recommendation-title'>{featured.get('title', 'Unknown song')}</div>", unsafe_allow_html=True)
            featured_details = f"Singer: {featured.get('artists', '')}"
            if featured.get("album"):
                featured_details += f" · {featured.get('album')}"
            if featured.get("year"):
                featured_details += f" · {featured.get('year')}"
            st.markdown(f"<div class='recommendation-meta'>{featured_details}</div>", unsafe_allow_html=True)
            if featured.get("recommendation_reason"):
                st.markdown(f"<div class='recommendation-reason'>✦ {featured['recommendation_reason']}</div>", unsafe_allow_html=True)
        with banner_right:
            if st.button("▶ Play next", key=f"featured_play_{featured.get('song_id') or featured.get('title')}", width="stretch"):
                st.session_state["jio_now_playing"] = featured
                st.session_state["scroll_to_player"] = True
                selected_keys = _recommendation_history_keys(featured)
                st.session_state["jio_queue"] = [
                    queued for queued in up_next
                    if not selected_keys.intersection(_recommendation_history_keys(queued))
                ]
                st.session_state["jio_current_track_id"] = str(featured.get("song_id") or featured.get("title", ""))
                st.session_state["jio_started_at"] = time.time()
                st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                defer_player_event("jio_play", st.session_state["jio_current_track_id"], metadata=featured)
                current_seed_id = str(featured.get("song_id") or featured.get("title", ""))
                st.session_state["jio_queue_attempted_id"] = current_seed_id
                st.session_state["jio_queue_refill_attempted_id"] = ""
                st.rerun()
            featured_feedback_id = str(featured.get("song_id") or featured.get("title", ""))
            featured_yes, featured_no = st.columns(2)
            if featured_yes.button("Yes", key=f"upnext_yes_featured_{featured_feedback_id}", width="stretch"):
                _recommendation_feedback(featured_feedback_id, "up_next", "good")
            if featured_no.button("No", key=f"upnext_no_featured_{featured_feedback_id}", width="stretch"):
                _recommendation_feedback(featured_feedback_id, "up_next", "bad")
        st.caption(" · ".join(str(item.get("title", "Unknown song")) for item in up_next[:5]))
        with st.expander(f"See all {len(up_next)} personalized recommendations", expanded=True):
            recommendation_cols = st.columns(4)
            for index, item in enumerate(up_next[1:], start=1):
                with recommendation_cols[index % 4]:
                    with st.container(border=True):
                        if item.get("image_url"):
                            st.image(item["image_url"], width="stretch", alt=f"Artwork for {item.get('title', 'recommended song')}")
                        st.caption(f"{index + 1}. {item.get('title', 'Unknown song')}")
                        rec_details = f"Singer: {item.get('artists', '')}"
                        if item.get("album"):
                            rec_details += f" · {item.get('album')}"
                        if item.get("year"):
                            rec_details += f" · {item.get('year')}"
                        st.caption(rec_details)
                        if item.get("recommendation_reason"):
                            st.caption(f"💡 {item['recommendation_reason']}")
                        if st.button(
                            "▶ Play",
                            key=f"up_next_play_{index}_{item.get('song_id') or item.get('title')}",
                            width="stretch",
                        ):
                            # Reuse the already-ranked queue so playback never
                            # waits for another provider discovery wave.
                            st.session_state["jio_now_playing"] = item
                            st.session_state["scroll_to_player"] = True
                            old_queue = list(up_next)
                            selected_keys = _recommendation_history_keys(item)
                            st.session_state["jio_queue"] = [
                                queued for queued in old_queue
                                if not selected_keys.intersection(_recommendation_history_keys(queued))
                            ]
                            st.session_state["jio_current_track_id"] = str(
                                item.get("song_id") or item.get("title", "")
                            )
                            st.session_state["jio_started_at"] = time.time()
                            st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                            defer_player_event(
                                "jio_play",
                                str(item.get("song_id") or item.get("title", "")),
                                metadata=item,
                            )
                            current_seed_id = str(item.get("song_id") or item.get("title", ""))
                            st.session_state["jio_queue_attempted_id"] = current_seed_id
                            st.session_state["jio_queue_refill_attempted_id"] = ""
                            st.rerun()
                        rec_feedback_id = str(item.get("song_id") or item.get("title", ""))
                        rec_yes, rec_no = st.columns(2)
                        if rec_yes.button("Yes", key=f"upnext_yes_{index}_{rec_feedback_id}", width="stretch"):
                            _recommendation_feedback(rec_feedback_id, "up_next", "good")
                        if rec_no.button("No", key=f"upnext_no_{index}_{rec_feedback_id}", width="stretch"):
                            _recommendation_feedback(rec_feedback_id, "up_next", "bad")
    else:
        status = st.session_state.get("jio_queue_status", {})
        if status.get("refresh_error"):
            st.error(f"Recommendation refresh failed: {status['refresh_error']}")
        st.info(
            "You’ve reached the fresh picks for this song in this session. "
            "Play a different song to get a new, relevant mix — we won’t pad it with repeats."
        )
        if status:
            with st.expander("Recommendation diagnostics"):
                st.write({
                    "Seed": status.get("seed", ""),
                    "Candidates retrieved": status.get("candidates_retrieved", 0),
                    "Relevant candidates accepted": status.get("relevant_candidates", 0),
                    "Provider failures": status.get("provider_query_failures", 0),
                    "Ranking error": status.get("ranking_error") or "none",
                    "Error detail": status.get("ranking_error_detail") or "none",
                    "Candidate sources": status.get("candidate_sources", []),
                })
    if st.session_state.pop("scroll_to_player", False):
        st.html(
            """
            <script>
            const root = window.parent && window.parent.document ? window.parent.document : document;
            const scrollToPlayer = () => {
              const anchor = root.getElementById('tune-match-player-anchor')
                || document.getElementById('tune-match-player-anchor');
              if (anchor) {
                anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                try { (window.top || window).location.hash = 'tune-match-player-anchor'; } catch (_) {}
              }
            };
            requestAnimationFrame(scrollToPlayer);
            setTimeout(scrollToPlayer, 180);
            setTimeout(scrollToPlayer, 650);
            </script>
            """,
            unsafe_allow_javascript=True,
        )
    if st.button("↻ Refresh recommendations", key="refresh_recommendations"):
        refresh_hybrid_model(force=True)
        st.session_state["recommendation_refresh_round"] = (
            int(st.session_state.get("recommendation_refresh_round", 0) or 0) + 1
        )
        cached_results = st.session_state.get("jio_search_cache_v4", {}).get(search_query.strip().lower(), [])
        existing_queue = list(st.session_state.get("jio_queue", []))
        candidate_pool = [*existing_queue, *cached_results]
        current_seed_id = str(jio_now.get("song_id") or jio_now.get("title", ""))
        try:
            refreshed_queue = build_jio_up_next(
                jio_now,
                candidate_pool,
                jio_client,
                hybrid_engine,
                local_user_id,
                data,
                preferred_track_ids,
                # A manual refresh should move to a different set of songs,
                # not simply rerank the cards already on screen.
                exclude_items=[jio_now, *existing_queue],
            )
            # Never put the old queue back after a refresh: that makes the
            # button look broken and repeatedly surfaces the same first song.
            st.session_state["jio_queue"] = refreshed_queue
            st.session_state.get("jio_queue_status", {}).pop("refresh_error", None)
        except Exception as exc:
            st.session_state["jio_queue"] = []
            st.session_state.setdefault("jio_queue_status", {})["refresh_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        st.session_state["jio_queue_seed_id"] = current_seed_id
        st.session_state["jio_queue_attempted_id"] = current_seed_id
        st.session_state["jio_queue_refill_attempted_id"] = ""
        st.rerun()

if st.session_state.get("local_is_admin", False):
    with st.expander("Your listening analytics", expanded=True):
        activity = cached_user_summary(
            local_user_id,
            session_id=st.session_state["local_session_id"],
        )
        st.caption("Your account only · choose a period to explore listening habits and recommendation feedback.")
        a, b, c, d = st.columns(4)
        a.metric("Events", activity["events"])
        b.metric("Tracks played", activity["tracks_played"])
        c.metric("Likes", activity["likes"])
        live_ms = activity["listening_ms"]
        if st.session_state.get("jio_started_at"):
            live_ms += int(
                max(0.0, time.time() - float(
                    st.session_state.get("jio_last_accounted_at", st.session_state["jio_started_at"])
                )) * 1000
            )
        d.metric("Listening time", f"{live_ms / 60000:.1f} min")
        st.subheader("Listening trends")
    
        analytics_events = pd.DataFrame(cached_user_events(local_user_id))
        if analytics_events.empty:
            st.info("Your charts will appear after you play or interact with a few tracks.")
        else:
            analytics_events["created_at"] = pd.to_datetime(
                analytics_events["created_at"], errors="coerce", utc=True
            )
            analytics_events["event_type"] = analytics_events["event_type"].fillna("").astype(str)
            analytics_events["duration_ms"] = pd.to_numeric(
                analytics_events.get("duration_ms", 0), errors="coerce"
            ).fillna(0)
            analytics_events["duration_min"] = analytics_events["duration_ms"].where(
                analytics_events["event_type"] == "listen_progress", 0
            ) / 60000
    
            period = st.selectbox(
                "Time range",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                index=1,
                key="listener_analytics_period",
            )
            period_days = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
            }.get(period)
            if period_days:
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
                analytics_events = analytics_events[
                    analytics_events["created_at"] >= cutoff
                ].copy()
    
            play_types = {"jio_play", "play_start"}
            like_types = {"jio_like", "like"}
            skip_types = {"jio_skip", "skip"}
            plays = analytics_events[analytics_events["event_type"].isin(play_types)]
            likes = analytics_events[analytics_events["event_type"].isin(like_types)]
            skips = analytics_events[analytics_events["event_type"].isin(skip_types)]
            unique_tracks = int(plays["track_id"].nunique()) if "track_id" in plays else 0
            play_count = len(plays)
            like_count = len(likes)
            skip_count = len(skips)
            listening_minutes = float(analytics_events["duration_min"].sum())
    
            kpis = st.columns(5)
            kpis[0].metric("Play starts", f"{play_count:,}")
            kpis[1].metric("Unique tracks", f"{unique_tracks:,}")
            kpis[2].metric("Likes", f"{like_count:,}")
            kpis[3].metric("Skips", f"{skip_count:,}")
            kpis[4].metric("Listening time", f"{listening_minutes:.1f} min")
    
            def parse_event_metadata(value: object) -> dict:
                if isinstance(value, dict):
                    return value
                try:
                    parsed = json.loads(str(value or "{}"))
                except (TypeError, ValueError):
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    
            analytics_events["metadata"] = analytics_events["metadata_json"].map(
                parse_event_metadata
            )
            analytics_events["date"] = analytics_events["created_at"].dt.strftime("%Y-%m-%d")
            daily = analytics_events.dropna(subset=["created_at"]).groupby("date").agg(
                plays=("event_type", lambda values: values.isin(play_types).sum()),
                likes=("event_type", lambda values: values.isin(like_types).sum()),
                skips=("event_type", lambda values: values.isin(skip_types).sum()),
                minutes=("duration_min", "sum"),
            )
    
            chart_left, chart_right = st.columns(2)
            with chart_left:
                if not daily.empty:
                    st.markdown("**Daily plays, likes and skips**")
                    st.line_chart(
                        daily[["plays", "likes", "skips"]],
                        height=240,
                        alt="Daily play, like and skip counts",
                    )
                else:
                    st.info("No dated activity in this period.")
            with chart_right:
                if not daily.empty:
                    st.markdown("**Listening time by day**")
                    st.area_chart(
                        daily[["minutes"]],
                        height=240,
                        alt="Listening minutes per day",
                    )
                event_mix = analytics_events["event_type"].value_counts()
                if not event_mix.empty:
                    st.markdown("**Activity mix**")
                    st.bar_chart(
                        event_mix.rename("events"),
                        horizontal=True,
                        height=220,
                        alt="Counts of each listener event type",
                    )
    
            def metadata_names(value: object) -> list[str]:
                if isinstance(value, list):
                    values = [
                        str(item.get("name", "") if isinstance(item, dict) else item).strip()
                        for item in value
                    ]
                else:
                    values = str(value or "").split(",")
                return [
                    name for name in values
                    if name and name.casefold() not in {"unknown", "n/a", "na", "-"}
                ]
    
            artist_values: list[str] = []
            track_values: list[str] = []
            mood_values: list[str] = []
            era_values: list[str] = []
            language_values: list[str] = []
            for event in plays.to_dict("records"):
                metadata = event.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                artist_values.extend(metadata_names(
                    metadata.get("artists") or metadata.get("artist") or metadata.get("singers")
                ))
                track_values.append(
                    str(metadata.get("title") or metadata.get("track_name") or event.get("track_id") or "Unknown track")
                )
                mood = metadata.get("mood_label") or metadata.get("mood")
                mood_values.extend(metadata_names(mood))
                year = str(metadata.get("year") or "")[:4]
                if year.isdigit() and 1900 <= int(year) <= 2100:
                    era_values.append(f"{int(year) // 10 * 10}s")
                language = str(metadata.get("language") or "").strip()
                if language:
                    language_values.append(language)
    
            def show_top_counts(title: str, values: list[str], label: str) -> None:
                counts = pd.Series(values, dtype="string").dropna().value_counts().head(10)
                counts = counts[counts.index.astype(str).str.strip() != ""]
                if not counts.empty:
                    st.markdown(f"**{title}**")
                    st.bar_chart(
                        counts.rename(label),
                        horizontal=True,
                        height=260,
                        alt=title,
                    )
    
            insight_left, insight_right = st.columns(2)
            with insight_left:
                show_top_counts("Most played artists", artist_values, "plays")
                show_top_counts("Most played tracks", track_values, "plays")
            with insight_right:
                show_top_counts("Mood mix", mood_values, "plays")
                show_top_counts("Listening by release era", era_values, "plays")
                show_top_counts("Listening by language", language_values, "plays")


with st.expander("How this recommender works"):
    st.markdown("""
    1. **Current-track relevance:** the playing track provides the primary recommendation context.
    2. **Audio and metadata:** verified audio features are distinguished from metadata-based mood estimates.
    3. **Listener personalization:** previous plays, likes, skips and listening preferences provide secondary signals.
    4. **Collaborative filtering:** TruncatedSVD estimates latent relationships from observed user-track interactions.
    5. **Sequential context:** previous listening transitions can contribute when sufficient session data exists.
    6. **Relevance filtering:** incompatible musical intents, duplicate recordings and weak candidates are rejected.
    7. **Diversity:** MMR balances recommendation relevance against repetitive artists and recordings.
    Learned ranking remains inactive until sufficient attributed recommendation feedback is available.
    """)
