from pathlib import Path
import os
import time
import uuid
import re
import json
import hashlib
from difflib import SequenceMatcher
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from jiosaavn_client import JioSaavnClient
from recommender import HYBRID_MODEL_VERSION, HybridRecommendationEngine, SpotifyRecommendationEngine
from spotify_api import SpotifyOAuth, new_oauth_state, verify_oauth_state
from user_store import UserStore

ERA_COMPATIBILITY_WINDOW_YEARS = 30

st.set_page_config(page_title="TuneMatch | Spotify Recommender", page_icon="🎧", layout="wide")

# A small bidirectional Streamlit component is used for the player because
# st.audio does not expose the browser's `ended` event to Python. The event
# lets the server advance the already-ranked queue without requiring a click.
TUNEMATCH_AUDIO = st.components.v2.component(
    "tunematch_audio_player",
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
  label.textContent = data.title || 'TuneMatch player';

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
.stApp { background: radial-gradient(circle at 8% 0%, #203f35 0%, #0d1718 30%, #090b10 67%, #160d20 100%); }
.block-container { max-width: 1440px; padding-top: 1rem; padding-bottom: 4rem; }
header[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
header[data-testid="stHeader"] { height: 2.5rem; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #12152a 0%, #0b1115 52%, #160d20 100%); border-right: 1px solid #3d4269; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(25,51,41,.92), rgba(24,18,43,.92)); border: 1px solid #3c6b61; padding: 14px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
[data-testid="stMetricValue"] { color: #d9ffe9; }
.stButton > button, .stLinkButton > a { border-radius: 999px; border: 1px solid #467b70; background: linear-gradient(100deg, rgba(30,215,96,.13), rgba(139,92,246,.13)); transition: all .15s ease; }
.stButton > button:hover, .stLinkButton > a:hover { border-color: #b7ff4a; color: #f2fff8; box-shadow: 0 0 22px rgba(30,215,96,.22); transform: translateY(-1px); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #38545a; border-radius: 18px; background: linear-gradient(145deg, rgba(17,35,29,.74), rgba(24,18,43,.62)); box-shadow: 0 12px 30px rgba(0,0,0,.12); }
[data-testid="stImage"] img { border-radius: 14px; }
.stTextInput input, .stMultiSelect div[data-baseweb="select"] { border-radius: 12px; }
[data-testid="stTextInputRootElement"] { border: 1px solid #3e6a66; border-radius: 14px; background: rgba(11,20,24,.75); }
[data-testid="stTextInputRootElement"]:focus-within { border-color: #ff4fd8; box-shadow: 0 0 0 3px rgba(255,79,216,.14); }
.shelf-title { color: #f4f7f5; font-size: 1.35rem; font-weight: 800; margin: 1.35rem 0 .4rem; letter-spacing: -.02em; }
.shelf-subtitle { color: #a8c7d0; margin-bottom: .8rem; }
.hero-panel { padding: 1.55rem 1.7rem; border-radius: 26px; border: 1px solid #315b58; background: linear-gradient(120deg, rgba(25,65,51,.76), rgba(42,21,65,.62) 72%, rgba(16,32,45,.7)); box-shadow: 0 18px 50px rgba(0,0,0,.2); }
.hero-title { font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: .95; font-weight: 900; letter-spacing: -.07em; background: linear-gradient(100deg, #ffffff 0%, #baffd2 42%, #ff8cec 86%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-copy { color: #b6d3d1; font-size: 1.05rem; max-width: 760px; margin-top: .65rem; }
.hero-chip { display: inline-block; margin: .95rem .45rem 0 0; padding: .32rem .72rem; border-radius: 999px; color: #eafff3; background: rgba(30,215,96,.12); border: 1px solid rgba(183,255,74,.32); font-size: .82rem; }
.hero-chip:nth-child(2) { background: rgba(255,79,216,.12); border-color: rgba(255,79,216,.35); }
.hero-chip:nth-child(3) { background: rgba(84,229,255,.1); border-color: rgba(84,229,255,.32); }
.recommendation-banner { padding: 1rem 1.15rem; border-radius: 20px; border: 1px solid #5b4779; background: linear-gradient(110deg, rgba(44,29,68,.9), rgba(16,42,39,.82)); margin: .5rem 0 1rem; }
.recommendation-kicker { color: #ff8cec; font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.recommendation-title { color: #f4f7f5; font-size: 1.35rem; font-weight: 850; margin-top: .2rem; }
.recommendation-meta { color: #b6d3d1; font-size: .9rem; }
.recommendation-reason { color: #d8c8ff; font-size: .84rem; }
@media (max-width: 760px) { .block-container { padding: 1rem .7rem 3rem; } }
</style>
""")

st.markdown('''<div class="hero-panel"><div class="hero-title">🎧 TuneMatch</div><div class="hero-copy">Your live, mood-aware music universe—personalized by every play, like and skip.</div><span class="hero-chip">✦ Fresh for you</span><span class="hero-chip">◉ Mood-aware</span><span class="hero-chip">⚡ Live discovery</span></div>''', unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "spotify_tracks_data.csv"
EVENT_DB_PATH = Path(__file__).parent / "tune_match_events.sqlite3"
HISTORY_PATH = Path(__file__).parent / "spotify_history.csv"


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
    "instrumentalness", "speechiness", "liveness", "loudness", "popularity",
)

INTENT_TERMS = {
    "romantic": {"love", "romantic", "ishq", "mohabbat", "pyaar", "pyar", "dil", "jaan", "saiyaara"},
    "ghazal": {"ghazal", "ghazals", "shayar", "shayari", "mehfil", "rekhta", "ghulam ali", "ghulam ali classics", "ustad bade ghulam ali", "ghulam ali khan", "chupke chupke classics"},
    "motivational": {"motivation", "motivational", "zinda", "jeet", "victory", "fighter", "warrior", "rise", "goal", "inspire"},
    "energetic": {"fight", "anthem", "rock", "power", "action", "dance", "party", "bhangra", "bang", "fire", "dhamaka", "dhamaal", "high", "energy"},
    "sad": {"sad", "heartbreak", "alone", "broken", "dard", "udaas", "tanha", "judai"},
    "devotional": {"bhajan", "chalisa", "gurbani", "hanuman", "aarti", "kirtan", "mantra", "om namah shivay", "krishna", "allah", "khuda"},
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
    text = " ".join(str(item.get(field, "")) for field in ("title", "album", "artists", "language")).lower()
    vector = {
        "energy": 0.50, "tempo": 100.0, "valence": 0.50, "danceability": 0.50,
        "acousticness": 0.40, "instrumentalness": 0.15, "speechiness": 0.10,
        "liveness": 0.20, "loudness": -12.0, "popularity": 35.0,
    }
    labels = []
    groups = {
        "energetic": {"fight", "anthem", "rock", "zinda", "sultan", "power", "action", "dance", "party", "bhangra", "bang", "fire", "dhamaka", "warrior", "bhaag", "milkha", "singham", "chaiyya", "ramta", "appadipodu", "mappillai"},
        "romantic": {"love", "romantic", "ishq", "mohabbat", "pyaar", "dil", "jaan", "saiyaara"},
        "sad": {"sad", "heartbreak", "alone", "broken", "dard", "udaas"},
        "instrumental": {"theme", "score", "background", "instrumental", "bgm", "soundtrack"},
        "acoustic_indie": {"acoustic", "unplugged", "indie", "folk", "singer songwriter", "singer-songwriter", "coke studio", "lofi", "slowed", "reverb"},
        "ghazal": {"ghazal", "ghazals", "shayar", "shayari", "mehfil", "rekhta", "ghulam ali", "ghulam ali classics", "ustad bade ghulam ali", "ghulam ali khan", "chupke chupke classics"},
    }
    tokens = set(re.findall(r"[a-z0-9]+", text))
    matched = {
        label for label, words in groups.items()
        if any((word in text if " " in word else word in tokens) for word in words)
    }
    if "energetic" in matched:
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
    labels = sorted(matched) or ["neutral"]
    return {
        "mood_vector": _mood_vector(vector),
        "mood_label": ", ".join(labels),
        "mood_confidence": 0.72 if matched else 0.22,
        "mood_source": "metadata_fallback",
    }


def infer_music_intents(item: dict) -> list[str]:
    """Infer a small intent set from live metadata in constant time."""
    text = " ".join(
        str(item.get(field, "")) for field in ("title", "album", "artists", "language")
    ).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    labels = [
        label for label, terms in INTENT_TERMS.items()
        if any((term in text if " " in term else term in tokens) for term in terms)
    ]
    # Numeric audio features provide a language/artist-independent drive
    # signal. This catches motivational tracks whose title contains no mood
    # keyword (for example a film title track).
    try:
        energy = float(item.get("energy", 0.0) or 0.0)
        valence = float(item.get("valence", 0.0) or 0.0)
        danceability = float(item.get("danceability", 0.0) or 0.0)
        tempo = min(1.0, float(item.get("tempo", 0.0) or 0.0) / 140.0)
        drive_score = 0.35 * energy + 0.30 * valence + 0.20 * danceability + 0.15 * tempo
        if drive_score >= 0.62:
            labels.append("motivational")
        if energy >= 0.62 and tempo >= 0.55:
            labels.append("energetic")
    except (TypeError, ValueError):
        pass
    return sorted(set(labels)) or ["general"]


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
    """Attach local audio features to live results when title/artist matches."""
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
        if match:
            item.update({key: value for key, value in match.items() if key != "catalog_artists"})
            if not str(item.get("artists", "")).strip():
                item["artists"] = match.get("catalog_artists", "")
            if not str(item.get("language", "")).strip() and match.get("catalog_language_hint"):
                item["language"] = match["catalog_language_hint"]
        else:
            item.setdefault("feature_source", "live_metadata_fallback")
            item.update({key: value for key, value in infer_metadata_mood(item).items() if key not in item})
        item["intent_labels"] = infer_music_intents(item)
    return items


@st.cache_resource
def build_hybrid_engine(_content_engine):
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
            "Cloud database is unavailable right now. TuneMatch is using temporary local storage; "
            "fix DATABASE_URL and reboot to persist users and activity."
        )
jio_client = JioSaavnClient()
hybrid_engine = build_hybrid_engine(engine)
st.session_state.setdefault("local_session_id", uuid.uuid4().hex)
if "recommendation_variant" not in st.session_state:
    variant_key = str(st.session_state.get("local_user_id") or st.session_state["local_session_id"])
    bucket = int(hashlib.sha256(variant_key.encode("utf-8")).hexdigest()[:8], 16) % 2
    st.session_state["recommendation_variant"] = "diverse" if bucket else "control"
st.session_state.setdefault("jio_search_cache_v2", {})
st.session_state.setdefault("jio_loop", False)
st.session_state.setdefault("scroll_to_player", False)
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
    query_key = _compact_key(query)
    if not query_key:
        return results
    query_tokens = set(query_key.split())
    language_terms = {"hindi", "tamil", "telugu", "punjabi", "bengali", "marathi", "malayalam", "kannada"}
    query_languages = query_tokens.intersection(language_terms)

    def score(item: dict) -> tuple[float, int]:
        title = _compact_key(item.get("title", ""))
        album = _compact_key(item.get("album", ""))
        artists = _compact_key(item.get("artists", ""))
        title_tokens = set(title.split())
        exact_title = float(title == query_key)
        title_match = float(query_key in title or query_tokens.issubset(title_tokens))
        album_exact = float(album == query_key)
        album_match = float(query_key in album or query_tokens.issubset(set(album.split())))
        entity_title_match = float(
            query_key in title and any(marker in title for marker in ("from", "title track", "soundtrack"))
        )
        artist_match = float(query_key in artists)
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
            - variant_penalty,
            0,
        )
    ordered = [item for _, item in sorted(enumerate(results), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)]
    unique: list[dict] = []
    seen: set[str] = set()
    for item in ordered:
        key = "|".join(
            _compact_key(item.get(field, ""))
            for field in ("title", "artists", "album")
        )
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def build_jio_up_next(
    seed: dict,
    search_results: list[dict],
    client: JioSaavnClient,
    hybrid: HybridRecommendationEngine,
    user_id: str,
    catalog: pd.DataFrame,
    preferred_track_ids: list[str],
    limit: int = 70,
    exclude_items: list[dict] | None = None,
) -> list[dict]:
    """Build a separate recommendation queue; never reuse search results."""
    # Ensure the playing seed has the same mood representation as candidates,
    # including when it came from the player refresh path rather than search.
    seed = enrich_live_items([seed])[0]
    title = str(seed.get("title", "")).strip()
    artists = str(seed.get("artists", "")).strip()
    album = str(seed.get("album", "")).strip()

    # Classify the seed independently. Search neighbours are candidate data,
    # not evidence about the current track's intent; using them here caused a
    # broad artist result (for example a motivational song) to redefine a
    # romantic/acoustic seed.
    verified_intents = set(seed.get("intent_labels", ["general"]))

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
    artist_names = existing_artists
    related_queries = []
    for artist in artist_names[:5]:
        related_queries.extend([
            f"{artist} songs",
            f"{artist} best songs",
            f"{artist} similar songs",
        ])
        related_queries.extend(f"{artist} {intent}" for intent in sorted(verified_intents) if intent != "general")
    if album:
        related_queries.extend([f"{album} songs", f"{album} soundtrack"])
    if title:
        related_queries.extend([
            f"{title} similar songs",
            f"songs like {title}",
            f"{title} soundtrack",
            f"{title} acoustic songs" if "acoustic_indie" in verified_intents else f"{title} songs",
        ])
        related_queries.extend(f"{title} {intent}" for intent in sorted(verified_intents) if intent != "general")
    try:
        seed_year = int(str(seed.get("year", ""))[:4])
    except ValueError:
        seed_year = 0
    if seed_year:
        decade = (seed_year // 10) * 10
        era_intents = [intent for intent in sorted(verified_intents) if intent != "general"] or ["songs"]
        related_queries.extend([
            f"bollywood {intent} songs {decade}s" for intent in era_intents
        ])
        related_queries.extend([
            f"hindi songs from {decade}s",
            f"bollywood songs from {seed_year}",
            f"songs like {title} {decade}s" if title else "",
        ])
    # Do not fill a seed-specific radio queue with global popular/romantic
    # pages. Those pages create the exact failure mode where an acoustic indie
    # song receives unrelated Bollywood recommendations. Every fallback must
    # retain at least the seed artist, intent or title context.
    fallback_queries = []
    if title:
        fallback_queries.extend([f"songs like {title}", f"{title} similar songs"])
    if verified_intents:
        fallback_queries.extend(f"{intent} hindi songs" for intent in sorted(verified_intents) if intent != "general")
    if not fallback_queries and artist_names:
        fallback_queries.extend(f"{artist} songs" for artist in artist_names[:2])
    collected: list[dict] = []
    seen: set[str] = set()
    normalize_title = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    search_ids = {
        str(item.get("song_id") or item.get("title", "")) for item in search_results
    }
    search_titles = {normalize_title(item.get("title", "")) for item in search_results}
    for item in exclude_items or []:
        search_ids.add(str(item.get("song_id") or item.get("title", "")))
        search_titles.add(normalize_title(item.get("title", "")))
    def collect_items(items: list[dict], source: str = "metadata_discovery") -> None:
        for item in items:
            key = str(item.get("song_id") or item.get("title", ""))
            title_key = normalize_title(item.get("title", ""))
            if not key or not title_key or key in seen or key in search_ids or title_key in search_titles:
                continue
            seen.add(key)
            seen.add(title_key)
            item.setdefault("candidate_source", source)
            collected.append(item)

    def collect_queries(queries: list[str], page_limit: int = 12, max_queries: int = 16, source: str = "metadata_discovery") -> None:
        # One bounded concurrent wave is faster than sequential batches. The
        # upstream client is cached, so repeated songs do not cause repeated
        # network work across Streamlit reruns.
        queries = list(dict.fromkeys(query for query in queries if query))[:max_queries]
        if not queries:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as executor:
            pages = executor.map(lambda value: client.search_direct(value, limit=page_limit), queries)
            for page in pages:
                collect_items(page, source)

    collect_queries(related_queries, page_limit=12, max_queries=16, source="seed_artist_and_intent")
    # A full expanded artist page is used only when the related pages have not
    # produced enough candidates for the large Up Next shelf.
    if len(collected) < 60:
        collect_queries(
            [f"{artist} songs" for artist in artist_names[:3]],
            page_limit=20,
            max_queries=3,
            source="artist_discovery",
        )
    if len(collected) < 60:
        collect_queries(fallback_queries, page_limit=15, max_queries=6, source="seed_metadata_similarity")
    # The public upstream service can return only a few rows per query. Use
    # broader but still seed-related pages as a final capacity pass so a
    # selected recommendation gets its own large queue instead of inheriting
    # the old queue's remainder.
    if len(collected) < 45:
        broad_queries = []
        for artist in artist_names[:2]:
            broad_queries.extend([
                f"{artist} film songs",
                f"{artist} playlist",
                f"{artist} evergreen songs",
                f"{artist} top 50 songs",
            ])
        if seed_year:
            decade = (seed_year // 10) * 10
            broad_queries.append(f"hindi songs {decade}s")
            broad_queries.extend(
                f"{intent} songs {decade}s"
                for intent in sorted(verified_intents) if intent != "general"
            )
        collect_queries(broad_queries, page_limit=20, max_queries=10, source="era_intent_discovery")
    enriched_collected = enrich_live_items(collected)
    try:
        ranked = hybrid.rank_jio_results(
            enriched_collected,
            user_id,
            catalog,
            preferred_track_ids,
            seed=seed,
        )
        # Never relax the musical relevance gate merely to fill the queue.
        # A shorter relevant radio is better than unrelated recommendations.
    except Exception:
        ranked = []

    if not ranked:
        # Never bypass the ranker with an era/artist-only fallback. That path
        # produced queues with zero diagnostics and unrelated recommendations.
        # The caller can show a shorter/empty radio instead.
        return []
    unique: list[dict] = []
    used_titles: set[str] = set()
    for item in ranked:
        title_key = normalize_title(item.get("title", ""))
        if title_key and title_key not in used_titles:
            used_titles.add(title_key)
            unique.append(item)
    queue = unique[:limit]
    for position, item in enumerate(queue, start=1):
        item["recommendation_rank"] = position
        item.setdefault("recommendation_source", item.get("candidate_source", "current_track_radio"))

    # One compact impression event per generated queue gives us exposure data
    # without inserting one database row per visible recommendation.
    if queue and st.session_state.get("local_user_id"):
        impression_key = "|".join([
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
                    "items": [
                        {
                            "track_id": str(item.get("song_id") or item.get("title", "")),
                            "title": item.get("title", ""),
                            "rank": item.get("recommendation_rank"),
                            "score": item.get("recommendation_score", 0.0),
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
    st.header("Your TuneMatch account")
    local_user_id = st.session_state.get("local_user_id")
    if local_user_id:
        summary = cached_user_summary(local_user_id)
        st.success("Signed in")
        if st.session_state.get("auth_storage_warning"):
            st.warning(st.session_state.pop("auth_storage_warning"))
        st.caption(f"{summary['tracks_played']} tracks played · {summary['likes']} likes")
        if st.session_state.get("local_session_started_at"):
            minutes = (time.time() - st.session_state["local_session_started_at"]) / 60
            st.caption(f"Current session: {minutes:.1f} min")
        if st.button("Log out of TuneMatch", width="stretch"):
            account_listening_time()
            started = st.session_state.pop("local_session_started_at", time.time())
            log_behavior("session_end", duration_ms=int((time.time() - started) * 1000))
            st.session_state.pop("local_user_id", None)
            st.session_state.pop("local_is_admin", None)
            st.session_state.pop("user_summary_cache", None)
            st.rerun()
    else:
        st.subheader("Welcome to TuneMatch")
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
                log_behavior("session_start")
                st.rerun()
            else:
                st.error(message)
    if not st.session_state.get("local_user_id"):
        st.info("Create a TuneMatch account to save likes, plays, skips, and session analytics.")
        st.stop()

account_listening_time()


@st.fragment(run_every="10s")
def render_admin_dashboard() -> None:
    """Admin-only observability view; normal listeners never see this data."""
    st.header("TuneMatch admin analytics")
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
        # Explainability audit: show the actual feature values that allowed
        # the latest recommendations through the gate.
        diagnostic_rows = []
        for row in reversed(metadata_rows):
            if not isinstance(row, dict) or not row.get("items"):
                continue
            for item in row.get("items", [])[:10]:
                if not isinstance(item, dict):
                    continue
                features = item.get("features") or {}
                diagnostic_rows.append({
                    "track": item.get("title", item.get("track_id", "")),
                    "rank": item.get("rank"),
                    "score": item.get("score"),
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
            break
        if diagnostic_rows:
            st.subheader("Latest recommendation diagnostics")
            st.caption("Feature evidence behind the latest queue; metadata-only mood is not audio proof.")
            st.dataframe(pd.DataFrame(diagnostic_rows), width="stretch", hide_index=True)
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
        workspace_view = st.radio("Workspace", ["TuneMatch", "Admin dashboard"], key="workspace_view")
    if workspace_view == "Admin dashboard":
        render_admin_dashboard()
        st.stop()

st.markdown(f"## Good evening, listener")
st.caption("Your recommendations are ready. Search only when you want something specific.")
search_query = st.text_input(
    "Search songs, artists, albums, or genres",
    placeholder="What do you want to listen to?",
    key="global_search",
)

genre_options = sorted(data["track_genre"].dropna().astype(str).unique().tolist())
popular = data.sort_values("popularity", ascending=False).head(100)
track_labels = {
    f"{row.track_name} — {row.artists}": str(row.track_id)
    for row in popular.itertuples()
}
with st.expander("Tune your taste (optional)"):
    onboarding_genres = st.multiselect(
        "Genres you like", genre_options, default=genre_options[:2], key="onboarding_genres"
    )
    onboarding_tracks = st.multiselect(
        "Artists or tracks you already enjoy", list(track_labels), key="onboarding_tracks"
    )

live_track_ids = local_ids_for_spotify_taste(spotify_taste or {}, data)
imported_track_ids = local_ids_for_imported_history(imported_history, data)
preferred_track_ids = (
    user_store.liked_track_ids(local_user_id)
    + live_track_ids
    + imported_track_ids
    + [track_labels[label] for label in onboarding_tracks]
)


def play_next_from_queue(mark_previous_as_skip: bool = True) -> bool:
    """Promote the next ranked item and rebuild its queue context."""
    queue = list(st.session_state.get("jio_queue", []))
    if not queue:
        return False
    current_id = st.session_state.get("jio_current_track_id")
    if current_id and mark_previous_as_skip:
        log_behavior("jio_skip", str(current_id))
    next_item = queue.pop(0)
    st.session_state["jio_queue"] = queue
    st.session_state["jio_now_playing"] = next_item
    st.session_state["jio_current_track_id"] = str(
        next_item.get("song_id") or next_item.get("title", "")
    )
    st.session_state["jio_started_at"] = time.time()
    st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
    log_behavior(
        "jio_play",
        str(next_item.get("song_id") or next_item.get("title", "")),
        metadata=next_item,
    )
    refresh_hybrid_model(force=True)
    cached_results = st.session_state.get("jio_search_cache_v2", {}).get(search_query.strip().lower(), [])
    try:
        st.session_state["jio_queue"] = build_jio_up_next(
            next_item,
            cached_results,
            jio_client,
            hybrid_engine,
            local_user_id,
            data,
            preferred_track_ids,
            exclude_items=[next_item, *queue],
        )
    except Exception:
        st.session_state["jio_queue"] = queue
    return True

if not search_query:
    st.info("Search a song, artist, album, or genre to see matching music and play it.")
else:
    st.markdown('<div class="shelf-title">Search results</div>', unsafe_allow_html=True)
    st.markdown('<div class="shelf-subtitle">Songs · ranked for your taste</div>', unsafe_allow_html=True)
    try:
        search_cache = st.session_state["jio_search_cache_v2"]
        search_cache_key = search_query.strip().lower()
        jio_results = search_cache.get(search_cache_key)
        # Older sessions may contain a sparse exact-title response from the
        # upstream mirror. Re-run the expanded search once instead of showing
        # a permanently truncated shelf from Streamlit session state.
        if jio_results is None or len(jio_results) < 8:
            jio_results = jio_client.search(search_query, limit=30)
        if not jio_results:
            local_guess = engine.search(search_query, limit=1)
            if not local_guess.empty:
                guess = local_guess.iloc[0]
                fallback_query = f"{guess['track_name']} {guess.get('album_name', '')}".strip()
                jio_results = jio_client.search(fallback_query, limit=30)
        raw_jio_results = list(jio_results or [])
        try:
            jio_results = hybrid_engine.rank_jio_results(
                enrich_live_items(raw_jio_results), local_user_id, data, preferred_track_ids
            )
            jio_results = prioritize_search_results(jio_results, search_query)
        except Exception:
            # Live catalog search must remain usable even if one malformed
            # metadata row breaks a personalization feature. Show the raw
            # upstream results and let the next rerun retry ranking.
            jio_results = raw_jio_results
            st.info("Live results loaded. Personalization is refreshing for this search.")
        if jio_results:
            search_cache[search_cache_key] = jio_results
        if jio_results:
            for row_start in range(0, len(jio_results), 4):
                jio_cols = st.columns(4)
                for column_index, item in enumerate(jio_results[row_start:row_start + 4]):
                    i = row_start + column_index + 1
                    with jio_cols[column_index], st.container(border=True):
                        if item["image_url"]:
                            st.image(item["image_url"], width="stretch")
                        st.caption(f"Search result {i}")
                        st.markdown(f"**{item['title']}**")
                        st.caption(f"{item['artists']}\n{item['album']}")
                        if st.button("▶ Play", key=f"jio_play_{i}_{item['song_id'] or item['title']}", width="stretch"):
                            # Set the selected song first; recommendation
                            # fetching must never make the player disappear.
                            st.session_state["jio_now_playing"] = item
                            st.session_state["scroll_to_player"] = True
                            st.session_state["jio_queue"] = []
                            st.session_state["jio_current_track_id"] = str(
                                item.get("song_id") or item.get("title", "")
                            )
                            st.session_state["jio_started_at"] = time.time()
                            st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                            log_behavior(
                                "jio_play",
                                str(item.get("song_id") or item.get("title", "")),
                                metadata=item,
                            )
                            try:
                                # Refit immediately so this play is included
                                # in the live collaborative/MF ranking.
                                refresh_hybrid_model(force=True)
                                st.session_state["jio_queue"] = build_jio_up_next(
                                    item,
                                    jio_results,
                                    jio_client,
                                    hybrid_engine,
                                    local_user_id,
                                    data,
                                    preferred_track_ids,
                                )
                            except Exception:
                                st.session_state["jio_queue"] = []
                            st.rerun()
                        if st.button("♡ Like", key=f"jio_like_{i}_{item['song_id'] or item['title']}", width="stretch"):
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
                        if item["perma_url"]:
                            st.link_button("Open song", item["perma_url"], width="stretch")
        else:
            st.caption("No songs found for this search.")
    except Exception:
        st.caption("Music search is temporarily unavailable. Try again in a moment.")

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
    # Rebuild the queue automatically whenever the actual player seed changes.
    # Manual refresh remains optional; autoplay, Next and Like all converge on
    # the same current-song context.
    current_queue_seed = str(jio_now.get("song_id") or jio_now.get("title", ""))
    queue_seed = st.session_state.get("jio_queue_seed_id")
    if queue_seed != current_queue_seed:
        current_queue = list(st.session_state.get("jio_queue", []))
        if current_queue:
            # A play/Next handler already built this queue in this rerun.
            st.session_state["jio_queue_seed_id"] = current_queue_seed
        elif st.session_state.get("jio_queue_attempted_id") != current_queue_seed:
            st.session_state["jio_queue_attempted_id"] = current_queue_seed
            try:
                cached_results = st.session_state.get("jio_search_cache_v2", {}).get(
                    search_query.strip().lower(), []
                )
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
                st.session_state["jio_queue_seed_id"] = current_queue_seed
            except Exception:
                st.session_state["jio_queue"] = []
    st.html("<div id='tune-match-player-anchor'></div>")
    st.markdown('<div class="shelf-title">TuneMatch player</div>', unsafe_allow_html=True)
    now_cols = st.columns([1, 3, 1, 1])
    with now_cols[0]:
        if jio_now.get("image_url"):
            st.image(jio_now["image_url"], width="stretch", alt=jio_now.get("title", "TuneMatch song"))
    with now_cols[1]:
        st.markdown(f"**{jio_now.get('title', 'Unknown song')}**")
        st.caption(f"{jio_now.get('artists', '')} · {jio_now.get('album', '')}")
        current_id = str(jio_now.get("song_id") or jio_now.get("title", ""))
        # Component-v2 IDs cannot contain its internal `__` delimiter. Live
        # providers occasionally return titles/IDs containing underscores.
        audio_component_key = re.sub(r"[^A-Za-z0-9]+", "-", current_id).strip("-") or "track"
        stream_url = jio_now.get("stream_url", "")
        perma_url = jio_now.get("perma_url", "")
        if stream_url:
            audio_result = TUNEMATCH_AUDIO(
                data={
                    "url": stream_url,
                    "title": jio_now.get("title", "TuneMatch player"),
                    "track_id": current_id,
                    "autoplay": True,
                    "loop": bool(st.session_state.get("jio_loop", False)),
                },
                key=f"tm-audio-{audio_component_key}",
                on_ended_change=lambda: None,
            )
            if getattr(audio_result, "ended", None) and not st.session_state.get("jio_loop", False):
                if play_next_from_queue(mark_previous_as_skip=False):
                    st.rerun()
        elif perma_url:
            st.warning("Audio preview unavailable for this song.")
            st.link_button("Open song", perma_url, width="stretch")
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
                            st.session_state.get("jio_search_cache_v2", {}).get(search_query.strip().lower(), []),
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
                cached_results = st.session_state.get("jio_search_cache_v2", {}).get(search_query.strip().lower(), [])
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
    seen_up_next_titles: set[str] = set()
    for item in raw_up_next:
        title_key = re.sub(r"[^a-z0-9]+", " ", str(item.get("title", "")).lower()).strip()
        if title_key and title_key not in seen_up_next_titles:
            seen_up_next_titles.add(title_key)
            up_next.append(item)
    st.session_state["jio_queue"] = up_next
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
            st.markdown(f"<div class='recommendation-meta'>{featured.get('artists', '')} · {featured.get('album', '')}</div>", unsafe_allow_html=True)
            if featured.get("recommendation_reason"):
                st.markdown(f"<div class='recommendation-reason'>✦ {featured['recommendation_reason']}</div>", unsafe_allow_html=True)
        with banner_right:
            if st.button("▶ Play next", key=f"featured_play_{featured.get('song_id') or featured.get('title')}", width="stretch"):
                st.session_state["jio_now_playing"] = featured
                st.session_state["scroll_to_player"] = True
                old_queue = list(up_next)
                st.session_state["jio_queue"] = []
                st.session_state["jio_current_track_id"] = str(featured.get("song_id") or featured.get("title", ""))
                st.session_state["jio_started_at"] = time.time()
                st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                log_behavior("jio_play", st.session_state["jio_current_track_id"], metadata=featured)
                refresh_hybrid_model(force=True)
                try:
                    st.session_state["jio_queue"] = build_jio_up_next(featured, jio_results if search_query else [], jio_client, hybrid_engine, local_user_id, data, preferred_track_ids, exclude_items=[featured, *old_queue])
                except Exception:
                    st.session_state["jio_queue"] = []
                st.rerun()
        st.caption(" · ".join(str(item.get("title", "Unknown song")) for item in up_next[:5]))
        with st.expander(f"See all {len(up_next)} personalized recommendations", expanded=True):
            recommendation_cols = st.columns(4)
            for index, item in enumerate(up_next[1:], start=1):
                with recommendation_cols[index % 4]:
                    with st.container(border=True):
                        if item.get("image_url"):
                            st.image(item["image_url"], width="stretch", alt=f"Artwork for {item.get('title', 'recommended song')}")
                        st.caption(f"{index + 1}. {item.get('title', 'Unknown song')}")
                        st.caption(f"{item.get('artists', '')} · {item.get('album', '')}")
                        if item.get("recommendation_reason"):
                            st.caption(f"💡 {item['recommendation_reason']}")
                        if st.button(
                            "▶ Play",
                            key=f"up_next_play_{index}_{item.get('song_id') or item.get('title')}",
                            width="stretch",
                        ):
                            # Selecting an Up Next item starts a new listening
                            # context. Rebuild the queue from that song instead
                            # of merely playing the old queue's remainder.
                            st.session_state["jio_now_playing"] = item
                            st.session_state["scroll_to_player"] = True
                            old_queue = list(up_next)
                            st.session_state["jio_queue"] = []
                            st.session_state["jio_current_track_id"] = str(
                                item.get("song_id") or item.get("title", "")
                            )
                            st.session_state["jio_started_at"] = time.time()
                            st.session_state["jio_last_accounted_at"] = st.session_state["jio_started_at"]
                            log_behavior(
                                "jio_play",
                                str(item.get("song_id") or item.get("title", "")),
                                metadata=item,
                            )
                            refresh_hybrid_model(force=True)
                            cached_results = st.session_state.get("jio_search_cache_v2", {}).get(
                                search_query, []
                            )
                            try:
                                st.session_state["jio_queue"] = build_jio_up_next(
                                    item,
                                    cached_results,
                                    jio_client,
                                    hybrid_engine,
                                    local_user_id,
                                    data,
                                    preferred_track_ids,
                                    exclude_items=[item, *old_queue],
                                )
                            except Exception:
                                st.session_state["jio_queue"] = []
                            st.rerun()
    else:
        st.caption("No more personalized recommendations in this queue.")
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
        cached_results = st.session_state.get("jio_search_cache_v2", {}).get(search_query.strip().lower(), [])
        current_seed_id = str(jio_now.get("song_id") or jio_now.get("title", ""))
        # Do not exclude the previous queue here. Those tracks are precisely
        # the only candidates available in the cached seed-related result
        # pool, and excluding them all can produce an empty radio. The ranker
        # still excludes the current track, recent history and duplicates.
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
        except Exception:
            st.session_state["jio_queue"] = []
        st.session_state["jio_queue_seed_id"] = current_seed_id
        st.session_state["jio_queue_attempted_id"] = current_seed_id
        st.rerun()

with st.expander("Current session activity and data-science metrics", expanded=True):
    activity = cached_user_summary(
        local_user_id,
        session_id=st.session_state["local_session_id"],
    )
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
    st.caption("Play starts, skips, likes, dislikes, session duration, and recommendation seeds are collected for future ranking evaluation.")

with st.expander("How this recommender works"):
    st.markdown("""
    1. **Current-song context first:** the playing song's artist, album, language, era, mood/style and local audio cluster drive the next-track candidates.
    2. **User taste injection:** likes and plays influence the queue, but only when the song also fits the current session context.
    3. **Collaborative filtering:** users with similar play/like behavior contribute implicit preference signals.
    4. **Matrix factorization:** TruncatedSVD learns latent user-song embeddings from the interaction matrix.
    5. **Audio-first hybrid ranking:** audio similarity (35%), metadata/mood (15%), collaborative/MF (15%), recent and long-term taste (20%), session transitions (10%), context (3%) and popularity (2%) are blended. Metadata-only mood is never described as verified audio similarity.
    6. **Freshness and diversity:** recently played/duplicate songs are removed, and artist/album spacing prevents repetitive queues.
    7. **Cold-start aware:** before enough events exist, context and onboarding signals carry the ranking.

    Every play, like, and skip improves the local implicit-feedback model for the next search.
    """)
