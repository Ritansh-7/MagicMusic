# Cozybeats

Cozybeats is a Spotify-inspired, live music recommendation platform built with Streamlit. It combines live JioSaavn discovery with a local audio-feature catalog, implicit user feedback, mood clustering and session-aware ranking.

It is not Spotify's proprietary system. It is an explainable, deployable recommendation-engine project demonstrating search, playback, personalization, feedback loops, diversity, cold start and analytics.

## What the project does

- Creates Cozybeats accounts with salted PBKDF2 password hashes.
- Searches live songs through an unofficial JioSaavn API without requiring a JioSaavn login.
- Shows live artwork, artist/album metadata and available audio previews.
- Provides search, an in-page audio player, Like, Next, Loop and Refresh controls; playback advances automatically when a track ends.
- Builds a separate personalized Up Next queue of up to 30 unique songs.
- Learns from plays, likes, skips and listening progress.
- Reduces repeated titles and consecutive artist/album repetition.
- Shows explainable recommendation reasons.
- Can seed personalization from an imported Spotify listening-history export, matched to local catalog tracks.
- Provides separate Hindi and English/global discovery shelves.
- Gives each listener private listening trends, artist/track rankings, mood and era charts.
- Provides an admin-only dashboard that auto-refreshes, audits recommendation features and shows diagnostics for the latest played song's queue.

## Recommendation architecture

```text
Live JioSaavn search/discovery + 4,002-track local catalog
                ↓
Candidate cleaning and parallel multi-channel retrieval
                ↓
Audio content, mood/intent, long/short preference, collaborative and sequence signals
                ↓
Seed-context relevance gates + hybrid ranker
                ↓
Optional learned ranking correction when enough feedback exists
                ↓
MMR-style artist/album/recent-item diversity reranking
                ↓
Personalized queue (up to 30 tracks)
```

### Content and live-feature enrichment

The local catalog contains 4,002 tracks with artist, genre, popularity and audio features. The local content engine builds a single normalized feature vector from genre TF-IDF, lower-weight artist TF-IDF and min-max-scaled audio features, then compares tracks with cosine similarity.

For the local track-to-track baseline, the score is:

```text
local_score = (1 - p) × cosine_similarity + p × normalized_popularity
```

The default popularity weight is `p = 0.15`; recommendation calls can override it. The listener dashboard also builds profile-based shelves from this local catalog.

Live JioSaavn results are normalized using title, artist, album, language, year, duration, song ID and version type. When a live result matches the local catalog by normalized title and artist, it inherits:

```text
energy, tempo, valence, danceability, acousticness,
instrumentalness, speechiness, liveness and loudness
```

When no local match exists, the system does not invent audio features. It uses metadata fallback signals: artist, album, language, year, title and session context.

For fast live ranking, unmatched JioSaavn songs also receive a low-confidence metadata mood vector inferred from title, album and artist terms (for example energetic, romantic, sad, instrumental or acoustic). This avoids downloading and analysing dozens of previews during every queue build. A metadata estimate is never reported as verified audio evidence.

### Mood/style clustering

`MiniBatchKMeans` clusters the local catalog using standardized energy, tempo, valence, danceability, acousticness, instrumentalness, speechiness, liveness and loudness. The cluster count is between 2 and 24, based on catalog size; a matched live song inherits its local cluster ID.

### Implicit feedback and matrix factorization

User actions become weighted interactions:

```text
Play start / play       +1
Like                    +3
Skip                    -1
```

For collaborative factorization, the summed user-track interaction matrix is clipped at zero before applying `TruncatedSVD`; explicit skips are handled separately. The component count is at most 32 and is bounded by the matrix dimensions. Candidate scores are user/item latent-factor dot products, normalized across the current candidate set; a previously skipped candidate receives an additional `0.35` score reduction.

### Session transitions

Adjacent play events inside a session create next-song signals:

```text
Song A → Song B
Song A → Song C
```

If Song B frequently follows Song A for the same listener in a session, the transition score for A → B increases. This gives the Next queue a session-aware signal instead of relying only on long-term taste.

## Ranking and scoring math

For a queue built from a currently playing seed, the current pre-normalization hybrid score for candidate `i` is:

```text
seed_score(i) =
    0.28 × verified_audio_similarity(i)
  + 0.14 × metadata_score(i)
  + 0.05 × collaborative_score(i)
  + 0.04 × (0.60 × short_term_preference(i) + 0.40 × medium_term_preference(i))
  + 0.03 × listener_mood_match(i)
  + 0.03 × listener_era_match(i)
  + 0.07 × session_transition(i)
  + 0.16 × seed_context(i)
  + 0.06 × seed_artist_match(i)
  + 0.08 × era_match(i)
  + 0.04 × metadata_mood_fallback(i)
  + 0.03 × intent_match(i)
  + 0.03 × exploration_score(i)
  + 0.01 × event_popularity(i)
```

These are coefficients in the implementation, not percentages; component values may be zero when evidence is unavailable. The score is min-max normalized over the candidate set before diversity reranking. Verified audio similarity is used only when both seed and candidate have matched local audio features. Metadata mood is a separate fallback signal.

The main sub-scores are formed as follows:

```text
metadata_score =
    0.35 × intent_match
  + 0.20 × era_match
  + 0.15 × language_match
  + 0.15 × style_match
  + 0.15 × metadata_token_similarity

seed_context =
    0.25 × seed_artist_match
  + 0.12 × album_match
  + 0.08 × era_match
  + 0.08 × language_match
  + 0.12 × mood_cluster_match
  + 0.20 × mood_similarity
  + 0.10 × metadata_token_similarity
  + 0.05 × intent_match
```

For candidates with known release years, era similarity is a Gaussian-shaped score:

```text
era_match = exp(-((seed_year - candidate_year) / 18)^2)
```

When the seed year is known, the candidate must also have a year within 15 years of the seed. Compilation seed albums do not provide a year anchor.

Recent, medium-term and long-term artist profiles use exponential time decay:

```text
preference_weight(age_days, horizon_days) = event_weight × exp(-age_days / horizon_days)
horizons: short = 14 days, medium = 30 days, long = 180 days
```

The seed-based queue blends short and medium profiles 60/40. The long-term profile contributes to ranking without a seed. Search-without-seed ranking uses a separate formula:

```text
search_score =
    0.45 × content_similarity
  + 0.22 × collaborative_score
  + 0.12 × session_transition
  + 0.08 × event_popularity
  + 0.08 × short_term_preference
  + 0.05 × long_term_preference
```

When exposed recommendation candidates later receive feedback, the optional `ranking_model.py` layer trains only after at least 100 labelled candidates, 20 positive labels and multi-outcome impression groups are available. Labels are `like = 2`, `play = 1`, `skip = 0`; unknown candidates are not treated as negatives. Feedback is attributed for up to 24 hours. If training is possible, LightGBM LambdaMART is used when installed; otherwise scikit-learn's HistGradientBoosting is used. The learned score contributes a 12% correction (`88%` hybrid + `12%` learned).

### Relevance gates and queue diversity

In addition to ranking, the seed-based path rejects the seed and duplicate recordings, applies language and intent conflict checks, and requires musical-context evidence. A weak candidate cannot enter solely because it is popular. The final greedy, MMR-style reranker starts from base relevance and adjusts each pick:

```text
adjusted_score =
    base_relevance
  - 0.08 × maximum_similarity_to_already_selected_tracks
  - 0.30 × prior_occurrences_of_the_candidate_artist
  - 0.25 × prior_occurrences_of_the_candidate_album
  - 0.10, if its artist matches the immediately previous pick

recently played candidate: base_relevance × 0.88
```

The first appearance of an artist or album has no repeat penalty. Duplicate recordings and tracks already in the current session queue/history are excluded; recent tracks found in the longer-term interaction model are also down-weighted by 12%.

For live songs without native audio features, metadata semantics are used only
as low-confidence evidence. A confident acoustic/indie, ghazal, motivational,
romantic or devotional seed rejects candidates with a conflicting intent, and
generic popular/trending pages are never used as a seed-specific fallback.

## Real-time behavior

- Search answers the user’s query.
- Playing a search result creates a fresh queue from that song.
- Selecting an Up Next song makes it the new seed and regenerates recommendations.
- Manual Next marks the current song as skipped; Next and automatic end-of-track playback both advance to the queued song.
- The browser audio component reports the ended event so playback can advance without a manual Next click; Loop disables that behavior.
- Like updates the model and reranks the existing queue without removing all recommendations.
- Refresh refits the model and builds a fresh queue excluding the current queue.
- Live API responses are cached and candidate pages are fetched in controlled parallel batches.
- The hybrid model is cached and refits only when new interaction events arrive.

### Import Spotify listening history

Spotify's **Download your data** privacy export can be used without connecting a Spotify account to the app. The importer accepts either Extended Streaming History or the older Streaming History JSON format, removes duplicates, and excludes technical/private fields such as IP addresses and user-agent strings.

```powershell
python import_spotify_history.py "C:\path\to\SpotifyData" --output spotify_history.csv
```

The app reads `spotify_history.csv` from the repository root and maps its most-listened tracks (by total `ms_played`) to matching title/artist pairs in the local catalog. Up to 100 matches seed personalization. Restart the app after importing so the cached CSV is reloaded. Imported history is a recommendation preference source; it is not copied into account event analytics.

## Data and admin analytics

Each behavior event contains:

```text
user_id, track_id, event_type, session_id,
duration_ms, metadata_json, created_at
```

Listeners can view private, date-filtered charts for daily plays, likes, skips and listening time, event mix, top artists and tracks, mood, release era and language. Available periods are 7, 30 and 90 days or all time. Listening duration counts persisted player-progress events, not total account-session duration.

The allowlisted admin dashboard refreshes every 10 seconds and displays registered users, total events, plays, likes, skips, listening hours, user-wise activity, listening trends, most-played artists, database backend, catalog size, mood clusters, live feature-enrichment match rate, metadata fallback count, interaction-matrix density, like rate, model weights and cluster distribution. Its recommendation diagnostics pair the latest play with that user's queue for the same session, rather than reusing a prior song's queue; if a queue is not ready, the dashboard says so.

## Database behavior

```text
No DATABASE_URL → local SQLite: tune_match_events.sqlite3
mysql://...      → MySQL
postgresql://... → PostgreSQL
```

Cloud deployment should use managed MySQL or PostgreSQL. This keeps accounts, password hashes, likes, skips, plays, listening time and recommendation events across restarts and redeploys. Never commit the real database URL or Streamlit secrets.

## Run locally on Windows

```powershell
cd D:\spotify_recommendation_engine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Configure secrets

Create `.streamlit/secrets.toml` locally and do not commit it. An example is available at `.streamlit/secrets.toml.example`.

```toml
DATABASE_URL = "replace-with-your-provider-url"
TUNEMATCH_ADMIN_EMAIL = "your-admin-email@example.com"
```

For PostgreSQL, paste the provider-issued URL into `DATABASE_URL`.

The current Streamlit UI does not start a Spotify OAuth flow, so Spotify API credentials are not required. Spotify personalization is available through the local history-export import described above.

## Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io).
2. Sign in with the GitHub account that owns `Ritansh-7/MagicMusic`.
3. Click **Create app**.
4. Select:

```text
Repository: Ritansh-7/MagicMusic
Branch: master
Main file: app.py
```

5. Deploy the app.
6. Open **Manage app → Settings → Secrets** and add:

```toml
DATABASE_URL = "replace-with-your-provider-url"
TUNEMATCH_ADMIN_EMAIL = "your-admin-email@example.com"
```

7. Reboot the app after saving secrets.

Without `DATABASE_URL`, local SQLite is used and cloud data may not survive a restart. Existing local SQLite data is not automatically migrated to the cloud database. `TUNEMATCH_ADMIN_EMAIL` must match the email address of the account that should see the admin workspace. The `TUNEMATCH_*` setting names and `tune_match_events.sqlite3` filename are retained for backward compatibility with existing deployments and local listening data.

## Deploy on Render

The repository includes `render.yaml`.

```text
Build command:
pip install -r requirements.txt

Start command:
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Add `DATABASE_URL` and `TUNEMATCH_ADMIN_EMAIL` as Render environment variables. `render.yaml` also sets the default JioSaavn API base URL; override `JIOSAAVN_API_URL` only when using a compatible API endpoint. Render is the recommended target for this Streamlit server; Vercel is not the preferred target for a long-running Streamlit application.

## Test

```powershell
python -m pytest -q
```

## Privacy and limitations

- Passwords are never stored in readable form; only salted PBKDF2 hashes are stored.
- Do not commit `.streamlit/secrets.toml` or real database credentials.
- JioSaavn integration uses an unofficial public API and may experience rate limits, missing metadata or temporary failures.
- Audio features are inherited only for reliable local catalog matches; unmatched live songs use metadata fallback.
- Spotify history is an optional local file; review it before importing and do not publish personal listening data.
- Production deployment should add consent, data export/deletion, stronger account recovery and database backups.

## Project structure

```text
app.py                  Streamlit UI, player, personalization and analytics
recommender.py          Local content engine, hybrid ranker, SVD and session signals
ranking_model.py        Optional feedback-trained learning-to-rank model
search_ranking.py       Search-result title matching and ordering
jiosaavn_client.py      Live JioSaavn API adapter and result normalization
spotify_api.py          Spotify OAuth/Web API helper (not used by current UI flow)
import_spotify_history.py Spotify privacy-export JSON to CSV importer
event_store.py          Minimal standalone SQLite recommendation-event store
user_store.py           SQLite/MySQL/PostgreSQL accounts and behavior events
spotify_tracks_data.csv Local catalog (4,002 tracks)
spotify_history.csv     Optional imported listening history
tests/                  Recommendation, search, provider and store tests
render.yaml             Render deployment configuration
.streamlit/config.toml  Streamlit theme and server configuration
```
