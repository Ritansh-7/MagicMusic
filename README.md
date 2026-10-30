# TuneMatch

TuneMatch is a Spotify-inspired, live music recommendation platform built with Streamlit. It combines live JioSaavn discovery with a local audio-feature catalog, implicit user feedback, mood clustering and session-aware ranking.

It is not Spotify's proprietary system. It is an explainable, deployable recommendation-engine project demonstrating search, playback, personalization, feedback loops, diversity, cold start and analytics.

## What the project does

- Creates TuneMatch accounts with salted PBKDF2 password hashes.
- Searches live songs through an unofficial JioSaavn API without requiring a JioSaavn login.
- Shows live artwork, artist/album metadata and available audio previews.
- Provides Search, Player, Like, Next, Loop and Refresh controls.
- Builds a separate personalized Up Next queue of up to 70 unique songs.
- Learns from plays, likes, skips and listening progress.
- Reduces repeated titles and consecutive artist/album repetition.
- Shows explainable recommendation reasons.
- Provides an admin-only, auto-refreshing data-science dashboard.

## Recommendation architecture

```text
Live JioSaavn results + local catalog
                ↓
Candidate cleaning and parallel multi-channel retrieval
                ↓
Audio content, mood/intent, long/short preference, collaborative and sequence signals
                ↓
Hybrid ranker + optional LambdaMART learning-to-rank correction
                ↓
MMR-style artist/album/recent-item diversity reranking
                ↓
Personalized 60–70 song queue
```

### Content and live-feature enrichment

The local catalog contains approximately 4,000 tracks with artist, genre, popularity and audio features. The local content engine uses genre TF-IDF, artist TF-IDF, scaled audio features and cosine similarity.

Live JioSaavn results are normalized using title, artist, album, language, year, duration, song ID and version type. When a live result matches the local catalog by normalized title and artist, it inherits:

```text
energy, tempo, valence, danceability, acousticness,
instrumentalness, speechiness, liveness, loudness, popularity
```

When no local match exists, the system does not invent audio features. It uses metadata fallback signals: artist, album, language, year, title and session context.

For fast live ranking, unmatched JioSaavn songs also receive a low-confidence metadata mood vector inferred from title, album and artist terms (for example energetic, romantic, sad, instrumental or acoustic). This avoids downloading and analysing dozens of previews during every queue build. Exact local matches use the catalog audio-feature vector with higher confidence.

### Mood/style clustering

`MiniBatchKMeans` clusters the local catalog using standardized audio features. The model learns up to 24 fast mood/style clusters, such as old acoustic, romantic Bollywood, high-energy dance, slow/sad and instrumental styles. A matched live song inherits its local cluster ID, and cluster similarity becomes part of ranking.

### Implicit feedback and matrix factorization

User actions become weighted interactions:

```text
Play start       +1
Like             +3
Skip             -1
```

The event store also records listening progress, session ID, timestamp and song metadata. TruncatedSVD factorizes the user-song interaction matrix into latent user and item factors. The collaborative score is the dot product between the user and candidate-song latent vectors.

### Session transitions

Adjacent play events inside a session create next-song signals:

```text
Song A → Song B
Song A → Song C
```

If Song B frequently follows Song A, the transition score for A → B increases. This gives the Next queue a session-aware signal instead of relying only on long-term taste.

## Ranking formula

For every candidate song `i`, the current hybrid score is:

```text
score(i) =
    0.35 × audio_similarity(i)
  + 0.15 × metadata_similarity(i)
  + 0.15 × collaborative_score(i)
  + 0.12 × recent_preference(i)
  + 0.08 × long_term_preference(i)
  + 0.10 × sequence_score(i)
  + 0.03 × session_context(i)
  + 0.02 × popularity_score(i)
```

The recent preference is a 60/40 blend of 14-day and 30-day artist profiles;
the long-term profile decays over approximately 180 days. The current song
remains the strongest semantic anchor, while popularity is only a tie-breaker.
When at least 30 impression-labelled examples and 5 positive outcomes exist,
the optional `ranking_model.py` layer applies a small learned correction. It
uses LightGBM LambdaMART when the optional package is installed; otherwise it
uses a fast HistGradientBoosting fallback and never blocks playback.

Recommendation impressions record the candidate feature vector, rank, score,
source and model version. Later plays, likes and skips provide labels for
Precision@10, NDCG@10 and exposure play-through. Users are assigned a stable
control/diverse experiment variant so model changes can be compared instead of
judged only by anecdotal sessions.

For the current live song, seed-content similarity uses:

```text
0.40 × artist match
0.17 × album match
0.16 × era similarity
0.10 × language match
0.17 × mood-cluster match
```

The final reranker applies:

```text
duplicate/recently played songs: removed
artist repeat penalty: 0.42 per previous occurrence
album repeat penalty: 0.10 per previous occurrence
consecutive same-artist penalty: 0.30
```

Weak fallback songs are removed when they do not meet the current mood/context gate. This prevents the same singer from filling the entire queue while preserving strong relevance.

For live songs without native audio features, metadata semantics are used only
as low-confidence evidence. A confident acoustic/indie, ghazal, motivational,
romantic or devotional seed rejects candidates with a conflicting intent, and
generic popular/trending pages are never used as a seed-specific fallback.

## Real-time behavior

- Search answers the user’s query.
- Playing a search result creates a fresh queue from that song.
- Selecting an Up Next song makes it the new seed and regenerates recommendations.
- Next logs the current song as skipped, starts the next song and rebuilds the queue.
- Like updates the model and reranks the existing queue without removing all recommendations.
- Refresh refits the model and builds a fresh queue excluding the current queue.
- API responses are cached and candidate pages are fetched in controlled parallel batches.
- The hybrid model is cached and refits only when new interaction events arrive.

## Data and admin analytics

Each behavior event contains:

```text
user_id, track_id, event_type, session_id,
duration_ms, metadata_json, created_at
```

The admin dashboard refreshes every 10 seconds and displays registered users, total events, plays, likes, skips, listening hours, user-wise activity, listening trends, most-played artists, database backend, catalog size, mood clusters, live feature-enrichment match rate, metadata fallback count, interaction-matrix density, like rate, model weights and cluster distribution.

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
DATABASE_URL = "mysql://username:password@host:3306/tunematch"
TUNEMATCH_ADMIN_EMAIL = "your-admin-email@example.com"
```

Optional Spotify credentials:

```toml
SPOTIFY_CLIENT_ID = "your-client-id"
SPOTIFY_CLIENT_SECRET = "your-client-secret"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8501"
```

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
DATABASE_URL = "mysql://username:password@host:3306/tunematch"
TUNEMATCH_ADMIN_EMAIL = "your-admin-email@example.com"
```

7. Reboot the app after saving secrets.

Without `DATABASE_URL`, local SQLite is used and cloud data may not survive a restart. Existing local SQLite data is not automatically migrated to the cloud database.

## Deploy on Render

The repository includes `render.yaml`.

```text
Build command:
pip install -r requirements.txt

Start command:
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Add `DATABASE_URL` and `TUNEMATCH_ADMIN_EMAIL` as Render environment variables. Render is the recommended target for this Streamlit server; Vercel is not the preferred target for a long-running Streamlit application.

## Test

```powershell
python -m pytest -q
```

Current test status: `4 passed`.

## Privacy and limitations

- Passwords are never stored in readable form; only salted PBKDF2 hashes are stored.
- Do not commit `.streamlit/secrets.toml` or real database credentials.
- JioSaavn integration uses an unofficial public API and may experience rate limits, missing metadata or temporary failures.
- Audio features are inherited only for reliable local catalog matches; unmatched live songs use metadata fallback.
- Production deployment should add consent, data export/deletion, stronger account recovery and database backups.

## Project structure

```text
app.py                  Streamlit UI and application flow
recommender.py          Content, hybrid, SVD and ranking logic
jiosaavn_client.py      Live API adapter and normalization
user_store.py           SQLite/MySQL/PostgreSQL accounts and events
spotify_tracks_data.csv Local feature catalog
tests/                  Recommendation tests
render.yaml             Render deployment configuration
.streamlit/config.toml  Theme and server configuration
```
