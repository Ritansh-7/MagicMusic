# TuneMatch Recommendation Engine — Team Report

## 1. Executive summary

TuneMatch is a Spotify-inspired Streamlit music discovery application. It has two recommendation layers:

1. A local catalog engine built from `spotify_tracks_data.csv`.
2. A live music discovery layer that retrieves external song metadata and audio URLs, then ranks those results for the signed-in TuneMatch user.

The current live ranking is a hybrid, context-first, sequence-aware recommender. It combines:

```text
current-song context     60%
user taste               15%
next-track transitions   15%
collaborative/MF score    5%
popularity                5%
```

The app records implicit feedback—plays, likes, skips, and listening duration—in SQLite. After a play, like, skip, or explicit refresh, the interaction model is refit and the `Up next for you` queue is rebuilt.

The important production caveat is that the current external catalog does not provide the same audio-feature vector as the local dataset. Exact live-result matches inherit the local audio-feature vector. Unmatched results receive a fast, low-confidence metadata mood vector; the app deliberately avoids downloading and analysing every live preview during queue construction because that would add significant latency.

## 2. Product behavior

The user flow is:

```text
Create/login to TuneMatch account
        ↓
Search for a song, artist, album, or genre
        ↓
Display matching live music results with artwork
        ↓
User clicks Play
        ↓
Current song starts in the in-page audio player
        ↓
Candidate songs are collected from artist, album, similar, and fallback searches
        ↓
Candidates are ranked by the hybrid model
        ↓
Up next for you queue is created
        ↓
Next plays from the recommendation queue, not from the original search results
```

Search results and recommendations are intentionally different:

- Search results answer: “What matches my query?”
- Recommendations answer: “What should this listener hear next?”
- The current search-result songs are excluded from the next queue.
- Repeated titles are removed even when the upstream service returns different IDs for the same song.

## 3. Repository structure

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, authentication flow, search, player, queue, event logging, live refresh |
| `recommender.py` | Local content engine and hybrid external-result ranker |
| `jiosaavn_client.py` | External music search/metadata/media adapter |
| `user_store.py` | Local users, behavior events, summaries, interaction extraction |
| `event_store.py` | Older explicit recommendation-event store; not the main live interaction source |
| `spotify_api.py` | Optional Spotify OAuth and Web API helper |
| `import_spotify_history.py` | Converts Spotify privacy-export JSON into clean CSV |
| `spotify_tracks_data.csv` | Local metadata/audio-feature catalog |
| `spotify_history.csv` | Optional imported listening history |
| `tune_match_events.sqlite3` | Local accounts and behavior database |
| `tests/test_recommender.py` | Unit tests for the local recommendation engine |

## 4. Data sources

### 4.1 Local track catalog

The current catalog contains:

- 4,002 unique tracks
- 2,106 unique artists
- 512 genres
- 20 columns

Important columns:

```text
track_id
track_name
artists
album_name
track_genre
popularity
explicit
duration_ms
danceability
energy
speechiness
acousticness
instrumentalness
liveness
valence
tempo
loudness
key
mode
time_signature
```

This dataset is suitable for a content-based baseline. It does not contain a reliable user-item interaction table, so it cannot independently train a production collaborative model.

### 4.2 Live music metadata

The live adapter normalizes external results into:

```text
title
artists
album
language
image_url
duration
year
perma_url
stream_url
stream_format
song_id
```

The client uses query expansion because the upstream endpoint often returns only a small page or fails for an exact title. It aggregates results, removes duplicate IDs, and returns up to the requested limit.

The live source is an unofficial adapter. It can return 500/429 errors, incomplete artist fields, duplicate song versions, expired media URLs, or no playable URL. The app therefore keeps search results cached in Streamlit session state and provides a link fallback when direct audio is unavailable.

### 4.3 User behavior data

`user_store.py` creates two SQLite tables:

```text
users
  id, email, password_hash, created_at

behavior_events
  id, user_id, track_id, event_type, session_id,
  duration_ms, metadata_json, created_at
```

Raw passwords are not stored. Passwords use salted PBKDF2-HMAC-SHA256 with 240,000 iterations.

The event metadata stores live song details so that an external song ID can still be interpreted later even if it is not present in the local catalog.

## 5. Local content-based engine

The `SpotifyRecommendationEngine` prepares the local catalog and builds a sparse feature representation.

### 5.1 Text features

Two TF-IDF blocks are created:

1. Genre TF-IDF
2. Artist TF-IDF

Both use lowercase unigrams and bigrams. Multi-value genre strings are converted into comparable text. Artist text is normalized for commas and ampersands.

### 5.2 Numeric audio features

Available audio features are converted to numbers and missing values are filled with column medians. `MinMaxScaler` maps numeric columns to comparable ranges.

### 5.3 Feature block weights

The combined sparse vector is:

```text
track_vector =
    1.00 × genre_tfidf
  + 0.35 × artist_tfidf
  + 0.80 × scaled_audio_features
```

The final vector is L2-normalized. Similarity is measured with cosine similarity.

### 5.4 Similar-track recommendation

For a seed track, the engine:

1. Finds the seed row.
2. Computes cosine similarity against every catalog track.
3. Removes the seed itself.
4. Optionally keeps only matching genres.
5. Optionally removes explicit tracks.
6. Combines similarity and popularity.

The score is:

```text
score = (1 - popularity_weight) × similarity
      + popularity_weight × normalized_popularity
```

Popularity is deliberately a tie-breaker rather than the main recommendation signal.

### 5.5 Profile recommendation

`build_user_profile()` averages feature vectors from:

- selected onboarding tracks
- imported listening-history matches
- Spotify signals when optional Spotify OAuth is available
- selected onboarding genres

The resulting profile lives in the same feature space as the local catalog. `recommend_for_profile()` compares that profile to all tracks, excludes already selected IDs, and applies a light artist-diversity penalty.

### 5.6 Vibe recommendation

The local engine also supports target preferences for:

- genre
- energy
- danceability
- valence

It scores a track by closeness to the requested numeric targets and blends that with popularity.

## 6. Collaborative filtering and matrix factorization

The live behavior model is implemented by `HybridRecommendationEngine`.

### 6.1 Implicit-feedback weights

Events are converted into preference weights:

| Event | Weight | Interpretation |
|---|---:|---|
| `play_start` | +1.0 | Positive exposure/listening signal |
| `jio_play` | +1.0 | Positive live-song listening signal |
| `like` | +3.0 | Strong positive preference |
| `jio_like` | +3.0 | Strong positive live-song preference |
| `skip` | -1.0 | Negative preference signal |
| `jio_skip` | -1.0 | Negative live-song signal |

Events such as session start and listening-progress are useful for analytics but do not directly create a preference weight in the factorization model.

### 6.2 User-item matrix

The model pivots events into:

```text
rows    = users
columns = external/local track IDs
values  = summed implicit-feedback weights
```

If a user likes a song after playing it, that item receives both signals. If a user skips it, the negative signal lowers its interaction value.

### 6.3 Matrix factorization

The interaction matrix is shifted row-wise so negative skips do not create an unstable factorization input. `TruncatedSVD` then learns latent factors.

The factor count is:

```text
min(32, min(number_of_users, number_of_items) - 1)
```

For each user and candidate item, the dot product between the user latent vector and item latent vector becomes the collaborative score. Scores are normalized to 0–1 when a range exists.

### 6.4 Cold-start behavior

When the current user has no learned factor or a candidate has never appeared in the event matrix, its collaborative score is zero. This is expected. The system then relies on content signals and popularity.

Collaborative filtering becomes useful only when:

- multiple TuneMatch users exist;
- those users interact with overlapping songs;
- songs are represented with stable IDs;
- the current user has enough behavior to map to a latent preference vector.

## 7. Hybrid ranking of live results

`rank_jio_results()` combines current-song context, user taste, session transitions, collaborative signals and a small popularity prior.

### 7.1 Fast live mood enrichment

For exact normalized title/artist matches, the live item inherits the local catalog's numeric feature vector and MiniBatchKMeans cluster. For unmatched items, a lightweight keyword mapper infers a mood vector from title, album, artist and language text. It labels the result as `metadata_fallback` with lower confidence instead of pretending that an audio model analysed the track.

This choice keeps search and queue latency low. A future offline worker can analyse previews with librosa or Essentia and replace the fallback vector without changing the ranker interface.

### 7.2 Live content score

For each external result, the engine checks:

- whether a preferred local artist appears in the result artist text;
- whether a preferred local track title matches the result title.

Artist matches contribute up to 0.25 each and exact preferred-title matches contribute up to 0.75. The content value is capped at 1.0.

This is a practical external-content bridge. It is not the same as full audio-vector similarity because the external result does not provide all local numeric audio features.

### 7.3 Collaborative score

The external `song_id` is looked up in the user-item factorization model. If that ID has appeared in behavior data, the latent user-item score is used. Otherwise it falls back to zero.

### 7.4 Popularity score

Observed interaction totals are normalized and used as a small popularity prior. Unknown items receive zero for this internal popularity signal.

### 7.5 Sequential transition score

The model records adjacent play transitions within a user session. If a user plays track A and then track B, the transition `A -> B` receives evidence. When A is the current seed, candidates previously observed after A receive an additional next-track score.

This is a lightweight sequential recommender. It is not a transformer or reinforcement-learning model, but it gives the queue a session-aware signal beyond static similarity.

### 7.6 Final formula

```text
hybrid_score =
    0.60 × current_song_context
  + 0.15 × user_taste
  + 0.15 × transition_score
  + 0.05 × collaborative_score
  + 0.05 × popularity_score
```

The results are sorted descending by this score.

## 8. How `Up next` is generated

When a user presses Play, the app does not reuse the search-result list. It creates new candidates from:

- each individual artist’s songs;
- artist similar-song searches;
- album songs;
- title similar-song searches;
- popular songs;
- trending songs;
- Bollywood fallback songs.

The current search-result IDs and normalized song titles are removed. The remaining candidates are ranked by the hybrid model and truncated to 70 items. The final reranker penalizes recently played tracks, repeated artists, and repeated albums.

The queue is rebuilt after:

- Play
- Like
- Next/skip
- `Refresh recommendations`

This is the app’s live-learning behavior. Streamlit does not continuously execute Python for every audio-progress second, so model refresh is event-driven rather than continuous per-second streaming.

## 9. Session and analytics metrics

The app distinguishes all-time account information from current-session metrics.

### All-time sidebar summary

The sidebar reads all behavior events for the account and displays total tracks played and total likes.

### Current-session summary

The lower metrics panel filters by the current `session_id` and shows:

- Events: all recorded events in this login session.
- Tracks played: distinct track IDs from `play_start` and `jio_play`.
- Likes: `like` and `jio_like` events.
- Listening time: accumulated duration events plus the currently elapsed player interval.

Listening duration is approximate because a normal Streamlit script cannot receive a browser audio progress callback every second. The app accounts for elapsed player time between Streamlit reruns, such as Play, Like, Next, Refresh, or page interaction.

## 10. Authentication and privacy

TuneMatch login is local and independent of Spotify. It supports:

- email/password account creation;
- salted password hashing;
- local session state;
- logout and session-end logging.

Optional Spotify OAuth support exists in `spotify_api.py`, but the current core experience does not require Spotify Premium or a Spotify login. Spotify access tokens are intended to remain in Streamlit session state rather than the SQLite event database.

For production, the project should add:

- HTTPS;
- secure secret management;
- rate limiting;
- email verification or a managed identity provider;
- account deletion and data-export controls;
- encryption and access control for the event database;
- a consent screen for behavioral tracking.

## 11. Current limitations

### Recommendation quality

The local catalog has no original multi-user listening dataset. The collaborative model is therefore only as good as events collected by TuneMatch users.

### External ID coverage

Jio/external song IDs and local catalog IDs are different namespaces. A local catalog like/Spotify-history signal cannot always be mapped to the external candidate ID.

### Sparse collaborative model

With one user and a few events, matrix factorization cannot discover reliable cross-user patterns. The app correctly falls back toward content and popularity in this situation.

### External API reliability

The unofficial external endpoint can fail, return duplicate versions, omit artist/language fields, rate-limit preview URLs, or provide a media URL that a browser cannot play. Search caching and fallback links reduce, but do not eliminate, this risk.

### Live content features

External results currently expose metadata, not the full local audio-feature vector. The live content component therefore uses artist/title matching rather than exact danceability/energy/valence similarity.

### Listening-time precision

Duration is inferred between reruns and should be treated as an analytics estimate, not a precise playback telemetry stream.

## 12. Testing and verification

The current automated test suite contains four passing tests:

1. Similar-track recommendations exclude the seed track.
2. Artist search returns matching tracks.
3. Explicit filtering removes explicit tracks when disabled.
4. Profile recommendations exclude onboarding tracks.

Manual checks completed during development include:

- Python compilation of app and recommender modules;
- Streamlit health check returning HTTP 200;
- live media URL normalization for MP4/MP3 sources;
- duplicate-title removal in recommendation queues;
- current-session metric aggregation;
- SVD collaborative-score smoke test.

Recommended next tests:

- `HybridRecommendationEngine` with one user and multiple items;
- multiple users sharing one item;
- like/skip weight ordering;
- queue exclusion of search results;
- duplicate title normalization;
- cache fallback when the external API is unavailable;
- session-summary filtering;
- media URL fallback behavior.

## 13. Evaluation plan for the team

The model should not be evaluated only by whether the UI renders. Use time-aware recommendation evaluation:

1. Split each user’s events chronologically.
2. Train on earlier events.
3. Hide later liked/played songs as validation targets.
4. Generate top-K recommendations.
5. Measure:
   - Precision@K
   - Recall@K
   - NDCG@K
   - coverage
   - catalog diversity
   - artist repetition rate
   - skip rate
   - play-through rate
6. Compare against baselines:
   - popularity-only;
   - content-only;
   - collaborative-only;
   - hybrid.

An online experiment should compare different weights, for example:

```text
Variant A: content 60 / collaborative 30 / popularity 10
Variant B: content 50 / collaborative 40 / popularity 10
Variant C: content 70 / collaborative 20 / popularity 10
```

The best production weights should be selected from held-out behavior, not intuition alone.

## 14. Recommended production roadmap

### Phase 1 — Stabilize the current prototype

- Add automated hybrid-model tests.
- Add structured API error logging.
- Cache candidate searches with TTL.
- Add a stable external-song canonical ID.
- Store recommendation impression events, not only clicks.

### Phase 2 — Improve personalization

- Add exponential time decay to events.
- Give completed plays more weight than short plays.
- Add skip position and listening percentage.
- Add artist, album, language, and genre embeddings.
- Map imported Spotify history to external IDs where possible.

### Phase 3 — Improve collaborative learning

- Collect more users and interaction overlap.
- Use implicit ALS or a production recommendation library.
- Add negative sampling and confidence weighting.
- Train offline and serve a versioned model.

### Phase 4 — Production platform

- Replace local SQLite with a protected database.
- Move API calls behind a backend service.
- Add monitoring, retries, rate limits, and provider fallbacks.
- Add user privacy controls and data deletion.
- Track model versions and recommendation explanations.

## 15. Team presentation summary

The short explanation to the team is:

> TuneMatch retrieves candidate music, represents the listener through onboarding and behavior, learns latent preferences from the user-item interaction matrix, and combines content similarity, collaborative signals, and popularity into one ranked list. Search results answer the user’s explicit query, while the separate Up Next queue is generated from related candidates and refreshed after new feedback. The system is cold-start aware: before enough multi-user behavior exists, it relies mostly on content signals and popularity.

The honest limitation is:

> The architecture supports collaborative filtering and matrix factorization, but their quality depends on collecting enough shared user-song interactions. The live external catalog also lacks the full audio-feature vector used by the local content engine, so external live content matching currently relies mainly on artist/title metadata.

