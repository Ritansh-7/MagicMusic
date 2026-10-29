# TuneMatch — Spotify Track Recommendation Engine

A Spotify-inspired hybrid recommender with live music search, implicit feedback, session-aware ranking, and a local content catalog.

## Features
- Local TuneMatch account signup/login with salted password hashing
- Single-page Spotify-style Home with immediate shelves, an optional global search bar, and a player queue
- Live JioSaavn song cards with artwork, artist/album metadata, and external play links (no JioSaavn login)
- Local event tracking for session duration, play starts, likes, dislikes, and recommendation seeds
- Cold-start onboarding and personalized shelves without forcing the user into separate search/vibe screens
- Unified listener profile from selected genres and favorite tracks
- Hybrid profile ranking with semantic/audio similarity, popularity, and artist diversity
- Explainable shelves for personalized picks, high-energy discoveries, and hidden gems
- Similar-song recommendations from a selected seed track
- TF-IDF representations for genre tags and artist names
- Min-max-scaled audio features and cosine similarity
- Hybrid ranking that combines similarity with popularity
- Search by song, artist, album, or genre
- Vibe-based recommendations from energy, danceability, valence, and genre
- Optional explicit-track filtering
- Export recommendations to CSV
- Unit tests

## Recommendation strategy and data limitation
The supplied catalog has metadata and audio features; user behavior is collected separately in the local event database. The app combines content similarity, implicit-feedback collaborative signals, TruncatedSVD matrix factorization, session transitions, popularity, and diversity penalties. Cold-start users rely more on content and onboarding signals.

The strategy is adapted from the themes in [Spotify Research's Search & Recommendations work](https://research.atspotify.com/search-recommendations): represent listeners and items in a shared feature space, combine behavioral-style profile signals with item semantics, make cold start explicit through onboarding choices, and serve recommendations as intent-driven shelves. In this local clone, onboarding genres and selected tracks stand in for behavioral signals; when event data becomes available, the same profile API can be extended with recency-weighted likes, skips, saves, and playlist context.

## Setup (Windows PowerShell)

Extract the ZIP, open PowerShell in the project folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run spotify_recommendation_engine\app.py
```

The app opens in your browser. Use **Similar to a song** to select a seed track and generate recommendations.

The app now starts with a normal TuneMatch account login. This local account is independent of Spotify and works without Spotify Premium. Playback controls log recommendation behavior and open a track in Spotify; the demo does not rehost or stream copyrighted audio.

JioSaavn live discovery is an optional metadata layer. It does not require a JioSaavn login, but it uses an unofficial API and can temporarily fail. The app falls back to the local recommendation catalog automatically.

## Import your downloaded Spotify history

After Spotify sends your ZIP, extract it to a folder and run:

```powershell
cd D:\spotify_recommendation_engine\spotify_recommendation_engine
python import_spotify_history.py "C:\Users\YOUR_NAME\Downloads\SpotifyData" --output spotify_history.csv
```

The importer supports Spotify Extended Streaming History JSON files and older Streaming History files. It creates a clean CSV with track, artist, album, timestamp, play duration, skip, and playback-context fields. It deliberately excludes IP addresses and technical log data. We can use this CSV next to build a time-aware personal profile and evaluate recommendations from your actual listening behavior.

## Enable real Spotify login and songs

1. Create an app in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2. Add this redirect URI to the app settings for local development:
   `http://127.0.0.1:8501`
3. Create `spotify_recommendation_engine/.streamlit/secrets.toml` with:

```toml
SPOTIFY_CLIENT_ID = "your-client-id"
SPOTIFY_CLIENT_SECRET = "your-client-secret"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8501"
```

4. Start the app and select **Log in with Spotify** in the sidebar.

Live mode uses Spotify OAuth and requests only `user-read-private`, `user-top-read`, `user-read-recently-played`, and `user-library-read`. It syncs the current user's profile, top tracks, recent plays, and saved tracks. Live catalog search links back to Spotify; this app does not download or rehost audio.

The app stores only explicit recommendation events (`recommendation_seed`, `like`, and `dislike`) in `tune_match_events.sqlite3`. Access and refresh tokens remain in the Streamlit session and are not written to the database. For production, move the event store to a protected server database, add a privacy/consent screen, encrypt secrets, and implement deletion/export controls.

Spotify data can improve personalization, but it is not automatically a good training set. Use explicit feedback and time-aware evaluation, separate users between train/validation/test, and never train on data without user consent.

## Deploy on Render

Use the repository root as the Render service directory and use:

```text
Build command: pip install -r requirements.txt
Start command: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

The included `render.yaml` contains the same configuration. Add `TUNEMATCH_ADMIN_EMAIL` as a Render environment variable for the admin dashboard. The local SQLite database is ephemeral on Render's free tier, so use an external persistent database before production; otherwise user accounts and events can disappear after a redeploy or restart.

## Run tests

```powershell
python -m pip install pytest
python -m pytest -q
```

## How the ranking works

The track representation combines three feature groups:
- genre TF-IDF (weight 1.0)
- artist TF-IDF (weight 0.35)
- scaled audio features (weight 0.8)

Cosine similarity measures how close the combined track vectors are. The final seed-track score is:

`score = (1 - p) * similarity + p * (popularity / 100)`

where `p` is the popularity influence chosen in the UI. Home shelves additionally apply a light artist-diversity penalty so a single artist does not dominate the row. This score is a practical baseline, not a model trained on listening outcomes.

## Dataset columns used
`track_id`, `track_name`, `artists`, `album_name`, `track_genre`, `popularity`, `explicit`, and available audio features including `danceability`, `energy`, `acousticness`, `instrumentalness`, `valence`, `tempo`, `loudness`, and `duration_ms`.
