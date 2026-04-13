# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spotify Playlist Splitter: a web app that authenticates with Spotify, loads a user's playlist, and suggests/creates sub-playlists based on era, popularity, listening history, and ML-based clustering. The ML model runs entirely client-side in the browser using a pre-trained KMeans model bundled as JSON.

## Running Locally

**Server** (Express, port 4000):
```
cd server && node index.js
```

Requires a `server/.env` with:
```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:4000/auth/callback
FRONTEND_URL=http://127.0.0.1:5173
SESSION_SECRET=...
MUSICBRAINZ_USER_AGENT=AppName/0.1 (email@example.com)
```

**Client** (React + Vite, port 5173):
```
cd client && npm run dev
```

**Lint client:**
```
cd client && npm run lint
```

## Architecture

### Client (`client/src/`)
- **`App.jsx`** — monolithic single-component React app. Contains all UI state, playlist logic, suggestion rules (`buildSuggestions`), and calls to the ML inference layer. The API base URL is hardcoded to `http://127.0.0.1:4000`.
- **`ml/kmeansInfer.js`** — pure-JS KMeans inference: `buildFeatureDict` → `vectorize` → `scaleVector` → `predictCluster`. Replicates the Python training pipeline's feature engineering at runtime.
- **`ml/kmeans_model.json`** — bundled model artifact (centroids, scale, feature_names, k). This file must match the training pipeline's output format exactly.

### Server (`server/index.js`)
Single Express file handling:
- **Spotify OAuth** (`/auth/login`, `/auth/callback`, `/auth/logout`) — authorization code flow with server-side cookie-session token storage. Tokens never reach the client.
- **Spotify API proxy** (`/api/me`, `/api/playlists`, `/api/playlists/:id/tracks`, `POST /api/playlists`, `/api/playlists/:id/remove-tracks`) — fetches tracks with ISRCs and audio features (danceability, energy, valence, tempo).
- **MusicBrainz/AcousticBrainz enrichment** (`POST /api/brainz/enrich`) — takes ISRCs, resolves to MBIDs, fetches tags/genres and AcousticBrainz high-level data. In-memory caches per session; rate-limited to ~1 req/sec for MusicBrainz.

### ML Pipeline (`ml_pipeline/scripts/`)
Offline Python pipeline for training new model versions. Run scripts in order:

1. `build_song_corpus.py` — normalize raw playlist exports → `song_records.ndjson`
2. `backfill_musicbrainz.py` — enrich with MusicBrainz/AcousticBrainz data
3. `featurize_song_records.py` — build numeric feature matrix (`feature_matrix.npz`)
4. `train_representation.py` — fit StandardScaler + PCA + KMeans → `representation_artifacts.npz`
5. `cluster_playlist.py` — cluster a new playlist using the trained representation
6. `build_display_groups.py` — map cluster names to display group buckets
7. `build_corpus_lookup.py` — build lookup table for fast corpus enrichment

Data directories: `ml_pipeline/data/raw/` (inputs), `ml_pipeline/data/processed/` (normalized), `ml_pipeline/data/artifacts/` (versioned training outputs like `features_v1_8/`, `representation_v1_8/`).

Requires scikit-learn + numpy. Each script accepts `--help` for usage.

## Key Design Constraints

- **Feature parity**: `kmeansInfer.js::buildFeatureDict` must exactly match the Python featurizer's column names and imputation logic (missing values → 0, presence flags named `sp__has_*`, AcousticBrainz features named `ab__<classifier>__<class>`). Changing either side requires updating the other and retraining.
- **Token security**: Spotify access/refresh tokens live only in server-side `cookie-session`. The client never sees them; all Spotify calls are proxied through the server.
- **MusicBrainz rate limiting**: The server enforces ≥1.1s between MB requests. The enrichment limit is capped at 120 ISRCs per request.
- **Audio features deprecation**: Spotify's `/audio-features` endpoint may return 403; the server handles this gracefully and continues without energy/valence/tempo data.
