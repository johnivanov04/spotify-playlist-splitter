# ML Pipeline

This folder is the new offline music-model foundation.

The goal is to keep model building **separate** from the Spotify playlist app.
The app should eventually call backend inference, while this pipeline owns:

- corpus building
- normalization
- validation
- feature extraction
- training
- export of reusable model artifacts

## Why this exists

The old `client/src/ml` path is a useful prototype, but it is tightly coupled to
frontend runtime inference. That makes it hard to:

- train on a much larger corpus
- version your data pipeline cleanly
- swap models without redeploying the frontend
- move playlist analysis to the backend
- add future audio embeddings

This new `ml_pipeline/` folder is the first step toward a cleaner architecture:

1. build a large normalized song corpus offline
2. train a representation model offline
3. serve playlist analysis from the backend
4. cluster each playlist in the learned embedding space

## Current scope of this first milestone

This first milestone does **not** train the final model yet.
It establishes the foundation:

- `schemas/song_record.schema.json`
  - canonical normalized song shape
- `scripts/build_song_corpus.py`
  - converts exported track JSON files into normalized records
- `data/processed/song_records.ndjson`
  - corpus output for later training steps
- `data/processed/corpus_summary.json`
  - diagnostics and vocabulary summary

## Canonical song record

Each normalized song record is designed to work for both metadata-only training
now and audio-embedding training later.

High-level fields:

- `track_id`
  - stable internal ID, preferring ISRC, then MBID, then Spotify ID, then
    normalized text fallback
- `title`, `artists`, `album`, `year`, `duration_ms`
- `ids`
  - `spotify_id`, `spotify_uri`, `isrc`, `musicbrainz_recording_id`
- `brainz`
  - `tags`, `genres`, `acoustic_high_level`, `acoustic_low_level`
- `runtime_metadata`
  - data that may be useful during app inference but should not define the core
    training identity of the song
- `audio_embeddings`
  - placeholder for future audio representation vectors
- `source`
  - provenance metadata
- `aliases`
  - normalized text keys for fallback matching

## Input expectations

`build_song_corpus.py` currently expects one or more JSON files where each file
contains a **list of track objects**. It is designed to accept the shape your
current app already produces or exports, including fields like:

- `id`
- `uri`
- `name`
- `artists`
- `album`
- `year`
- `durationMs`
- `popularity`
- `spotifyUrl`
- `previewUrl`
- `imageUrl`
- `isrc`
- `brainz.tags`
- `brainz.genres`
- `brainz.acousticHighLevel`
- `brainz.acousticLowLevel`
- optional Brainz MBID fields

This means you can bootstrap the new pipeline using the same enriched playlist
JSON you already know how to generate.

## Example usage

From the project root:

```bash
python3 ml_pipeline/scripts/build_song_corpus.py \
  --input data/bootstrap/*.json \
  --dataset-name bootstrap-playlists-v1 \
  --out-dir ml_pipeline/data/processed \
  --write-array-json \
  --include-debug
```

You can also point it at a single file:

```bash
python3 ml_pipeline/scripts/build_song_corpus.py \
  --input data/bootstrap/my_playlist_export.json \
  --dataset-name bootstrap-playlists-v1 \
  --out-dir ml_pipeline/data/processed
```

## Output files

### `song_records.ndjson`

Primary machine-friendly corpus file.
One normalized record per line.

### `song_records.json`

Optional inspection-friendly array output, written only when
`--write-array-json` is passed.

### `corpus_summary.json`

Includes:

- number of input files
- raw tracks seen
- normalized records written
- duplicates skipped
- missing ISRC count
- missing MBID count
- counts for tags / genres / acoustic data coverage
- top tags and genres

## Dedupe strategy

Current dedupe priority is:

1. ISRC
2. MusicBrainz recording ID
3. Spotify track ID
4. normalized title + artists fallback

That is a practical starting point for building a clean corpus from repeated
playlist exports.

## Recommended workflow right now

Use the current Spotify app as a **bootstrap data collector**, not as the final
ML system.

A good short-term workflow is:

1. export or save enriched playlist track JSON files
2. place them in a bootstrap data folder
3. run `build_song_corpus.py`
4. inspect `corpus_summary.json`
5. iterate on normalization until the corpus shape is stable
6. only then move on to feature extraction and training

## What comes next after this milestone

After the corpus builder is working, the next files to add should be:

- `scripts/validate_song_records.py`
- `scripts/featurize_song_records.py`
- `scripts/train_representation.py`
- `scripts/cluster_playlist.py`

At that point, the backend can start moving from:

- frontend-imported fixed clusters

into:

- backend playlist analysis using learned song representations

## Note on legacy code

Do **not** delete the old `client/src/ml` folder yet.
Keep it as a reference prototype until the new backend-driven path is working
end to end.
