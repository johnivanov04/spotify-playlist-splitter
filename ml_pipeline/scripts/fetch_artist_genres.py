#!/usr/bin/env python3
"""
Fetch Spotify artist genres for corpus records and add them as features.

Requires the local server to be running and the user to be logged in.

Usage:
    # 1. Start the server: cd server && node index.js
    # 2. Log in via the app at http://127.0.0.1:5173
    # 3. Run:
    python ml_pipeline/scripts/fetch_artist_genres.py \
      --input  ml_pipeline/data/processed/song_records.backfilled.ndjson \
      --output ml_pipeline/data/processed/song_records.backfilled.ndjson
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None

SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def get_token_from_server(server_url: str) -> str:
    url = f"{server_url.rstrip('/')}/api/internal/token"
    try:
        with urllib.request.urlopen(url) as resp:
            body = json.loads(resp.read().decode())
        token = body.get("access_token")
        if not token:
            raise RuntimeError(f"No token in response: {body}")
        return token
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode()
        if exc.code == 401:
            raise RuntimeError(
                "Not logged in — open http://127.0.0.1:5173 and log in with Spotify first"
            ) from exc
        raise RuntimeError(f"Server error {exc.code}: {body_text[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach server at {server_url} — run: cd server && node index.js"
        ) from exc


def spotify_get(path: str, token: str) -> Any:
    url = path if path.startswith("https://") else f"{SPOTIFY_API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry_after = int(exc.headers.get("Retry-After", "5"))
            print(f"  rate limited, waiting {retry_after}s...")
            time.sleep(retry_after + 1)
            return spotify_get(path, token)
        body = exc.read().decode()
        raise RuntimeError(f"Spotify API {exc.code} for {path}: {body[:200]}") from exc


def fetch_artist_ids_for_tracks(
    track_ids: list[str], token: str
) -> dict[str, list[str]]:
    """Returns {track_id: [artist_id, ...]}."""
    result: dict[str, list[str]] = {}
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i : i + 50]
        data = spotify_get(f"/tracks?ids={','.join(chunk)}", token)
        for t in data.get("tracks") or []:
            if not t or not t.get("id"):
                continue
            result[t["id"]] = [a["id"] for a in (t.get("artists") or []) if a.get("id")]
        print(f"  track artist IDs: {min(i + 50, len(track_ids))}/{len(track_ids)}")
    return result


def fetch_genres_for_artists(
    artist_ids: list[str], token: str
) -> dict[str, list[str]]:
    """Returns {artist_id: [genre_string, ...]}."""
    result: dict[str, list[str]] = {}
    for i in range(0, len(artist_ids), 50):
        chunk = artist_ids[i : i + 50]
        data = spotify_get(f"/artists?ids={','.join(chunk)}", token)
        for a in data.get("artists") or []:
            if not a or not a.get("id"):
                continue
            result[a["id"]] = list(a.get("genres") or [])
        print(f"  artist genres: {min(i + 50, len(artist_ids))}/{len(artist_ids)}")
    return result


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".ndjson":
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich corpus records with Spotify artist genres."
    )
    parser.add_argument("--input", required=True, help="Path to backfilled corpus JSON/NDJSON")
    parser.add_argument("--output", required=True, help="Output path (can be same as input)")
    parser.add_argument("--server", default="http://127.0.0.1:4000", help="Local server URL")
    parser.add_argument(
        "--cache",
        default="ml_pipeline/data/cache/spotify_genres_cache.json",
        help="Cache file for Spotify API responses",
    )
    args = parser.parse_args(argv or sys.argv[1:])

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)

    # Load cache
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    track_to_artists: dict[str, list[str]] = cache.get("track_to_artists") or {}
    artist_genres: dict[str, list[str]] = cache.get("artist_genres") or {}

    print(f"Loading records from {input_path}...")
    records = load_records(input_path)
    print(f"  {len(records)} records")

    # Collect track IDs not yet in cache
    all_track_ids = list(
        {
            (r.get("ids") or {}).get("spotify_id") or r.get("id")
            for r in records
        }
        - {None}
        - set(track_to_artists.keys())
    )

    if all_track_ids:
        print(f"Getting token from server...")
        token = get_token_from_server(args.server)

        print(f"Fetching artist IDs for {len(all_track_ids)} new tracks...")
        new_mappings = fetch_artist_ids_for_tracks(all_track_ids, token)
        track_to_artists.update(new_mappings)

        # Collect artist IDs not yet in cache
        all_artist_ids = list(
            {aid for aids in track_to_artists.values() for aid in aids}
            - set(artist_genres.keys())
        )
        if all_artist_ids:
            print(f"Fetching genres for {len(all_artist_ids)} new artists...")
            new_genres = fetch_genres_for_artists(all_artist_ids, token)
            artist_genres.update(new_genres)

        # Save cache
        cache["track_to_artists"] = track_to_artists
        cache["artist_genres"] = artist_genres
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"Cache saved: {cache_path}")
    else:
        print("All tracks already in cache.")

    # Enrich records
    enriched = 0
    for record in records:
        spotify_id = (record.get("ids") or {}).get("spotify_id") or record.get("id")
        if not spotify_id:
            continue
        artist_ids = track_to_artists.get(spotify_id, [])
        seen: set[str] = set()
        genres: list[str] = []
        for aid in artist_ids:
            for g in artist_genres.get(aid, []):
                if g and g not in seen:
                    seen.add(g)
                    genres.append(g)
        if genres:
            # Store as name-count dicts compatible with iter_name_count_items
            record["spotify_artist_genres"] = [{"name": g, "count": 1} for g in genres]
            enriched += 1

    print(f"Enriched {enriched}/{len(records)} records with Spotify genres")

    # Sample output
    sample = next(
        (r for r in records if r.get("spotify_artist_genres")), None
    )
    if sample:
        genres_sample = [x["name"] for x in sample["spotify_artist_genres"][:5]]
        print(f"Sample ({sample.get('title', '?')}): {genres_sample}")

    write_records(output_path, records)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
