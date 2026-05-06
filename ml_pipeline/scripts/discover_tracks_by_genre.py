#!/usr/bin/env python3
"""
Discover tracks for underrepresented genres via Spotify search.

For each genre, paginates `GET /v1/search?q=genre:"<genre>"&type=track`,
filters by popularity, deduplicates against existing corpus by ISRC and
Spotify ID, and writes a raw playlist-style JSON file consumable by
build_song_corpus.py.

Requires the server to be running and the user logged in (uses /api/internal/token).
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None


def get_token(server: str) -> str:
    url = f"{server.rstrip('/')}/api/internal/token"
    try:
        with urllib.request.urlopen(url) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            token = body.get("access_token")
            if not token:
                raise RuntimeError(f"No token in response: {body}")
            return token
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Not logged in — open http://127.0.0.1:5173 and log in with Spotify first"
        ) from exc


def spotify_get(path: str, token: str, retries: int = 3) -> Any:
    url = f"https://api.spotify.com/v1{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After") or 2)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if exc.code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            raise


def load_existing_keys(corpus_path: Path) -> tuple[set[str], set[str]]:
    """Return (isrcs, spotify_ids) already in corpus."""
    isrcs: set[str] = set()
    spotify_ids: set[str] = set()
    if not corpus_path.exists():
        return isrcs, spotify_ids
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = r.get("ids") or {}
        if ids.get("isrc"):
            isrcs.add(ids["isrc"].upper())
        if ids.get("spotify_id"):
            spotify_ids.add(ids["spotify_id"])
    return isrcs, spotify_ids


def build_track_record(track: dict[str, Any]) -> dict[str, Any]:
    artists = [a.get("name") for a in (track.get("artists") or []) if a.get("name")]
    album = track.get("album") or {}
    images = album.get("images") or []
    image_url = images[0].get("url") if images else None
    release_date = album.get("release_date") or ""
    year = None
    if release_date:
        try:
            year = int(release_date[:4])
        except ValueError:
            year = None
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": artists,
        "album": album.get("name"),
        "year": year,
        "spotifyUrl": (track.get("external_urls") or {}).get("spotify"),
        "imageUrl": image_url,
        "popularity": track.get("popularity"),
        "previewUrl": track.get("preview_url"),
        "durationMs": track.get("duration_ms"),
        "isrc": (track.get("external_ids") or {}).get("isrc"),
    }


def search_genre(
    genre: str,
    token: str,
    *,
    max_pages: int,
    min_popularity: int,
    seen_isrcs: set[str],
    seen_spotify_ids: set[str],
) -> list[dict[str, Any]]:
    """Find artists matching the genre, then collect their top tracks.

    Spotify's `genre:"x"` operator is unreliable. We use plain text artist
    search, then validate by checking the artist's actual `genres` tags
    contain the target as a substring. Top tracks come from /artists/{id}/top-tracks.
    """
    out: list[dict[str, Any]] = []
    genre_l = genre.lower()
    q = urllib.parse.quote(genre)

    matched_artists: list[tuple[str, str]] = []  # (id, name)
    for page in range(max_pages):
        offset = page * 50
        path = f"/search?q={q}&type=artist&limit=50&offset={offset}"
        data = spotify_get(path, token)
        items = ((data or {}).get("artists") or {}).get("items") or []
        if not items:
            break
        kept = 0
        for a in items:
            if not a.get("id"):
                continue
            artist_genres = [g.lower() for g in (a.get("genres") or [])]
            if not artist_genres:
                continue
            if any(genre_l in g for g in artist_genres):
                matched_artists.append((a["id"], a.get("name", "?")))
                kept += 1
        print(f"  {genre}: artist page {page + 1} → {len(items)} found, {kept} matched genre")
        if len(items) < 50:
            break
        time.sleep(0.1)

    print(f"  {genre}: {len(matched_artists)} artists matched, fetching top tracks...")
    for i, (artist_id, artist_name) in enumerate(matched_artists):
        path = f"/artists/{artist_id}/top-tracks?market=US"
        try:
            data = spotify_get(path, token)
        except urllib.error.HTTPError as exc:
            print(f"    {artist_name}: HTTP {exc.code}, skipping")
            continue
        tracks = (data or {}).get("tracks") or []
        for t in tracks:
            if not t or not t.get("id"):
                continue
            sp_id = t["id"]
            isrc = (t.get("external_ids") or {}).get("isrc") or ""
            isrc_upper = isrc.upper()
            if sp_id in seen_spotify_ids:
                continue
            if isrc_upper and isrc_upper in seen_isrcs:
                continue
            if (t.get("popularity") or 0) < min_popularity:
                continue
            seen_spotify_ids.add(sp_id)
            if isrc_upper:
                seen_isrcs.add(isrc_upper)
            out.append(build_track_record(t))
        if (i + 1) % 25 == 0:
            print(f"    {genre}: {i + 1}/{len(matched_artists)} artists processed, {len(out)} tracks kept")
        time.sleep(0.05)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover tracks for underrepresented genres via Spotify search.")
    parser.add_argument("--genres", nargs="+", required=True, help="Genres to fetch, e.g. 'k-pop' 'j-pop' 'hyperpop'.")
    parser.add_argument("--server", default="http://127.0.0.1:4000", help="Local server for /api/internal/token.")
    parser.add_argument("--corpus", default="ml_pipeline/data/processed/song_records.backfilled.ndjson",
                        help="Existing corpus to dedupe against.")
    parser.add_argument("--out-dir", default="ml_pipeline/data/raw",
                        help="Where to write the raw JSON output.")
    parser.add_argument("--name", default="discovered", help="Filename stem for output.")
    parser.add_argument("--max-pages", type=int, default=10, help="Max search pages per genre (50 tracks/page, max 20).")
    parser.add_argument("--min-popularity", type=int, default=25, help="Skip tracks below this popularity (0-100).")
    args = parser.parse_args(argv or sys.argv[1:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading existing corpus: {args.corpus}")
    seen_isrcs, seen_spotify_ids = load_existing_keys(Path(args.corpus))
    print(f"  existing: {len(seen_isrcs)} ISRCs, {len(seen_spotify_ids)} Spotify IDs")

    print("Getting Spotify token...")
    token = get_token(args.server)

    all_new: list[dict[str, Any]] = []
    per_genre: dict[str, int] = {}
    for genre in args.genres:
        print(f"\nSearching: {genre}")
        new_tracks = search_genre(
            genre, token,
            max_pages=args.max_pages,
            min_popularity=args.min_popularity,
            seen_isrcs=seen_isrcs,
            seen_spotify_ids=seen_spotify_ids,
        )
        per_genre[genre] = len(new_tracks)
        all_new.extend(new_tracks)
        print(f"  → {len(new_tracks)} new tracks for '{genre}'")

    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(all_new, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = out_dir / f"{args.name}.summary.json"
    summary_path.write_text(
        json.dumps({"total_new_tracks": len(all_new), "per_genre": per_genre}, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== Discovery Summary ===")
    print(f"Total new tracks: {len(all_new)}")
    for g, n in per_genre.items():
        print(f"  {n:5d}  {g}")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
