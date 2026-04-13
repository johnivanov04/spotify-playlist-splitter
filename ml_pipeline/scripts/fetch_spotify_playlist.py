#!/usr/bin/env python3
"""
Fetch a Spotify playlist and save it as a bootstrap JSON file
compatible with build_song_corpus.py.

Requires the local server to be running and the user to be logged in via the app.
The script retrieves the Spotify access token from the server's local-only
/api/internal/token endpoint, then fetches the playlist directly from Spotify.

Usage:
    # 1. Start the server: cd server && node index.js
    # 2. Log in via the app at http://127.0.0.1:5173
    # 3. Run:
    python ml_pipeline/scripts/fetch_spotify_playlist.py \
      --playlist https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U \
      --out ml_pipeline/data/bootstrap/rock_classics.json
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_playlist_id(value: str) -> str:
    m = re.search(r"playlist/([A-Za-z0-9]+)", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]+", value):
        return value
    raise ValueError(f"Cannot parse playlist ID from: {value!r}")


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
        with urllib.request.urlopen(req, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"Spotify API {exc.code} for {path}: {body[:200]}") from exc


def fetch_all_tracks(playlist_id: str, token: str) -> list[dict[str, Any]]:
    url: str | None = f"/playlists/{playlist_id}/tracks?limit=100"
    items: list[dict[str, Any]] = []
    while url:
        data = spotify_get(url, token)
        items.extend(data.get("items") or [])
        url = data.get("next")
    return items


def fetch_isrcs(track_ids: list[str], token: str) -> dict[str, str]:
    isrc_map: dict[str, str] = {}
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i : i + 50]
        data = spotify_get(f"/tracks?ids={','.join(chunk)}", token)
        for t in data.get("tracks") or []:
            if not t or not t.get("id"):
                continue
            isrc = (t.get("external_ids") or {}).get("isrc")
            if isrc:
                isrc_map[t["id"]] = isrc.strip().upper()
    return isrc_map


def items_to_bootstrap(items: list[dict[str, Any]], isrc_map: dict[str, str]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for item in items:
        t = item.get("track")
        if not t or not t.get("id") or t.get("is_local"):
            continue
        album = t.get("album") or {}
        artists = [a["name"] for a in (t.get("artists") or []) if a.get("name")]
        year = None
        rd = album.get("release_date", "")
        if rd:
            m = re.match(r"(\d{4})", rd)
            if m:
                year = int(m.group(1))
        images = album.get("images") or []
        tracks.append({
            "id": t["id"],
            "name": t.get("name", ""),
            "artists": artists,
            "album": album.get("name", ""),
            "year": year,
            "spotifyUrl": (t.get("external_urls") or {}).get("spotify", ""),
            "imageUrl": images[0]["url"] if images else None,
            "popularity": t.get("popularity"),
            "previewUrl": t.get("preview_url"),
            "durationMs": t.get("duration_ms"),
            "isrc": isrc_map.get(t["id"]),
            "energy": None,
            "tempo": None,
            "valence": None,
            "danceability": None,
        })
    return tracks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Spotify playlist for corpus training."
    )
    parser.add_argument("--playlist", required=True, help="Spotify playlist URL or ID")
    parser.add_argument("--out", required=True, help="Output bootstrap JSON path")
    parser.add_argument("--server", default="http://127.0.0.1:4000", help="Local server URL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        playlist_id = extract_playlist_id(args.playlist)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    print(f"playlist_id: {playlist_id}")

    try:
        print("Getting token from server...")
        token = get_token_from_server(args.server)

        print("Fetching playlist tracks...")
        items = fetch_all_tracks(playlist_id, token)
        print(f"  raw items: {len(items)}")

        track_ids = [
            item["track"]["id"]
            for item in items
            if item.get("track") and item["track"].get("id") and not item["track"].get("is_local")
        ]

        print("Fetching ISRCs...")
        isrc_map = fetch_isrcs(track_ids, token)
        print(f"  isrcs resolved: {len(isrc_map)}/{len(track_ids)}")

        tracks = items_to_bootstrap(items, isrc_map)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tracks, indent=2, ensure_ascii=False), encoding="utf-8")

    with_isrc = sum(1 for t in tracks if t.get("isrc"))
    print(f"tracks: {len(tracks)}")
    print(f"with_isrc: {with_isrc}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
