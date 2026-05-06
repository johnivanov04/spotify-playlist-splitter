#!/usr/bin/env python3
"""
Merge a delta corpus into a base corpus.

For each delta record:
- If the dedupe key is new, append the record.
- If the dedupe key already exists, upsert fields where the delta is richer:
  missing MBID, longer brainz.tags / brainz.genres, missing acoustic payloads,
  longer spotify_artist_genres.

Dedupe key precedence: ISRC > MBID > Spotify ID > track_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def dedupe_key(r: dict[str, Any]) -> str:
    ids = r["ids"]
    return (
        ids.get("isrc")
        or ids.get("musicbrainz_recording_id")
        or ids.get("spotify_id")
        or r["track_id"]
    )


def list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def upsert(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, str]:
    """Mutate `existing` in place. Return a count of fields updated."""
    updates: dict[str, str] = {}

    if not existing["ids"].get("musicbrainz_recording_id") and incoming["ids"].get("musicbrainz_recording_id"):
        existing["ids"]["musicbrainz_recording_id"] = incoming["ids"]["musicbrainz_recording_id"]
        updates["mbid"] = "filled"

    eb, ib = existing.get("brainz") or {}, incoming.get("brainz") or {}

    if list_len(ib.get("tags")) > list_len(eb.get("tags")):
        existing.setdefault("brainz", {})["tags"] = ib["tags"]
        updates["tags"] = f"{list_len(eb.get('tags'))}->{list_len(ib['tags'])}"

    if list_len(ib.get("genres")) > list_len(eb.get("genres")):
        existing.setdefault("brainz", {})["genres"] = ib["genres"]
        updates["genres"] = f"{list_len(eb.get('genres'))}->{list_len(ib['genres'])}"

    if not eb.get("acoustic_high_level") and ib.get("acoustic_high_level"):
        existing.setdefault("brainz", {})["acoustic_high_level"] = ib["acoustic_high_level"]
        updates["acoustic_high_level"] = "filled"

    if not eb.get("acoustic_low_level") and ib.get("acoustic_low_level"):
        existing.setdefault("brainz", {})["acoustic_low_level"] = ib["acoustic_low_level"]
        updates["acoustic_low_level"] = "filled"

    if list_len(incoming.get("spotify_artist_genres")) > list_len(existing.get("spotify_artist_genres")):
        existing["spotify_artist_genres"] = incoming["spotify_artist_genres"]
        updates["spotify_artist_genres"] = (
            f"{list_len(existing.get('spotify_artist_genres'))}->{list_len(incoming['spotify_artist_genres'])}"
        )

    return updates


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a delta corpus into a base corpus with upsert semantics.")
    parser.add_argument("--base", required=True, help="Existing corpus ndjson (will be overwritten).")
    parser.add_argument("--delta", required=True, help="New delta corpus ndjson.")
    parser.add_argument("--output", help="Defaults to --base.")
    args = parser.parse_args(argv or sys.argv[1:])

    base_path = Path(args.base)
    delta_path = Path(args.delta)
    out_path = Path(args.output) if args.output else base_path

    base = load_ndjson(base_path)
    delta = load_ndjson(delta_path)

    by_key = {dedupe_key(r): r for r in base}
    added = 0
    updated_records = 0
    field_counts: dict[str, int] = {}

    for r in delta:
        k = dedupe_key(r)
        if k in by_key:
            updates = upsert(by_key[k], r)
            if updates:
                updated_records += 1
                for field in updates:
                    field_counts[field] = field_counts.get(field, 0) + 1
        else:
            base.append(r)
            by_key[k] = r
            added += 1

    write_ndjson(out_path, base)

    print(f"base: {len(base) - added} | delta: {len(delta)} | added: {added} | updated: {updated_records} | total: {len(base)}")
    if field_counts:
        print("field updates:", ", ".join(f"{k}={v}" for k, v in sorted(field_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
