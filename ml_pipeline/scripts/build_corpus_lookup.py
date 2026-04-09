#!/usr/bin/env python3
"""
Build a local lookup index from the backfilled corpus so runtime playlist prep
can reuse already-known MBIDs / tags / genres / acoustic metadata before making
live MusicBrainz requests.

Lookup priority later at runtime:
1. spotify_id
2. isrc
3. musicbrainz_recording_id
4. normalized title + artists
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_token(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("&", " and ")
    value = _TOKEN_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if path.suffix.lower() == ".ndjson":
        rows: list[dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid NDJSON at line {i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"NDJSON line {i} is not an object")
            rows.append(obj)
        return rows

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Corpus input must be a JSON array or NDJSON of record objects")
    return payload


def normalize_artist_key(artists: list[str]) -> str | None:
    vals = [normalize_token(a) for a in artists if is_nonempty_string(a)]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return "::".join(vals)


def alias_key(record: dict[str, Any]) -> str | None:
    aliases = record.get("aliases") or {}
    title = aliases.get("normalized_title")
    artists_key = aliases.get("normalized_artists_key")

    if is_nonempty_string(title) and is_nonempty_string(artists_key):
        return f"{title}::{artists_key}"

    raw_title = record.get("title")
    raw_artists = record.get("artists") or []
    if is_nonempty_string(raw_title):
        title_n = normalize_token(raw_title)
        artists_n = normalize_artist_key(raw_artists)
        if title_n and artists_n:
            return f"{title_n}::{artists_n}"

    return None


def make_enrichment_payload(record: dict[str, Any]) -> dict[str, Any]:
    ids = record.get("ids") or {}
    brainz = record.get("brainz") or {}
    runtime = record.get("runtime_metadata") or {}

    return {
        "track_id": record.get("track_id"),
        "title": record.get("title"),
        "artists": record.get("artists") or [],
        "album": record.get("album"),
        "year": record.get("year"),
        "duration_ms": record.get("duration_ms"),
        "ids": {
            "spotify_id": ids.get("spotify_id"),
            "spotify_uri": ids.get("spotify_uri"),
            "isrc": ids.get("isrc"),
            "musicbrainz_recording_id": ids.get("musicbrainz_recording_id"),
        },
        "brainz": {
            "tags": brainz.get("tags") or [],
            "genres": brainz.get("genres") or [],
            "acoustic_high_level": brainz.get("acoustic_high_level"),
            "acoustic_low_level": brainz.get("acoustic_low_level"),
        },
        "runtime_metadata": {
            "spotify_popularity": runtime.get("spotify_popularity"),
            "spotify_url": runtime.get("spotify_url"),
            "preview_url": runtime.get("preview_url"),
            "image_url": runtime.get("image_url"),
        },
        "aliases": {
            "normalized_title": (record.get("aliases") or {}).get("normalized_title"),
            "normalized_artists_key": (record.get("aliases") or {}).get("normalized_artists_key"),
        },
        "source": {
            "dataset_name": (record.get("source") or {}).get("dataset_name"),
            "source_files": (record.get("source") or {}).get("source_files") or [],
        },
    }


def richness_score(record: dict[str, Any]) -> tuple[int, int, int, int]:
    ids = record.get("ids") or {}
    brainz = record.get("brainz") or {}

    score = 0
    if is_nonempty_string(ids.get("musicbrainz_recording_id")):
        score += 10
    if brainz.get("tags"):
        score += 4
    if brainz.get("genres"):
        score += 3
    if isinstance(brainz.get("acoustic_high_level"), dict) and brainz.get("acoustic_high_level"):
        score += 5

    # tie breakers
    tag_count = len(brainz.get("tags") or [])
    genre_count = len(brainz.get("genres") or [])
    has_acoustic = 1 if isinstance(brainz.get("acoustic_high_level"), dict) and brainz.get("acoustic_high_level") else 0
    return (score, tag_count, genre_count, has_acoustic)


def choose_better(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return candidate
    return candidate if richness_score(candidate) > richness_score(existing) else existing


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local corpus lookup index for runtime enrichment.")
    parser.add_argument(
        "--input",
        default="ml_pipeline/data/processed/song_records.backfilled.json",
        help="Path to backfilled corpus JSON/NDJSON",
    )
    parser.add_argument(
        "--output",
        default="ml_pipeline/data/artifacts/corpus_lookup_v1.json",
        help="Path to write lookup artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        records = load_records(input_path)
    except Exception as exc:
        print(f"Failed to load corpus: {exc}", file=sys.stderr)
        return 1

    by_spotify_id: dict[str, dict[str, Any]] = {}
    by_isrc: dict[str, dict[str, Any]] = {}
    by_mbid: dict[str, dict[str, Any]] = {}
    by_alias: dict[str, dict[str, Any]] = {}

    for record in records:
        payload = make_enrichment_payload(record)
        ids = payload["ids"]

        spotify_id = ids.get("spotify_id")
        isrc = ids.get("isrc")
        mbid = ids.get("musicbrainz_recording_id")
        alias = alias_key(record)

        if is_nonempty_string(spotify_id):
            by_spotify_id[spotify_id] = choose_better(by_spotify_id.get(spotify_id), payload)
        if is_nonempty_string(isrc):
            by_isrc[isrc] = choose_better(by_isrc.get(isrc), payload)
        if is_nonempty_string(mbid):
            by_mbid[mbid] = choose_better(by_mbid.get(mbid), payload)
        if is_nonempty_string(alias):
            by_alias[alias] = choose_better(by_alias.get(alias), payload)

    artifact = {
        "schema_version": 1,
        "generated_from": str(input_path),
        "counts": {
            "records_seen": len(records),
            "spotify_id_keys": len(by_spotify_id),
            "isrc_keys": len(by_isrc),
            "mbid_keys": len(by_mbid),
            "alias_keys": len(by_alias),
        },
        "by_spotify_id": by_spotify_id,
        "by_isrc": by_isrc,
        "by_mbid": by_mbid,
        "by_alias": by_alias,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Corpus Lookup Summary ===")
    print(f"records_seen: {len(records)}")
    print(f"spotify_id_keys: {len(by_spotify_id)}")
    print(f"isrc_keys: {len(by_isrc)}")
    print(f"mbid_keys: {len(by_mbid)}")
    print(f"alias_keys: {len(by_alias)}")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())