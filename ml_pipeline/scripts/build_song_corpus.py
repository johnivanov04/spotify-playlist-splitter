#!/usr/bin/env python3
"""
Build a normalized song corpus for the offline music-model pipeline.

This script is intentionally lightweight and stdlib-only. It does not call
external APIs yet. Its job is to turn one or more exported playlist/dataset
JSON files into a stable, normalized corpus format that later training and
backend inference can rely on.

Supported input shape (current app-compatible):
- A JSON list of track objects, where each track can include fields like:
  id, uri, name, artists, album, year, durationMs, popularity, spotifyUrl,
  previewUrl, imageUrl, isrc, brainz.tags, brainz.genres,
  brainz.acousticHighLevel, brainz.acousticLowLevel, brainz.mbid

Outputs:
- song_records.ndjson  (one normalized record per line)
- song_records.json    (array form, optional)
- corpus_summary.json  (counts and diagnostics)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DATASET_VERSION = "0.1.0"


@dataclass
class NormalizeStats:
    files_seen: int = 0
    raw_tracks_seen: int = 0
    normalized_records_written: int = 0
    duplicate_records_skipped: int = 0
    missing_isrc: int = 0
    missing_mbid: int = 0
    with_brainz_tags: int = 0
    with_brainz_genres: int = 0
    with_acoustic_high_level: int = 0
    with_acoustic_low_level: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {path} ({exc})") from exc


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_isrc(value: Any) -> str | None:
    value = normalize_text(value).upper()
    if not value:
        return None
    return value


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [normalize_text(v) for v in value if normalize_text(v)]
        return out
    if isinstance(value, str) and normalize_text(value):
        return [normalize_text(value)]
    return []


def normalize_name_count_items(value: Any) -> list[dict[str, Any]]:
    """
    Accepts a few possible shapes:
    - [{"name": "indie", "count": 4}, ...]
    - ["indie", "dream pop"]
    - {"indie": 4, "dream pop": 2}
    """
    items: list[dict[str, Any]] = []

    if isinstance(value, dict):
        for name, count in value.items():
            norm_name = normalize_text(name)
            if not norm_name:
                continue
            items.append({"name": norm_name, "count": safe_float(count) or 1.0})
        return sorted(items, key=lambda x: (-x["count"], x["name"]))

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = normalize_text(item.get("name"))
                if not name:
                    continue
                count = safe_float(item.get("count"))
                if count is None:
                    count = safe_float(item.get("value"))
                items.append({"name": name, "count": count if count is not None else 1.0})
            else:
                name = normalize_text(item)
                if name:
                    items.append({"name": name, "count": 1.0})
        items.sort(key=lambda x: (-x["count"], x["name"]))
        return items

    return []


def normalized_artists_key(artists: Iterable[str]) -> str:
    return "|".join(slugify(a) for a in artists if slugify(a))


def track_fallback_key(title: str, artists: list[str]) -> str:
    return f"{slugify(title)}__{normalized_artists_key(artists)}"


def choose_track_id(ids: dict[str, Any], title: str, artists: list[str]) -> str:
    isrc = ids.get("isrc")
    if isrc:
        return f"isrc:{isrc}"

    mbid = ids.get("musicbrainz_recording_id")
    if mbid:
        return f"mbid:{mbid}"

    spotify_id = ids.get("spotify_id")
    if spotify_id:
        return f"spotify:{spotify_id}"

    return f"fallback:{track_fallback_key(title, artists)}"


def extract_mbid(track: dict[str, Any]) -> str | None:
    brainz = track.get("brainz") or {}
    for key in ("mbid", "recordingId", "musicbrainz_recording_id"):
        value = normalize_text(brainz.get(key))
        if value:
            return value
    ids = track.get("ids") or {}
    value = normalize_text(ids.get("musicbrainz_recording_id"))
    return value or None


def normalize_track(track: dict[str, Any], dataset_name: str, source_file: str, include_debug: bool) -> dict[str, Any]:
    title = normalize_text(track.get("name") or track.get("title"))
    artists = ensure_list_of_strings(track.get("artists"))
    album = normalize_text(track.get("album")) or None
    year = safe_int(track.get("year"))
    duration_ms = safe_int(track.get("durationMs") or track.get("duration_ms"))

    ids = {
        "spotify_id": normalize_text(track.get("id")) or None,
        "spotify_uri": normalize_text(track.get("uri")) or None,
        "isrc": normalize_isrc(track.get("isrc")),
        "musicbrainz_recording_id": extract_mbid(track),
    }

    brainz = track.get("brainz") or {}
    tags = normalize_name_count_items(brainz.get("tags"))
    genres = normalize_name_count_items(brainz.get("genres"))
    acoustic_high = brainz.get("acousticHighLevel")
    acoustic_low = brainz.get("acousticLowLevel")

    record = {
        "schema_version": SCHEMA_VERSION,
        "track_id": choose_track_id(ids, title, artists),
        "title": title,
        "artists": artists,
        "album": album,
        "year": year,
        "duration_ms": duration_ms,
        "ids": ids,
        "brainz": {
            "tags": tags,
            "genres": genres,
            "acoustic_high_level": acoustic_high if isinstance(acoustic_high, dict) else None,
            "acoustic_low_level": acoustic_low if isinstance(acoustic_low, dict) else None,
        },
        "runtime_metadata": {
            "spotify_popularity": safe_float(track.get("popularity")),
            "spotify_url": normalize_text(track.get("spotifyUrl")) or None,
            "preview_url": normalize_text(track.get("previewUrl")) or None,
            "image_url": normalize_text(track.get("imageUrl")) or None,
        },
        "audio_embeddings": {
            "model_name": None,
            "vector": None,
        },
        "source": {
            "dataset_name": dataset_name,
            "dataset_version": DATASET_VERSION,
            "source_files": [source_file],
            "ingested_at_utc": utc_now_iso(),
        },
        "aliases": {
            "normalized_title": slugify(title),
            "normalized_artists_key": normalized_artists_key(artists),
        },
        "debug": {
            "raw_track_id": track.get("id"),
            "raw_track_name": track.get("name"),
        } if include_debug else None,
    }

    if not record["title"]:
        raise ValueError("Track missing title/name")
    if not record["artists"]:
        raise ValueError(f"Track '{record['title']}' missing artists")

    return record


def iter_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif any(ch in raw for ch in "*?[]"):
            files.extend(sorted(Path().glob(raw)))
        else:
            files.append(p)
    deduped = []
    seen = set()
    for f in files:
        resolved = str(f.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(f)
    return deduped


def dedupe_key(record: dict[str, Any]) -> str:
    ids = record["ids"]
    return (
        ids.get("isrc")
        or ids.get("musicbrainz_recording_id")
        or ids.get("spotify_id")
        or record["track_id"]
    )


def build_corpus(files: list[Path], dataset_name: str, include_debug: bool) -> tuple[list[dict[str, Any]], NormalizeStats, dict[str, int]]:
    stats = NormalizeStats()
    records: list[dict[str, Any]] = []
    seen = set()
    tag_counter: Counter[str] = Counter()
    genre_counter: Counter[str] = Counter()

    for path in files:
        stats.files_seen += 1
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Input must be a JSON list of tracks: {path}")

        for track in payload:
            stats.raw_tracks_seen += 1
            record = normalize_track(
                track=track,
                dataset_name=dataset_name,
                source_file=path.name,
                include_debug=include_debug,
            )

            key = dedupe_key(record)
            if key in seen:
                stats.duplicate_records_skipped += 1
                continue
            seen.add(key)

            if not record["ids"]["isrc"]:
                stats.missing_isrc += 1
            if not record["ids"]["musicbrainz_recording_id"]:
                stats.missing_mbid += 1
            if record["brainz"]["tags"]:
                stats.with_brainz_tags += 1
            if record["brainz"]["genres"]:
                stats.with_brainz_genres += 1
            if record["brainz"]["acoustic_high_level"] is not None:
                stats.with_acoustic_high_level += 1
            if record["brainz"]["acoustic_low_level"] is not None:
                stats.with_acoustic_low_level += 1

            for item in record["brainz"]["tags"]:
                tag_counter[item["name"]] += 1
            for item in record["brainz"]["genres"]:
                genre_counter[item["name"]] += 1

            records.append(record)
            stats.normalized_records_written += 1

    vocab_summary = {
        "unique_tags": len(tag_counter),
        "unique_genres": len(genre_counter),
        "top_tags": tag_counter.most_common(25),
        "top_genres": genre_counter.most_common(25),
    }
    return records, stats, vocab_summary


def write_outputs(records: list[dict[str, Any]], stats: NormalizeStats, vocab_summary: dict[str, Any], out_dir: Path, write_array_json: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ndjson_path = out_dir / "song_records.ndjson"
    with ndjson_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if write_array_json:
        (out_dir / "song_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": utc_now_iso(),
        "counts": {
            "files_seen": stats.files_seen,
            "raw_tracks_seen": stats.raw_tracks_seen,
            "normalized_records_written": stats.normalized_records_written,
            "duplicate_records_skipped": stats.duplicate_records_skipped,
            "missing_isrc": stats.missing_isrc,
            "missing_mbid": stats.missing_mbid,
            "with_brainz_tags": stats.with_brainz_tags,
            "with_brainz_genres": stats.with_brainz_genres,
            "with_acoustic_high_level": stats.with_acoustic_high_level,
            "with_acoustic_low_level": stats.with_acoustic_low_level,
        },
        "vocabulary": vocab_summary,
    }
    (out_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized song corpus from exported track JSON files.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input JSON files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--dataset-name",
        default="spotify-playlist-splitter-bootstrap",
        help="Human-readable name for the dataset recorded in source metadata.",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_pipeline/data/processed",
        help="Directory to write song_records.ndjson and summary files.",
    )
    parser.add_argument(
        "--write-array-json",
        action="store_true",
        help="Also write song_records.json as a JSON array for easier inspection.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include a small debug payload on each record.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    files = iter_input_files(args.input)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 1

    records, stats, vocab_summary = build_corpus(
        files=files,
        dataset_name=args.dataset_name,
        include_debug=args.include_debug,
    )
    write_outputs(
        records=records,
        stats=stats,
        vocab_summary=vocab_summary,
        out_dir=Path(args.out_dir),
        write_array_json=args.write_array_json,
    )

    print(f"Built corpus from {stats.files_seen} file(s).")
    print(f"Raw tracks seen: {stats.raw_tracks_seen}")
    print(f"Normalized records written: {stats.normalized_records_written}")
    print(f"Duplicates skipped: {stats.duplicate_records_skipped}")
    print(f"Missing ISRC: {stats.missing_isrc}")
    print(f"Missing MBID: {stats.missing_mbid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
