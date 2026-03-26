# ml_pipeline/scripts/inspect_corpus.py
#!/usr/bin/env python3
"""
Inspect a normalized song corpus and print quality / coverage stats.

Supports:
- JSON array input (song_records.json)
- NDJSON input (song_records.ndjson)

Typical usage:
    python3 ml_pipeline/scripts/inspect_corpus.py \
      --input ml_pipeline/data/processed/song_records.json

    python3 ml_pipeline/scripts/inspect_corpus.py \
      --input ml_pipeline/data/processed/song_records.json \
      --write-report ml_pipeline/data/processed/inspection_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("JSON input must be an array of record objects")

    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            raise ValueError(f"JSON array item {i} is not an object")

    return payload


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 2)


def nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def canonical_key(record: dict[str, Any]) -> str:
    ids = record.get("ids") or {}
    aliases = record.get("aliases") or {}

    isrc = ids.get("isrc")
    mbid = ids.get("musicbrainz_recording_id")
    spotify_id = ids.get("spotify_id")
    normalized_title = aliases.get("normalized_title")
    normalized_artists_key = aliases.get("normalized_artists_key")

    if nonempty_str(isrc):
        return f"isrc:{isrc.strip().upper()}"
    if nonempty_str(mbid):
        return f"mbid:{mbid.strip()}"
    if nonempty_str(spotify_id):
        return f"spotify:{spotify_id.strip()}"
    if nonempty_str(normalized_title) and nonempty_str(normalized_artists_key):
        return f"alias:{normalized_title.strip()}::{normalized_artists_key.strip()}"
    return f"track_id:{record.get('track_id')}"


def summarize(records: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    total = len(records)

    tag_counter: Counter[str] = Counter()
    genre_counter: Counter[str] = Counter()
    source_file_counter: Counter[str] = Counter()
    dataset_counter: Counter[str] = Counter()
    year_counter: Counter[int] = Counter()
    artist_counter: Counter[str] = Counter()

    unique_canonical_keys: set[str] = set()

    with_isrc = 0
    with_mbid = 0
    with_spotify_id = 0
    with_tags = 0
    with_genres = 0
    with_acoustic_high_level = 0
    with_acoustic_low_level = 0
    with_audio_embedding = 0
    with_popularity = 0
    with_year = 0

    missing_all_ids = 0
    missing_brainz = 0

    for r in records:
        ids = r.get("ids") or {}
        brainz = r.get("brainz") or {}
        runtime_metadata = r.get("runtime_metadata") or {}
        audio_embeddings = r.get("audio_embeddings") or {}
        source = r.get("source") or {}

        unique_canonical_keys.add(canonical_key(r))

        isrc = ids.get("isrc")
        mbid = ids.get("musicbrainz_recording_id")
        spotify_id = ids.get("spotify_id")

        if nonempty_str(isrc):
            with_isrc += 1
        if nonempty_str(mbid):
            with_mbid += 1
        if nonempty_str(spotify_id):
            with_spotify_id += 1
        if not any(nonempty_str(x) for x in (isrc, mbid, spotify_id)):
            missing_all_ids += 1

        tags = brainz.get("tags") or []
        genres = brainz.get("genres") or []
        acoustic_high_level = brainz.get("acoustic_high_level")
        acoustic_low_level = brainz.get("acoustic_low_level")

        if tags:
            with_tags += 1
            for item in tags:
                if isinstance(item, dict):
                    name = item.get("name")
                    count = item.get("count", 1)
                    if nonempty_str(name):
                        try:
                            weight = float(count)
                        except Exception:
                            weight = 1.0
                        tag_counter[name.strip().lower()] += weight

        if genres:
            with_genres += 1
            for item in genres:
                if isinstance(item, dict):
                    name = item.get("name")
                    count = item.get("count", 1)
                    if nonempty_str(name):
                        try:
                            weight = float(count)
                        except Exception:
                            weight = 1.0
                        genre_counter[name.strip().lower()] += weight

        if isinstance(acoustic_high_level, dict) and acoustic_high_level:
            with_acoustic_high_level += 1
        if isinstance(acoustic_low_level, dict) and acoustic_low_level:
            with_acoustic_low_level += 1

        if not tags and not genres and not (
            isinstance(acoustic_high_level, dict) and acoustic_high_level
        ):
            missing_brainz += 1

        vector = audio_embeddings.get("vector")
        if isinstance(vector, list) and len(vector) > 0:
            with_audio_embedding += 1

        popularity = runtime_metadata.get("spotify_popularity")
        if isinstance(popularity, (int, float)):
            with_popularity += 1

        year = r.get("year")
        if isinstance(year, int):
            with_year += 1
            year_counter[year] += 1

        artists = r.get("artists") or []
        for artist in artists:
            if nonempty_str(artist):
                artist_counter[artist.strip()] += 1

        dataset_name = source.get("dataset_name")
        if nonempty_str(dataset_name):
            dataset_counter[dataset_name.strip()] += 1

        for sf in source.get("source_files") or []:
            if nonempty_str(sf):
                source_file_counter[sf.strip()] += 1

    duplicate_estimate = total - len(unique_canonical_keys)

    coverage = {
        "records": total,
        "unique_canonical_records_estimate": len(unique_canonical_keys),
        "duplicate_estimate": duplicate_estimate,
        "with_isrc": with_isrc,
        "with_isrc_pct": pct(with_isrc, total),
        "with_mbid": with_mbid,
        "with_mbid_pct": pct(with_mbid, total),
        "with_spotify_id": with_spotify_id,
        "with_spotify_id_pct": pct(with_spotify_id, total),
        "with_tags": with_tags,
        "with_tags_pct": pct(with_tags, total),
        "with_genres": with_genres,
        "with_genres_pct": pct(with_genres, total),
        "with_acoustic_high_level": with_acoustic_high_level,
        "with_acoustic_high_level_pct": pct(with_acoustic_high_level, total),
        "with_acoustic_low_level": with_acoustic_low_level,
        "with_acoustic_low_level_pct": pct(with_acoustic_low_level, total),
        "with_audio_embedding": with_audio_embedding,
        "with_audio_embedding_pct": pct(with_audio_embedding, total),
        "with_popularity": with_popularity,
        "with_popularity_pct": pct(with_popularity, total),
        "with_year": with_year,
        "with_year_pct": pct(with_year, total),
        "missing_all_ids": missing_all_ids,
        "missing_all_ids_pct": pct(missing_all_ids, total),
        "missing_brainz": missing_brainz,
        "missing_brainz_pct": pct(missing_brainz, total),
    }

    year_min = min(year_counter.keys()) if year_counter else None
    year_max = max(year_counter.keys()) if year_counter else None

    return {
        "coverage": coverage,
        "vocabulary": {
            "unique_tags": len(tag_counter),
            "unique_genres": len(genre_counter),
            "top_tags": tag_counter.most_common(top_n),
            "top_genres": genre_counter.most_common(top_n),
            "top_artists": artist_counter.most_common(top_n),
            "top_years": year_counter.most_common(top_n),
            "year_min": year_min,
            "year_max": year_max,
        },
        "sources": {
            "dataset_names": dataset_counter.most_common(),
            "source_files": source_file_counter.most_common(top_n),
            "source_file_count": len(source_file_counter),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    coverage = report["coverage"]
    vocab = report["vocabulary"]
    sources = report["sources"]

    print("=== Corpus Coverage ===")
    for key in [
        "records",
        "unique_canonical_records_estimate",
        "duplicate_estimate",
        "with_isrc",
        "with_isrc_pct",
        "with_mbid",
        "with_mbid_pct",
        "with_spotify_id",
        "with_spotify_id_pct",
        "with_tags",
        "with_tags_pct",
        "with_genres",
        "with_genres_pct",
        "with_acoustic_high_level",
        "with_acoustic_high_level_pct",
        "with_acoustic_low_level",
        "with_acoustic_low_level_pct",
        "with_audio_embedding",
        "with_audio_embedding_pct",
        "with_popularity",
        "with_popularity_pct",
        "with_year",
        "with_year_pct",
        "missing_all_ids",
        "missing_all_ids_pct",
        "missing_brainz",
        "missing_brainz_pct",
    ]:
        print(f"{key}: {coverage[key]}")

    print("\n=== Vocabulary ===")
    print(f"unique_tags: {vocab['unique_tags']}")
    print(f"unique_genres: {vocab['unique_genres']}")
    print(f"year_min: {vocab['year_min']}")
    print(f"year_max: {vocab['year_max']}")

    print("\nTop tags:")
    for name, count in vocab["top_tags"]:
        print(f"  {name}: {count}")

    print("\nTop genres:")
    for name, count in vocab["top_genres"]:
        print(f"  {name}: {count}")

    print("\nTop artists:")
    for name, count in vocab["top_artists"]:
        print(f"  {name}: {count}")

    print("\nTop years:")
    for year, count in vocab["top_years"]:
        print(f"  {year}: {count}")

    print("\n=== Sources ===")
    print(f"source_file_count: {sources['source_file_count']}")
    print("dataset_names:")
    for name, count in sources["dataset_names"]:
        print(f"  {name}: {count}")

    print("source_files:")
    for name, count in sources["source_files"]:
        print(f"  {name}: {count}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect normalized song corpus quality.")
    parser.add_argument(
        "--input",
        default="ml_pipeline/data/processed/song_records.json",
        help="Path to song_records.json or song_records.ndjson",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many top items to show for tags / genres / artists / years",
    )
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional path to write a JSON report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    path = Path(args.input)

    if not path.exists():
        print(f"Input file not found: {path}", file=sys.stderr)
        return 1

    try:
        records = load_records(path)
    except Exception as exc:
        print(f"Failed to load records: {exc}", file=sys.stderr)
        return 1

    report = summarize(records, top_n=args.top_n)
    print_summary(report)

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote inspection report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())