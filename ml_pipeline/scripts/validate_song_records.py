#!/usr/bin/env python3
"""
Validate normalized song records produced by build_song_corpus.py.

This is a lightweight stdlib-only validator. It does not require jsonschema.
Its purpose is to catch bad record shapes early and give a clean summary before
training or backend inference code depends on the corpus.

Supported inputs:
- song_records.ndjson
- song_records.json (JSON array)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def is_string_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)


def is_int_or_none(value: Any) -> bool:
    return value is None or isinstance(value, int)


def is_number_or_none(value: Any) -> bool:
    return value is None or isinstance(value, (int, float))


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if path.suffix.lower() == ".ndjson":
        rows = []
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


def validate_name_count_list(
    value: Any,
    field_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return

    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{field_name}[{i}] must be an object")
            continue

        name = item.get("name")
        count = item.get("count")

        if not is_nonempty_string(name):
            errors.append(f"{field_name}[{i}].name must be a non-empty string")

        if not isinstance(count, (int, float)):
            errors.append(f"{field_name}[{i}].count must be numeric")
        elif count < 0:
            warnings.append(f"{field_name}[{i}].count is negative")


def validate_record(record: dict[str, Any], index: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    # top-level required shape
    if not isinstance(record.get("schema_version"), int):
        errors.append("schema_version must be an integer")

    if not is_nonempty_string(record.get("track_id")):
        errors.append("track_id must be a non-empty string")

    if not is_nonempty_string(record.get("title")):
        errors.append("title must be a non-empty string")

    artists = record.get("artists")
    if not isinstance(artists, list) or not artists:
        errors.append("artists must be a non-empty list")
    else:
        for i, artist in enumerate(artists):
            if not is_nonempty_string(artist):
                errors.append(f"artists[{i}] must be a non-empty string")

    if not is_string_or_none(record.get("album")):
        errors.append("album must be a string or null")

    if not is_int_or_none(record.get("year")):
        errors.append("year must be an integer or null")

    if not is_int_or_none(record.get("duration_ms")):
        errors.append("duration_ms must be an integer or null")

    # ids
    ids = record.get("ids")
    if not isinstance(ids, dict):
        errors.append("ids must be an object")
    else:
        for key in ("spotify_id", "spotify_uri", "isrc", "musicbrainz_recording_id"):
            if not is_string_or_none(ids.get(key)):
                errors.append(f"ids.{key} must be a string or null")

        if not ids.get("isrc") and not ids.get("musicbrainz_recording_id") and not ids.get("spotify_id"):
            warnings.append("record has no canonical ID (isrc, mbid, or spotify_id)")

    # brainz
    brainz = record.get("brainz")
    if not isinstance(brainz, dict):
        errors.append("brainz must be an object")
    else:
        validate_name_count_list(brainz.get("tags"), "brainz.tags", errors, warnings)
        validate_name_count_list(brainz.get("genres"), "brainz.genres", errors, warnings)

        ah = brainz.get("acoustic_high_level")
        if ah is not None and not isinstance(ah, dict):
            errors.append("brainz.acoustic_high_level must be an object or null")

        al = brainz.get("acoustic_low_level")
        if al is not None and not isinstance(al, dict):
            errors.append("brainz.acoustic_low_level must be an object or null")

    # runtime_metadata
    runtime_metadata = record.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        errors.append("runtime_metadata must be an object")
    else:
        if not is_number_or_none(runtime_metadata.get("spotify_popularity")):
            errors.append("runtime_metadata.spotify_popularity must be numeric or null")
        for key in ("spotify_url", "preview_url", "image_url"):
            if not is_string_or_none(runtime_metadata.get(key)):
                errors.append(f"runtime_metadata.{key} must be a string or null")

    # audio_embeddings
    audio_embeddings = record.get("audio_embeddings")
    if not isinstance(audio_embeddings, dict):
        errors.append("audio_embeddings must be an object")
    else:
        if not is_string_or_none(audio_embeddings.get("model_name")):
            errors.append("audio_embeddings.model_name must be a string or null")

        vector = audio_embeddings.get("vector")
        if vector is not None:
            if not isinstance(vector, list):
                errors.append("audio_embeddings.vector must be a list or null")
            else:
                for i, v in enumerate(vector):
                    if not isinstance(v, (int, float)):
                        errors.append(f"audio_embeddings.vector[{i}] must be numeric")

    # source
    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        if not is_nonempty_string(source.get("dataset_name")):
            errors.append("source.dataset_name must be a non-empty string")

        if not is_nonempty_string(source.get("dataset_version")):
            errors.append("source.dataset_version must be a non-empty string")

        if not is_string_or_none(source.get("ingested_at_utc")):
            errors.append("source.ingested_at_utc must be a string or null")

        source_files = source.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            errors.append("source.source_files must be a non-empty list")
        else:
            for i, sf in enumerate(source_files):
                if not is_nonempty_string(sf):
                    errors.append(f"source.source_files[{i}] must be a non-empty string")

    # aliases
    aliases = record.get("aliases")
    if not isinstance(aliases, dict):
        errors.append("aliases must be an object")
    else:
        for key in ("normalized_title", "normalized_artists_key"):
            if not is_string_or_none(aliases.get(key)):
                errors.append(f"aliases.{key} must be a string or null")

    # debug is optional
    debug = record.get("debug")
    if debug is not None and not isinstance(debug, dict):
        errors.append("debug must be an object or null")

    return errors, warnings


def summarize_quality(records: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "records": len(records),
        "with_isrc": 0,
        "with_mbid": 0,
        "with_tags": 0,
        "with_genres": 0,
        "with_acoustic_high_level": 0,
        "with_audio_embedding": 0,
    }

    for r in records:
        ids = r.get("ids") or {}
        brainz = r.get("brainz") or {}
        audio = r.get("audio_embeddings") or {}

        if ids.get("isrc"):
            out["with_isrc"] += 1
        if ids.get("musicbrainz_recording_id"):
            out["with_mbid"] += 1
        if brainz.get("tags"):
            out["with_tags"] += 1
        if brainz.get("genres"):
            out["with_genres"] += 1
        if isinstance(brainz.get("acoustic_high_level"), dict) and brainz.get("acoustic_high_level"):
            out["with_acoustic_high_level"] += 1
        if isinstance(audio.get("vector"), list) and audio.get("vector"):
            out["with_audio_embedding"] += 1

    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate normalized song record corpus.")
    parser.add_argument(
        "--input",
        default="ml_pipeline/data/processed/song_records.json",
        help="Path to song_records.json or song_records.ndjson",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=25,
        help="Maximum number of detailed errors/warnings to print",
    )
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional path to write a validation report JSON",
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

    total_errors = 0
    total_warnings = 0
    printed = 0
    valid_records = 0

    detailed_issues: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        errors, warnings = validate_record(record, idx)

        if not errors:
            valid_records += 1

        total_errors += len(errors)
        total_warnings += len(warnings)

        if errors or warnings:
            detailed_issues.append(
                {
                    "index": idx,
                    "track_id": record.get("track_id"),
                    "title": record.get("title"),
                    "errors": errors,
                    "warnings": warnings,
                }
            )

        if printed < args.max_errors and (errors or warnings):
            print(f"\nRecord {idx}: {record.get('title')!r} ({record.get('track_id')})")
            for err in errors:
                if printed >= args.max_errors:
                    break
                print(f"  ERROR: {err}")
                printed += 1
            for warn in warnings:
                if printed >= args.max_errors:
                    break
                print(f"  WARN:  {warn}")
                printed += 1

    quality = summarize_quality(records)

    report = {
        "input": str(path),
        "records_loaded": len(records),
        "valid_records": valid_records,
        "records_with_any_issue": len(detailed_issues),
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "quality_summary": quality,
        "sample_issues": detailed_issues[:100],
    }

    print("\n=== Validation Summary ===")
    print(f"Records loaded: {len(records)}")
    print(f"Valid records: {valid_records}")
    print(f"Records with any issue: {len(detailed_issues)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    print("\n=== Quality Summary ===")
    for k, v in quality.items():
        print(f"{k}: {v}")

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote validation report to: {out_path}")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())