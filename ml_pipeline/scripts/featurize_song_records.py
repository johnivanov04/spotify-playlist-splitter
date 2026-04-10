#!/usr/bin/env python3
"""
Featurize normalized/backfilled song records into a dense numeric matrix.

Design goals:
- sound / vibe / mood features should dominate
- genre should help somewhat
- year and popularity should help only a little
- noisy chart/popularity tags should be excluded
- redundant acoustic complement/probability fields should be excluded

Inputs:
- song_records.json
- song_records.ndjson

Outputs:
- feature_matrix.npz
- feature_names.json
- row_index.json
- featurizer_report.json

Recommended usage:
    python ml_pipeline/scripts/featurize_song_records.py \
      --input ml_pipeline/data/processed/song_records.backfilled.json \
      --out-dir ml_pipeline/data/artifacts/features_v1_1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ----------------------------
# loading
# ----------------------------

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


# ----------------------------
# normalization helpers
# ----------------------------

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_token(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("&", " and ")
    value = _TOKEN_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def canonical_key(record: dict[str, Any]) -> str:
    ids = record.get("ids") or {}
    aliases = record.get("aliases") or {}

    isrc = ids.get("isrc")
    mbid = ids.get("musicbrainz_recording_id")
    spotify_id = ids.get("spotify_id")
    normalized_title = aliases.get("normalized_title")
    normalized_artists_key = aliases.get("normalized_artists_key")

    if is_nonempty_string(isrc):
        return f"isrc:{isrc.strip().upper()}"
    if is_nonempty_string(mbid):
        return f"mbid:{mbid.strip()}"
    if is_nonempty_string(spotify_id):
        return f"spotify:{spotify_id.strip()}"
    if is_nonempty_string(normalized_title) and is_nonempty_string(normalized_artists_key):
        return f"alias:{normalized_title.strip()}::{normalized_artists_key.strip()}"
    return f"track_id:{record.get('track_id')}"


# ----------------------------
# feature filtering rules
# ----------------------------

_SKIP_TAG_SUBSTRINGS = {
    "billboard",
    "hot_100",
    "chart",
    "charts",
    "offizielle",
}

_SKIP_TAG_EXACT = {
    "american",
    "american_rock",
    "english",
    "tempo_change",
    "vocal",
    "rap_hip_hop",
    "hip_hop_rap",
    "hip_hop_underground_hip_hop",
    "contemporary_rap_gangsta_rap_hardcore_rap_rap_west_coast_rap",
}

_TAG_ALIAS_MAP = {
    "rhythm_and_blues": "r_and_b",
    "westcoast_rap": "west_coast_hip_hop",
    "west_coast_rap": "west_coast_hip_hop",
}

_SKIP_ACOUSTIC_KEYS = {
    "metadata",
    "version",
    "versions",
    "recording_id",
    "mbid",
    "analysis_time",
    "timestamp",
}

_SKIP_ACOUSTIC_SUBSTRINGS = {
    "__all__not_",
    "__gender__",
    "__genre_dortmund__",
    "__genre_rosamerica__",
    "__genre_tzanetakis__",
    "__genre_electronic__",
    "__ismir04_rhythm__",
    "__moods_mirex__",
    "__tonal_atonal__",
}

_SKIP_ACOUSTIC_SUFFIXES = {
    "__probability",
}


def canonicalize_tag_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    return _TAG_ALIAS_MAP.get(token, token)



def keep_tag_token(token: str) -> bool:
    token = canonicalize_tag_token(token)
    if not token:
        return False
    if token in _SKIP_TAG_EXACT:
        return False
    for bad in _SKIP_TAG_SUBSTRINGS:
        if bad in token:
            return False
    return True


def keep_acoustic_feature(name: str) -> bool:
    if not name:
        return False

    for bad in _SKIP_ACOUSTIC_SUBSTRINGS:
        if bad in name:
            return False

    for suffix in _SKIP_ACOUSTIC_SUFFIXES:
        if name.endswith(suffix):
            return False

    return True


# ----------------------------
# field extraction
# ----------------------------

def iter_name_count_items(items: Any) -> Iterable[tuple[str, float]]:
    if not isinstance(items, list):
        return

    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        count = item.get("count", 1)
        if not is_nonempty_string(name):
            continue

        token = canonicalize_tag_token(normalize_token(name))
        if not keep_tag_token(token):
            continue

        try:
            weight = float(count)
        except Exception:
            weight = 1.0
        if weight < 0:
            continue

        yield token, weight


def iter_acoustic_numeric_leaves(obj: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    """
    Recursively walk acoustic_high_level and yield numeric leaf features.

    We keep numeric leaves and ignore string leaves like "value": "danceable".
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).strip()
            if not key:
                continue
            if key.lower() in _SKIP_ACOUSTIC_KEYS:
                continue

            next_prefix = f"{prefix}__{normalize_token(key)}" if prefix else normalize_token(key)

            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if keep_acoustic_feature(next_prefix):
                    yield next_prefix, float(v)
            elif isinstance(v, (dict, list)):
                yield from iter_acoustic_numeric_leaves(v, next_prefix)
            else:
                continue

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            next_prefix = f"{prefix}__{i}" if prefix else str(i)

            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if keep_acoustic_feature(next_prefix):
                    yield next_prefix, float(v)
            elif isinstance(v, (dict, list)):
                yield from iter_acoustic_numeric_leaves(v, next_prefix)


def get_year(record: dict[str, Any]) -> int | None:
    year = record.get("year")
    return year if isinstance(year, int) else None


def get_popularity(record: dict[str, Any]) -> float | None:
    runtime = record.get("runtime_metadata") or {}
    value = runtime.get("spotify_popularity")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


# ----------------------------
# vocab building
# ----------------------------

def select_vocab(counter: Counter[str], min_df: int, max_features: int | None) -> list[str]:
    items = [(k, v) for k, v in counter.items() if v >= min_df]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_features is not None:
        items = items[:max_features]
    return [k for k, _ in items]


def build_vocabs(
    records: list[dict[str, Any]],
    min_tag_df: int,
    min_genre_df: int,
    min_acoustic_df: int,
    max_tags: int | None,
    max_genres: int | None,
    max_acoustic: int | None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    tag_df: Counter[str] = Counter()
    genre_df: Counter[str] = Counter()
    acoustic_df: Counter[str] = Counter()

    for record in records:
        brainz = record.get("brainz") or {}

        tag_keys = {name for name, _ in iter_name_count_items(brainz.get("tags"))}
        genre_keys = {name for name, _ in iter_name_count_items(brainz.get("genres"))}

        acoustic_keys = set()
        acoustic = brainz.get("acoustic_high_level")
        if isinstance(acoustic, dict):
            for name, value in iter_acoustic_numeric_leaves(acoustic):
                if isinstance(value, float):
                    acoustic_keys.add(name)

        tag_df.update(tag_keys)
        genre_df.update(genre_keys)
        acoustic_df.update(acoustic_keys)

    tag_vocab = select_vocab(tag_df, min_df=min_tag_df, max_features=max_tags)
    genre_vocab = select_vocab(genre_df, min_df=min_genre_df, max_features=max_genres)
    acoustic_vocab = select_vocab(acoustic_df, min_df=min_acoustic_df, max_features=max_acoustic)

    vocab_report = {
        "tag_vocab_size": len(tag_vocab),
        "genre_vocab_size": len(genre_vocab),
        "acoustic_vocab_size": len(acoustic_vocab),
        "top_tag_df": tag_df.most_common(25),
        "top_genre_df": genre_df.most_common(25),
        "top_acoustic_df": acoustic_df.most_common(25),
    }

    return tag_vocab, genre_vocab, acoustic_vocab, vocab_report


# ----------------------------
# matrix building
# ----------------------------

def compute_year_stats(records: list[dict[str, Any]]) -> tuple[float, float]:
    years = [y for r in records if (y := get_year(r)) is not None]
    if not years:
        return 2000.0, 1.0
    mean = float(sum(years) / len(years))
    var = float(sum((y - mean) ** 2 for y in years) / len(years))
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return mean, std


def build_feature_names(
    tag_vocab: list[str],
    genre_vocab: list[str],
    acoustic_vocab: list[str],
) -> list[str]:
    base = [
        "meta__has_tags",
        "meta__has_genres",
        "meta__has_acoustic_high_level",
        "meta__has_year",
        "meta__has_popularity",
        "meta__year_z",
        "meta__popularity_scaled",
    ]
    tag_features = [f"tag__{t}" for t in tag_vocab]
    genre_features = [f"genre__{g}" for g in genre_vocab]
    acoustic_features = [f"acoustic__{a}" for a in acoustic_vocab]
    return base + tag_features + genre_features + acoustic_features


def featurize_records(
    records: list[dict[str, Any]],
    tag_vocab: list[str],
    genre_vocab: list[str],
    acoustic_vocab: list[str],
    *,
    tag_weight: float,
    genre_weight: float,
    acoustic_weight: float,
    year_weight: float,
    popularity_weight: float,
    binary_weight: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    year_mean, year_std = compute_year_stats(records)

    feature_names = build_feature_names(tag_vocab, genre_vocab, acoustic_vocab)
    n_rows = len(records)
    n_cols = len(feature_names)

    X = np.zeros((n_rows, n_cols), dtype=np.float32)
    row_index: list[dict[str, Any]] = []

    tag_offset = 7
    genre_offset = tag_offset + len(tag_vocab)
    acoustic_offset = genre_offset + len(genre_vocab)

    tag_index = {name: i for i, name in enumerate(tag_vocab)}
    genre_index = {name: i for i, name in enumerate(genre_vocab)}
    acoustic_index = {name: i for i, name in enumerate(acoustic_vocab)}

    nonzero_rows = 0

    for row_i, record in enumerate(records):
        brainz = record.get("brainz") or {}
        ids = record.get("ids") or {}

        tags_present = False
        genres_present = False
        acoustic_present = False

        tags = list(iter_name_count_items(brainz.get("tags")))
        if tags:
            tags_present = True
            X[row_i, 0] = binary_weight

        genres = list(iter_name_count_items(brainz.get("genres")))
        if genres:
            genres_present = True
            X[row_i, 1] = binary_weight

        acoustic_vals: list[tuple[str, float]] = []
        acoustic_obj = brainz.get("acoustic_high_level")
        if isinstance(acoustic_obj, dict):
            acoustic_vals = list(iter_acoustic_numeric_leaves(acoustic_obj))
            if acoustic_vals:
                acoustic_present = True
                X[row_i, 2] = binary_weight

        year = get_year(record)
        if year is not None:
            X[row_i, 3] = binary_weight
            X[row_i, 5] = np.float32(((year - year_mean) / year_std) * year_weight)

        popularity = get_popularity(record)
        if popularity is not None:
            X[row_i, 4] = binary_weight
            X[row_i, 6] = np.float32((popularity / 100.0) * popularity_weight)

        for name, count in tags:
            idx = tag_index.get(name)
            if idx is None:
                continue
            value = math.log1p(max(count, 0.0)) * tag_weight
            X[row_i, tag_offset + idx] += np.float32(value)

        for name, count in genres:
            idx = genre_index.get(name)
            if idx is None:
                continue
            value = math.log1p(max(count, 0.0)) * genre_weight
            X[row_i, genre_offset + idx] += np.float32(value)

        for name, value in acoustic_vals:
            idx = acoustic_index.get(name)
            if idx is None:
                continue
            clipped = float(max(min(value, 1.0), -1.0))
            X[row_i, acoustic_offset + idx] += np.float32(clipped * acoustic_weight)

        if np.count_nonzero(X[row_i]) > 0:
            nonzero_rows += 1

        row_index.append(
            {
                "row": row_i,
                "track_id": record.get("track_id"),
                "canonical_key": canonical_key(record),
                "title": record.get("title"),
                "artists": record.get("artists") or [],
                "year": record.get("year"),
                "spotify_id": ids.get("spotify_id"),
                "spotify_uri": ids.get("spotify_uri"),
                "isrc": ids.get("isrc"),
                "musicbrainz_recording_id": ids.get("musicbrainz_recording_id"),
                "source_files": (record.get("source") or {}).get("source_files") or [],
                "has_tags": tags_present,
                "has_genres": genres_present,
                "has_acoustic_high_level": acoustic_present,
            }
        )

    matrix_report = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "dtype": str(X.dtype),
        "nonzero_rows": int(nonzero_rows),
        "nonzero_rows_pct": round((nonzero_rows / len(records)) * 100.0, 2) if records else 0.0,
        "year_mean": year_mean,
        "year_std": year_std,
    }

    return X, row_index, matrix_report


# ----------------------------
# cli
# ----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Featurize song records into a dense matrix.")

    parser.add_argument(
        "--input",
        default="ml_pipeline/data/processed/song_records.backfilled.json",
        help="Path to song_records.backfilled.json or .ndjson",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_pipeline/data/artifacts/features_v1_1",
        help="Directory to write feature artifacts into",
    )

    parser.add_argument("--min-tag-df", type=int, default=2)
    parser.add_argument("--min-genre-df", type=int, default=2)
    parser.add_argument("--min-acoustic-df", type=int, default=5)
    parser.add_argument("--max-tags", type=int, default=500)
    parser.add_argument("--max-genres", type=int, default=300)
    parser.add_argument("--max-acoustic", type=int, default=1000)

    parser.add_argument("--tag-weight", type=float, default=1.0)
    parser.add_argument("--genre-weight", type=float, default=0.7)
    parser.add_argument("--acoustic-weight", type=float, default=1.2)
    parser.add_argument("--year-weight", type=float, default=0.25)
    parser.add_argument("--popularity-weight", type=float, default=0.15)
    parser.add_argument("--binary-weight", type=float, default=0.10)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        records = load_records(input_path)
    except Exception as exc:
        print(f"Failed to load input records: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    tag_vocab, genre_vocab, acoustic_vocab, vocab_report = build_vocabs(
        records=records,
        min_tag_df=args.min_tag_df,
        min_genre_df=args.min_genre_df,
        min_acoustic_df=args.min_acoustic_df,
        max_tags=args.max_tags,
        max_genres=args.max_genres,
        max_acoustic=args.max_acoustic,
    )

    feature_names = build_feature_names(tag_vocab, genre_vocab, acoustic_vocab)

    X, row_index, matrix_report = featurize_records(
        records=records,
        tag_vocab=tag_vocab,
        genre_vocab=genre_vocab,
        acoustic_vocab=acoustic_vocab,
        tag_weight=args.tag_weight,
        genre_weight=args.genre_weight,
        acoustic_weight=args.acoustic_weight,
        year_weight=args.year_weight,
        popularity_weight=args.popularity_weight,
        binary_weight=args.binary_weight,
    )

    feature_matrix_path = out_dir / "feature_matrix.npz"
    feature_names_path = out_dir / "feature_names.json"
    row_index_path = out_dir / "row_index.json"
    report_path = out_dir / "featurizer_report.json"

    np.savez_compressed(feature_matrix_path, X=X)
    feature_names_path.write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    row_index_path.write_text(
        json.dumps(row_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "input": str(input_path),
        "records": len(records),
        "feature_count": len(feature_names),
        "vocab": vocab_report,
        "matrix": matrix_report,
        "weights": {
            "tag_weight": args.tag_weight,
            "genre_weight": args.genre_weight,
            "acoustic_weight": args.acoustic_weight,
            "year_weight": args.year_weight,
            "popularity_weight": args.popularity_weight,
            "binary_weight": args.binary_weight,
        },
        "thresholds": {
            "min_tag_df": args.min_tag_df,
            "min_genre_df": args.min_genre_df,
            "min_acoustic_df": args.min_acoustic_df,
            "max_tags": args.max_tags,
            "max_genres": args.max_genres,
            "max_acoustic": args.max_acoustic,
        },
        "filters": {
            "skip_tag_substrings": sorted(_SKIP_TAG_SUBSTRINGS),
            "skip_tag_exact": sorted(_SKIP_TAG_EXACT),
            "tag_alias_map": dict(sorted(_TAG_ALIAS_MAP.items())),
            "skip_acoustic_substrings": sorted(_SKIP_ACOUSTIC_SUBSTRINGS),
            "skip_acoustic_suffixes": sorted(_SKIP_ACOUSTIC_SUFFIXES),
        },
        "outputs": {
            "feature_matrix_npz": str(feature_matrix_path),
            "feature_names_json": str(feature_names_path),
            "row_index_json": str(row_index_path),
            "featurizer_report_json": str(report_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Featurization Summary ===")
    print(f"records: {len(records)}")
    print(f"feature_count: {len(feature_names)}")
    print(f"matrix_shape: {X.shape}")
    print(f"tag_vocab_size: {len(tag_vocab)}")
    print(f"genre_vocab_size: {len(genre_vocab)}")
    print(f"acoustic_vocab_size: {len(acoustic_vocab)}")
    print(f"wrote: {feature_matrix_path}")
    print(f"wrote: {feature_names_path}")
    print(f"wrote: {row_index_path}")
    print(f"wrote: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())