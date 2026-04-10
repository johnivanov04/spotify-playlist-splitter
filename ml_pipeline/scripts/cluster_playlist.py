#!/usr/bin/env python3
"""
Cluster a single playlist in the learned embedding space.

This script:
1. loads normalized/backfilled playlist records
2. featurizes them against the frozen training vocabulary
3. applies the saved scaler + PCA representation
4. tries several KMeans values
5. merges tiny clusters before scoring
6. penalizes degenerate "one giant bucket" solutions
7. writes cluster assignments + summaries

Key fixes vs the earlier version:
- stronger penalty for oversized dominant clusters
- candidate k is biased upward for medium/large playlists
- small clusters are merged before scoring, not only after selection
- user-facing cluster names avoid cryptic acoustic abbreviations when possible
- missing acoustic data is neutralized using training means
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

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with:\n"
        "  python -m pip install scikit-learn\n"
        f"Original import error: {exc}"
    )


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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_representation(path: Path) -> dict[str, np.ndarray | list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Representation artifact not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = [
        "scaler_mean",
        "scaler_scale",
        "pca_components",
        "pca_mean",
        "feature_names",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key in representation artifact: {key}")

    feature_names_arr = data["feature_names"]
    feature_names = [str(x) for x in feature_names_arr.tolist()]

    return {
        "scaler_mean": data["scaler_mean"].astype(np.float64),
        "scaler_scale": data["scaler_scale"].astype(np.float64),
        "pca_components": data["pca_components"].astype(np.float64),
        "pca_mean": data["pca_mean"].astype(np.float64),
        "feature_names": feature_names,
    }


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
# featurization rules
# ----------------------------

_SKIP_TAG_SUBSTRINGS = {
    "billboard",
    "hot_100",
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
}

_SKIP_ACOUSTIC_SUFFIXES = {
    "__probability",
}

_ACOUSTIC_ABBREV_MAP = {
    "reg": "reggae",
    "cou": "country",
    "blu": "blues",
    "met": "metal",
    "dis": "disco",
    "hip": "hip hop",
    "cla": "classical",
    "dan": "dance",
    "jaz": "jazz",
    "roc": "rock",
    "spe": "speech",
    "funksoulrnb": "funk soul rnb",
}


def keep_tag_token(token: str) -> bool:
    if not token:
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

        token = normalize_token(name)
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


def parse_feature_names(feature_names: list[str]) -> dict[str, Any]:
    meta_index: dict[str, int] = {}
    tag_index: dict[str, int] = {}
    genre_index: dict[str, int] = {}
    acoustic_index: dict[str, int] = {}

    for idx, name in enumerate(feature_names):
        if name.startswith("meta__"):
            meta_index[name] = idx
        elif name.startswith("tag__"):
            tag_index[name[len("tag__"):]] = idx
        elif name.startswith("genre__"):
            genre_index[name[len("genre__"):]] = idx
        elif name.startswith("acoustic__"):
            acoustic_index[name[len("acoustic__"):]] = idx

    return {
        "meta_index": meta_index,
        "tag_index": tag_index,
        "genre_index": genre_index,
        "acoustic_index": acoustic_index,
        "acoustic_columns": sorted(acoustic_index.values()),
    }


def featurize_against_frozen_vocab(
    records: list[dict[str, Any]],
    feature_names: list[str],
    scaler_mean: np.ndarray,
    *,
    tag_weight: float,
    genre_weight: float,
    acoustic_weight: float,
    year_weight: float,
    popularity_weight: float,
    binary_weight: float,
    neutralize_missing_acoustic: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    parsed = parse_feature_names(feature_names)
    meta_index: dict[str, int] = parsed["meta_index"]
    tag_index: dict[str, int] = parsed["tag_index"]
    genre_index: dict[str, int] = parsed["genre_index"]
    acoustic_index: dict[str, int] = parsed["acoustic_index"]
    acoustic_columns: list[int] = parsed["acoustic_columns"]

    n_rows = len(records)
    n_cols = len(feature_names)
    X = np.zeros((n_rows, n_cols), dtype=np.float64)
    row_meta: list[dict[str, Any]] = []

    years = [y for r in records if (y := get_year(r)) is not None]
    if years:
        local_year_mean = float(sum(years) / len(years))
        local_year_var = float(sum((y - local_year_mean) ** 2 for y in years) / len(years))
        local_year_std = math.sqrt(local_year_var) if local_year_var > 1e-12 else 1.0
    else:
        local_year_mean = 2000.0
        local_year_std = 1.0

    for row_i, record in enumerate(records):
        brainz = record.get("brainz") or {}
        ids = record.get("ids") or {}

        tags = list(iter_name_count_items(brainz.get("tags")))
        genres = list(iter_name_count_items(brainz.get("genres")))

        acoustic_vals: list[tuple[str, float]] = []
        acoustic_obj = brainz.get("acoustic_high_level")
        if isinstance(acoustic_obj, dict):
            acoustic_vals = list(iter_acoustic_numeric_leaves(acoustic_obj))

        has_acoustic = bool(acoustic_vals)

        if tags and "meta__has_tags" in meta_index:
            X[row_i, meta_index["meta__has_tags"]] = binary_weight

        if genres and "meta__has_genres" in meta_index:
            X[row_i, meta_index["meta__has_genres"]] = binary_weight

        if "meta__has_acoustic_high_level" in meta_index:
            if has_acoustic:
                X[row_i, meta_index["meta__has_acoustic_high_level"]] = binary_weight
            elif neutralize_missing_acoustic:
                X[row_i, meta_index["meta__has_acoustic_high_level"]] = scaler_mean[
                    meta_index["meta__has_acoustic_high_level"]
                ]

        year = get_year(record)
        if "meta__has_year" in meta_index and year is not None:
            X[row_i, meta_index["meta__has_year"]] = binary_weight

        if "meta__year_z" in meta_index and year is not None:
            X[row_i, meta_index["meta__year_z"]] = ((year - local_year_mean) / local_year_std) * year_weight

        popularity = get_popularity(record)
        if "meta__has_popularity" in meta_index and popularity is not None:
            X[row_i, meta_index["meta__has_popularity"]] = binary_weight

        if "meta__popularity_scaled" in meta_index and popularity is not None:
            X[row_i, meta_index["meta__popularity_scaled"]] = (popularity / 100.0) * popularity_weight

        for name, count in tags:
            idx = tag_index.get(name)
            if idx is None:
                continue
            X[row_i, idx] += math.log1p(max(count, 0.0)) * tag_weight

        for name, count in genres:
            idx = genre_index.get(name)
            if idx is None:
                continue
            X[row_i, idx] += math.log1p(max(count, 0.0)) * genre_weight

        if has_acoustic:
            for name, value in acoustic_vals:
                idx = acoustic_index.get(name)
                if idx is None:
                    continue
                clipped = float(max(min(value, 1.0), -1.0))
                X[row_i, idx] += clipped * acoustic_weight
        elif neutralize_missing_acoustic:
            for idx in acoustic_columns:
                X[row_i, idx] = scaler_mean[idx]

        row_meta.append(
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
                "has_tags": bool(tags),
                "has_genres": bool(genres),
                "has_acoustic_high_level": has_acoustic,
            }
        )

    return X, row_meta


# ----------------------------
# representation transform
# ----------------------------

def apply_representation(
    X: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = (X - scaler_mean) / scaler_scale

    if pca_components.size == 0:
        return X_scaled, X_scaled

    embeddings = (X_scaled - pca_mean) @ pca_components.T
    return X_scaled, embeddings


# ----------------------------
# clustering
# ----------------------------

def build_candidate_ks(n_rows: int, min_k: int, max_k: int, min_cluster_size: int) -> list[int]:
    if n_rows < 2:
        return []

    lower = max(2, min_k)

    if n_rows >= 40:
        lower = max(lower, 3)
    if n_rows >= 120:
        lower = max(lower, 4)
    if n_rows >= 250:
        lower = max(lower, 5)

    upper_from_size = max(2, n_rows // max(1, min_cluster_size))
    upper = min(max_k, n_rows - 1, upper_from_size)

    if upper < lower:
        upper = min(max_k, n_rows - 1)
        lower = min(lower, upper)

    if upper < 2:
        return []

    return list(range(lower, upper + 1))


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(np.unique(labels).tolist())
    mapping = {old: new for new, old in enumerate(uniq)}
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def cluster_centers_from_labels(labels: np.ndarray, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        member_ix = np.where(labels == lab)[0]
        out[int(lab)] = embeddings[member_ix].mean(axis=0)
    return out


def merge_small_clusters(labels: np.ndarray, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    labels = labels.copy()

    while True:
        counts = Counter(labels.tolist())
        small = [lab for lab, c in counts.items() if c < min_cluster_size]
        large = [lab for lab, c in counts.items() if c >= min_cluster_size]

        if not small or not large:
            break

        centers = cluster_centers_from_labels(labels, embeddings)

        # merge the smallest cluster first
        small.sort(key=lambda lab: counts[lab])
        small_lab = small[0]
        small_ix = np.where(labels == small_lab)[0]

        target_lab = min(
            large,
            key=lambda lab: float(np.linalg.norm(centers[small_lab] - centers[lab])),
        )

        for ix in small_ix:
            labels[ix] = target_lab

    return relabel_contiguous(labels)


def score_solution(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_size: int,
) -> tuple[float | None, float, dict[str, Any]]:
    uniq = np.unique(labels)
    counts = Counter(labels.tolist())
    sizes = sorted((int(v) for v in counts.values()), reverse=True)
    n = int(len(labels))

    if uniq.size < 2:
        silhouette = None
    else:
        silhouette = float(silhouette_score(embeddings, labels))

    largest_frac = (sizes[0] / n) if sizes else 1.0
    second_frac = (sizes[1] / n) if len(sizes) > 1 else 0.0

    small_clusters = sum(1 for c in sizes if c < min_cluster_size)
    small_items = sum(c for c in sizes if c < min_cluster_size)

    base = silhouette if silhouette is not None else -1.0
    adjusted = base

    # strong penalty for one giant cluster
    if largest_frac > 0.55:
        adjusted -= (largest_frac - 0.55) * 2.2
    if largest_frac > 0.70:
        adjusted -= (largest_frac - 0.70) * 2.5
    if largest_frac > 0.85:
        adjusted -= (largest_frac - 0.85) * 4.0

    # penalty if the second cluster is tiny
    if second_frac < 0.10:
        adjusted -= (0.10 - second_frac) * 0.8

    # penalty for undersized clusters
    adjusted -= 0.05 * small_clusters
    adjusted -= 0.01 * small_items

    # slight reward for having 3+ meaningful clusters
    meaningful_clusters = sum(1 for c in sizes if c >= min_cluster_size)
    if meaningful_clusters >= 3:
        adjusted += min(0.06, 0.02 * (meaningful_clusters - 2))

    details = {
        "cluster_sizes": sizes,
        "largest_fraction": largest_frac,
        "second_fraction": second_frac,
        "small_cluster_count": small_clusters,
        "small_item_count": small_items,
        "meaningful_cluster_count": meaningful_clusters,
    }

    return silhouette, adjusted, details


def choose_best_clustering(
    embeddings: np.ndarray,
    candidate_ks: list[int],
    min_cluster_size: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, list[dict[str, Any]], int, float | None]:
    diagnostics: list[dict[str, Any]] = []

    if not candidate_ks:
        labels = np.zeros(embeddings.shape[0], dtype=int)
        diagnostics.append(
            {
                "k": 1,
                "silhouette": None,
                "adjusted_score": None,
                "cluster_sizes": [int(embeddings.shape[0])],
                "largest_fraction": 1.0,
                "meaningful_cluster_count": 1,
                "merged_before_scoring": False,
            }
        )
        return labels, diagnostics, 1, None

    best_labels = None
    best_adjusted = -1e18
    best_k = None

    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        raw_labels = model.fit_predict(embeddings)
        merged_labels = merge_small_clusters(raw_labels, embeddings, min_cluster_size=min_cluster_size)
        merged_labels = relabel_contiguous(merged_labels)

        silhouette, adjusted, details = score_solution(
            merged_labels,
            embeddings,
            min_cluster_size=min_cluster_size,
        )

        diagnostics.append(
            {
                "k": int(k),
                "silhouette": silhouette,
                "adjusted_score": adjusted,
                "cluster_sizes": details["cluster_sizes"],
                "largest_fraction": details["largest_fraction"],
                "second_fraction": details["second_fraction"],
                "meaningful_cluster_count": details["meaningful_cluster_count"],
                "small_cluster_count": details["small_cluster_count"],
                "small_item_count": details["small_item_count"],
                "merged_before_scoring": not np.array_equal(raw_labels, merged_labels),
            }
        )

        if adjusted > best_adjusted:
            best_adjusted = adjusted
            best_labels = merged_labels
            best_k = int(np.unique(merged_labels).size)

    assert best_labels is not None
    return best_labels, diagnostics, int(best_k), float(best_adjusted)


def apply_subcluster_mapping(
    labels: np.ndarray,
    member_ix: np.ndarray,
    sub_labels: np.ndarray,
    parent_label: int,
) -> np.ndarray:
    labels = labels.copy()
    uniq_sub = sorted(np.unique(sub_labels).tolist())
    if len(uniq_sub) < 2:
        return labels

    sub_counts = Counter(sub_labels.tolist())
    keep_parent = max(uniq_sub, key=lambda lab: (sub_counts[int(lab)], -int(lab)))
    next_label = int(labels.max()) + 1
    mapping: dict[int, int] = {}
    for sub_lab in uniq_sub:
        if int(sub_lab) == int(keep_parent):
            mapping[int(sub_lab)] = int(parent_label)
        else:
            mapping[int(sub_lab)] = next_label
            next_label += 1

    for local_pos, global_ix in enumerate(member_ix.tolist()):
        labels[int(global_ix)] = mapping[int(sub_labels[local_pos])]

    return relabel_contiguous(labels)


def recursively_split_oversized_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_k: int,
    max_k: int,
    random_state: int,
    n_init: int,
    max_cluster_fraction: float,
    min_split_size: int,
    min_adjusted_score: float,
    max_subcluster_fraction: float,
    max_rounds: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = relabel_contiguous(labels)
    n_total = int(len(labels))
    diagnostics: list[dict[str, Any]] = []

    if n_total <= 1 or max_rounds <= 0:
        return labels, diagnostics

    split_size_threshold = max(int(min_split_size), int(math.ceil(n_total * max_cluster_fraction)))

    for round_idx in range(max_rounds):
        counts = Counter(labels.tolist())
        candidate_labels = [
            int(lab)
            for lab, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
            if int(count) >= split_size_threshold
        ]

        if not candidate_labels:
            break

        round_diag: dict[str, Any] = {
            "round": int(round_idx + 1),
            "split_size_threshold": int(split_size_threshold),
            "candidates": [],
            "accepted": None,
        }
        applied = False

        for parent_label in candidate_labels:
            member_ix = np.where(labels == parent_label)[0]
            cluster_size = int(member_ix.size)
            sub_embeddings = embeddings[member_ix]

            local_candidate_ks = build_candidate_ks(
                n_rows=cluster_size,
                min_k=max(2, min_k),
                max_k=min(max_k, max(2, cluster_size - 1)),
                min_cluster_size=min_cluster_size,
            )

            candidate_diag: dict[str, Any] = {
                "cluster_id": int(parent_label),
                "cluster_size": int(cluster_size),
                "candidate_ks": local_candidate_ks,
                "accepted": False,
            }

            if not local_candidate_ks:
                candidate_diag["reason"] = "no_candidate_ks"
                round_diag["candidates"].append(candidate_diag)
                continue

            sub_labels, sub_diagnostics, sub_final_k, sub_best_adjusted = choose_best_clustering(
                embeddings=sub_embeddings,
                candidate_ks=local_candidate_ks,
                min_cluster_size=min_cluster_size,
                random_state=random_state,
                n_init=n_init,
            )
            sub_silhouette, sub_adjusted, sub_details = score_solution(
                sub_labels,
                sub_embeddings,
                min_cluster_size=min_cluster_size,
            )

            candidate_diag.update(
                {
                    "sub_final_k": int(sub_final_k),
                    "sub_best_adjusted_score": sub_best_adjusted,
                    "sub_silhouette": sub_silhouette,
                    "sub_adjusted_score": sub_adjusted,
                    "sub_cluster_sizes": sub_details["cluster_sizes"],
                    "sub_largest_fraction": sub_details["largest_fraction"],
                    "sub_meaningful_cluster_count": sub_details["meaningful_cluster_count"],
                    "diagnostics": sub_diagnostics,
                }
            )

            accept = (
                int(sub_final_k) >= 2
                and int(sub_details["meaningful_cluster_count"]) >= 2
                and float(sub_adjusted) >= float(min_adjusted_score)
                and float(sub_details["largest_fraction"]) <= float(max_subcluster_fraction)
            )

            if not accept:
                reasons = []
                if int(sub_final_k) < 2:
                    reasons.append("single_cluster")
                if int(sub_details["meaningful_cluster_count"]) < 2:
                    reasons.append("not_enough_meaningful_clusters")
                if float(sub_adjusted) < float(min_adjusted_score):
                    reasons.append("adjusted_score_too_low")
                if float(sub_details["largest_fraction"]) > float(max_subcluster_fraction):
                    reasons.append("largest_subcluster_still_too_large")
                candidate_diag["reason"] = ",".join(reasons) if reasons else "rejected"
                round_diag["candidates"].append(candidate_diag)
                continue

            labels = apply_subcluster_mapping(
                labels=labels,
                member_ix=member_ix,
                sub_labels=sub_labels,
                parent_label=int(parent_label),
            )
            candidate_diag["accepted"] = True
            candidate_diag["reason"] = "accepted"
            round_diag["accepted"] = candidate_diag
            round_diag["candidates"].append(candidate_diag)
            applied = True
            break

        diagnostics.append(round_diag)

        if not applied:
            break

    return relabel_contiguous(labels), diagnostics


# ----------------------------
# summary / explainability
# ----------------------------

def pretty_feature_name(feature: str) -> str:
    if feature.startswith("tag__"):
        return feature[len("tag__"):].replace("_", " ")

    if feature.startswith("genre__"):
        return feature[len("genre__"):].replace("_", " ")

    if feature.startswith("acoustic__"):
        s = feature[len("acoustic__"):]
        s = s.replace("highlevel__", "")
        s = s.replace("__all__", "__")
        parts = [p for p in s.split("__") if p]
        if not parts:
            return feature.replace("_", " ")

        last = parts[-1]
        if last in _ACOUSTIC_ABBREV_MAP:
            last = _ACOUSTIC_ABBREV_MAP[last]

        if len(parts) >= 2 and parts[-2] in {"mood_happy", "mood_sad", "mood_relaxed", "mood_party", "mood_electronic", "voice_instrumental", "timbre"}:
            prefix = parts[-2].replace("_", " ")
            return f"{prefix} / {last.replace('_', ' ')}"

        return last.replace("_", " ")

    if feature.startswith("meta__"):
        return feature[len("meta__"):].replace("_", " ")

    return feature.replace("_", " ")


def is_good_naming_feature(feature: str) -> bool:
    if feature.startswith("tag__") or feature.startswith("genre__"):
        return True

    if not feature.startswith("acoustic__"):
        return False

    bad_substrings = [
        "genre_tzanetakis",
        "genre_dortmund",
        "genre_rosamerica",
        "ismir04_rhythm",
        "moods_mirex",
        "tonal_atonal",
    ]
    return not any(bad in feature for bad in bad_substrings)


def top_cluster_features(
    cluster_mean_scaled: np.ndarray,
    feature_names: list[str],
    top_n: int,
    exclude_meta: bool = True,
) -> list[dict[str, Any]]:
    pairs = []
    for idx, value in enumerate(cluster_mean_scaled.tolist()):
        name = feature_names[idx]
        if exclude_meta and name.startswith("meta__"):
            continue
        pairs.append((name, float(value)))

    pairs.sort(key=lambda x: x[1], reverse=True)

    out = []
    for name, value in pairs[:top_n]:
        out.append(
            {
                "feature": name,
                "pretty_feature": pretty_feature_name(name),
                "score": value,
            }
        )
    return out


def build_cluster_name(top_features: list[dict[str, Any]]) -> str:
    tokens = []
    seen = set()

    preferred = [f for f in top_features if is_good_naming_feature(f["feature"])]
    fallback = [f for f in top_features if f not in preferred]

    for feat in preferred + fallback:
        label = feat["pretty_feature"]
        if label in seen:
            continue
        seen.add(label)
        tokens.append(label)
        if len(tokens) == 2:
            break

    if not tokens:
        return "mixed cluster"
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0]} / {tokens[1]}"


def build_cluster_summary(
    labels: np.ndarray,
    row_meta: list[dict[str, Any]],
    X_scaled: np.ndarray,
    feature_names: list[str],
    sample_titles_per_cluster: int,
    top_features_per_cluster: int,
) -> list[dict[str, Any]]:
    summary = []
    uniq = sorted(np.unique(labels).tolist())

    for cluster_id in uniq:
        member_ix = np.where(labels == cluster_id)[0]
        cluster_mean = X_scaled[member_ix].mean(axis=0)

        top_features = top_cluster_features(
            cluster_mean_scaled=cluster_mean,
            feature_names=feature_names,
            top_n=top_features_per_cluster,
            exclude_meta=True,
        )

        sample_tracks = []
        for ix in member_ix[:sample_titles_per_cluster]:
            meta = row_meta[int(ix)]
            artist_str = ", ".join(meta.get("artists") or [])
            sample_tracks.append(f"{meta.get('title')} — {artist_str}")

        summary.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": build_cluster_name(top_features),
                "size": int(member_ix.size),
                "sample_tracks": sample_tracks,
                "top_features": top_features,
            }
        )

    summary.sort(key=lambda x: x["size"], reverse=True)
    return summary


# ----------------------------
# cli
# ----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster a playlist in the learned embedding space.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to normalized/backfilled playlist records JSON/NDJSON",
    )
    parser.add_argument(
        "--representation",
        default="ml_pipeline/data/artifacts/representation_v1/representation_artifacts.npz",
        help="Path to representation_artifacts.npz",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_pipeline/data/artifacts/playlist_cluster_test",
        help="Output directory",
    )

    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kmeans-n-init", type=int, default=20)

    parser.add_argument("--tag-weight", type=float, default=1.0)
    parser.add_argument("--genre-weight", type=float, default=0.7)
    parser.add_argument("--acoustic-weight", type=float, default=0.9)
    parser.add_argument("--year-weight", type=float, default=0.25)
    parser.add_argument("--popularity-weight", type=float, default=0.15)
    parser.add_argument("--binary-weight", type=float, default=0.10)

    parser.add_argument(
        "--disable-neutralize-missing-acoustic",
        action="store_true",
        help="Do not replace missing acoustic block with training means",
    )
    parser.add_argument("--sample-titles-per-cluster", type=int, default=10)
    parser.add_argument("--top-features-per-cluster", type=int, default=10)

    parser.add_argument(
        "--disable-recursive-splitting",
        action="store_true",
        help="Do not recursively split oversized first-pass clusters",
    )
    parser.add_argument(
        "--recursive-split-max-fraction",
        type=float,
        default=0.28,
        help="Recursively split a cluster if it exceeds this fraction of the playlist",
    )
    parser.add_argument(
        "--recursive-split-min-size",
        type=int,
        default=24,
        help="Minimum absolute cluster size required before recursive splitting is considered",
    )
    parser.add_argument(
        "--recursive-split-min-adjusted-score",
        type=float,
        default=0.02,
        help="Minimum adjusted score a local split must achieve to be kept",
    )
    parser.add_argument(
        "--recursive-split-max-subcluster-fraction",
        type=float,
        default=0.78,
        help="Reject a local split if its largest resulting subcluster is still above this fraction",
    )
    parser.add_argument(
        "--recursive-split-max-rounds",
        type=int,
        default=2,
        help="Maximum recursive split rounds to apply after the first clustering pass",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    representation_path = Path(args.representation)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        records = load_records(input_path)
    except Exception as exc:
        print(f"Failed to load playlist records: {exc}", file=sys.stderr)
        return 1

    try:
        rep = load_representation(representation_path)
    except Exception as exc:
        print(f"Failed to load representation artifact: {exc}", file=sys.stderr)
        return 1

    feature_names = rep["feature_names"]
    scaler_mean = rep["scaler_mean"]
    scaler_scale = rep["scaler_scale"]
    pca_components = rep["pca_components"]
    pca_mean = rep["pca_mean"]

    X, row_meta = featurize_against_frozen_vocab(
        records=records,
        feature_names=feature_names,
        scaler_mean=scaler_mean,
        tag_weight=args.tag_weight,
        genre_weight=args.genre_weight,
        acoustic_weight=args.acoustic_weight,
        year_weight=args.year_weight,
        popularity_weight=args.popularity_weight,
        binary_weight=args.binary_weight,
        neutralize_missing_acoustic=not args.disable_neutralize_missing_acoustic,
    )

    X_scaled, embeddings = apply_representation(
        X=X,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )

    candidate_ks = build_candidate_ks(
        n_rows=len(records),
        min_k=args.min_k,
        max_k=args.max_k,
        min_cluster_size=args.min_cluster_size,
    )

    labels, diagnostics, final_k, best_adjusted_score = choose_best_clustering(
        embeddings=embeddings,
        candidate_ks=candidate_ks,
        min_cluster_size=args.min_cluster_size,
        random_state=args.random_state,
        n_init=args.kmeans_n_init,
    )

    recursive_split_diagnostics: list[dict[str, Any]] = []
    if not args.disable_recursive_splitting:
        labels, recursive_split_diagnostics = recursively_split_oversized_clusters(
            labels=labels,
            embeddings=embeddings,
            min_cluster_size=args.min_cluster_size,
            min_k=args.min_k,
            max_k=args.max_k,
            random_state=args.random_state,
            n_init=args.kmeans_n_init,
            max_cluster_fraction=args.recursive_split_max_fraction,
            min_split_size=args.recursive_split_min_size,
            min_adjusted_score=args.recursive_split_min_adjusted_score,
            max_subcluster_fraction=args.recursive_split_max_subcluster_fraction,
            max_rounds=args.recursive_split_max_rounds,
        )
        final_k = int(np.unique(labels).size)

    assignments = []
    for meta, label in zip(row_meta, labels, strict=False):
        assignments.append({**meta, "cluster": int(label)})

    cluster_summary = build_cluster_summary(
        labels=labels,
        row_meta=row_meta,
        X_scaled=X_scaled,
        feature_names=feature_names,
        sample_titles_per_cluster=args.sample_titles_per_cluster,
        top_features_per_cluster=args.top_features_per_cluster,
    )

    report = {
        "input": str(input_path),
        "representation": str(representation_path),
        "records": len(records),
        "feature_count": int(X.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "candidate_ks": candidate_ks,
        "final_k": int(final_k),
        "min_cluster_size": int(args.min_cluster_size),
        "cluster_sizes": {
            str(cluster_id): int((labels == cluster_id).sum())
            for cluster_id in sorted(np.unique(labels).tolist())
        },
        "diagnostics": diagnostics,
        "best_adjusted_score": best_adjusted_score,
        "weights": {
            "tag_weight": args.tag_weight,
            "genre_weight": args.genre_weight,
            "acoustic_weight": args.acoustic_weight,
            "year_weight": args.year_weight,
            "popularity_weight": args.popularity_weight,
            "binary_weight": args.binary_weight,
            "neutralize_missing_acoustic": not args.disable_neutralize_missing_acoustic,
        },
        "recursive_splitting": {
            "enabled": not args.disable_recursive_splitting,
            "max_fraction": args.recursive_split_max_fraction,
            "min_size": args.recursive_split_min_size,
            "min_adjusted_score": args.recursive_split_min_adjusted_score,
            "max_subcluster_fraction": args.recursive_split_max_subcluster_fraction,
            "max_rounds": args.recursive_split_max_rounds,
            "rounds": recursive_split_diagnostics,
        },
        "outputs": {
            "assignments_json": str(out_dir / "playlist_cluster_assignments.json"),
            "summary_json": str(out_dir / "playlist_cluster_summary.json"),
            "report_json": str(out_dir / "playlist_clustering_report.json"),
            "embeddings_npy": str(out_dir / "playlist_embeddings.npy"),
        },
    }

    save_json(out_dir / "playlist_cluster_assignments.json", assignments)
    save_json(out_dir / "playlist_cluster_summary.json", cluster_summary)
    save_json(out_dir / "playlist_clustering_report.json", report)
    np.save(out_dir / "playlist_embeddings.npy", embeddings.astype(np.float32))

    print("=== Playlist Clustering Summary ===")
    print(f"records: {len(records)}")
    print(f"feature_count: {X.shape[1]}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    print(f"candidate_ks: {candidate_ks}")
    applied_recursive_splits = sum(1 for round_info in recursive_split_diagnostics if round_info.get("accepted"))
    print(f"final_k: {final_k}")
    print(f"cluster_sizes: {report['cluster_sizes']}")
    if not args.disable_recursive_splitting:
        print(f"recursive_splits_applied: {applied_recursive_splits}")
    print(f"wrote: {out_dir / 'playlist_cluster_assignments.json'}")
    print(f"wrote: {out_dir / 'playlist_cluster_summary.json'}")
    print(f"wrote: {out_dir / 'playlist_clustering_report.json'}")
    print(f"wrote: {out_dir / 'playlist_embeddings.npy'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())