#!/usr/bin/env python3
"""
Train a reusable song representation from featurized song records.

This script:
1. loads the featurized matrix
2. standardizes the features
3. optionally applies PCA to build a compact embedding space
4. runs diagnostic KMeans clustering across a range of k values
5. saves reusable representation artifacts for later playlist-time inference

Primary output:
- representation_artifacts.npz
  Contains scaler stats + PCA parameters needed to embed new songs later.

Diagnostic outputs:
- embeddings.npy
- cluster_labels_k*.json
- training_report.json
- cluster_diagnostics.json

Recommended usage:
    python ml_pipeline/scripts/train_representation.py \
      --artifacts-dir ml_pipeline/data/artifacts/features_v1_1 \
      --out-dir ml_pipeline/data/artifacts/representation_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with:\n"
        "  python -m pip install scikit-learn\n"
        f"Original import error: {exc}"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feature_artifacts(artifacts_dir: Path) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, Any] | None]:
    matrix_path = artifacts_dir / "feature_matrix.npz"
    feature_names_path = artifacts_dir / "feature_names.json"
    row_index_path = artifacts_dir / "row_index.json"
    featurizer_report_path = artifacts_dir / "featurizer_report.json"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {matrix_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Missing feature names: {feature_names_path}")
    if not row_index_path.exists():
        raise FileNotFoundError(f"Missing row index: {row_index_path}")

    X = np.load(matrix_path)["X"].astype(np.float64)
    feature_names = load_json(feature_names_path)
    row_index = load_json(row_index_path)
    featurizer_report = load_json(featurizer_report_path) if featurizer_report_path.exists() else None

    if not isinstance(feature_names, list):
        raise ValueError("feature_names.json must contain a list")
    if not isinstance(row_index, list):
        raise ValueError("row_index.json must contain a list")
    if X.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {X.shape}")
    if X.shape[0] != len(row_index):
        raise ValueError(f"Row count mismatch: matrix has {X.shape[0]} rows, row_index has {len(row_index)} rows")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: matrix has {X.shape[1]} cols, feature_names has {len(feature_names)} names")

    return X, feature_names, row_index, featurizer_report


def compute_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def choose_pca_components(
    n_features: int,
    n_samples: int,
    requested: int,
) -> int:
    max_valid = max(1, min(n_features, n_samples))
    return max(1, min(requested, max_valid))


def build_pca(X_scaled: np.ndarray, n_components: int, random_state: int) -> PCA:
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)
    return pca


def top_feature_contributors(
    center_original_space: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(center_original_space))[::-1][:top_n]
    out: list[dict[str, Any]] = []
    for idx in order:
        out.append(
            {
                "feature": feature_names[int(idx)],
                "weight": float(center_original_space[int(idx)]),
                "abs_weight": float(abs(center_original_space[int(idx)])),
            }
        )
    return out


def cluster_diagnostics(
    embeddings: np.ndarray,
    ks: list[int],
    random_state: int,
    n_init: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    n_samples = embeddings.shape[0]
    for k in ks:
        if k < 2 or k >= n_samples:
            continue

        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(embeddings)

        unique = np.unique(labels)
        if unique.size < 2:
            sil = None
        else:
            sil = float(silhouette_score(embeddings, labels))

        results.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": sil,
            }
        )

    return results


def choose_best_k(results: list[dict[str, Any]]) -> int:
    if not results:
        raise ValueError("No clustering diagnostics were produced")

    with_sil = [r for r in results if r["silhouette"] is not None]
    if with_sil:
        with_sil.sort(key=lambda r: (r["silhouette"], -r["inertia"]), reverse=True)
        return int(with_sil[0]["k"])

    results.sort(key=lambda r: r["inertia"])
    return int(results[0]["k"])


def build_cluster_summary(
    X_scaled: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    pca: PCA | None,
    feature_names: list[str],
    row_index: list[dict[str, Any]],
    top_features_per_cluster: int,
    sample_titles_per_cluster: int,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []

    centers_embed = model.cluster_centers_
    if pca is not None:
        centers_scaled = pca.inverse_transform(centers_embed)
    else:
        centers_scaled = centers_embed

    for cluster_id in range(model.n_clusters):
        member_ix = np.where(labels == cluster_id)[0]
        titles = []
        for ix in member_ix[:sample_titles_per_cluster]:
            meta = row_index[int(ix)]
            title = meta.get("title")
            artists = meta.get("artists") or []
            artist_str = ", ".join(artists[:2]) if artists else "Unknown artist"
            titles.append(f"{title} — {artist_str}")

        top_features = top_feature_contributors(
            center_original_space=centers_scaled[cluster_id],
            feature_names=feature_names,
            top_n=top_features_per_cluster,
        )

        summary.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(member_ix.size),
                "sample_tracks": titles,
                "top_features": top_features,
            }
        )

    summary.sort(key=lambda x: x["size"], reverse=True)
    return summary


def parse_ks(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("k values list cannot be empty")
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reusable song representation and run clustering diagnostics.")

    parser.add_argument(
        "--artifacts-dir",
        default="ml_pipeline/data/artifacts/features_v1_1",
        help="Directory containing feature_matrix.npz, feature_names.json, and row_index.json",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_pipeline/data/artifacts/representation_v1",
        help="Directory to write representation outputs into",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=32,
        help="Target PCA dimensionality for the reusable embedding space",
    )
    parser.add_argument(
        "--diagnostic-ks",
        type=parse_ks,
        default=[4, 5, 6, 7, 8, 9, 10],
        help='Comma-separated diagnostic K values, e.g. "4,5,6,7,8"',
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for PCA/KMeans reproducibility",
    )
    parser.add_argument(
        "--kmeans-n-init",
        type=int,
        default=20,
        help="KMeans n_init value",
    )
    parser.add_argument(
        "--top-features-per-cluster",
        type=int,
        default=12,
        help="How many strongest cluster features to report",
    )
    parser.add_argument(
        "--sample-titles-per-cluster",
        type=int,
        default=10,
        help="How many example tracks to report per cluster",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, feature_names, row_index, featurizer_report = load_feature_artifacts(artifacts_dir)
    except Exception as exc:
        print(f"Failed to load feature artifacts: {exc}", file=sys.stderr)
        return 1

    mean, std = compute_scaler(X)
    X_scaled = apply_scaler(X, mean, std)

    actual_pca_components = choose_pca_components(
        n_features=X_scaled.shape[1],
        n_samples=X_scaled.shape[0],
        requested=args.pca_components,
    )

    pca: PCA | None = None
    if actual_pca_components < X_scaled.shape[1]:
        pca = build_pca(X_scaled, n_components=actual_pca_components, random_state=args.random_state)
        embeddings = pca.transform(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float64)
        explained_variance_cumulative = np.cumsum(explained_variance_ratio)
    else:
        embeddings = X_scaled.copy()
        explained_variance_ratio = np.array([], dtype=np.float64)
        explained_variance_cumulative = np.array([], dtype=np.float64)

    diagnostics = cluster_diagnostics(
        embeddings=embeddings,
        ks=args.diagnostic_ks,
        random_state=args.random_state,
        n_init=args.kmeans_n_init,
    )
    best_k = choose_best_k(diagnostics)

    final_model = KMeans(
        n_clusters=best_k,
        random_state=args.random_state,
        n_init=args.kmeans_n_init,
    )
    final_labels = final_model.fit_predict(embeddings)

    cluster_summary = build_cluster_summary(
        X_scaled=X_scaled,
        embeddings=embeddings,
        labels=final_labels,
        model=final_model,
        pca=pca,
        feature_names=feature_names,
        row_index=row_index,
        top_features_per_cluster=args.top_features_per_cluster,
        sample_titles_per_cluster=args.sample_titles_per_cluster,
    )

    representation_npz_path = out_dir / "representation_artifacts.npz"
    embeddings_npy_path = out_dir / "embeddings.npy"
    cluster_labels_json_path = out_dir / f"cluster_labels_k{best_k}.json"
    diagnostics_json_path = out_dir / "cluster_diagnostics.json"
    cluster_summary_json_path = out_dir / f"cluster_summary_k{best_k}.json"
    training_report_json_path = out_dir / "training_report.json"

    if pca is not None:
        np.savez_compressed(
            representation_npz_path,
            scaler_mean=mean.astype(np.float32),
            scaler_scale=std.astype(np.float32),
            pca_components=pca.components_.astype(np.float32),
            pca_mean=pca.mean_.astype(np.float32),
            pca_explained_variance=pca.explained_variance_.astype(np.float32),
            pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            feature_names=np.array(feature_names, dtype=object),
        )
    else:
        np.savez_compressed(
            representation_npz_path,
            scaler_mean=mean.astype(np.float32),
            scaler_scale=std.astype(np.float32),
            pca_components=np.empty((0, 0), dtype=np.float32),
            pca_mean=np.empty((0,), dtype=np.float32),
            pca_explained_variance=np.empty((0,), dtype=np.float32),
            pca_explained_variance_ratio=np.empty((0,), dtype=np.float32),
            feature_names=np.array(feature_names, dtype=object),
        )

    np.save(embeddings_npy_path, embeddings.astype(np.float32))

    labeled_rows = []
    for meta, label in zip(row_index, final_labels, strict=False):
        labeled_rows.append(
            {
                **meta,
                "cluster": int(label),
            }
        )

    save_json(cluster_labels_json_path, labeled_rows)
    save_json(diagnostics_json_path, diagnostics)
    save_json(cluster_summary_json_path, cluster_summary)

    report = {
        "input_artifacts_dir": str(artifacts_dir),
        "output_dir": str(out_dir),
        "records": int(X.shape[0]),
        "input_feature_count": int(X.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "used_pca": pca is not None,
        "pca_components_requested": int(args.pca_components),
        "pca_components_used": int(actual_pca_components),
        "pca_total_explained_variance_ratio": float(explained_variance_ratio.sum()) if explained_variance_ratio.size else None,
        "pca_explained_variance_ratio_first_10": explained_variance_ratio[:10].tolist(),
        "pca_explained_variance_ratio_cumulative_first_10": explained_variance_cumulative[:10].tolist(),
        "diagnostic_ks": args.diagnostic_ks,
        "best_k": int(best_k),
        "diagnostics": diagnostics,
        "cluster_sizes": {
            str(cluster_id): int((final_labels == cluster_id).sum())
            for cluster_id in sorted(np.unique(final_labels).tolist())
        },
        "feature_artifacts": {
            "representation_artifacts_npz": str(representation_npz_path),
            "embeddings_npy": str(embeddings_npy_path),
            "cluster_labels_json": str(cluster_labels_json_path),
            "cluster_diagnostics_json": str(diagnostics_json_path),
            "cluster_summary_json": str(cluster_summary_json_path),
        },
        "featurizer_report": featurizer_report,
    }

    save_json(training_report_json_path, report)

    print("=== Representation Training Summary ===")
    print(f"records: {X.shape[0]}")
    print(f"input_feature_count: {X.shape[1]}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    print(f"used_pca: {pca is not None}")
    if pca is not None:
        print(f"pca_total_explained_variance_ratio: {explained_variance_ratio.sum():.4f}")
    print(f"diagnostic_ks: {args.diagnostic_ks}")
    print(f"best_k: {best_k}")
    print(f"wrote: {representation_npz_path}")
    print(f"wrote: {embeddings_npy_path}")
    print(f"wrote: {cluster_labels_json_path}")
    print(f"wrote: {diagnostics_json_path}")
    print(f"wrote: {cluster_summary_json_path}")
    print(f"wrote: {training_report_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())