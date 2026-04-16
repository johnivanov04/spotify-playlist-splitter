#!/usr/bin/env python3
"""
Export v1_9 representation artifacts to kmeans_model.json for the client.

Usage:
    python ml_pipeline/scripts/export_model_json.py \
        --representation ml_pipeline/data/artifacts/representation_v1_9 \
        --featurizer-report ml_pipeline/data/artifacts/features_v1_9/featurizer_report.json \
        --output client/src/ml/kmeans_model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--featurizer-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--float-precision", type=int, default=6)
    args = parser.parse_args()

    rep_path = Path(args.representation) / "representation_artifacts.npz"
    d = np.load(rep_path, allow_pickle=True)

    report = json.loads(Path(args.featurizer_report).read_text(encoding="utf-8"))

    def f(arr: np.ndarray, precision: int = args.float_precision) -> list:
        return [round(float(v), precision) for v in arr.ravel()]

    feature_names = [str(n) for n in d["feature_names"]]

    model = {
        "version": "v1_9",
        "feature_names": feature_names,
        "scaler_mean": f(d["scaler_mean"]),
        "scaler_scale": f(d["scaler_scale"]),
        "pca_components": [f(row) for row in d["pca_components"]],
        "pca_mean": f(d["pca_mean"]),
        "year_mean": report["matrix"]["year_mean"],
        "year_std": report["matrix"]["year_std"],
        "weights": {
            "tag": float(d["input_tag_weight"]),
            "genre": float(d["input_genre_weight"]),
            "acoustic": float(d["input_acoustic_weight"]),
            "year": float(d["input_year_weight"]),
            "popularity": float(d["input_popularity_weight"]),
            "binary": float(d["input_binary_weight"]),
        },
    }

    out = Path(args.output)
    out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"Exported {len(feature_names)} features, {len(model['pca_components'])} PCA components")
    print(f"Output: {out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
