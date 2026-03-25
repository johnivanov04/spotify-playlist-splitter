import json
import argparse
from pathlib import Path
import random

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def safe_num(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def extract_ab_features(track):
    """
    Expects:
      track["brainz"]["acousticHighLevel"] to look like:
        { "highlevel": { classifier: { "all": {class: prob, ...}, ... }, ... }, ... }
    We robustly handle slight shape differences.
    """
    brainz = track.get("brainz") or {}
    ab = brainz.get("acousticHighLevel")
    if not isinstance(ab, dict):
        return {}

    high = ab.get("highlevel")
    if not isinstance(high, dict):
        # sometimes the API shape differs; try treating top-level as highlevel
        high = ab

    out = {}
    for clf_name, clf_obj in high.items():
        if not isinstance(clf_obj, dict):
            continue

        probs = clf_obj.get("all") or clf_obj.get("probabilities")
        if isinstance(probs, dict):
            for cls_name, p in probs.items():
                p = safe_num(p)
                if p is None:
                    continue
                out[f"ab__{clf_name}__{cls_name}"] = p
        else:
            # fallback: value/probability form
            val = clf_obj.get("value")
            p = safe_num(clf_obj.get("probability"))
            if val is not None and p is not None:
                out[f"ab__{clf_name}__{val}"] = p

    # a simple “has AB” flag helps the model deal with missing AB
    out["ab__has_acousticbrainz"] = 1.0
    return out


def extract_spotify_features(track):
    out = {}

    # core numeric features (some may be null if spotify audio-features is restricted)
    for k in ["energy", "valence", "danceability", "tempo", "popularity", "year"]:
        v = safe_num(track.get(k))
        out[f"sp__{k}"] = 0.0 if v is None else v
        out[f"sp__has_{k}"] = 0.0 if v is None else 1.0

    return out


def featurize_track(track):
    f = {}
    f.update(extract_spotify_features(track))
    f.update(extract_ab_features(track))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to exported playlist_..._dataset.json")
    ap.add_argument("--k", type=int, default=6, help="Number of clusters")
    ap.add_argument("--out_dir", default="artifacts", help="Output directory")
    ap.add_argument("--silhouette_sample", type=int, default=1500, help="Sample size for silhouette")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = json.loads(inp.read_text())
    if not isinstance(tracks, list):
        raise ValueError("Input JSON must be a list of tracks")

    # build feature dicts
    feat_dicts = [featurize_track(t) for t in tracks]

    # vectorize
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)

    # scale (works with sparse if with_mean=False)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)

    # train kmeans
    km = MiniBatchKMeans(n_clusters=args.k, random_state=42, batch_size=2048)
    labels = km.fit_predict(Xs)

    # silhouette (sampled)
    n = Xs.shape[0]
    sample_n = min(args.silhouette_sample, n)
    sil = None
    if sample_n >= 50 and args.k >= 2:
        idx = random.sample(range(n), sample_n)
        try:
            sil = float(silhouette_score(Xs[idx], labels[idx], metric="euclidean"))
        except Exception:
            sil = None

    # save assignments
    assignments = []
    for t, lab in zip(tracks, labels):
        assignments.append({
            "spotifyId": t.get("id"),
            "name": t.get("name"),
            "artists": t.get("artists"),
            "cluster": int(lab),
        })
    (out_dir / "cluster_assignments.json").write_text(json.dumps(assignments, indent=2))

    # export a portable JSON model for Node/React inference
    model = {
        "schema_version": 1,
        "k": int(args.k),
        "feature_names": vec.get_feature_names_out().tolist(),
        "scale": scaler.scale_.tolist(),              # for x_scaled = x_raw / scale
        "centroids": km.cluster_centers_.tolist(),    # in scaled space
        "silhouette_sample": sil,
    }
    (out_dir / "kmeans_model.json").write_text(json.dumps(model, indent=2))

    print(f"Saved to: {out_dir}")
    print(f"Tracks: {len(tracks)} | k={args.k} | silhouette(sample)={sil}")


if __name__ == "__main__":
    main()
