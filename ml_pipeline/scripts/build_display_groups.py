#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def norm(text: str) -> str:
    return (text or "").strip().lower()


def dedupe_preserve_order(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def pick_display_group(cluster_name: str, top_features: list[dict[str, Any]]) -> str:
    name = norm(cluster_name)

    # Primary routing: trust the cluster name first.
    if "mixed vibe cluster" in name:
        return "More Picks"

    if "breakbeat" in name or "soundtrack" in name:
        return "Instrumental / Breakbeat"

    if "cool jazz" in name or name == "jazz / cool jazz" or "jazz /" in name or "/ jazz" in name:
        return "Jazz / Nujabes / Downtempo"

    if any(k in name for k in ["singer songwriter", "acoustic"]):
        return "Acoustic / Chill"

    if any(k in name for k in [
        "rock", "indie", "alternative", "punk", "grunge", "emo"
    ]):
        return "Rock / Indie"

    if any(k in name for k in [
        "r and b", "r b", "soul", "neo soul", "alternative r and b", "contemporary r and b",
        "psychedelic soul",
    ]):
        return "R&B / Soul"

    if any(k in name for k in [
        "trap", "thug rap", "southern hip hop"
    ]):
        return "Trap"

    if any(k in name for k in [
        "rap", "hip hop", "pop rap", "conscious hip hop",
        "underground hip hop", "west coast hip hop", "hardcore hip hop"
    ]):
        return "Rap / Hip-Hop"

    # Only use top features as fallback when the cluster name is weak/generic.
    feat_names = [norm(f.get("pretty_feature", "")) for f in top_features]
    feat_text = " ".join(feat_names)

    if any(k in feat_text for k in [
        "breakbeat", "soundtrack", "trip hop"
    ]):
        return "Instrumental / Breakbeat"

    if any(k in feat_text for k in [
        "cool jazz", "jazz rap", "jazz", "downtempo"
    ]):
        return "Jazz / Nujabes / Downtempo"

    if any(k in feat_text for k in [
        "singer songwriter", "folk", "acoustic guitar"
    ]):
        return "Acoustic / Chill"

    if any(k in feat_text for k in [
        "rock", "indie", "alternative", "punk", "grunge", "britpop", "post punk"
    ]):
        return "Rock / Indie"

    if any(k in feat_text for k in [
        "r and b", "soul", "neo soul", "alternative r and b", "contemporary r and b",
        "psychedelic soul",
    ]):
        return "R&B / Soul"

    if any(k in feat_text for k in [
        "trap", "thug rap", "southern hip hop"
    ]):
        return "Trap"

    if any(k in feat_text for k in [
        "rap", "hip hop", "pop rap", "conscious hip hop",
        "underground hip hop", "west coast hip hop", "hardcore hip hop"
    ]):
        return "Rap / Hip-Hop"

    if any(k in feat_text for k in [
        "mood party", "danceable", "voice instrumental",
        "timbre / dark", "timbre / bright", "aggressive"
    ]):
        return "Melodic / Party Rap"

    return "More Picks"


def build_display_groups(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cluster in clusters:
        display_group = pick_display_group(
            cluster.get("cluster_name", ""),
            cluster.get("top_features", []),
        )
        grouped[display_group].append(cluster)

    merged: list[dict[str, Any]] = []
    for display_group, members in grouped.items():
        members = sorted(members, key=lambda c: c.get("size", 0), reverse=True)
        size = sum(c.get("size", 0) for c in members)

        sample_tracks: list[str] = []
        for cluster in members:
            sample_tracks.extend(cluster.get("sample_tracks", []))
        sample_tracks = dedupe_preserve_order(sample_tracks, limit=12)

        merged_top_features: list[str] = []
        for cluster in members:
            for f in cluster.get("top_features", []):
                pretty = f.get("pretty_feature", "")
                score = float(f.get("score", 0.0) or 0.0)
                if pretty and score >= 0.15:
                    merged_top_features.append(pretty)
        merged_top_features = dedupe_preserve_order(merged_top_features, limit=10)

        merged.append({
            "display_group": display_group,
            "size": size,
            "member_cluster_ids": [c["cluster_id"] for c in members],
            "member_cluster_names": [c["cluster_name"] for c in members],
            "top_descriptors": merged_top_features,
            "sample_tracks": sample_tracks,
        })

    merged.sort(key=lambda x: x["size"], reverse=True)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge backend playlist clusters into cleaner display groups."
    )
    parser.add_argument("--input", required=True, help="Path to playlist_cluster_summary.json")
    parser.add_argument("--output", help="Optional output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    clusters = json.loads(input_path.read_text(encoding="utf-8"))

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name("playlist_display_groups.json")
    )

    merged = build_display_groups(clusters)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print("=== Display Group Summary ===")
    for group in merged:
        print(f"{group['display_group']}: {group['size']} tracks")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
