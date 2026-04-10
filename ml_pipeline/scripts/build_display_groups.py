#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def norm(text: str) -> str:
    return (text or "").strip().lower()


def pick_display_group(cluster_name: str, top_features: list[dict[str, Any]]) -> str:
    name = norm(cluster_name)
    feats = " ".join(norm(f.get("pretty_feature", "")) for f in top_features)
    text = f"{name} {feats}"

    if "mixed vibe cluster" in text:
        return "More Vibes"

    if any(k in text for k in [
        "breakbeat", "soundtrack", "trip hop", "downtempo", "instrumentalhiphop"
    ]):
        return "Instrumental / Breakbeat"

    if any(k in text for k in [
        "cool jazz", "jazz rap", "jazz", "nujabes"
    ]):
        return "Jazz / Nujabes / Downtempo"

    if any(k in text for k in [
        "r and b", "soul", "neo soul", "alternative r and b", "contemporary r and b"
    ]):
        return "R&B / Soul"

    if any(k in text for k in [
        "trap", "thug rap", "southern hip hop"
    ]):
        return "Trap"

    if any(k in text for k in [
        "mood party", "danceable", "voice instrumental", "timbre / dark",
        "timbre / bright", "aggressive"
    ]):
        return "Melodic / Party Rap"

    if any(k in text for k in [
        "pop rap", "hip hop", "rap", "conscious hip hop",
        "underground hip hop", "west coast hip hop", "hardcore hip hop"
    ]):
        return "Rap / Hip-Hop"

    return "More Vibes"


def dedupe_preserve_order(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


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
            merged_top_features.extend(
                f.get("pretty_feature", "")
                for f in cluster.get("top_features", [])
                if f.get("pretty_feature")
            )
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