#!/usr/bin/env python3
"""
Audit which genres/tags are well-covered vs underrepresented in the corpus.

Counts spotify_artist_genres, brainz.genres, and brainz.tags occurrences,
shows distribution, and flags reference genres below a target threshold so
we know where the corpus has gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# A pragmatic reference of common music genres a general-purpose model should
# cover. Each is paired with a target track count we'd like to see in the
# corpus before considering it "well-covered."
REFERENCE_GENRES: dict[str, int] = {
    # Popular / mainstream
    "pop": 200, "rock": 200, "indie": 150, "alternative rock": 100,
    "hip hop": 200, "rap": 200, "r&b": 100, "soul": 80,
    "electronic": 100, "house": 80, "techno": 50, "trance": 30,
    "edm": 50, "drum and bass": 30, "dubstep": 30, "idm": 30,
    # Genre families
    "jazz": 100, "classical": 80, "country": 100, "metal": 80,
    "folk": 50, "blues": 50, "reggae": 30, "punk": 50,
    # Regional / international
    "k-pop": 50, "j-pop": 30, "latin": 80, "reggaeton": 50,
    "afrobeat": 30, "afrobeats": 30, "salsa": 30, "bossa nova": 20,
    "bollywood": 20, "indian": 30, "arabic": 20,
    # Subgenres / niches
    "shoegaze": 30, "emo": 30, "grunge": 30, "post punk": 30,
    "synth pop": 50, "psychedelic rock": 50, "neo soul": 30,
    "ambient": 30, "lo-fi": 50, "chillwave": 30, "vaporwave": 20,
    "trap": 80, "drill": 30, "boom bap": 30, "conscious hip hop": 30,
    "bedroom pop": 50, "dream pop": 50, "indie pop": 80,
    "art pop": 30, "hyperpop": 20, "dance pop": 50,
}


def iter_genre_names(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for g in (record.get("spotify_artist_genres") or []):
        name = g.get("name") if isinstance(g, dict) else g
        if isinstance(name, str) and name.strip():
            out.append(name.strip().lower())
    for g in ((record.get("brainz") or {}).get("genres") or []):
        name = g.get("name") if isinstance(g, dict) else g
        if isinstance(name, str) and name.strip():
            out.append(name.strip().lower())
    return out


def iter_tag_names(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for t in ((record.get("brainz") or {}).get("tags") or []):
        name = t.get("name") if isinstance(t, dict) else t
        if isinstance(name, str) and name.strip():
            out.append(name.strip().lower())
    return out


def bucket(count: int) -> str:
    if count >= 100: return ">=100"
    if count >= 50: return "50-99"
    if count >= 20: return "20-49"
    if count >= 5: return "5-19"
    return "<5"


def print_distribution(label: str, counter: Counter[str]) -> None:
    print(f"\n{label} distribution ({len(counter)} unique terms):")
    bands = ["<5", "5-19", "20-49", "50-99", ">=100"]
    band_counts = Counter()
    for name, c in counter.items():
        band_counts[bucket(c)] += 1
    for b in bands:
        print(f"  {b:>6}: {band_counts.get(b, 0)} terms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit corpus genre/tag coverage.")
    parser.add_argument(
        "--input",
        default="ml_pipeline/data/processed/song_records.backfilled.ndjson",
        help="Corpus ndjson to audit.",
    )
    parser.add_argument("--top", type=int, default=30, help="Show top N most-covered genres.")
    parser.add_argument("--report", help="Optional JSON output path with full counts and gaps.")
    args = parser.parse_args(argv or sys.argv[1:])

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    genre_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    for r in records:
        for g in set(iter_genre_names(r)):
            genre_counts[g] += 1
        for t in set(iter_tag_names(r)):
            tag_counts[t] += 1

    print(f"=== Corpus Coverage Audit ===")
    print(f"Records: {len(records)}")

    print_distribution("Genres (spotify + brainz)", genre_counts)
    print_distribution("Brainz tags", tag_counts)

    print(f"\nTop {args.top} best-covered genres:")
    for name, c in genre_counts.most_common(args.top):
        print(f"  {c:5d}  {name}")

    print(f"\nReference-genre coverage gaps (have < target):")
    gaps = []
    for ref, target in sorted(REFERENCE_GENRES.items()):
        have = genre_counts.get(ref, 0)
        if have < target:
            gaps.append((ref, have, target))
            print(f"  {ref:<22s} have={have:4d}  target={target:4d}  short by {target - have}")
    if not gaps:
        print("  (none — all reference genres meet target)")

    if args.report:
        report = {
            "total_records": len(records),
            "genre_counts": dict(genre_counts),
            "tag_counts": dict(tag_counts),
            "reference_gaps": [
                {"genre": ref, "have": have, "target": target, "short_by": target - have}
                for ref, have, target in gaps
            ],
        }
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
