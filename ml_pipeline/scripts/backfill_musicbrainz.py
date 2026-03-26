#!/usr/bin/env python3
"""
Backfill MusicBrainz / AcousticBrainz metadata into normalized song records.

Reads a normalized corpus produced by build_song_corpus.py and attempts to fill
missing:
- ids.musicbrainz_recording_id
- brainz.tags
- brainz.genres
- brainz.acoustic_high_level
- brainz.acoustic_low_level (optional)

Resolution order:
1) ISRC -> MusicBrainz recording
2) Title + primary artist search fallback

The script is intentionally stdlib-only and caches API responses to reduce
repeated calls across runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

MB_BASE = "https://musicbrainz.org/ws/2"
AB_BASE = "https://acousticbrainz.org/api/v1"
DEFAULT_USER_AGENT = "spotify-playlist-splitter-backfill/0.1 (local script)"
DEFAULT_SLEEP_SECONDS = 1.1


@dataclass
class BackfillStats:
    records_seen: int = 0
    records_changed: int = 0
    records_unchanged: int = 0
    skipped_no_lookup_key: int = 0

    mbid_resolved_isrc: int = 0
    mbid_resolved_search: int = 0
    mbid_already_present: int = 0
    mbid_lookup_failed: int = 0

    recording_payload_fetched: int = 0
    tags_filled: int = 0
    genres_filled: int = 0
    acoustic_high_filled: int = 0
    acoustic_low_filled: int = 0

    api_errors: int = 0
    cache_hits: int = 0


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def slugify(text: Any) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def normalize_isrc(value: Any) -> str | None:
    text = normalize_text(value).upper()
    return text or None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if path.suffix.lower() == ".ndjson":
        rows: list[dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"NDJSON line {i} is not an object")
            rows.append(obj)
        return rows, "ndjson"

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("JSON input must be an array of record objects")
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            raise ValueError(f"JSON array item {i} is not an object")
    return payload, "json"


def write_records(path: Path, records: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "ndjson":
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class CacheStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = load_json_if_exists(
            path,
            {
                "schema_version": 1,
                "isrc_to_mbid": {},
                "search_to_mbid": {},
                "recording": {},
                "ab_high": {},
                "ab_low": {},
            },
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


class ApiClient:
    def __init__(self, user_agent: str, sleep_seconds: float, timeout: float, verbose: bool = False):
        self.user_agent = user_agent
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.verbose = verbose
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.sleep_seconds:
            time.sleep(self.sleep_seconds - elapsed)

    def get_json(self, url: str, accept_404: bool = True, retries: int = 3) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            req = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                    self._last_request_ts = time.monotonic()
                    if self.verbose:
                        print(f"GET {url} -> {getattr(resp, 'status', 'ok')}")
                    return json.loads(body)
            except HTTPError as exc:
                self._last_request_ts = time.monotonic()
                if accept_404 and exc.code == 404:
                    return None
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep((attempt + 1) * 1.5)
                    last_exc = exc
                    continue
                raise
            except URLError as exc:
                self._last_request_ts = time.monotonic()
                if attempt < retries - 1:
                    time.sleep((attempt + 1) * 1.5)
                    last_exc = exc
                    continue
                raise
            except Exception as exc:
                self._last_request_ts = time.monotonic()
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep((attempt + 1) * 1.5)
                    continue
                raise
        if last_exc:
            raise last_exc
        return None


def normalize_name_count_items(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if isinstance(value, dict):
        for name, count in value.items():
            norm_name = normalize_text(name)
            if norm_name:
                items.append({"name": norm_name, "count": safe_float(count) or 1.0})
        return sorted(items, key=lambda x: (-x["count"], x["name"]))

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = normalize_text(item.get("name"))
                if not name:
                    continue
                count = safe_float(item.get("count"))
                if count is None:
                    count = safe_float(item.get("value"))
                items.append({"name": name, "count": count if count is not None else 1.0})
            else:
                name = normalize_text(item)
                if name:
                    items.append({"name": name, "count": 1.0})
        return sorted(items, key=lambda x: (-x["count"], x["name"]))

    return []


def extract_primary_artist(record: dict[str, Any]) -> str | None:
    artists = record.get("artists") or []
    if not artists:
        return None
    artist = normalize_text(artists[0])
    return artist or None


def text_match_score(record_title: str, record_artist: str | None, candidate: dict[str, Any]) -> int:
    score = 0
    cand_title = normalize_text(candidate.get("title")).lower()
    cand_artists = []

    for ac in candidate.get("artist-credit") or []:
        if isinstance(ac, dict):
            name = normalize_text((ac.get("artist") or {}).get("name") or ac.get("name"))
            if name:
                cand_artists.append(name.lower())

    norm_title = normalize_text(record_title).lower()
    if cand_title == norm_title:
        score += 50
    elif slugify(cand_title) == slugify(norm_title):
        score += 35
    elif norm_title and norm_title in cand_title:
        score += 20

    if record_artist:
        ra = normalize_text(record_artist).lower()
        if ra in cand_artists:
            score += 40
        elif any(slugify(ra) == slugify(a) for a in cand_artists):
            score += 25

    try:
        score += int(candidate.get("score") or 0)
    except Exception:
        pass

    return score


def search_mbid_by_isrc(isrc: str, api: ApiClient, cache: CacheStore, stats: BackfillStats) -> str | None:
    cached = cache.data["isrc_to_mbid"].get(isrc)
    if cached is not None:
        stats.cache_hits += 1
        return cached or None

    url = f"{MB_BASE}/isrc/{quote(isrc)}?fmt=json"
    data = api.get_json(url, accept_404=True)
    recs = []
    if isinstance(data, dict):
        recs = data.get("recordings") or data.get("recording_list") or data.get("recording-list") or []

    mbid = None
    if isinstance(recs, list) and recs:
        first = recs[0]
        if isinstance(first, dict):
            mbid = normalize_text(first.get("id")) or None

    cache.data["isrc_to_mbid"][isrc] = mbid
    return mbid


def search_mbid_by_title_artist(title: str, artist: str | None, api: ApiClient, cache: CacheStore, stats: BackfillStats) -> str | None:
    query_key = f"{slugify(title)}::{slugify(artist or '')}"
    cached = cache.data["search_to_mbid"].get(query_key)
    if cached is not None:
        stats.cache_hits += 1
        return cached or None

    clauses = []
    if normalize_text(title):
        clauses.append(f'recording:"{normalize_text(title)}"')
    if normalize_text(artist):
        clauses.append(f'artist:"{normalize_text(artist)}"')
    if not clauses:
        cache.data["search_to_mbid"][query_key] = None
        return None

    query = " AND ".join(clauses)
    params = urlencode({"query": query, "fmt": "json", "limit": 5})
    url = f"{MB_BASE}/recording/?{params}"
    data = api.get_json(url, accept_404=True)

    candidates = []
    if isinstance(data, dict):
        candidates = data.get("recordings") or []

    best_mbid = None
    best_score = -10**9
    for cand in candidates if isinstance(candidates, list) else []:
        if not isinstance(cand, dict):
            continue
        score = text_match_score(title, artist, cand)
        cand_mbid = normalize_text(cand.get("id")) or None
        if cand_mbid and score > best_score:
            best_score = score
            best_mbid = cand_mbid

    # avoid locking in a very weak match
    if best_score < 35:
        best_mbid = None

    cache.data["search_to_mbid"][query_key] = best_mbid
    return best_mbid


def fetch_recording_payload(mbid: str, api: ApiClient, cache: CacheStore, stats: BackfillStats) -> dict[str, Any] | None:
    cached = cache.data["recording"].get(mbid)
    if cached is not None:
        stats.cache_hits += 1
        return cached or None

    url = f"{MB_BASE}/recording/{quote(mbid)}?fmt=json&inc=tags+genres+artist-credits"
    data = api.get_json(url, accept_404=True)
    cache.data["recording"][mbid] = data
    return data if isinstance(data, dict) else None


def fetch_ab_payload(kind: str, mbid: str, api: ApiClient, cache: CacheStore, stats: BackfillStats) -> dict[str, Any] | None:
    cache_key = "ab_high" if kind == "high" else "ab_low"
    cached = cache.data[cache_key].get(mbid)
    if cached is not None:
        stats.cache_hits += 1
        return cached or None

    suffix = "high-level" if kind == "high" else "low-level"
    url = f"{AB_BASE}/{quote(mbid)}/{suffix}"
    data = api.get_json(url, accept_404=True)
    cache.data[cache_key][mbid] = data
    return data if isinstance(data, dict) else None


def should_fetch_recording(record: dict[str, Any], overwrite_existing: bool) -> bool:
    brainz = record.get("brainz") or {}
    tags = brainz.get("tags") or []
    genres = brainz.get("genres") or []
    return overwrite_existing or not tags or not genres


def should_fetch_ab_high(record: dict[str, Any], overwrite_existing: bool) -> bool:
    brainz = record.get("brainz") or {}
    ah = brainz.get("acoustic_high_level")
    return overwrite_existing or not (isinstance(ah, dict) and ah)


def should_fetch_ab_low(record: dict[str, Any], overwrite_existing: bool) -> bool:
    brainz = record.get("brainz") or {}
    al = brainz.get("acoustic_low_level")
    return overwrite_existing or not (isinstance(al, dict) and al)


def backfill_record(
    record: dict[str, Any],
    api: ApiClient,
    cache: CacheStore,
    stats: BackfillStats,
    *,
    use_search_fallback: bool,
    include_acoustic: bool,
    include_low_level: bool,
    overwrite_existing: bool,
    verbose: bool,
) -> bool:
    stats.records_seen += 1
    changed = False

    ids = record.setdefault("ids", {})
    brainz = record.setdefault("brainz", {})
    debug = record.get("debug")
    if debug is None:
        debug = {}
        record["debug"] = debug

    mbid = normalize_text(ids.get("musicbrainz_recording_id")) or None
    if mbid:
        stats.mbid_already_present += 1
    else:
        isrc = normalize_isrc(ids.get("isrc"))
        title = normalize_text(record.get("title"))
        artist = extract_primary_artist(record)

        if isrc:
            try:
                mbid = search_mbid_by_isrc(isrc, api, cache, stats)
            except Exception as exc:
                stats.api_errors += 1
                if verbose:
                    print(f"ISRC lookup failed for {isrc}: {exc}", file=sys.stderr)
            if mbid:
                ids["musicbrainz_recording_id"] = mbid
                debug["mbid_source"] = "isrc"
                stats.mbid_resolved_isrc += 1
                changed = True

        if not mbid and use_search_fallback and title and artist:
            try:
                mbid = search_mbid_by_title_artist(title, artist, api, cache, stats)
            except Exception as exc:
                stats.api_errors += 1
                if verbose:
                    print(f"Search lookup failed for {title} / {artist}: {exc}", file=sys.stderr)
            if mbid:
                ids["musicbrainz_recording_id"] = mbid
                debug["mbid_source"] = "search"
                stats.mbid_resolved_search += 1
                changed = True

        if not mbid:
            if not isrc and not (title and artist):
                stats.skipped_no_lookup_key += 1
            stats.mbid_lookup_failed += 1
            stats.records_unchanged += 1
            return changed

    if should_fetch_recording(record, overwrite_existing):
        try:
            payload = fetch_recording_payload(mbid, api, cache, stats)
        except Exception as exc:
            stats.api_errors += 1
            payload = None
            if verbose:
                print(f"Recording fetch failed for {mbid}: {exc}", file=sys.stderr)

        if isinstance(payload, dict):
            stats.recording_payload_fetched += 1
            new_tags = normalize_name_count_items(payload.get("tags"))
            new_genres = normalize_name_count_items(payload.get("genres"))

            if overwrite_existing or not (brainz.get("tags") or []):
                if new_tags:
                    brainz["tags"] = new_tags
                    stats.tags_filled += 1
                    changed = True
                else:
                    brainz.setdefault("tags", [])

            if overwrite_existing or not (brainz.get("genres") or []):
                if new_genres:
                    brainz["genres"] = new_genres
                    stats.genres_filled += 1
                    changed = True
                else:
                    brainz.setdefault("genres", [])

            debug.setdefault("musicbrainz_title", payload.get("title"))

    if include_acoustic and should_fetch_ab_high(record, overwrite_existing):
        try:
            high = fetch_ab_payload("high", mbid, api, cache, stats)
        except Exception as exc:
            stats.api_errors += 1
            high = None
            if verbose:
                print(f"Acoustic high-level fetch failed for {mbid}: {exc}", file=sys.stderr)

        if isinstance(high, dict) and high:
            brainz["acoustic_high_level"] = high
            stats.acoustic_high_filled += 1
            changed = True
        else:
            brainz.setdefault("acoustic_high_level", None)

    if include_low_level and should_fetch_ab_low(record, overwrite_existing):
        try:
            low = fetch_ab_payload("low", mbid, api, cache, stats)
        except Exception as exc:
            stats.api_errors += 1
            low = None
            if verbose:
                print(f"Acoustic low-level fetch failed for {mbid}: {exc}", file=sys.stderr)

        if isinstance(low, dict) and low:
            brainz["acoustic_low_level"] = low
            stats.acoustic_low_filled += 1
            changed = True
        else:
            brainz.setdefault("acoustic_low_level", None)

    if changed:
        stats.records_changed += 1
    else:
        stats.records_unchanged += 1
    return changed


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for r in records:
        ids = r.get("ids") or {}
        brainz = r.get("brainz") or {}
        if ids.get("musicbrainz_recording_id"):
            counts["with_mbid"] += 1
        if ids.get("isrc"):
            counts["with_isrc"] += 1
        if brainz.get("tags"):
            counts["with_tags"] += 1
        if brainz.get("genres"):
            counts["with_genres"] += 1
        if isinstance(brainz.get("acoustic_high_level"), dict) and brainz.get("acoustic_high_level"):
            counts["with_acoustic_high_level"] += 1
        if isinstance(brainz.get("acoustic_low_level"), dict) and brainz.get("acoustic_low_level"):
            counts["with_acoustic_low_level"] += 1
    counts["records"] = len(records)
    return dict(counts)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill MusicBrainz metadata into normalized song records.")
    ap.add_argument("--input", default="ml_pipeline/data/processed/song_records.json")
    ap.add_argument("--output", default=None, help="Where to write the updated corpus. Defaults to input path.")
    ap.add_argument(
        "--cache",
        default="ml_pipeline/data/cache/musicbrainz_backfill_cache.json",
        help="Persistent cache file to reduce repeated API calls.",
    )
    ap.add_argument("--report", default=None, help="Optional JSON report output path.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N records before processing.")
    ap.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    ap.add_argument("--no-search-fallback", action="store_true", help="Disable title+artist search fallback.")
    ap.add_argument("--include-acoustic", action="store_true", help="Fetch AcousticBrainz high-level data.")
    ap.add_argument("--include-low-level", action="store_true", help="Fetch AcousticBrainz low-level data too.")
    ap.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing tags/genres/acoustic payloads.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-every", type=int, default=25, help="Persist cache every N processed records.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    cache_path = Path(args.cache)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        records, fmt = load_records(input_path)
    except Exception as exc:
        print(f"Failed to load records: {exc}", file=sys.stderr)
        return 1

    api = ApiClient(
        user_agent=args.user_agent,
        sleep_seconds=max(0.0, args.sleep_seconds),
        timeout=max(1.0, args.timeout),
        verbose=args.verbose,
    )
    cache = CacheStore(cache_path)
    stats = BackfillStats()

    start = max(0, args.offset)
    end = len(records) if args.limit is None else min(len(records), start + max(0, args.limit))

    if start >= len(records):
        print("Offset is past the end of the corpus; nothing to do.")
        return 0

    for idx in range(start, end):
        rec = records[idx]
        try:
            backfill_record(
                rec,
                api,
                cache,
                stats,
                use_search_fallback=not args.no_search_fallback,
                include_acoustic=args.include_acoustic,
                include_low_level=args.include_low_level,
                overwrite_existing=args.overwrite_existing,
                verbose=args.verbose,
            )
        except Exception as exc:
            stats.api_errors += 1
            if args.verbose:
                title = normalize_text(rec.get("title"))
                print(f"Record failed: {title} ({exc})", file=sys.stderr)

        processed_count = idx - start + 1
        if not args.dry_run and processed_count % max(1, args.save_every) == 0:
            cache.save()

    summary = summarize_records(records)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "processed_range": {"offset": start, "end_exclusive": end},
        "stats": stats.__dict__,
        "summary_after_backfill": summary,
    }

    if not args.dry_run:
        write_records(output_path, records, fmt)
        cache.save()

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Backfill Summary ===")
    print(f"records_seen: {stats.records_seen}")
    print(f"records_changed: {stats.records_changed}")
    print(f"records_unchanged: {stats.records_unchanged}")
    print(f"mbid_resolved_isrc: {stats.mbid_resolved_isrc}")
    print(f"mbid_resolved_search: {stats.mbid_resolved_search}")
    print(f"mbid_already_present: {stats.mbid_already_present}")
    print(f"mbid_lookup_failed: {stats.mbid_lookup_failed}")
    print(f"tags_filled: {stats.tags_filled}")
    print(f"genres_filled: {stats.genres_filled}")
    print(f"acoustic_high_filled: {stats.acoustic_high_filled}")
    print(f"acoustic_low_filled: {stats.acoustic_low_filled}")
    print(f"cache_hits: {stats.cache_hits}")
    print(f"api_errors: {stats.api_errors}")

    print("\n=== Coverage After Backfill ===")
    for key in [
        "records",
        "with_isrc",
        "with_mbid",
        "with_tags",
        "with_genres",
        "with_acoustic_high_level",
        "with_acoustic_low_level",
    ]:
        print(f"{key}: {summary.get(key, 0)}")

    if args.dry_run:
        print("\nDry run only: no files were written.")
    else:
        print(f"\nWrote updated corpus to: {output_path}")
        print(f"Wrote cache to: {cache_path}")
        if args.report:
            print(f"Wrote report to: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
