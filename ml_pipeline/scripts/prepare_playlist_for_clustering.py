#!/usr/bin/env python3
"""
Prepare one playlist for playlist-time clustering.

Workflow:
1. load raw playlist export
2. normalize into canonical song records
3. enrich from local corpus lookup first
4. optionally do live MusicBrainz / AcousticBrainz backfill only for remaining gaps
5. write prepared playlist JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi


# ----------------------------
# general helpers
# ----------------------------

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_token(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("&", " and ")
    value = _TOKEN_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def extract_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\b(19|20)\d{2}\b", value)
        if m:
            return int(m.group(0))
    return None


def normalize_isrc(value: Any) -> str | None:
    if not is_nonempty_string(value):
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", value).upper().strip()
    return s or None


def normalize_name_count_list(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out

    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        count = item.get("count", 1)
        if not is_nonempty_string(name):
            continue
        token = name.strip()
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            count_num = float(count)
        except Exception:
            count_num = 1.0
        out.append({"name": token, "count": count_num})
    return out


def get_artists(raw: dict[str, Any]) -> list[str]:
    artists = raw.get("artists")
    out: list[str] = []

    if isinstance(artists, list):
        for a in artists:
            if isinstance(a, str) and a.strip():
                out.append(a.strip())
            elif isinstance(a, dict):
                name = a.get("name")
                if is_nonempty_string(name):
                    out.append(name.strip())

    if out:
        return out

    artist = raw.get("artist")
    if is_nonempty_string(artist):
        return [artist.strip()]

    artist_name = raw.get("artistName")
    if is_nonempty_string(artist_name):
        return [artist_name.strip()]

    return []


def get_album_name(raw: dict[str, Any]) -> str | None:
    album = raw.get("album")
    if isinstance(album, dict):
        name = album.get("name")
        if is_nonempty_string(name):
            return name.strip()
    if is_nonempty_string(raw.get("album")):
        return raw["album"].strip()
    if is_nonempty_string(raw.get("albumName")):
        return raw["albumName"].strip()
    return None


def get_year_from_raw(raw: dict[str, Any]) -> int | None:
    for key in ("year", "release_year", "releaseYear"):
        y = extract_year(raw.get(key))
        if y is not None:
            return y

    album = raw.get("album")
    if isinstance(album, dict):
        for key in ("release_date", "releaseDate"):
            y = extract_year(album.get(key))
            if y is not None:
                return y

    for key in ("release_date", "releaseDate"):
        y = extract_year(raw.get(key))
        if y is not None:
            return y

    return None


def get_duration_ms(raw: dict[str, Any]) -> int | None:
    for key in ("duration_ms", "durationMs"):
        val = safe_int(raw.get(key))
        if val is not None:
            return val
    return None


def get_spotify_url(raw: dict[str, Any]) -> str | None:
    external_urls = raw.get("external_urls")
    if isinstance(external_urls, dict) and is_nonempty_string(external_urls.get("spotify")):
        return external_urls["spotify"].strip()
    if is_nonempty_string(raw.get("spotifyUrl")):
        return raw["spotifyUrl"].strip()
    return None


def get_preview_url(raw: dict[str, Any]) -> str | None:
    if is_nonempty_string(raw.get("preview_url")):
        return raw["preview_url"].strip()
    if is_nonempty_string(raw.get("previewUrl")):
        return raw["previewUrl"].strip()
    return None


def get_image_url(raw: dict[str, Any]) -> str | None:
    album = raw.get("album")
    if isinstance(album, dict):
        images = album.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict) and is_nonempty_string(first.get("url")):
                return first["url"].strip()
    if is_nonempty_string(raw.get("imageUrl")):
        return raw["imageUrl"].strip()
    if is_nonempty_string(raw.get("image_url")):
        return raw["image_url"].strip()
    return None


def normalized_artists_key(artists: list[str]) -> str | None:
    vals = [normalize_token(a) for a in artists if is_nonempty_string(a)]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return "::".join(vals)


def make_track_id(
    spotify_id: str | None,
    isrc: str | None,
    title: str | None,
    artists: list[str],
) -> str:
    if spotify_id:
        return f"spotify:{spotify_id}"
    if isrc:
        return f"isrc:{isrc}"
    payload = json.dumps(
        {"title": title or "", "artists": artists},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"track:{digest}"


# ----------------------------
# loading raw playlist payload
# ----------------------------

def load_input_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_track_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("tracks", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise ValueError("Input JSON must be a list of track objects or an object containing a 'tracks' or 'items' list")


def looks_normalized(record: dict[str, Any]) -> bool:
    return (
        isinstance(record.get("ids"), dict)
        and isinstance(record.get("brainz"), dict)
        and isinstance(record.get("runtime_metadata"), dict)
        and isinstance(record.get("source"), dict)
    )


# ----------------------------
# normalization
# ----------------------------

def normalize_raw_record(
    raw: dict[str, Any],
    *,
    dataset_name: str,
    source_file: str,
) -> dict[str, Any]:
    if looks_normalized(raw):
        record = dict(raw)
        source = dict(record.get("source") or {})
        source.setdefault("dataset_name", dataset_name)
        source.setdefault("dataset_version", "0.1.0")
        source.setdefault("ingested_at_utc", now_utc_iso())
        source_files = source.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            source["source_files"] = [source_file]
        record["source"] = source
        return record

    title = raw.get("title") if is_nonempty_string(raw.get("title")) else raw.get("name")
    title = title.strip() if is_nonempty_string(title) else None

    artists = get_artists(raw)
    album_name = get_album_name(raw)
    year = get_year_from_raw(raw)
    duration_ms = get_duration_ms(raw)

    spotify_id = raw.get("spotify_id") if is_nonempty_string(raw.get("spotify_id")) else raw.get("id")
    spotify_id = spotify_id.strip() if is_nonempty_string(spotify_id) else None

    spotify_uri = raw.get("spotify_uri") if is_nonempty_string(raw.get("spotify_uri")) else raw.get("uri")
    spotify_uri = spotify_uri.strip() if is_nonempty_string(spotify_uri) else None

    isrc = normalize_isrc(raw.get("isrc"))
    if isrc is None:
        external_ids = raw.get("external_ids")
        if isinstance(external_ids, dict):
            isrc = normalize_isrc(external_ids.get("isrc"))

    brainz = raw.get("brainz") or {}
    mbid = None
    if isinstance(brainz, dict):
        for key in ("mbid", "musicbrainz_recording_id", "recording_mbid"):
            if is_nonempty_string(brainz.get(key)):
                mbid = brainz[key].strip()
                break
    if mbid is None and is_nonempty_string(raw.get("musicbrainz_recording_id")):
        mbid = raw["musicbrainz_recording_id"].strip()

    acoustic_high = None
    acoustic_low = None
    tags: list[dict[str, Any]] = []
    genres: list[dict[str, Any]] = []

    if isinstance(brainz, dict):
        tags = normalize_name_count_list(brainz.get("tags"))
        genres = normalize_name_count_list(brainz.get("genres"))

        ah = brainz.get("acoustic_high_level")
        if not isinstance(ah, dict):
            ah = brainz.get("acousticHighLevel")
        if isinstance(ah, dict):
            acoustic_high = ah

        al = brainz.get("acoustic_low_level")
        if not isinstance(al, dict):
            al = brainz.get("acousticLowLevel")
        if isinstance(al, dict):
            acoustic_low = al

    popularity = safe_float(raw.get("popularity"))
    if popularity is None:
        popularity = safe_float((raw.get("runtime_metadata") or {}).get("spotify_popularity"))

    norm_title = normalize_token(title) if title else None
    norm_artists = normalized_artists_key(artists)

    return {
        "schema_version": 1,
        "track_id": make_track_id(spotify_id=spotify_id, isrc=isrc, title=title, artists=artists),
        "title": title,
        "artists": artists,
        "album": album_name,
        "year": year,
        "duration_ms": duration_ms,
        "ids": {
            "spotify_id": spotify_id,
            "spotify_uri": spotify_uri,
            "isrc": isrc,
            "musicbrainz_recording_id": mbid,
        },
        "brainz": {
            "tags": tags,
            "genres": genres,
            "acoustic_high_level": acoustic_high,
            "acoustic_low_level": acoustic_low,
        },
        "runtime_metadata": {
            "spotify_popularity": popularity,
            "spotify_url": get_spotify_url(raw),
            "preview_url": get_preview_url(raw),
            "image_url": get_image_url(raw),
        },
        "audio_embeddings": {
            "model_name": None,
            "vector": None,
        },
        "source": {
            "dataset_name": dataset_name,
            "dataset_version": "0.1.0",
            "ingested_at_utc": now_utc_iso(),
            "source_files": [source_file],
        },
        "aliases": {
            "normalized_title": norm_title,
            "normalized_artists_key": norm_artists,
        },
        "debug": {
            "prepared_from_raw": True,
            "enriched_from_local_lookup": False,
        },
    }


def normalize_playlist(
    tracks: list[dict[str, Any]],
    *,
    dataset_name: str,
    source_file: str,
) -> list[dict[str, Any]]:
    out = []
    for raw in tracks:
        record = normalize_raw_record(
            raw,
            dataset_name=dataset_name,
            source_file=source_file,
        )
        if not is_nonempty_string(record.get("title")):
            continue
        if not isinstance(record.get("artists"), list) or not record["artists"]:
            continue
        out.append(record)
    return out


# ----------------------------
# local corpus lookup
# ----------------------------

def load_lookup(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Lookup file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def record_alias_key(record: dict[str, Any]) -> str | None:
    aliases = record.get("aliases") or {}
    title = aliases.get("normalized_title")
    artists_key = aliases.get("normalized_artists_key")
    if is_nonempty_string(title) and is_nonempty_string(artists_key):
        return f"{title}::{artists_key}"
    return None


def choose_local_match(record: dict[str, Any], lookup: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    ids = record.get("ids") or {}

    spotify_id = ids.get("spotify_id")
    if is_nonempty_string(spotify_id):
        payload = (lookup.get("by_spotify_id") or {}).get(spotify_id)
        if payload:
            return "spotify_id", payload

    isrc = ids.get("isrc")
    if is_nonempty_string(isrc):
        payload = (lookup.get("by_isrc") or {}).get(isrc)
        if payload:
            return "isrc", payload

    mbid = ids.get("musicbrainz_recording_id")
    if is_nonempty_string(mbid):
        payload = (lookup.get("by_mbid") or {}).get(mbid)
        if payload:
            return "musicbrainz_recording_id", payload

    alias = record_alias_key(record)
    if is_nonempty_string(alias):
        payload = (lookup.get("by_alias") or {}).get(alias)
        if payload:
            return "alias", payload

    return None, None


def merge_enrichment_into_record(record: dict[str, Any], payload: dict[str, Any], match_type: str) -> bool:
    changed = False

    ids = dict(record.get("ids") or {})
    p_ids = payload.get("ids") or {}

    if not is_nonempty_string(ids.get("musicbrainz_recording_id")) and is_nonempty_string(p_ids.get("musicbrainz_recording_id")):
        ids["musicbrainz_recording_id"] = p_ids.get("musicbrainz_recording_id")
        changed = True

    brainz = dict(record.get("brainz") or {})
    p_brainz = payload.get("brainz") or {}

    if not normalize_name_count_list(brainz.get("tags")) and normalize_name_count_list(p_brainz.get("tags")):
        brainz["tags"] = normalize_name_count_list(p_brainz.get("tags"))
        changed = True

    if not normalize_name_count_list(brainz.get("genres")) and normalize_name_count_list(p_brainz.get("genres")):
        brainz["genres"] = normalize_name_count_list(p_brainz.get("genres"))
        changed = True

    if not isinstance(brainz.get("acoustic_high_level"), dict) and isinstance(p_brainz.get("acoustic_high_level"), dict):
        brainz["acoustic_high_level"] = p_brainz.get("acoustic_high_level")
        changed = True

    if not isinstance(brainz.get("acoustic_low_level"), dict) and isinstance(p_brainz.get("acoustic_low_level"), dict):
        brainz["acoustic_low_level"] = p_brainz.get("acoustic_low_level")
        changed = True

    runtime = dict(record.get("runtime_metadata") or {})
    p_runtime = payload.get("runtime_metadata") or {}

    for key in ("spotify_popularity", "spotify_url", "preview_url", "image_url"):
        if runtime.get(key) is None and p_runtime.get(key) is not None:
            runtime[key] = p_runtime.get(key)
            changed = True

    debug = dict(record.get("debug") or {})
    debug["enriched_from_local_lookup"] = True
    debug["local_lookup_match_type"] = match_type

    record["ids"] = ids
    record["brainz"] = brainz
    record["runtime_metadata"] = runtime
    record["debug"] = debug
    return changed


def enrich_from_lookup(records: list[dict[str, Any]], lookup: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "lookup_hits_total": 0,
        "lookup_hits_spotify_id": 0,
        "lookup_hits_isrc": 0,
        "lookup_hits_mbid": 0,
        "lookup_hits_alias": 0,
        "records_changed_from_lookup": 0,
    }

    if lookup is None:
        return records, stats

    out = []
    for record in records:
        match_type, payload = choose_local_match(record, lookup)
        if payload is not None:
            stats["lookup_hits_total"] += 1
            key_name = f"lookup_hits_{match_type}"
            if key_name in stats:
                stats[key_name] += 1
            if merge_enrichment_into_record(record, payload, match_type):
                stats["records_changed_from_lookup"] += 1
        out.append(record)

    return out, stats


# ----------------------------
# API client for backfill
# ----------------------------

class ApiClient:
    def __init__(self, user_agent: str, sleep_seconds: float, timeout: float, verbose: bool = False):
        self.user_agent = user_agent
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.verbose = verbose
        self._last_request_ts = 0.0
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        remaining = self.sleep_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get_json(self, url: str) -> Any:
        self._throttle()
        req = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        if self.verbose:
            print(f"GET {url}")
        with urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
            body = resp.read().decode("utf-8")
        self._last_request_ts = time.time()
        return json.loads(body)


# ----------------------------
# backfill helpers
# ----------------------------

def load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_get(cache: dict[str, Any], key: str) -> Any:
    return cache.get(key)


def cache_set(cache: dict[str, Any], key: str, value: Any) -> None:
    cache[key] = value


def musicbrainz_recording_search_url(title: str, artist: str) -> str:
    query = f'recording:"{title}" AND artist:"{artist}"'
    return f"https://musicbrainz.org/ws/2/recording/?query={quote(query)}&fmt=json&limit=5"


def musicbrainz_recording_isrc_url(isrc: str) -> str:
    return f"https://musicbrainz.org/ws/2/isrc/{quote(isrc)}?fmt=json"


def musicbrainz_recording_lookup_url(mbid: str) -> str:
    return f"https://musicbrainz.org/ws/2/recording/{quote(mbid)}?fmt=json&inc=tags+genres+artist-credits"


def acoustic_high_url(mbid: str) -> str:
    return f"https://acousticbrainz.org/api/v1/{quote(mbid)}/high-level"


def extract_mbid_from_isrc_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    recordings = payload.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        return None
    first = recordings[0]
    if isinstance(first, dict) and is_nonempty_string(first.get("id")):
        return first["id"].strip()
    return None


def extract_best_mbid_from_search_payload(payload: Any, wanted_title: str, wanted_artist: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    recordings = payload.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        return None

    wanted_title_n = normalize_token(wanted_title)
    wanted_artist_n = normalize_token(wanted_artist)

    def score(rec: dict[str, Any]) -> tuple[int, int]:
        score = 0
        title = rec.get("title")
        if is_nonempty_string(title) and normalize_token(title) == wanted_title_n:
            score += 3

        artist_credit = rec.get("artist-credit")
        matched_artist = False
        if isinstance(artist_credit, list):
            for entry in artist_credit:
                if isinstance(entry, dict):
                    artist = entry.get("artist")
                    if isinstance(artist, dict) and is_nonempty_string(artist.get("name")):
                        if normalize_token(artist["name"]) == wanted_artist_n:
                            matched_artist = True
                            break
        if matched_artist:
            score += 3

        mb_score = safe_int(rec.get("score")) or 0
        return (score, mb_score)

    recordings_sorted = sorted(
        [r for r in recordings if isinstance(r, dict) and is_nonempty_string(r.get("id"))],
        key=score,
        reverse=True,
    )
    if not recordings_sorted:
        return None

    return recordings_sorted[0]["id"].strip()


def extract_tags_and_genres_from_recording(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return [], []
    return normalize_name_count_list(payload.get("tags")), normalize_name_count_list(payload.get("genres"))


def maybe_fetch_json(
    client: ApiClient,
    cache: dict[str, Any],
    cache_key: str,
    url: str,
) -> Any:
    cached = cache_get(cache, cache_key)
    if cached is not None:
        return cached
    try:
        payload = client.get_json(url)
    except Exception:
        payload = None
    cache_set(cache, cache_key, payload)
    return payload


def should_backfill_record(record: dict[str, Any], include_acoustic: bool) -> bool:
    ids = record.get("ids") or {}
    brainz = record.get("brainz") or {}

    missing_mbid = not is_nonempty_string(ids.get("musicbrainz_recording_id"))
    missing_tags = not normalize_name_count_list(brainz.get("tags"))
    missing_genres = not normalize_name_count_list(brainz.get("genres"))
    missing_acoustic = include_acoustic and not isinstance(brainz.get("acoustic_high_level"), dict)

    return missing_mbid or missing_tags or missing_genres or missing_acoustic


def backfill_playlist(
    records: list[dict[str, Any]],
    *,
    client: ApiClient,
    cache: dict[str, Any],
    include_acoustic: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "records_considered_for_backfill": 0,
        "records_seen": 0,
        "records_changed": 0,
        "mbid_resolved_isrc": 0,
        "mbid_resolved_search": 0,
        "mbid_already_present": 0,
        "mbid_lookup_failed": 0,
        "tags_filled": 0,
        "genres_filled": 0,
        "acoustic_high_filled": 0,
    }

    out = []
    backfill_seen = 0

    for record in records:
        if not should_backfill_record(record, include_acoustic):
            out.append(record)
            continue

        stats["records_considered_for_backfill"] += 1

        if limit is not None and backfill_seen >= limit:
            out.append(record)
            continue

        backfill_seen += 1
        stats["records_seen"] += 1
        changed = False

        ids = dict(record.get("ids") or {})
        brainz = dict(record.get("brainz") or {})

        mbid = ids.get("musicbrainz_recording_id")
        if is_nonempty_string(mbid):
            stats["mbid_already_present"] += 1
        else:
            isrc = ids.get("isrc")
            title = record.get("title")
            artists = record.get("artists") or []
            primary_artist = artists[0] if artists else None

            if is_nonempty_string(isrc):
                payload = maybe_fetch_json(
                    client,
                    cache,
                    f"isrc::{isrc}",
                    musicbrainz_recording_isrc_url(isrc),
                )
                mbid = extract_mbid_from_isrc_payload(payload)
                if mbid:
                    ids["musicbrainz_recording_id"] = mbid
                    stats["mbid_resolved_isrc"] += 1
                    changed = True

            if not is_nonempty_string(mbid) and is_nonempty_string(title) and is_nonempty_string(primary_artist):
                payload = maybe_fetch_json(
                    client,
                    cache,
                    f"search::{normalize_token(title)}::{normalize_token(primary_artist)}",
                    musicbrainz_recording_search_url(title, primary_artist),
                )
                mbid = extract_best_mbid_from_search_payload(payload, title, primary_artist)
                if mbid:
                    ids["musicbrainz_recording_id"] = mbid
                    stats["mbid_resolved_search"] += 1
                    changed = True

            if not is_nonempty_string(mbid):
                stats["mbid_lookup_failed"] += 1

        mbid = ids.get("musicbrainz_recording_id")
        if is_nonempty_string(mbid):
            need_tags = not normalize_name_count_list(brainz.get("tags"))
            need_genres = not normalize_name_count_list(brainz.get("genres"))
            need_acoustic = include_acoustic and not isinstance(brainz.get("acoustic_high_level"), dict)

            if need_tags or need_genres:
                payload = maybe_fetch_json(
                    client,
                    cache,
                    f"recording::{mbid}",
                    musicbrainz_recording_lookup_url(mbid),
                )
                tags, genres = extract_tags_and_genres_from_recording(payload)

                if need_tags and tags:
                    brainz["tags"] = tags
                    stats["tags_filled"] += 1
                    changed = True
                if need_genres and genres:
                    brainz["genres"] = genres
                    stats["genres_filled"] += 1
                    changed = True

            if need_acoustic:
                payload = maybe_fetch_json(
                    client,
                    cache,
                    f"acoustic_high::{mbid}",
                    acoustic_high_url(mbid),
                )
                if isinstance(payload, dict) and payload:
                    brainz["acoustic_high_level"] = payload
                    stats["acoustic_high_filled"] += 1
                    changed = True

        if changed:
            stats["records_changed"] += 1

        record["ids"] = ids
        record["brainz"] = brainz
        out.append(record)

    return out, stats


# ----------------------------
# summary
# ----------------------------

def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "records": len(records),
        "with_isrc": 0,
        "with_mbid": 0,
        "with_tags": 0,
        "with_genres": 0,
        "with_acoustic_high_level": 0,
    }
    for r in records:
        ids = r.get("ids") or {}
        brainz = r.get("brainz") or {}

        if is_nonempty_string(ids.get("isrc")):
            out["with_isrc"] += 1
        if is_nonempty_string(ids.get("musicbrainz_recording_id")):
            out["with_mbid"] += 1
        if normalize_name_count_list(brainz.get("tags")):
            out["with_tags"] += 1
        if normalize_name_count_list(brainz.get("genres")):
            out["with_genres"] += 1
        if isinstance(brainz.get("acoustic_high_level"), dict) and brainz.get("acoustic_high_level"):
            out["with_acoustic_high_level"] += 1

    return out


# ----------------------------
# cli
# ----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one playlist for clustering.")

    parser.add_argument("--input", required=True, help="Path to raw playlist export JSON")
    parser.add_argument("--output", required=True, help="Path to write prepared playlist JSON")
    parser.add_argument("--dataset-name", default="playlist-runtime")

    parser.add_argument(
        "--lookup",
        default="ml_pipeline/data/artifacts/corpus_lookup_v1.json",
        help="Optional local corpus lookup artifact",
    )
    parser.add_argument("--disable-lookup", action="store_true")

    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--include-acoustic", action="store_true")
    parser.add_argument("--cache", default=None, help="Optional cache path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for live backfill requests")
    parser.add_argument("--user-agent", default="PlaylistSplitterML/0.1 (contact: local-dev)")
    parser.add_argument("--sleep-seconds", type=float, default=1.05)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache) if args.cache else None
    lookup_path = None if args.disable_lookup else Path(args.lookup)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        payload = load_input_payload(input_path)
        tracks = extract_track_list(payload)
    except Exception as exc:
        print(f"Failed to load input playlist: {exc}", file=sys.stderr)
        return 1

    prepared = normalize_playlist(
        tracks,
        dataset_name=args.dataset_name,
        source_file=input_path.name,
    )

    try:
        lookup = load_lookup(lookup_path) if lookup_path is not None else None
    except Exception as exc:
        print(f"Failed to load lookup artifact: {exc}", file=sys.stderr)
        return 1

    prepared, lookup_stats = enrich_from_lookup(prepared, lookup)

    if args.backfill:
        cache = load_cache(cache_path)
        client = ApiClient(
            user_agent=args.user_agent,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        prepared, backfill_stats = backfill_playlist(
            prepared,
            client=client,
            cache=cache,
            include_acoustic=args.include_acoustic,
            limit=args.limit,
        )
        save_cache(cache_path, cache)
    else:
        backfill_stats = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize(prepared)

    print("=== Prepare Playlist Summary ===")
    print(f"input_tracks: {len(tracks)}")
    print(f"prepared_records: {len(prepared)}")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Local Lookup Stats ===")
    for k, v in lookup_stats.items():
        print(f"{k}: {v}")

    if backfill_stats:
        print("\n=== Backfill Stats ===")
        for k, v in backfill_stats.items():
            print(f"{k}: {v}")

    print(f"\nwrote: {output_path}")
    if lookup_path is not None:
        print(f"lookup: {lookup_path}")
    if cache_path:
        print(f"cache: {cache_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())