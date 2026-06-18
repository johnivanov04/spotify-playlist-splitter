import { describe, it, expect } from "vitest";
import vibesLib from "../lib/vibes.js";

const { vibeCacheKey, summarizeTrackForPrompt, mapIndicesToTrackIds } = vibesLib;

describe("vibeCacheKey", () => {
  it("is deterministic for the same input", () => {
    const tracks = [{ id: "a" }, { id: "b" }, { id: "c" }];
    expect(vibeCacheKey(tracks, "")).toBe(vibeCacheKey(tracks, ""));
  });

  it("ignores track order (sorts by ID before hashing)", () => {
    const a = [{ id: "a" }, { id: "b" }, { id: "c" }];
    const b = [{ id: "c" }, { id: "a" }, { id: "b" }];
    expect(vibeCacheKey(a, "")).toBe(vibeCacheKey(b, ""));
  });

  it("produces a different key for different track sets", () => {
    const a = [{ id: "a" }, { id: "b" }];
    const b = [{ id: "a" }, { id: "c" }];
    expect(vibeCacheKey(a, "")).not.toBe(vibeCacheKey(b, ""));
  });

  it("produces a different key for different steer prompts", () => {
    const tracks = [{ id: "a" }, { id: "b" }];
    expect(vibeCacheKey(tracks, "late night")).not.toBe(vibeCacheKey(tracks, "summer"));
  });

  it("treats empty / whitespace / nullish steer as equivalent", () => {
    const tracks = [{ id: "a" }, { id: "b" }];
    const base = vibeCacheKey(tracks, "");
    expect(vibeCacheKey(tracks, "   ")).toBe(base);
    expect(vibeCacheKey(tracks, undefined)).toBe(base);
    expect(vibeCacheKey(tracks, null)).toBe(base);
  });

  it("case-folds and trims the steer text", () => {
    const tracks = [{ id: "a" }, { id: "b" }];
    expect(vibeCacheKey(tracks, "Late Night")).toBe(vibeCacheKey(tracks, "  late night  "));
  });

  it("ignores falsy / missing IDs", () => {
    const a = [{ id: "a" }, { id: null }, { id: undefined }, {}, null];
    const b = [{ id: "a" }];
    expect(vibeCacheKey(a, "")).toBe(vibeCacheKey(b, ""));
  });

  it("returns a 64-char hex string (sha256)", () => {
    const key = vibeCacheKey([{ id: "x" }], "");
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("handles empty track list", () => {
    expect(vibeCacheKey([], "")).toMatch(/^[0-9a-f]{64}$/);
    expect(vibeCacheKey(undefined, "")).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe("summarizeTrackForPrompt", () => {
  it("includes the ordinal index", () => {
    expect(summarizeTrackForPrompt({ id: "x", name: "S" }, 7).index).toBe(7);
  });

  it("does NOT include the Spotify ID", () => {
    const out = summarizeTrackForPrompt({ id: "spotify-id-xyz", name: "S" }, 0);
    expect(out.id).toBeUndefined();
  });

  it("joins multiple artists with comma-space", () => {
    expect(summarizeTrackForPrompt({ name: "S", artists: ["A", "B", "C"] }, 0).artist).toBe(
      "A, B, C"
    );
  });

  it("falls back to .title when .name is missing", () => {
    expect(summarizeTrackForPrompt({ title: "Title!" }, 0).title).toBe("Title!");
  });

  it("emits empty title for tracks missing both name and title", () => {
    expect(summarizeTrackForPrompt({}, 0).title).toBe("");
  });

  it("emits null year when missing or non-numeric", () => {
    expect(summarizeTrackForPrompt({}, 0).year).toBe(null);
    expect(summarizeTrackForPrompt({ year: "2024" }, 0).year).toBe(null);
  });

  it("caps genres at 8 and tags at 6", () => {
    const t = {
      spotifyArtistGenres: Array.from({ length: 20 }, (_, i) => `g${i}`),
      brainz: { tags: Array.from({ length: 20 }, (_, i) => ({ name: `t${i}` })) },
    };
    const out = summarizeTrackForPrompt(t, 0);
    expect(out.genres.length).toBe(8);
    expect(out.tags.length).toBe(6);
  });

  it("handles brainz tags as plain strings or {name} objects", () => {
    const out = summarizeTrackForPrompt(
      { brainz: { tags: ["a", { name: "b" }, { name: "" }, null] } },
      0
    );
    expect(out.tags).toEqual(["a", "b"]);
  });

  it("returns empty arrays when sources are missing", () => {
    const out = summarizeTrackForPrompt({}, 0);
    expect(out.genres).toEqual([]);
    expect(out.tags).toEqual([]);
  });
});

describe("mapIndicesToTrackIds", () => {
  const tracks = [
    { id: "id-0" }, { id: "id-1" }, { id: "id-2" }, { id: "id-3" }, { id: "id-4" },
  ];

  it("maps valid indices to the corresponding track IDs", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [0, 2, 4] }],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-2", "id-4"]);
  });

  it("preserves group name and description", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "VibeName", description: "VibeDesc", track_indices: [0, 1] }],
      tracks
    );
    expect(out[0].name).toBe("VibeName");
    expect(out[0].description).toBe("VibeDesc");
  });

  it("filters out negative indices", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [-1, 0, 1] }],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-1"]);
  });

  it("filters out out-of-range indices", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [0, 99, 100, 1] }],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-1"]);
  });

  it("filters out non-integer indices (floats, strings, null)", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [0, 1.5, "2", null, undefined, 3] }],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-3"]);
  });

  it("deduplicates indices within the same group", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [0, 0, 1, 1] }],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-1"]);
  });

  it("deduplicates indices ACROSS groups (each track assigned at most once)", () => {
    // Group B references 1 (already used) plus 2 fresh — should retain only the fresh ones.
    const out = mapIndicesToTrackIds(
      [
        { name: "A", description: "d", track_indices: [0, 1] },
        { name: "B", description: "d", track_indices: [1, 2, 3] },
      ],
      tracks
    );
    expect(out[0].track_ids).toEqual(["id-0", "id-1"]);
    expect(out[1].track_ids).toEqual(["id-2", "id-3"]);
  });

  it("a group that loses all its indices to cross-group dedup gets dropped", () => {
    const out = mapIndicesToTrackIds(
      [
        { name: "A", description: "d", track_indices: [0, 1] },
        { name: "B", description: "d", track_indices: [0, 1] }, // all taken → would be empty
      ],
      tracks
    );
    expect(out).toHaveLength(1);
    expect(out[0].name).toBe("A");
  });

  it("drops groupings with fewer than 2 valid track_ids", () => {
    const out = mapIndicesToTrackIds(
      [
        { name: "Lonely", description: "d", track_indices: [0] },
        { name: "OK", description: "d", track_indices: [1, 2, 3] },
      ],
      tracks
    );
    expect(out).toHaveLength(1);
    expect(out[0].name).toBe("OK");
  });

  it("skips indices pointing at tracks with no ID", () => {
    const tracksWithGaps = [{ id: "a" }, { id: null }, { id: "c" }];
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [0, 1, 2] }],
      tracksWithGaps
    );
    expect(out[0].track_ids).toEqual(["a", "c"]);
  });

  it("returns [] for missing groupings input", () => {
    expect(mapIndicesToTrackIds(undefined, tracks)).toEqual([]);
    expect(mapIndicesToTrackIds(null, tracks)).toEqual([]);
  });

  it("returns [] when no group has enough valid tracks", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "A", description: "d", track_indices: [-1, -2, 99] }],
      tracks
    );
    expect(out).toEqual([]);
  });

  it("regression: output uses only server-supplied IDs (no LLM-hallucinated values)", () => {
    const out = mapIndicesToTrackIds(
      [{ name: "X", description: "d", track_indices: [0, 1, 2] }],
      [{ id: "real-spotify-id-A" }, { id: "real-spotify-id-B" }, { id: "real-spotify-id-C" }]
    );
    expect(out[0].track_ids).toEqual([
      "real-spotify-id-A",
      "real-spotify-id-B",
      "real-spotify-id-C",
    ]);
  });
});
