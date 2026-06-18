const crypto = require("crypto");

/**
 * Deterministic cache key for a vibe analysis request.
 * Sorts track IDs so the order doesn't matter; folds in a normalized steer
 * prompt so different steers cache separately.
 */
function vibeCacheKey(tracks, steer) {
  const ids = (tracks || []).map((t) => t && t.id).filter(Boolean).slice().sort();
  const steerKey = (steer || "").trim().toLowerCase();
  const hash = crypto.createHash("sha256");
  hash.update(ids.join(","));
  if (steerKey) hash.update("\nSTEER:" + steerKey);
  return hash.digest("hex");
}

/**
 * Compact per-track payload for the LLM, with an explicit ordinal index.
 * Avoids sending raw Spotify IDs (the model used to hallucinate them).
 */
function summarizeTrackForPrompt(t, index) {
  return {
    index,
    title: t.name || t.title || "",
    artist: Array.isArray(t.artists) ? t.artists.join(", ") : (t.artist || ""),
    year: typeof t.year === "number" ? t.year : null,
    genres: Array.isArray(t.spotifyArtistGenres) ? t.spotifyArtistGenres.slice(0, 8) : [],
    tags: Array.isArray(t?.brainz?.tags)
      ? t.brainz.tags.slice(0, 6).map((x) => (typeof x === "string" ? x : x?.name)).filter(Boolean)
      : []
  };
}

/**
 * Map ordinal indices from the model's response back to real Spotify track IDs
 * using the input track order. Filters out:
 *   - non-integer / negative / out-of-range indices
 *   - duplicate indices (across all groupings, not just within one)
 *   - tracks with no Spotify ID
 *   - groupings that end up with fewer than 2 valid track IDs
 */
function mapIndicesToTrackIds(groupings, tracks) {
  const inputIds = (tracks || []).map((t) => t && t.id);
  const seenIndices = new Set();
  return (groupings || []).map((g) => {
    const indices = Array.isArray(g.track_indices) ? g.track_indices : [];
    const trackIds = [];
    for (const idx of indices) {
      if (typeof idx !== "number" || !Number.isInteger(idx)) continue;
      if (idx < 0 || idx >= inputIds.length) continue;
      if (seenIndices.has(idx)) continue;
      const id = inputIds[idx];
      if (!id) continue;
      seenIndices.add(idx);
      trackIds.push(id);
    }
    return {
      name: g.name,
      description: g.description,
      track_ids: trackIds,
    };
  }).filter((g) => g.track_ids.length >= 2);
}

module.exports = { vibeCacheKey, summarizeTrackForPrompt, mapIndicesToTrackIds };
