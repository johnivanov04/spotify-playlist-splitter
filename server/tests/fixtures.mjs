// Reusable test fixtures.

const FAKE_USER = {
  id: "00000000-0000-0000-0000-000000000001",
  spotifyUserId: "spotify-user-1",
  email: "test@example.com",
  displayName: "Test User",
  accessToken: "fake-access-token",
  refreshToken: "fake-refresh-token",
  expiresInSeconds: 3600,
  tokenObtainedAt: new Date("2026-01-01T12:00:00Z"),
  subscriptionStatus: null,
  monthlyVibeQuotaUsed: 0,
  quotaResetAt: null,
  createdAt: new Date("2026-01-01T00:00:00Z"),
  updatedAt: new Date("2026-01-01T12:00:00Z"),
};

const FAKE_USER_EXPIRED = {
  ...FAKE_USER,
  // obtained 2 hours ago → past expiry
  tokenObtainedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
};

function makeTrack(overrides = {}) {
  return {
    id: "track1",
    name: "Test Track",
    artists: ["Test Artist"],
    album: "Test Album",
    year: 2024,
    popularity: 50,
    spotifyArtistGenres: [{ name: "indie", count: 1 }],
    brainz: { tags: [{ name: "mellow", count: 1 }] },
    ...overrides,
  };
}

function makePlaylistTrackItem(overrides = {}) {
  return {
    track: {
      id: "track1",
      name: "Test Track",
      artists: [{ name: "Test Artist" }],
      album: { name: "Test Album", release_date: "2024-01-01", images: [{ url: "https://example.com/a.jpg" }] },
      external_urls: { spotify: "https://open.spotify.com/track/track1" },
      external_ids: { isrc: "USRC11111111" },
      duration_ms: 200_000,
      popularity: 50,
      preview_url: null,
      ...overrides,
    },
  };
}

function makeRelinkedPlaylistTrackItem(originalId, relinkedId, overrides = {}) {
  return {
    track: {
      id: relinkedId,
      name: "Relinked Track",
      artists: [{ name: "Original Artist" }],
      album: { name: "Album", release_date: "2024-01-01", images: [] },
      external_urls: { spotify: `https://open.spotify.com/track/${relinkedId}` },
      external_ids: { isrc: "USRC22222222" },
      duration_ms: 180_000,
      popularity: 40,
      preview_url: null,
      linked_from: {
        id: originalId,
        external_urls: { spotify: `https://open.spotify.com/track/${originalId}` },
      },
      ...overrides,
    },
  };
}

// LLM responses
const VALID_LLM_RESPONSE = {
  groupings: [
    {
      name: "Late Night Drive Home",
      description: "Subdued tracks for the empty highway at 2am.",
      track_indices: [0, 1, 2],
    },
    {
      name: "Cookout Sunday",
      description: "Warm, communal energy.",
      track_indices: [3, 4, 5],
    },
  ],
};

const LLM_RESPONSE_WITH_BAD_INDICES = {
  groupings: [
    {
      name: "Out-of-range and duplicate test",
      description: "Mix of valid, OOB, dupes, negative, non-int.",
      track_indices: [0, 100, -1, "abc", 0, 1.5, 1, 2],
    },
  ],
};

const LLM_RESPONSE_TOO_FEW_TRACKS_IN_GROUP = {
  groupings: [
    { name: "Lonely", description: "only one track", track_indices: [0] },
    { name: "OK", description: "enough tracks", track_indices: [1, 2, 3] },
  ],
};

export {
  FAKE_USER,
  FAKE_USER_EXPIRED,
  makeTrack,
  makePlaylistTrackItem,
  makeRelinkedPlaylistTrackItem,
  VALID_LLM_RESPONSE,
  LLM_RESPONSE_WITH_BAD_INDICES,
  LLM_RESPONSE_TOO_FEW_TRACKS_IN_GROUP,
};
