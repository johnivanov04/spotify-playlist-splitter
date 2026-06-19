import { describe, it, expect, beforeEach, beforeAll, vi } from "vitest";
import { makeSessionCookie, makeFetchResponse } from "./helpers.mjs";
import {
  FAKE_USER,
  makePlaylistTrackItem,
  makeRelinkedPlaylistTrackItem,
} from "./fixtures.mjs";

// Reuse the same mock infra as integration.test.mjs but isolated to this file.
const dbState = {
  selectResults: [],
  insertResults: [],
  updateResults: [],
  insertCalls: [],
  updateCalls: [],
  onConflictCalls: [],
};

function selectChain() {
  return {
    from() { return this; },
    where() { return this; },
    limit() { return Promise.resolve(dbState.selectResults.shift() ?? []); },
  };
}
function insertChain() {
  let captured;
  return {
    values(v) { captured = v; dbState.insertCalls.push(v); return this; },
    returning() { return Promise.resolve(dbState.insertResults.shift() ?? []); },
    onConflictDoNothing() {
      dbState.onConflictCalls.push(captured);
      return Promise.resolve(undefined);
    },
  };
}
function updateChain() {
  return {
    set(v) { dbState.updateCalls.push(v); return this; },
    where() { return this; },
    returning() { return Promise.resolve(dbState.updateResults.shift() ?? []); },
  };
}

globalThis.__TEST_DB_MOCK__ = {
  db: { select: () => selectChain(), insert: () => insertChain(), update: () => updateChain() },
  schema: { users: { id: {}, spotifyUserId: {} }, vibeCaches: { cacheKey: {} } },
  pool: { end: () => {} },
};

class FakeAnthropic { constructor() { this.messages = { stream: vi.fn() }; } }
globalThis.__TEST_ANTHROPIC_CLASS__ = Object.assign(FakeAnthropic, { default: FakeAnthropic });

const fetchMock = vi.fn();
globalThis.fetch = fetchMock;

let request, app;
beforeAll(async () => {
  request = (await import("supertest")).default;
  app = (await import("../index.js")).default;
});

function freshUser() { return { ...FAKE_USER, tokenObtainedAt: new Date() }; }
const AUTH_COOKIE = makeSessionCookie({ userId: FAKE_USER.id });

beforeEach(() => {
  dbState.selectResults = [];
  dbState.insertResults = [];
  dbState.updateResults = [];
  dbState.insertCalls = [];
  dbState.updateCalls = [];
  dbState.onConflictCalls = [];
  fetchMock.mockReset();
});

// ============================================================
// GET /api/playlists/:id/tracks  — pagination + relinking
// ============================================================
describe("GET /api/playlists/:id/tracks", () => {
  it("rejects unauthenticated requests", async () => {
    const res = await request(app).get("/api/playlists/p1/tracks");
    expect(res.status).toBe(401);
  });

  it("uses linked_from.id as canonical id when track is relinked", async () => {
    dbState.selectResults.push([freshUser()]);
    // playlist metadata call
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "My Playlist", images: [] }));
    // playlist tracks call — returns one normal and one relinked track
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      items: [
        makePlaylistTrackItem({ id: "normal-track-id", name: "Normal" }),
        makeRelinkedPlaylistTrackItem("ORIGINAL-id", "RELINKED-id"),
      ],
      next: null,
    }));
    // tracks lookup (for ISRCs)
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      tracks: [
        { id: "normal-track-id", external_ids: { isrc: "USRC11111111" } },
        { id: "ORIGINAL-id", external_ids: { isrc: "USRC22222222" } },
      ],
    }));
    // audio-features (deprecated; expect 403)
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ error: { status: 403 } }, { status: 403 }));

    const res = await request(app)
      .get("/api/playlists/p1/tracks")
      .set("Cookie", AUTH_COOKIE);

    expect(res.status).toBe(200);
    const ids = res.body.tracks.map((t) => t.id);
    expect(ids).toContain("normal-track-id");
    // Relinked track's canonical id is the linked_from.id, NOT the relinked id.
    expect(ids).toContain("ORIGINAL-id");
    expect(ids).not.toContain("RELINKED-id");
  });

  it("passes market=from_token on the playlist tracks call (track relinking enabled)", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "p" }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ items: [], next: null }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ tracks: [] }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ error: { status: 403 } }, { status: 403 }));

    await request(app).get("/api/playlists/p1/tracks").set("Cookie", AUTH_COOKIE);

    // Find the tracks call and check that it includes market=from_token
    const tracksCall = fetchMock.mock.calls.find(([url]) =>
      url.includes("/playlists/p1/tracks") && url.includes("market=from_token")
    );
    expect(tracksCall).toBeDefined();
  });

  it("handles an empty playlist without erroring", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "empty" }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ items: [], next: null }));
    // no tracks → no /tracks?ids lookup needed, but the code might still call

    const res = await request(app).get("/api/playlists/empty/tracks").set("Cookie", AUTH_COOKIE);
    expect(res.status).toBe(200);
    expect(res.body.tracks).toEqual([]);
  });

  it("skips items with no track (e.g. local files)", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "p" }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      items: [
        { track: null }, // unavailable
        makePlaylistTrackItem({ id: "valid-id", name: "Valid" }),
        { track: { id: null } }, // local file (no ID)
      ],
      next: null,
    }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ tracks: [{ id: "valid-id", external_ids: { isrc: "I" } }] }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ error: { status: 403 } }, { status: 403 }));

    const res = await request(app).get("/api/playlists/p1/tracks").set("Cookie", AUTH_COOKIE);
    expect(res.status).toBe(200);
    expect(res.body.tracks).toHaveLength(1);
    expect(res.body.tracks[0].id).toBe("valid-id");
  });

  it("paginates through multiple pages of tracks (Spotify 'next' cursor)", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "p" }));
    // First page with `next` set
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      items: [makePlaylistTrackItem({ id: "page1-track" })],
      next: "https://api.spotify.com/v1/next-page",
    }));
    // Second page with `next` null
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      items: [makePlaylistTrackItem({ id: "page2-track" })],
      next: null,
    }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      tracks: [
        { id: "page1-track", external_ids: { isrc: "A" } },
        { id: "page2-track", external_ids: { isrc: "B" } },
      ],
    }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ error: { status: 403 } }, { status: 403 }));

    const res = await request(app).get("/api/playlists/p1/tracks").set("Cookie", AUTH_COOKIE);
    expect(res.status).toBe(200);
    expect(res.body.tracks.map((t) => t.id)).toEqual(["page1-track", "page2-track"]);
  });

  it("never exposes the access token to the client", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ name: "p" }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ items: [], next: null }));

    const res = await request(app).get("/api/playlists/p1/tracks").set("Cookie", AUTH_COOKIE);
    const body = JSON.stringify(res.body);
    expect(body).not.toContain("fake-access-token");
    expect(body).not.toContain("fake-refresh-token");
  });
});

// ============================================================
// POST /api/playlists/:id/remove-tracks  — relinking-aware
// ============================================================
describe("POST /api/playlists/:id/remove-tracks", () => {
  it("rejects unauthenticated requests", async () => {
    const res = await request(app).post("/api/playlists/p1/remove-tracks").send({ trackIds: ["a"] });
    expect(res.status).toBe(401);
  });

  it("returns 400 when no trackIds provided", async () => {
    dbState.selectResults.push([freshUser()]);
    const res = await request(app)
      .post("/api/playlists/p1/remove-tracks")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: [] });
    expect(res.status).toBe(400);
  });

  it("sends the correct DELETE to Spotify with proper URIs", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ snapshot_id: "snap-abc" }));

    const res = await request(app)
      .post("/api/playlists/playlist-id/remove-tracks")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: ["id-a", "id-b"] });

    expect(res.status).toBe(200);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toContain("/playlists/playlist-id/tracks");
    expect(opts.method).toBe("DELETE");
    const body = JSON.parse(opts.body);
    expect(body.tracks).toEqual([
      { uri: "spotify:track:id-a" },
      { uri: "spotify:track:id-b" },
    ]);
  });

  it("returns Spotify's error status when DELETE fails", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ error: "forbidden" }, { status: 403 }));

    const res = await request(app)
      .post("/api/playlists/p1/remove-tracks")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: ["x"] });
    expect(res.status).toBe(403);
  });
});

// ============================================================
// POST /api/playlists  — create playlist
// ============================================================
describe("POST /api/playlists (create)", () => {
  it("rejects unauthenticated requests", async () => {
    const res = await request(app).post("/api/playlists").send({ name: "x", trackIds: ["a"] });
    expect(res.status).toBe(401);
  });

  it("creates a playlist and adds tracks using server-supplied IDs", async () => {
    dbState.selectResults.push([freshUser()]);
    // /me to get current user id
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ id: "spotify-user-1" }));
    // POST /users/:id/playlists
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ id: "new-playlist-id", external_urls: { spotify: "https://..." } })
    );
    // POST /playlists/:id/tracks (add)
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ snapshot_id: "snap" }));

    const res = await request(app)
      .post("/api/playlists")
      .set("Cookie", AUTH_COOKIE)
      .send({ name: "My New Vibe", trackIds: ["track-a", "track-b"] });

    expect(res.status).toBe(200);
    // The add-tracks call must use the supplied IDs as Spotify URIs
    const addCall = fetchMock.mock.calls.find(([url, opts]) =>
      url.includes("/tracks") && opts?.method === "POST" && url.includes("new-playlist-id")
    );
    expect(addCall).toBeDefined();
    const body = JSON.parse(addCall[1].body);
    expect(body.uris).toEqual([
      "spotify:track:track-a",
      "spotify:track:track-b",
    ]);
  });

  it("returns 400 when name or trackIds missing", async () => {
    dbState.selectResults.push([freshUser()]);
    let res = await request(app).post("/api/playlists").set("Cookie", AUTH_COOKIE).send({});
    expect(res.status).toBe(400);

    dbState.selectResults.push([freshUser()]);
    res = await request(app).post("/api/playlists").set("Cookie", AUTH_COOKIE).send({ name: "x" });
    expect(res.status).toBe(400);
  });
});

// ============================================================
// POST /api/spotify/artist-genres  — batched enrichment
// ============================================================
describe("POST /api/spotify/artist-genres", () => {
  it("rejects unauthenticated requests", async () => {
    const res = await request(app).post("/api/spotify/artist-genres").send({ trackIds: ["a"] });
    expect(res.status).toBe(401);
  });

  it("returns 400 when no trackIds provided", async () => {
    dbState.selectResults.push([freshUser()]);
    const res = await request(app)
      .post("/api/spotify/artist-genres")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: [] });
    expect(res.status).toBe(400);
  });

  it("returns per-track genres assembled from track→artist→genre lookups", async () => {
    dbState.selectResults.push([freshUser()]);
    // tracks lookup
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      tracks: [
        { id: "t1", artists: [{ id: "a1" }] },
        { id: "t2", artists: [{ id: "a1" }, { id: "a2" }] },
      ],
    }));
    // artists lookup
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      artists: [
        { id: "a1", genres: ["indie", "bedroom pop"] },
        { id: "a2", genres: ["rock"] },
      ],
    }));

    const res = await request(app)
      .post("/api/spotify/artist-genres")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: ["t1", "t2"] });

    expect(res.status).toBe(200);
    expect(res.body.byTrackId.t1).toEqual(["indie", "bedroom pop"]);
    // t2 has a1 + a2 → genres should merge (order is insertion-stable)
    expect(res.body.byTrackId.t2).toContain("indie");
    expect(res.body.byTrackId.t2).toContain("rock");
  });

  it("deduplicates genres when an artist appears under multiple tracks", async () => {
    dbState.selectResults.push([freshUser()]);
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      tracks: [
        { id: "t1", artists: [{ id: "a1" }] },
        { id: "t2", artists: [{ id: "a1" }] },
      ],
    }));
    fetchMock.mockResolvedValueOnce(makeFetchResponse({
      artists: [{ id: "a1", genres: ["indie", "indie"] }], // duplicate within one artist's genres
    }));

    const res = await request(app)
      .post("/api/spotify/artist-genres")
      .set("Cookie", AUTH_COOKIE)
      .send({ trackIds: ["t1", "t2"] });

    expect(res.status).toBe(200);
    // t1 should have unique entries (the endpoint dedupes per track)
    expect(res.body.byTrackId.t1).toEqual(["indie"]);
  });
});
