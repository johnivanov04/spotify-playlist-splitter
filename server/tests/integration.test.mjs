import { describe, it, expect, beforeEach, beforeAll, vi } from "vitest";
import { makeSessionCookie, makeFetchResponse } from "./helpers.mjs";
import { FAKE_USER, FAKE_USER_EXPIRED, VALID_LLM_RESPONSE } from "./fixtures.mjs";

// ---- Mock state -------------------------------------------------------
const dbState = {
  selectResults: [],
  insertResults: [],
  updateResults: [],
  insertCalls: [],
  updateCalls: [],
  onConflictCalls: [],
};

function resetDb() {
  dbState.selectResults = [];
  dbState.insertResults = [];
  dbState.updateResults = [];
  dbState.insertCalls = [];
  dbState.updateCalls = [];
  dbState.onConflictCalls = [];
}

function selectChain() {
  return {
    from() { return this; },
    where() { return this; },
    limit() {
      return Promise.resolve(dbState.selectResults.shift() ?? []);
    },
  };
}
function insertChain() {
  let captured;
  return {
    values(v) { captured = v; dbState.insertCalls.push(v); return this; },
    returning() {
      return Promise.resolve(dbState.insertResults.shift() ?? []);
    },
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
    returning() {
      return Promise.resolve(dbState.updateResults.shift() ?? []);
    },
  };
}

// Drizzle-shaped fake db module.
globalThis.__TEST_DB_MOCK__ = {
  db: {
    select: () => selectChain(),
    insert: () => insertChain(),
    update: () => updateChain(),
  },
  schema: {
    users: { id: {}, spotifyUserId: {} },
    vibeCaches: { cacheKey: {} },
  },
  pool: { end: () => {} },
};

// Anthropic SDK fake — index.js does `new Anthropic({...})` and calls
// client.messages.stream(...). We intercept the constructor to give it a
// .messages.stream that delegates to a swap-in vi.fn() per test.
const anthropicStreamMock = vi.fn();
class FakeAnthropic {
  constructor() {
    this.messages = { stream: anthropicStreamMock };
  }
}
// The SDK exports the class as default, but CJS interop expects it accessible
// as both the module itself (Anthropic = require(...)) and as .default.
globalThis.__TEST_ANTHROPIC_CLASS__ = Object.assign(FakeAnthropic, { default: FakeAnthropic });

// Fetch override (Spotify API)
const fetchMock = vi.fn();
globalThis.fetch = fetchMock;

// Load supertest and the app AFTER globals are set so the require hooks fire.
let request;
let app;
beforeAll(async () => {
  request = (await import("supertest")).default;
  app = (await import("../index.js")).default;
});

// ---- Helpers -----------------------------------------------------------
function setUserLookup(user) {
  dbState.selectResults.push(user ? [user] : []);
}
function setCacheLookup(cached) {
  dbState.selectResults.push(cached ? [cached] : []);
}
function makeFakeAnthropicStream(message) {
  return { async finalMessage() { return message; } };
}
function llmTextMessage(jsonObj) {
  return {
    stop_reason: "end_turn",
    content: [{ type: "text", text: JSON.stringify(jsonObj) }],
    usage: { input_tokens: 100, output_tokens: 50 },
  };
}

beforeEach(() => {
  resetDb();
  fetchMock.mockReset();
  anthropicStreamMock.mockReset();
});

// ============================================================
// AUTH MIDDLEWARE
// ============================================================
describe("requireSpotifyAuth middleware", () => {
  it("rejects unauthenticated request with 401", async () => {
    const res = await request(app).get("/api/me");
    expect(res.status).toBe(401);
    expect(res.body.error).toBeDefined();
  });

  it("rejects request with session but no userId", async () => {
    const cookie = makeSessionCookie({});
    const res = await request(app).get("/api/me").set("Cookie", cookie);
    expect(res.status).toBe(401);
  });

  it("rejects when session userId points to non-existent user", async () => {
    setUserLookup(null);
    const cookie = makeSessionCookie({ userId: "ghost-user-id" });
    const res = await request(app).get("/api/me").set("Cookie", cookie);
    expect(res.status).toBe(401);
  });

  it("rejects when user record has no access token", async () => {
    setUserLookup({ ...FAKE_USER, accessToken: null });
    const cookie = makeSessionCookie({ userId: FAKE_USER.id });
    const res = await request(app).get("/api/me").set("Cookie", cookie);
    expect(res.status).toBe(401);
  });

  it("allows authenticated request to proceed and proxies /me to Spotify", async () => {
    setUserLookup({ ...FAKE_USER, tokenObtainedAt: new Date() });
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ id: "spotify-user-1", display_name: "Test User", email: "t@e.com" })
    );

    const cookie = makeSessionCookie({ userId: FAKE_USER.id });
    const res = await request(app).get("/api/me").set("Cookie", cookie);
    expect(res.status).toBe(200);
    expect(res.body.id).toBe("spotify-user-1");
  });

  it("refreshes Spotify token when near expiry, persists to DB", async () => {
    setUserLookup(FAKE_USER_EXPIRED);
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({
        access_token: "new-access-token",
        refresh_token: "new-refresh-token",
        expires_in: 3600,
      })
    );
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ id: "spotify-user-1" }));

    dbState.updateResults.push([{
      ...FAKE_USER_EXPIRED,
      accessToken: "new-access-token",
      refreshToken: "new-refresh-token",
      expiresInSeconds: 3600,
      tokenObtainedAt: new Date(),
    }]);

    const cookie = makeSessionCookie({ userId: FAKE_USER_EXPIRED.id });
    const res = await request(app).get("/api/me").set("Cookie", cookie);

    expect(res.status).toBe(200);
    const tokenCall = fetchMock.mock.calls.find(([url]) => url.includes("/api/token"));
    expect(tokenCall).toBeDefined();
    expect(tokenCall[1].body).toContain("refresh_token");
    expect(dbState.updateCalls[0].accessToken).toBe("new-access-token");
    expect(dbState.updateCalls[0].refreshToken).toBe("new-refresh-token");
    expect(dbState.updateCalls[0].expiresInSeconds).toBe(3600);
  });

  it("preserves existing refresh token when Spotify doesn't return a new one", async () => {
    setUserLookup(FAKE_USER_EXPIRED);
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ access_token: "new-access-token", expires_in: 3600 })
    );
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ id: "spotify-user-1" }));
    dbState.updateResults.push([{ ...FAKE_USER_EXPIRED, accessToken: "new-access-token" }]);

    const cookie = makeSessionCookie({ userId: FAKE_USER_EXPIRED.id });
    await request(app).get("/api/me").set("Cookie", cookie);

    expect("refreshToken" in (dbState.updateCalls[0] || {})).toBe(false);
  });

  it("returns 401 when token refresh fails (e.g. revoked)", async () => {
    setUserLookup(FAKE_USER_EXPIRED);
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ error: "invalid_grant" }, { status: 400 })
    );

    const cookie = makeSessionCookie({ userId: FAKE_USER_EXPIRED.id });
    const res = await request(app).get("/api/me").set("Cookie", cookie);
    expect(res.status).toBe(401);
  });

  it("does NOT refresh a token that's still well within its lifetime", async () => {
    setUserLookup({ ...FAKE_USER, tokenObtainedAt: new Date() });
    fetchMock.mockResolvedValueOnce(makeFetchResponse({ id: "spotify-user-1" }));

    const cookie = makeSessionCookie({ userId: FAKE_USER.id });
    await request(app).get("/api/me").set("Cookie", cookie);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain("/me");
    expect(dbState.updateCalls).toHaveLength(0);
  });
});

// ============================================================
// AUTH CALLBACK
// ============================================================
describe("GET /auth/callback", () => {
  it("rejects on state mismatch (CSRF protection)", async () => {
    const res = await request(app).get("/auth/callback?state=evil&code=abc");
    expect(res.status).toBe(400);
  });

  it("creates a new user, stores tokens in DB, redirects, never puts tokens in cookie", async () => {
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({
        access_token: "new-access",
        refresh_token: "new-refresh",
        expires_in: 3600,
      })
    );
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ id: "new-spotify-id", email: "new@e.com", display_name: "New User" })
    );

    setUserLookup(null); // no existing user
    dbState.insertResults.push([{ ...FAKE_USER, spotifyUserId: "new-spotify-id" }]);

    const agent = request.agent(app);
    const loginRes = await agent.get("/auth/login").redirects(0);
    expect(loginRes.status).toBe(302);
    const state = new URL(loginRes.headers.location).searchParams.get("state");
    expect(state).toBeTruthy();

    const cbRes = await agent.get(`/auth/callback?state=${state}&code=test-code`);
    expect(cbRes.status).toBe(302);

    expect(dbState.insertCalls).toHaveLength(1);
    expect(dbState.insertCalls[0].accessToken).toBe("new-access");
    expect(dbState.insertCalls[0].refreshToken).toBe("new-refresh");
    expect(dbState.insertCalls[0].spotifyUserId).toBe("new-spotify-id");

    const setCookie = cbRes.headers["set-cookie"];
    expect(setCookie).toBeDefined();
    const sessionCookie = setCookie.find(c => c.startsWith("session="));
    expect(sessionCookie).toBeDefined();
    const value = sessionCookie.split("=")[1].split(";")[0];
    // cookie-session strips trailing '=' padding, add back for decode
    const padded = value + "=".repeat((4 - (value.length % 4)) % 4);
    const decoded = JSON.parse(Buffer.from(padded, "base64").toString("utf-8"));
    expect(decoded.userId).toBeDefined();
    expect(decoded.accessToken).toBeUndefined();
    expect(decoded.refreshToken).toBeUndefined();
  });

  it("updates an existing user's tokens on re-login", async () => {
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ access_token: "newer-access", refresh_token: "newer-refresh", expires_in: 3600 })
    );
    fetchMock.mockResolvedValueOnce(
      makeFetchResponse({ id: FAKE_USER.spotifyUserId, email: FAKE_USER.email, display_name: FAKE_USER.displayName })
    );

    setUserLookup(FAKE_USER); // existing user
    dbState.updateResults.push([{ ...FAKE_USER, accessToken: "newer-access" }]);

    const agent = request.agent(app);
    const loginRes = await agent.get("/auth/login").redirects(0);
    const state = new URL(loginRes.headers.location).searchParams.get("state");

    await agent.get(`/auth/callback?state=${state}&code=test-code`);

    expect(dbState.insertCalls).toHaveLength(0);
    expect(dbState.updateCalls).toHaveLength(1);
    expect(dbState.updateCalls[0].accessToken).toBe("newer-access");
  });
});

// ============================================================
// VIBE ENDPOINT
// ============================================================
describe("POST /api/vibes/analyze", () => {
  const AUTH_COOKIE = makeSessionCookie({ userId: FAKE_USER.id });
  function freshUser() {
    return { ...FAKE_USER, tokenObtainedAt: new Date() };
  }
  function vibePost() {
    return request(app).post("/api/vibes/analyze").set("Cookie", AUTH_COOKIE);
  }

  it("returns 400 when no tracks provided", async () => {
    setUserLookup(freshUser());
    const res = await vibePost().send({ tracks: [] });
    expect(res.status).toBe(400);
  });

  it("returns 400 for too many tracks (>1000)", async () => {
    setUserLookup(freshUser());
    const tracks = Array.from({ length: 1001 }, (_, i) => ({ id: `t${i}`, name: `Track ${i}` }));
    const res = await vibePost().send({ tracks });
    expect(res.status).toBe(400);
  });

  it("returns cached result without calling Anthropic", async () => {
    setUserLookup(freshUser());
    setCacheLookup({
      cacheKey: "any",
      groupings: [
        { name: "A", description: "d", track_ids: ["id-0", "id-1"] },
        { name: "B", description: "d", track_ids: ["id-2", "id-3"] },
      ],
    });
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));

    const res = await vibePost().send({ tracks });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(true);
    expect(res.body.groupings).toHaveLength(2);
    expect(anthropicStreamMock).not.toHaveBeenCalled();
  });

  it("on cache miss, calls Anthropic, maps indices to real IDs, stores result", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({
      id: `spotify-id-${i}`, name: `Track ${i}`, artists: ["Artist"],
    }));

    const res = await vibePost().send({ tracks });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(false);
    expect(res.body.groupings.length).toBeGreaterThanOrEqual(1);
    for (const g of res.body.groupings) {
      for (const id of g.track_ids) {
        expect(id).toMatch(/^spotify-id-\d$/);
      }
    }
    expect(dbState.onConflictCalls.length).toBeGreaterThan(0);
    const inserted = dbState.onConflictCalls[0];
    expect(inserted.createdByUserId).toBe(FAKE_USER.id);
    expect(Array.isArray(inserted.groupings)).toBe(true);
  });

  it("force=true bypasses cache lookup and calls Anthropic", async () => {
    setUserLookup(freshUser());
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));

    const res = await vibePost().send({ tracks, force: true });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(false);
    expect(anthropicStreamMock).toHaveBeenCalledTimes(1);
  });

  it("handles LLM refusal cleanly (502 with explanatory error)", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream({
        stop_reason: "refusal",
        stop_details: { category: "cyber" },
        content: [],
        usage: {},
      })
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));
    const res = await vibePost().send({ tracks });
    expect(res.status).toBe(502);
    expect(res.body.error).toMatch(/refus/i);
  });

  it("handles LLM response with no text block (502)", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream({
        stop_reason: "end_turn",
        content: [{ type: "thinking", thinking: "..." }],
        usage: {},
      })
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));
    const res = await vibePost().send({ tracks });
    expect(res.status).toBe(502);
  });

  it("handles invalid JSON from LLM (502)", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream({
        stop_reason: "end_turn",
        content: [{ type: "text", text: "not valid json {{{" }],
        usage: {},
      })
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));
    const res = await vibePost().send({ tracks });
    expect(res.status).toBe(502);
  });

  it("regression: bad indices in LLM output don't leak hallucinated IDs", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage({
        groupings: [
          { name: "Bad", description: "garbage", track_indices: [-1, 99, "abc", 1.5] },
          { name: "Good", description: "valid", track_indices: [0, 1, 2] },
        ],
      }))
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `real-${i}`, name: `T${i}` }));

    const res = await vibePost().send({ tracks });
    expect(res.status).toBe(200);
    expect(res.body.groupings).toHaveLength(1);
    expect(res.body.groupings[0].name).toBe("Good");
    for (const id of res.body.groupings[0].track_ids) {
      expect(id).toMatch(/^real-\d$/);
    }
  });

  it("includes steer text in the LLM prompt when provided", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));

    await vibePost().send({ tracks, steer: "lean late-night" });

    expect(anthropicStreamMock).toHaveBeenCalledTimes(1);
    const callArgs = anthropicStreamMock.mock.calls[0][0];
    expect(callArgs.messages[0].content).toContain("lean late-night");
  });

  it("treats whitespace-only steer as no steer (same cache key)", async () => {
    setUserLookup(freshUser());
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );
    const tracks = Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}`, name: `T${i}` }));
    await vibePost().send({ tracks, steer: "" });
    const firstInsert = dbState.onConflictCalls[dbState.onConflictCalls.length - 1];

    setUserLookup(freshUser());
    setCacheLookup({ cacheKey: firstInsert.cacheKey, groupings: firstInsert.groupings });
    const res = await vibePost().send({ tracks, steer: "   " });

    expect(res.body.cached).toBe(true);
    expect(anthropicStreamMock).toHaveBeenCalledTimes(1);
  });
});

// ============================================================
// QUOTA + RATE LIMIT ENFORCEMENT
// ============================================================
describe("POST /api/vibes/analyze — quota + rate limit", () => {
  const AUTH_COOKIE = makeSessionCookie({ userId: FAKE_USER.id });
  function vibePost() {
    return request(app).post("/api/vibes/analyze").set("Cookie", AUTH_COOKIE);
  }
  function userWith(overrides) {
    return {
      ...FAKE_USER,
      tokenObtainedAt: new Date(),
      quotaResetAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // future = no lazy reset
      monthlyVibeQuotaUsed: 0,
      lastFreshAnalysisAt: null,
      subscriptionStatus: null,
      ...overrides,
    };
  }
  const tracks = () => Array.from({ length: 6 }, (_, i) => ({ id: `id-${i}` }));

  it("includes quota.used/limit/reset_at on a cache hit (without incrementing)", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 1 }));
    setCacheLookup({
      cacheKey: "k",
      groupings: [{ name: "A", description: "d", track_ids: ["id-0", "id-1"] }],
    });

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(true);
    expect(res.body.quota).toBeDefined();
    expect(res.body.quota.used).toBe(1);
    expect(res.body.quota.limit).toBe(3);
    const incs = dbState.updateCalls.filter((u) => u.monthlyVibeQuotaUsed !== undefined);
    expect(incs).toHaveLength(0);
  });

  it("increments the counter after a successful fresh analysis", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 0 }));
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(false);
    const inc = dbState.updateCalls.find((u) => u.monthlyVibeQuotaUsed === 1);
    expect(inc).toBeDefined();
    expect(inc.lastFreshAnalysisAt).toBeInstanceOf(Date);
    expect(res.body.quota.used).toBe(1);
  });

  it("returns 429 (quota exceeded) when used >= limit; does NOT call Anthropic", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 3 }));
    setCacheLookup(null);

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(429);
    expect(res.body.error).toMatch(/quota/i);
    expect(res.body.quota.used).toBe(3);
    expect(res.body.quota.limit).toBe(3);
    expect(anthropicStreamMock).not.toHaveBeenCalled();
    const incs = dbState.updateCalls.filter((u) => u.monthlyVibeQuotaUsed !== undefined);
    expect(incs).toHaveLength(0);
  });

  it("Pro subscribers get the 50/mo limit", async () => {
    setUserLookup(userWith({ subscriptionStatus: "active", monthlyVibeQuotaUsed: 10 }));
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(200);
    expect(res.body.quota.limit).toBe(50);
  });

  it("lazy-resets the counter when quota_reset_at is in the past", async () => {
    const expired = userWith({
      monthlyVibeQuotaUsed: 3, // would otherwise be over free limit
      quotaResetAt: new Date(Date.now() - 1000),
    });
    setUserLookup(expired);
    dbState.updateResults.push([{
      ...expired,
      monthlyVibeQuotaUsed: 0,
      quotaResetAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
    }]);
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream(llmTextMessage(VALID_LLM_RESPONSE))
    );

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(200);
    // First UPDATE was the lazy reset (used→0)
    expect(dbState.updateCalls[0].monthlyVibeQuotaUsed).toBe(0);
    expect(dbState.updateCalls[0].quotaResetAt).toBeInstanceOf(Date);
    // Second UPDATE incremented to 1
    const incCall = dbState.updateCalls.find((u) => u.monthlyVibeQuotaUsed === 1);
    expect(incCall).toBeDefined();
  });

  it("does NOT count a refused Anthropic call against the quota", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 0 }));
    setCacheLookup(null);
    anthropicStreamMock.mockReturnValueOnce(
      makeFakeAnthropicStream({
        stop_reason: "refusal",
        stop_details: { category: "cyber" },
        content: [],
        usage: {},
      })
    );

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(502);
    const incs = dbState.updateCalls.filter((u) => u.monthlyVibeQuotaUsed !== undefined);
    expect(incs).toHaveLength(0);
  });

  it("does NOT count an Anthropic exception against the quota", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 0 }));
    setCacheLookup(null);
    anthropicStreamMock.mockImplementationOnce(() => ({
      async finalMessage() { throw new Error("network down"); },
    }));

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(500);
    const incs = dbState.updateCalls.filter((u) => u.monthlyVibeQuotaUsed !== undefined);
    expect(incs).toHaveLength(0);
  });

  it("returns 429 (rate-limited) when called again within the 30s cooldown; does NOT call Anthropic", async () => {
    setUserLookup(userWith({ lastFreshAnalysisAt: new Date(Date.now() - 2000) }));
    setCacheLookup(null);

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(429);
    expect(res.body.error).toMatch(/rate limited/i);
    expect(res.body.retry_after_ms).toBeGreaterThan(0);
    expect(res.headers["retry-after"]).toBeDefined();
    expect(anthropicStreamMock).not.toHaveBeenCalled();
  });

  it("rate limit ignores cache hits — a cached request returns immediately even right after a fresh one", async () => {
    setUserLookup(userWith({ lastFreshAnalysisAt: new Date(Date.now() - 1000) }));
    setCacheLookup({
      cacheKey: "k",
      groupings: [{ name: "A", description: "d", track_ids: ["id-0", "id-1"] }],
    });

    const res = await vibePost().send({ tracks: tracks() });

    expect(res.status).toBe(200);
    expect(res.body.cached).toBe(true);
  });

  it("force=true still respects the monthly quota", async () => {
    setUserLookup(userWith({ monthlyVibeQuotaUsed: 3 }));

    const res = await vibePost().send({ tracks: tracks(), force: true });

    expect(res.status).toBe(429);
    expect(anthropicStreamMock).not.toHaveBeenCalled();
  });

  it("force=true still respects the rate limit", async () => {
    setUserLookup(userWith({ lastFreshAnalysisAt: new Date(Date.now() - 1000) }));

    const res = await vibePost().send({ tracks: tracks(), force: true });

    expect(res.status).toBe(429);
    expect(res.body.error).toMatch(/rate limited/i);
    expect(anthropicStreamMock).not.toHaveBeenCalled();
  });
});
