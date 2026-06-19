import { describe, it, expect, beforeAll, vi } from "vitest";
import { makeSessionCookie } from "./helpers.mjs";

// Minimal stub mocks so index.js loads — the health endpoint shouldn't touch
// any of these but the module-level wiring needs the globals to exist.
function selectChain() {
  return { from() { return this; }, where() { return this; }, limit() { return Promise.resolve([]); } };
}
function insertChain() {
  return {
    values() { return this; },
    returning() { return Promise.resolve([]); },
    onConflictDoNothing() { return Promise.resolve(); },
  };
}
function updateChain() {
  return { set() { return this; }, where() { return this; }, returning() { return Promise.resolve([]); } };
}
globalThis.__TEST_DB_MOCK__ = {
  db: { select: () => selectChain(), insert: () => insertChain(), update: () => updateChain() },
  schema: { users: { id: {}, spotifyUserId: {} }, vibeCaches: { cacheKey: {} } },
  pool: { end: () => {} },
};
class FakeAnthropic { constructor() { this.messages = { stream: vi.fn() }; } }
globalThis.__TEST_ANTHROPIC_CLASS__ = Object.assign(FakeAnthropic, { default: FakeAnthropic });
globalThis.fetch = vi.fn();

let request, app;
beforeAll(async () => {
  request = (await import("supertest")).default;
  app = (await import("../index.js")).default;
});

describe("GET /api/health", () => {
  it("is publicly accessible (no auth required)", async () => {
    const res = await request(app).get("/api/health");
    expect(res.status).toBe(200);
  });

  it("returns a JSON body with status=ok", async () => {
    const res = await request(app).get("/api/health");
    expect(res.body.status).toBe("ok");
  });

  it("reports a numeric uptime (process uptime in seconds)", async () => {
    const res = await request(app).get("/api/health");
    expect(typeof res.body.uptime).toBe("number");
    expect(res.body.uptime).toBeGreaterThanOrEqual(0);
  });

  it("includes an ISO 8601 timestamp", async () => {
    const res = await request(app).get("/api/health");
    expect(res.body.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
  });

  it("does NOT leak secrets, env vars, or DB connection info", async () => {
    const res = await request(app).get("/api/health");
    const serialized = JSON.stringify(res.body);
    expect(serialized).not.toContain("SPOTIFY_CLIENT_SECRET");
    expect(serialized).not.toContain("SESSION_SECRET");
    expect(serialized).not.toContain("ANTHROPIC_API_KEY");
    expect(serialized).not.toContain("DATABASE_URL");
    expect(serialized).not.toContain("postgresql://");
    expect(serialized).not.toContain("sk-ant-");
    // The fake access token from fixtures shouldn't accidentally leak either
    expect(serialized).not.toContain("fake-access-token");
  });

  it("does not require or read a session cookie", async () => {
    // Send a bogus session — should still 200 (handler ignores it)
    const res = await request(app)
      .get("/api/health")
      .set("Cookie", makeSessionCookie({ userId: "anything" }));
    expect(res.status).toBe(200);
  });

  it("supports HEAD requests for deploy probes", async () => {
    const res = await request(app).head("/api/health");
    // Express derives HEAD from GET; status should be 200, body empty.
    expect(res.status).toBe(200);
  });

  it("responds quickly (sanity: under 100ms for the request itself)", async () => {
    const start = Date.now();
    await request(app).get("/api/health");
    const elapsed = Date.now() - start;
    // Probes hit every few seconds; this should be cheap.
    expect(elapsed).toBeLessThan(100);
  });
});
