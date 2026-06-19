// Verifies the env-gated production hardening (trust proxy, secure cookies)
// without polluting the env for the other test files.
//
// We re-import index.js with NODE_ENV temporarily set to "production" so the
// IS_PRODUCTION branch runs. vi.resetModules() drops the cached module from
// other suites; the patched Module.prototype.require from setup.mjs stays.

import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";

// Same stub mocks as the other integration files so the app loads.
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
globalThis.__TEST_DB_MOCK__ = globalThis.__TEST_DB_MOCK__ || {
  db: { select: () => selectChain(), insert: () => insertChain(), update: () => updateChain() },
  schema: { users: { id: {}, spotifyUserId: {} }, vibeCaches: { cacheKey: {} } },
  pool: { end: () => {} },
};
if (!globalThis.__TEST_ANTHROPIC_CLASS__) {
  class FakeAnthropic { constructor() { this.messages = { stream: vi.fn() }; } }
  globalThis.__TEST_ANTHROPIC_CLASS__ = Object.assign(FakeAnthropic, { default: FakeAnthropic });
}
globalThis.fetch = globalThis.fetch || vi.fn();

const ORIGINAL_NODE_ENV = process.env.NODE_ENV;

describe("Production hardening (NODE_ENV=production)", () => {
  let prodApp;
  beforeAll(async () => {
    process.env.NODE_ENV = "production";
    vi.resetModules();
    prodApp = (await import("../index.js")).default;
  });
  afterAll(() => {
    process.env.NODE_ENV = ORIGINAL_NODE_ENV;
    vi.resetModules();
  });

  it("enables Express 'trust proxy' (req.secure/req.ip honor the load balancer)", () => {
    expect(prodApp.get("trust proxy")).toBe(1);
  });

  it("the app still exports the configured Express instance", () => {
    expect(typeof prodApp.use).toBe("function");
    expect(typeof prodApp.listen).toBe("function");
  });

  it("session cookie is SameSite=None + Secure (cross-site XHR support, Vercel→Render)", async () => {
    const request = (await import("supertest")).default;
    // Hitting /auth/login mutates req.session.spotifyState, which forces
    // cookie-session to emit a Set-Cookie header on the response.
    // We forge X-Forwarded-Proto: https so the trust-proxy-aware cookie-session
    // treats the request as secure (production load balancers do this).
    const res = await request(prodApp)
      .get("/auth/login")
      .set("X-Forwarded-Proto", "https")
      .redirects(0);
    const setCookie = res.headers["set-cookie"] || [];
    const sessionCookie = setCookie.find((c) => c.startsWith("session="));
    expect(sessionCookie, "expected a session cookie to be set").toBeDefined();
    expect(sessionCookie.toLowerCase()).toContain("samesite=none");
    expect(sessionCookie.toLowerCase()).toContain("secure");
    expect(sessionCookie.toLowerCase()).toContain("httponly");
  });
});

describe("Local/test config (NODE_ENV != production)", () => {
  let localApp;
  beforeAll(async () => {
    // Reset to default test env (setup.mjs sets various vars but leaves
    // NODE_ENV unset — Vitest sets it to "test" by default).
    process.env.NODE_ENV = ORIGINAL_NODE_ENV || "test";
    vi.resetModules();
    localApp = (await import("../index.js")).default;
  });

  it("does NOT enable trust proxy in non-production env", () => {
    // Express default is false when never set.
    expect(localApp.get("trust proxy")).toBe(false);
  });
});
