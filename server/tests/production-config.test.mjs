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
