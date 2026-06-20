// Verifies the production-only static client + SPA fallback.
//
// In production the same Render service hosts the API and the built client.
// We exercise the static middleware against a fake client/dist directory so
// these tests don't depend on `vite build` having been run.

import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import { mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";

// Same DB / Anthropic stubs as the rest of the integration tests so the app
// can be imported.
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

// The server resolves `clientDist` as path.resolve(__dirname, "../client/dist"),
// where __dirname is the `server/` directory. We populate that exact path so
// the prod-mode app finds the fake build artifacts.
const SERVER_DIR = resolve(import.meta.dirname, "..");
const REPO_ROOT = resolve(SERVER_DIR, "..");
const CLIENT_DIST = resolve(REPO_ROOT, "client", "dist");

const INDEX_HTML = `<!doctype html>
<html><head><title>Playlist Splitter Test</title></head>
<body><div id="root">spa-root-test-marker</div></body></html>`;
const HASHED_JS = `console.log("hashed-bundle-test-marker");`;

let createdAssetsDir = false;
let preExisting = false;
let preExistingIndex = "";
let preExistingHashed = null;

describe("Production: server hosts client/dist", () => {
  let prodApp;

  beforeAll(async () => {
    // If the developer has actually run `vite build`, don't clobber their
    // artifacts. We snapshot index.html and replace it for the test, then
    // restore afterward. assets/ is created only if missing.
    if (existsSync(join(CLIENT_DIST, "index.html"))) {
      preExisting = true;
      preExistingIndex = readFileSync(join(CLIENT_DIST, "index.html"), "utf8");
    } else {
      mkdirSync(CLIENT_DIST, { recursive: true });
    }
    if (!existsSync(join(CLIENT_DIST, "assets"))) {
      mkdirSync(join(CLIENT_DIST, "assets"), { recursive: true });
      createdAssetsDir = true;
    }
    if (existsSync(join(CLIENT_DIST, "assets", "test-bundle.js"))) {
      preExistingHashed = readFileSync(join(CLIENT_DIST, "assets", "test-bundle.js"), "utf8");
    }
    writeFileSync(join(CLIENT_DIST, "index.html"), INDEX_HTML);
    writeFileSync(join(CLIENT_DIST, "assets", "test-bundle.js"), HASHED_JS);

    process.env.NODE_ENV = "production";
    vi.resetModules();
    prodApp = (await import("../index.js")).default;
  });

  afterAll(() => {
    process.env.NODE_ENV = ORIGINAL_NODE_ENV;
    vi.resetModules();

    if (preExisting) {
      writeFileSync(join(CLIENT_DIST, "index.html"), preExistingIndex);
    } else {
      try { rmSync(join(CLIENT_DIST, "index.html")); } catch {}
    }
    if (preExistingHashed !== null) {
      writeFileSync(join(CLIENT_DIST, "assets", "test-bundle.js"), preExistingHashed);
    } else {
      try { rmSync(join(CLIENT_DIST, "assets", "test-bundle.js")); } catch {}
    }
    if (createdAssetsDir) {
      try { rmSync(join(CLIENT_DIST, "assets"), { recursive: true, force: true }); } catch {}
    }
  });

  it("serves index.html for GET /", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/html/);
    expect(res.text).toContain("spa-root-test-marker");
  });

  it("serves hashed JS bundles directly from /assets/*", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/assets/test-bundle.js");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/javascript/);
    expect(res.text).toContain("hashed-bundle-test-marker");
  });

  it("SPA fallback serves index.html for unknown client-side routes", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/some/deep/spa/route");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/html/);
    expect(res.text).toContain("spa-root-test-marker");
  });

  it("does NOT swallow unknown /api/* paths — they 404 instead of returning HTML", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/api/does-not-exist");
    expect(res.status).toBe(404);
    expect(res.text).not.toContain("spa-root-test-marker");
  });

  it("does NOT swallow unknown /auth/* paths — they 404 instead of returning HTML", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/auth/does-not-exist");
    expect(res.status).toBe(404);
    expect(res.text).not.toContain("spa-root-test-marker");
  });

  it("registered API routes still take precedence over the static handler", async () => {
    const request = (await import("supertest")).default;
    const res = await request(prodApp).get("/api/health");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/json/);
    expect(res.body).toMatchObject({ status: "ok" });
  });

  it("non-GET requests are not intercepted by the SPA fallback (POST without auth still 401)", async () => {
    const request = (await import("supertest")).default;
    // POST /api/playlists requires auth; without a session this should be a
    // JSON 401 from requireSpotifyAuth, NOT the index.html.
    const res = await request(prodApp).post("/api/playlists").send({});
    expect(res.status).toBe(401);
    expect(res.headers["content-type"]).toMatch(/json/);
    expect(res.text).not.toContain("spa-root-test-marker");
  });
});

describe("Local/test mode: no static client middleware", () => {
  let localApp;

  beforeAll(async () => {
    process.env.NODE_ENV = ORIGINAL_NODE_ENV || "test";
    vi.resetModules();
    localApp = (await import("../index.js")).default;
  });

  it("does NOT serve index.html — unknown routes 404 like a plain API", async () => {
    const request = (await import("supertest")).default;
    const res = await request(localApp).get("/some/route/that/doesnt/exist");
    expect(res.status).toBe(404);
  });
});
