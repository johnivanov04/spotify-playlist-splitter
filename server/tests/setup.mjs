// Test-wide setup: provide harmless env vars so server/index.js loads without
// trying to talk to the real Spotify/Anthropic/Neon.
process.env.SPOTIFY_CLIENT_ID = process.env.SPOTIFY_CLIENT_ID || "test-client-id";
process.env.SPOTIFY_CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET || "test-client-secret";
process.env.SPOTIFY_REDIRECT_URI = process.env.SPOTIFY_REDIRECT_URI || "http://127.0.0.1:4000/auth/callback";
process.env.FRONTEND_URL = process.env.FRONTEND_URL || "http://127.0.0.1:5173";
process.env.SESSION_SECRET = process.env.SESSION_SECRET || "test-session-secret";
process.env.MUSICBRAINZ_USER_AGENT = process.env.MUSICBRAINZ_USER_AGENT || "Test/0.0 (test@example.com)";
process.env.DATABASE_URL = process.env.DATABASE_URL || "postgresql://test:test@localhost:5432/test";
process.env.ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || "test-anthropic-key";

// CommonJS `require` interception. vi.mock in Vitest 4 has trouble
// intercepting `require()` calls made by CJS modules loaded from ESM test
// files. We patch Module.prototype.require so tests can set
// globalThis.__TEST_DB_MOCK__ / __TEST_ANTHROPIC_MOCK__ before loading the
// app, and routes through the test's mock instead of hitting real Postgres.
import { Module } from "module";

const originalRequire = Module.prototype.require;
Module.prototype.require = function patchedRequire(specifier) {
  if (specifier === "./db" && globalThis.__TEST_DB_MOCK__) {
    return globalThis.__TEST_DB_MOCK__;
  }
  if (specifier === "@anthropic-ai/sdk" && globalThis.__TEST_ANTHROPIC_CLASS__) {
    return globalThis.__TEST_ANTHROPIC_CLASS__;
  }
  return originalRequire.apply(this, arguments);
};
