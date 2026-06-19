// Shared helpers for integration tests.
import crypto from "crypto";

/**
 * Build a valid cookie-session header for supertest. Matches the server's
 * cookie-session config: name "session", key = SESSION_SECRET.
 *
 * cookie-session uses Keygrip for the signature, which produces URL-safe
 * base64 (`-_` instead of `+/`, no padding) for both the value and the sig.
 */
export function makeSessionCookie(sessionData, secret = process.env.SESSION_SECRET || "test-session-secret") {
  const value = Buffer.from(JSON.stringify(sessionData)).toString("base64")
    .replace(/=+$/, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
  const sig = crypto
    .createHmac("sha1", secret)
    .update(`session=${value}`)
    .digest("base64")
    .replace(/=+$/, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
  return `session=${value}; session.sig=${sig}`;
}

/**
 * Build a fake `fetch` Response so endpoint code can `await response.json()` /
 * `.text()` / read `.ok` and `.status`.
 */
export function makeFetchResponse(body, { status = 200, headers = {} } = {}) {
  const isJson = typeof body === "object" && body !== null;
  const text = isJson ? JSON.stringify(body) : String(body ?? "");
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Map(Object.entries(headers)),
    json: async () => (isJson ? body : JSON.parse(text)),
    text: async () => text,
  };
}

/**
 * Build a Drizzle-shaped mock that supports the chained query patterns the
 * server uses:
 *   db.select().from(t).where(c).limit(n)         -> resolves to array
 *   db.insert(t).values(v).returning()            -> resolves to array
 *   db.update(t).set(v).where(c).returning()      -> resolves to array
 *   db.insert(t).values(v).onConflictDoNothing()  -> resolves to undefined
 *
 * stubs is a hash of { selects: [...], inserts: [...], updates: [...] } where
 * each entry is consumed in order by the matching chain. If stubs are
 * exhausted, returns [] / undefined.
 */
export function makeMockDb(stubs = {}) {
  const selects = [...(stubs.selects || [])];
  const inserts = [...(stubs.inserts || [])];
  const updates = [...(stubs.updates || [])];
  const calls = { selects: [], inserts: [], updates: [], onConflict: [] };

  function selectChain() {
    return {
      from() { return this; },
      where() { return this; },
      limit() {
        const next = selects.shift();
        return Promise.resolve(next ?? []);
      },
      then(resolve) {
        const next = selects.shift();
        return Promise.resolve(next ?? []).then(resolve);
      },
    };
  }

  function insertChain() {
    let captured;
    const chain = {
      values(v) { captured = v; calls.inserts.push(v); return this; },
      returning() {
        const next = inserts.shift();
        return Promise.resolve(next ?? []);
      },
      onConflictDoNothing() {
        calls.onConflict.push(captured);
        return Promise.resolve(undefined);
      },
    };
    return chain;
  }

  function updateChain() {
    let captured;
    const chain = {
      set(v) { captured = v; calls.updates.push(v); return this; },
      where() { return this; },
      returning() {
        const next = updates.shift();
        return Promise.resolve(next ?? []);
      },
    };
    return chain;
  }

  return {
    db: {
      select: () => selectChain(),
      insert: () => insertChain(),
      update: () => updateChain(),
    },
    calls,
  };
}
