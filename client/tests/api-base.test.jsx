// Verifies VITE_API_BASE_URL is honored at module load.
//
// The localhost fallback is covered implicitly by every other test in this
// suite — the standard test environment leaves VITE_API_BASE_URL unset, so the
// constant resolves to "http://127.0.0.1:4000" and all fetch routes assume
// that prefix.

import { describe, it, expect, beforeAll, beforeEach, afterAll, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";

describe("VITE_API_BASE_URL", () => {
  const PROD_URL = "https://api.example.com";
  let App;
  let fetchMock;

  beforeAll(async () => {
    vi.stubEnv("VITE_API_BASE_URL", PROD_URL);
    vi.resetModules();
    App = (await import("../src/App.jsx")).default;
  });

  afterAll(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: false,
        status: 401,
        headers: new Map(),
        json: async () => ({ error: "Not authenticated" }),
        text: async () => "",
      })
    );
    globalThis.fetch = fetchMock;
  });

  it("uses VITE_API_BASE_URL as the prefix for API calls when set", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const firstUrl = fetchMock.mock.calls[0][0];
    expect(firstUrl.startsWith(PROD_URL)).toBe(true);
  });

  it("does NOT fall back to localhost when VITE_API_BASE_URL is set", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    for (const [url] of fetchMock.mock.calls) {
      expect(url).not.toContain("127.0.0.1");
      expect(url).not.toContain("localhost");
    }
  });

  it("the initial /api/me probe uses the production prefix", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const meCall = fetchMock.mock.calls.find(([u]) => u.endsWith("/api/me"));
    expect(meCall).toBeDefined();
    expect(meCall[0]).toBe(`${PROD_URL}/api/me`);
  });
});

describe("VITE_API_BASE_URL fallback (default test env)", () => {
  // No stubEnv here — explicitly verify the default fallback.
  let App;
  let fetchMock;

  beforeAll(async () => {
    vi.resetModules();
    App = (await import("../src/App.jsx")).default;
  });

  afterAll(() => {
    vi.resetModules();
  });

  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: false, status: 401, headers: new Map(),
        json: async () => ({}), text: async () => "",
      })
    );
    globalThis.fetch = fetchMock;
  });

  it("falls back to http://127.0.0.1:4000 when VITE_API_BASE_URL is unset", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const firstUrl = fetchMock.mock.calls[0][0];
    expect(firstUrl.startsWith("http://127.0.0.1:4000")).toBe(true);
  });
});

describe("VITE_API_BASE_URL set to empty string (production same-origin via Vercel rewrites)", () => {
  // When Vercel rewrites /api/* and /auth/* to the Render origin, the client
  // should make RELATIVE calls so the browser sees them as same-origin.
  let App;
  let fetchMock;

  beforeAll(async () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    vi.resetModules();
    App = (await import("../src/App.jsx")).default;
  });

  afterAll(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: false, status: 401, headers: new Map(),
        json: async () => ({}), text: async () => "",
      })
    );
    globalThis.fetch = fetchMock;
  });

  it("treats an empty VITE_API_BASE_URL as 'use relative paths' (does NOT fall back to localhost)", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    for (const [url] of fetchMock.mock.calls) {
      // Relative URLs (e.g. "/api/me") or absolute paths — but NOT localhost.
      expect(url).not.toContain("127.0.0.1");
      expect(url).not.toContain("localhost");
    }
  });

  it("/api/me probe is requested as a relative path", async () => {
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const meCall = fetchMock.mock.calls.find(([u]) => u.endsWith("/api/me"));
    expect(meCall).toBeDefined();
    // No protocol/host prefix when API_BASE is empty string.
    expect(meCall[0]).toBe("/api/me");
  });
});
