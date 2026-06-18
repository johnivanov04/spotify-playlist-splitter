import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import tokenLib from "../lib/token.js";

const { isTokenNearExpiry } = tokenLib;

describe("isTokenNearExpiry", () => {
  const NOW = new Date("2026-06-18T12:00:00Z").getTime();

  beforeAll(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterAll(() => vi.useRealTimers());

  it("returns true when user has no token timestamps (fail-open to refresh)", () => {
    expect(isTokenNearExpiry({})).toBe(true);
    expect(isTokenNearExpiry({ tokenObtainedAt: null })).toBe(true);
    expect(isTokenNearExpiry({ tokenObtainedAt: new Date(NOW) })).toBe(true);
    expect(isTokenNearExpiry({ expiresInSeconds: 3600 })).toBe(true);
    expect(isTokenNearExpiry(null)).toBe(true);
  });

  it("returns false for a freshly obtained token with full lifetime ahead", () => {
    expect(
      isTokenNearExpiry({ tokenObtainedAt: new Date(NOW), expiresInSeconds: 3600 })
    ).toBe(false);
  });

  it("returns false when 10 minutes remain (outside refresh window)", () => {
    const obtainedAt = new Date(NOW - 50 * 60 * 1000);
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(false);
  });

  it("returns false at exactly 5 minutes remaining (boundary; strict >)", () => {
    // Implementation refreshes only when ageMs strictly > expiresInMs - 5min,
    // so exactly 5 minutes remaining does NOT trigger refresh.
    const obtainedAt = new Date(NOW - 55 * 60 * 1000);
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(false);
  });

  it("returns true at 5 minutes minus 1ms remaining (just past threshold)", () => {
    const obtainedAt = new Date(NOW - (55 * 60 * 1000 + 1));
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(true);
  });

  it("returns true when 3 minutes remain (within window)", () => {
    const obtainedAt = new Date(NOW - 57 * 60 * 1000);
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(true);
  });

  it("returns true for an already-expired token", () => {
    const obtainedAt = new Date(NOW - 2 * 60 * 60 * 1000);
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(true);
  });

  it("accepts ISO string tokenObtainedAt (from DB driver)", () => {
    const obtainedAt = new Date(NOW).toISOString();
    expect(
      isTokenNearExpiry({ tokenObtainedAt: obtainedAt, expiresInSeconds: 3600 })
    ).toBe(false);
  });

  it("returns true when tokenObtainedAt parses to NaN", () => {
    expect(
      isTokenNearExpiry({ tokenObtainedAt: "not-a-date", expiresInSeconds: 3600 })
    ).toBe(true);
  });
});
