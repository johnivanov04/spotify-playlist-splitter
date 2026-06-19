import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import quotaLib from "../lib/quota.js";

const {
  FREE_MONTHLY_LIMIT,
  PRO_MONTHLY_LIMIT,
  QUOTA_RESET_INTERVAL_MS,
  RATE_LIMIT_INTERVAL_MS,
  effectiveQuotaLimit,
  quotaNeedsReset,
  nextQuotaResetDate,
  checkMonthlyQuota,
  rateLimitWaitMs,
} = quotaLib;

describe("effectiveQuotaLimit", () => {
  it("returns the free limit for users with no subscription_status", () => {
    expect(effectiveQuotaLimit({})).toBe(FREE_MONTHLY_LIMIT);
    expect(effectiveQuotaLimit({ subscriptionStatus: null })).toBe(FREE_MONTHLY_LIMIT);
    expect(effectiveQuotaLimit(null)).toBe(FREE_MONTHLY_LIMIT);
    expect(effectiveQuotaLimit(undefined)).toBe(FREE_MONTHLY_LIMIT);
  });

  it("returns the free limit for explicit 'free' status", () => {
    expect(effectiveQuotaLimit({ subscriptionStatus: "free" })).toBe(FREE_MONTHLY_LIMIT);
  });

  it("returns the pro limit only for 'active' status", () => {
    expect(effectiveQuotaLimit({ subscriptionStatus: "active" })).toBe(PRO_MONTHLY_LIMIT);
  });

  it("treats 'past_due' and 'canceled' as free tier (no Pro perks once payment lapses)", () => {
    expect(effectiveQuotaLimit({ subscriptionStatus: "past_due" })).toBe(FREE_MONTHLY_LIMIT);
    expect(effectiveQuotaLimit({ subscriptionStatus: "canceled" })).toBe(FREE_MONTHLY_LIMIT);
  });

  it("treats unknown subscription values as free tier (defensive default)", () => {
    expect(effectiveQuotaLimit({ subscriptionStatus: "garbage" })).toBe(FREE_MONTHLY_LIMIT);
  });
});

describe("quotaNeedsReset", () => {
  const NOW = new Date("2026-06-19T12:00:00Z");

  it("returns true when quotaResetAt is null/undefined (never reset before)", () => {
    expect(quotaNeedsReset({}, NOW)).toBe(true);
    expect(quotaNeedsReset({ quotaResetAt: null }, NOW)).toBe(true);
    expect(quotaNeedsReset(null, NOW)).toBe(true);
  });

  it("returns false when quotaResetAt is in the future", () => {
    const future = new Date(NOW.getTime() + 5 * 24 * 60 * 60 * 1000);
    expect(quotaNeedsReset({ quotaResetAt: future }, NOW)).toBe(false);
  });

  it("returns true when quotaResetAt is in the past", () => {
    const past = new Date(NOW.getTime() - 1000);
    expect(quotaNeedsReset({ quotaResetAt: past }, NOW)).toBe(true);
  });

  it("returns true exactly at quotaResetAt (boundary: >=)", () => {
    expect(quotaNeedsReset({ quotaResetAt: NOW }, NOW)).toBe(true);
  });

  it("accepts ISO-string quotaResetAt (DB driver-provided)", () => {
    const future = new Date(NOW.getTime() + 1000).toISOString();
    expect(quotaNeedsReset({ quotaResetAt: future }, NOW)).toBe(false);
  });

  it("returns true when quotaResetAt is not a valid date", () => {
    expect(quotaNeedsReset({ quotaResetAt: "garbage" }, NOW)).toBe(true);
  });
});

describe("nextQuotaResetDate", () => {
  it("returns now + 30 days", () => {
    const now = new Date("2026-06-19T12:00:00Z");
    const next = nextQuotaResetDate(now);
    expect(next.getTime() - now.getTime()).toBe(QUOTA_RESET_INTERVAL_MS);
  });

  it("defaults to using the real current time when no argument is supplied", () => {
    const before = Date.now();
    const next = nextQuotaResetDate();
    const after = Date.now();
    const delta = next.getTime() - before;
    // Tolerance: it should be roughly the interval, never less, never wildly more.
    expect(delta).toBeGreaterThanOrEqual(QUOTA_RESET_INTERVAL_MS);
    expect(delta).toBeLessThanOrEqual(QUOTA_RESET_INTERVAL_MS + (after - before) + 5);
  });
});

describe("checkMonthlyQuota", () => {
  it("allows when used < limit (free tier)", () => {
    const out = checkMonthlyQuota({ monthlyVibeQuotaUsed: 1 });
    expect(out.allowed).toBe(true);
    expect(out.used).toBe(1);
    expect(out.limit).toBe(FREE_MONTHLY_LIMIT);
  });

  it("denies when used === limit (boundary)", () => {
    const out = checkMonthlyQuota({ monthlyVibeQuotaUsed: FREE_MONTHLY_LIMIT });
    expect(out.allowed).toBe(false);
  });

  it("denies when used > limit (shouldn't happen but defensive)", () => {
    const out = checkMonthlyQuota({ monthlyVibeQuotaUsed: 999 });
    expect(out.allowed).toBe(false);
  });

  it("treats null/undefined used as 0", () => {
    expect(checkMonthlyQuota({ monthlyVibeQuotaUsed: null }).used).toBe(0);
    expect(checkMonthlyQuota({}).used).toBe(0);
    expect(checkMonthlyQuota(null).used).toBe(0);
  });

  it("uses Pro limit for active subscribers", () => {
    const out = checkMonthlyQuota({
      subscriptionStatus: "active",
      monthlyVibeQuotaUsed: 5,
    });
    expect(out.allowed).toBe(true);
    expect(out.limit).toBe(PRO_MONTHLY_LIMIT);
  });

  it("returns the user's reset_at unchanged", () => {
    const resetAt = new Date("2026-07-19T00:00:00Z");
    const out = checkMonthlyQuota({ monthlyVibeQuotaUsed: 0, quotaResetAt: resetAt });
    expect(out.resetAt).toBe(resetAt);
  });
});

describe("rateLimitWaitMs", () => {
  const NOW = new Date("2026-06-19T12:00:00Z").getTime();

  beforeAll(() => { vi.useFakeTimers(); vi.setSystemTime(NOW); });
  afterAll(() => vi.useRealTimers());

  it("returns 0 when the user has never made a fresh analysis", () => {
    expect(rateLimitWaitMs({})).toBe(0);
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: null })).toBe(0);
    expect(rateLimitWaitMs(null)).toBe(0);
  });

  it("returns 0 when the cooldown has fully elapsed", () => {
    const lastAt = new Date(NOW - (RATE_LIMIT_INTERVAL_MS + 1));
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: lastAt })).toBe(0);
  });

  it("returns 0 exactly at the cooldown boundary", () => {
    const lastAt = new Date(NOW - RATE_LIMIT_INTERVAL_MS);
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: lastAt })).toBe(0);
  });

  it("returns the remaining ms when still within the cooldown", () => {
    const lastAt = new Date(NOW - 10 * 1000); // 10s ago, 30s cooldown
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: lastAt })).toBe(20 * 1000);
  });

  it("returns the full interval when the call just happened", () => {
    const lastAt = new Date(NOW);
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: lastAt })).toBe(RATE_LIMIT_INTERVAL_MS);
  });

  it("returns 0 if lastFreshAnalysisAt is not a valid date", () => {
    expect(rateLimitWaitMs({ lastFreshAnalysisAt: "garbage" })).toBe(0);
  });
});
