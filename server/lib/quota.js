// Per-user quota + rate-limit policy. Pure functions, no DB.
// The actual writes (counter increment, reset timestamps) live in the
// vibe endpoint so we don't fan out DB coupling here.

// Tier limits — easy to bump later. NULL subscription_status is treated as 'free'.
const FREE_MONTHLY_LIMIT = 3;
const PRO_MONTHLY_LIMIT = 50;
// How often the monthly counter resets.
const QUOTA_RESET_INTERVAL_MS = 30 * 24 * 60 * 60 * 1000;
// Anti-spam: minimum gap between two fresh (non-cached) analyses per user.
const RATE_LIMIT_INTERVAL_MS = 30 * 1000;

/**
 * Returns the monthly fresh-analysis cap for a user, based on subscription status.
 * NULL or 'free' or any unknown value → free tier.
 */
function effectiveQuotaLimit(user) {
  if (user?.subscriptionStatus === "active") return PRO_MONTHLY_LIMIT;
  return FREE_MONTHLY_LIMIT;
}

/**
 * Returns true if the user's quota window has expired (or was never set), so
 * the counter should be zeroed and `quotaResetAt` bumped to `now + 30 days`.
 */
function quotaNeedsReset(user, now = new Date()) {
  if (!user?.quotaResetAt) return true;
  const resetMs = new Date(user.quotaResetAt).getTime();
  if (!Number.isFinite(resetMs)) return true;
  return now.getTime() >= resetMs;
}

/**
 * Compute the next quota reset timestamp = now + 30 days.
 */
function nextQuotaResetDate(now = new Date()) {
  return new Date(now.getTime() + QUOTA_RESET_INTERVAL_MS);
}

/**
 * Given the user's current quota state (already lazy-reset if needed),
 * decide whether they can issue ONE more fresh analysis.
 * Returns { allowed, used, limit, resetAt }.
 */
function checkMonthlyQuota(user) {
  const used = typeof user?.monthlyVibeQuotaUsed === "number" ? user.monthlyVibeQuotaUsed : 0;
  const limit = effectiveQuotaLimit(user);
  return {
    allowed: used < limit,
    used,
    limit,
    resetAt: user?.quotaResetAt ?? null,
  };
}

/**
 * Returns the number of milliseconds the user still has to wait before issuing
 * another fresh analysis. 0 means "good to go".
 */
function rateLimitWaitMs(user, now = new Date()) {
  if (!user?.lastFreshAnalysisAt) return 0;
  const lastMs = new Date(user.lastFreshAnalysisAt).getTime();
  if (!Number.isFinite(lastMs)) return 0;
  const elapsed = now.getTime() - lastMs;
  if (elapsed >= RATE_LIMIT_INTERVAL_MS) return 0;
  return RATE_LIMIT_INTERVAL_MS - elapsed;
}

module.exports = {
  FREE_MONTHLY_LIMIT,
  PRO_MONTHLY_LIMIT,
  QUOTA_RESET_INTERVAL_MS,
  RATE_LIMIT_INTERVAL_MS,
  effectiveQuotaLimit,
  quotaNeedsReset,
  nextQuotaResetDate,
  checkMonthlyQuota,
  rateLimitWaitMs,
};
