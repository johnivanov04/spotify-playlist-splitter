const { pgTable, text, timestamp, integer, uuid, jsonb } = require("drizzle-orm/pg-core");

// Users — one row per Spotify account that has logged in.
// Spotify tokens live here so they survive server restarts and are
// properly scoped to a single user (no shared global state).
const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),

  // Spotify-side identity
  spotifyUserId: text("spotify_user_id").notNull().unique(),
  email: text("email"),
  displayName: text("display_name"),

  // Spotify OAuth tokens — refreshed automatically on the server
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  expiresInSeconds: integer("expires_in_seconds"),
  tokenObtainedAt: timestamp("token_obtained_at", { withTimezone: true }),

  // Subscription / quota — wired up when Stripe lands. Nullable subscription
  // status is treated as 'free' by the quota helpers.
  subscriptionStatus: text("subscription_status"), // 'free' | 'active' | 'past_due' | 'canceled' | null
  monthlyVibeQuotaUsed: integer("monthly_vibe_quota_used").default(0),
  quotaResetAt: timestamp("quota_reset_at", { withTimezone: true }),
  // When the user most recently triggered a FRESH (non-cached) analysis.
  // Used by the per-user rate limit (anti-spam, separate from monthly quota).
  lastFreshAnalysisAt: timestamp("last_fresh_analysis_at", { withTimezone: true }),

  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

// Vibe caches — shared across users. Key is sha256(sorted_track_ids + steer).
// Two requests with the same playlist tracks + steer produce the same key and
// share the cached result. created_by_user_id records who paid for the
// original analysis (for cost attribution / future audit).
const vibeCaches = pgTable("vibe_caches", {
  id: uuid("id").primaryKey().defaultRandom(),
  cacheKey: text("cache_key").notNull().unique(),
  groupings: jsonb("groupings").notNull(), // [{ name, description, track_ids: [string] }]
  createdByUserId: uuid("created_by_user_id").references(() => users.id, { onDelete: "set null" }),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

module.exports = { users, vibeCaches };
