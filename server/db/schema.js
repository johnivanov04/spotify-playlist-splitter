const { pgTable, text, timestamp, integer, uuid } = require("drizzle-orm/pg-core");

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

  // Subscription / quota — wired up when Stripe lands. Nullable for now.
  subscriptionStatus: text("subscription_status"), // 'free' | 'active' | 'past_due' | 'canceled' | null
  monthlyVibeQuotaUsed: integer("monthly_vibe_quota_used").default(0),
  quotaResetAt: timestamp("quota_reset_at", { withTimezone: true }),

  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

module.exports = { users };
