CREATE TABLE "users" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"spotify_user_id" text NOT NULL,
	"email" text,
	"display_name" text,
	"access_token" text,
	"refresh_token" text,
	"expires_in_seconds" integer,
	"token_obtained_at" timestamp with time zone,
	"subscription_status" text,
	"monthly_vibe_quota_used" integer DEFAULT 0,
	"quota_reset_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "users_spotify_user_id_unique" UNIQUE("spotify_user_id")
);
