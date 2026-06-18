CREATE TABLE "vibe_caches" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"cache_key" text NOT NULL,
	"groupings" jsonb NOT NULL,
	"created_by_user_id" uuid,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "vibe_caches_cache_key_unique" UNIQUE("cache_key")
);
--> statement-breakpoint
ALTER TABLE "vibe_caches" ADD CONSTRAINT "vibe_caches_created_by_user_id_users_id_fk" FOREIGN KEY ("created_by_user_id") REFERENCES "public"."users"("id") ON DELETE set null ON UPDATE no action;