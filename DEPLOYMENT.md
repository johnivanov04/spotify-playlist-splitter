# Deployment

Hosting checklist for taking Playlist Splitter from local-only to a publicly
accessible HTTPS app. Required for the Spotify Extended Quota Mode application.

## Architecture

| Tier | Recommended host | Why |
|---|---|---|
| Client (Vite + React) | **Vercel** | Zero-config static deploys, free hobby tier, automatic HTTPS, instant preview deploys per PR. |
| Server (Express + Node) | **Render** or **Fly.io** | Long-running Node server (vibe analysis can take 60–90s — incompatible with serverless-function timeouts on free tiers). |
| Database | **Neon** (already set up) | Same `DATABASE_URL` works in prod; just add a separate Neon branch if you want prod/dev isolation. |

Both Render and Fly let you set environment variables in the dashboard and
read them via `process.env` exactly like local. No code changes between local
and prod — only env vars.

## Required environment variables (server)

| Var | Local | Production |
|---|---|---|
| `NODE_ENV` | unset (defaults to "test"/"development") | **`production`** — enables `trust proxy`, secure cookies, SameSite=Lax |
| `SPOTIFY_CLIENT_ID` | from `.env` | same — copy from Spotify dev dashboard |
| `SPOTIFY_CLIENT_SECRET` | from `.env` | same |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:4000/auth/callback` | `https://<your-server-host>/auth/callback` |
| `FRONTEND_URL` | `http://127.0.0.1:5173` | `https://<your-client-host>` |
| `SESSION_SECRET` | dev value | **rotate to a fresh, long random string** — do not reuse the dev secret |
| `ANTHROPIC_API_KEY` | from `.env` | same key OR a separate prod key with a tighter spend cap |
| `DATABASE_URL` | Neon dev branch | Neon prod branch (or same — both work) |
| `MUSICBRAINZ_USER_AGENT` | as set | a value that identifies your prod deployment (e.g. `PlaylistSplitter/1.0 (you@example.com)`) — only used by the dormant brainz endpoint, but Spotify reviewers may notice |
| `PORT` | 4000 | whatever the host assigns (Render sets `PORT` automatically) |

## Required environment variables (client)

The client currently hard-codes `const API_BASE = "http://127.0.0.1:4000"` in
`client/src/App.jsx`. **Before deploying:**

1. Change that constant to read from `import.meta.env.VITE_API_BASE_URL`,
   falling back to `http://127.0.0.1:4000` for local dev.
2. Set `VITE_API_BASE_URL=https://<your-server-host>` in Vercel's project
   env vars.

(That's the one source-level change you'll want to make right before deploying.
I've intentionally not done it now so local dev keeps working with the current
constant; flag it when you're ready.)

## Spotify dev dashboard config

1. Go to <https://developer.spotify.com/dashboard> → your app.
2. Under **Redirect URIs**, add the production callback:
   `https://<your-server-host>/auth/callback`. Leave the local one in place.
3. Save.
4. When you submit for Extended Quota Mode, the demo URL in the form should
   be the **frontend** host (`https://<your-client-host>`).

## Health check endpoint

`GET /api/health` is wired up — public, no auth, returns
`{status, uptime, timestamp}`. Configure your host:

- **Render**: Settings → "Health Check Path" → `/api/health`.
- **Fly.io**: in `fly.toml`, add `[checks.health] path = "/api/health"`.
- **Both**: 5–10s interval is plenty; the endpoint is constant-time.

## CORS

`server/index.js` already builds the allowed-origin list from `FRONTEND_URL`
plus the local-dev hosts. Setting `FRONTEND_URL=https://<your-client-host>`
in prod is enough — no code change needed.

## Cookie / session behavior in production

When `NODE_ENV=production`:

- `app.set("trust proxy", 1)` — Express respects `X-Forwarded-Proto` / `X-Forwarded-For` from the load balancer.
- Cookie `secure: true` — browsers only send the session cookie over HTTPS.
- Cookie `httpOnly: true` — JS can never read it (already true locally too).
- Cookie `sameSite: "lax"` — sends the cookie on top-level cross-site navigations (needed for the Spotify OAuth redirect) but not on other cross-site requests.

These gates are covered by `server/tests/production-config.test.mjs`. Running
the test suite verifies the prod config without actually deploying.

## What to do once you have URLs

1. Update Spotify dev dashboard with the production redirect URI.
2. Set `VITE_API_BASE_URL` on Vercel.
3. Set all server env vars on Render/Fly (especially `NODE_ENV=production`,
   `SPOTIFY_REDIRECT_URI`, `FRONTEND_URL`, `SESSION_SECRET`).
4. Deploy server first, confirm `/api/health` returns 200 from the public URL.
5. Deploy client, confirm it loads and the Login button redirects through to
   Spotify and back.
6. Run a real vibe analysis end-to-end to verify Neon + Anthropic are reachable
   from the prod server.
7. Submit Extended Quota Mode application with the public client URL + privacy
   policy + ToS.

## Not yet done (deferred to actual deploy day)

- The client API base URL refactor (`VITE_API_BASE_URL`). Trivial; left in
  place to keep local dev working.
- A `Dockerfile` or `fly.toml` — depends on the chosen host; we'll write the
  one that fits when you pick.
- Sentry / error tracking — recommended after first deploy.
- Express-rate-limit for general anti-DoS — the per-user vibe rate limit handles
  the expensive call; broader rate limiting is a nice-to-have, not a blocker.
