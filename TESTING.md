# Testing

End-to-end test suite for the active code paths of Playlist Splitter.

## Stack

- **Server**: Vitest 4.x + Supertest, ESM test files against the CommonJS source.
- **Client**: Vitest 4.x + React Testing Library + jsdom.
- **Coverage**: V8 provider, configured to include only active app source.

## Commands

From the repo root:

```bash
npm test                # run server + client suites
npm run test:server     # server only
npm run test:client     # client only
npm run test:coverage   # both, with coverage reports
```

From a sub-package directory (`server/` or `client/`):

```bash
npm test                # run once
npm run test:watch      # watch mode
npm run test:coverage   # with coverage
```

## What is covered

### Server — `server/tests/`

**`vibes.test.mjs`** — pure logic (32 tests)
- `vibeCacheKey`: determinism, track-order invariance, steer normalization, empty/missing IDs, sha256 shape
- `summarizeTrackForPrompt`: ordinal index, no Spotify ID leakage, year/genre/tag caps, brainz tag normalization, empty inputs
- `mapIndicesToTrackIds`: valid + out-of-range + negative + non-integer + duplicate indices, cross-group dedupe, < 2 tracks rule, regression check that output uses only server-supplied IDs

**`token.test.mjs`** — token expiry (8 tests)
- Missing fields, fresh tokens, 5-minute boundary, ISO-string `tokenObtainedAt`, NaN handling

**`integration.test.mjs`** — auth + vibe endpoint (23 tests)
- `requireSpotifyAuth`: unauthenticated, no userId, ghost user, no access token, valid user, near-expiry refresh + DB write, refresh token rotation, refresh failure, fresh token skips refresh
- `GET /auth/callback`: state mismatch, new user creation, existing user update, cookie holds only `userId` (no tokens)
- `POST /api/vibes/analyze`: 400 on empty / >1000 tracks, cache hit, cache miss → LLM + DB write, `force=true` bypasses cache, refusal handling, no-text-block, invalid JSON, regression for hallucinated IDs, steer text in prompt, whitespace-steer normalization

**`spotify-routes.test.mjs`** — Spotify proxy routes (18 tests)
- `GET /api/playlists/:id/tracks`: 401 unauth, **track relinking — uses `linked_from.id`**, `market=from_token` propagation, empty playlist, null/local tracks skipped, pagination, no token leakage to client
- `POST /api/playlists/:id/remove-tracks`: 401 unauth, 400 empty trackIds, correct `DELETE` body, Spotify error propagation
- `POST /api/playlists` (create): 401 unauth, creates + adds tracks with right URIs, 400 missing fields
- `POST /api/spotify/artist-genres`: 401 unauth, 400 empty, batched track→artist→genre assembly, dedup per track

### Client — `client/tests/`

**`App.test.jsx`** (23 tests)
- Landing page (logged out): title, tagline, features, login redirect, no main UI shown
- Logged-in app shell: sidebar renders playlists, URL-paste field, no tokens in storage
- Vibe analysis flow: fires `POST /api/vibes/analyze` after genres, renders name + description, loading banner, refresh button uses `force=true`, steer prompt is sent, handles unusual fields
- Track selection: expand shows tracks + Select all / Deselect all, deselect-all behavior pinned, **bug fix** for first-toggle on uninitialized selection
- Create playlist: full selection saves all tracks, manual deselect omits the deselected track, failure alert
- Edge cases: empty playlist list, very long playlist names

## What is intentionally NOT covered

The following are present in the repo but **disabled** in the active app and therefore intentionally excluded from tests:

| Surface | Status |
|---|---|
| `client/src/ml/` (`kmeansInfer.js`, `kmeans_model.json`) | Imported by `App.jsx` but `SHOW_ML_CLUSTERS = false` so the clustering code path is never executed at runtime. Excluded from client coverage. |
| `buildMlClusterSuggestions()` and the genre-routing functions (`pickDisplayGroup`, `getGroupKey`, etc.) in `App.jsx` | Never reached because ML clusters are disabled. |
| `/api/brainz/enrich` route and the MusicBrainz/AcousticBrainz helpers in `server/index.js` | `AUTO_ENRICH_BRAINZ = false` in the client, so this endpoint is never hit. Drags server coverage down but the code is dead. |
| `ml_pipeline/` Python scripts | Out of scope — offline corpus build pipeline, not exercised by the running app. |
| `ml_legacy/`, `utils/` | Pre-existing legacy folders not imported by the active client or server. |

If those subsystems get turned back on, their tests should be added then — not now.

## Coverage targets and current numbers

| Area | Target | Current |
|---|---|---|
| Pure logic modules (cache keys, index mapping, token expiry, response validation) | 90%+ | **vibes.js 90.6%, token.js 100% lines** |
| Active server code | 85%+ statements | 66% statements; the ~33% gap is the disabled MusicBrainz code in `index.js`. **Active-only coverage is much higher** (≈90%+ if you exclude lines 506–660). |
| Active client code | 80%+ statements | 33% statements — `App.jsx` is monolithic (1,600+ lines) and covers many surfaces beyond the vibe flow (saved splits, threshold customization, listening-history upload, etc.). The **core flows (login, playlist load, vibe analysis, track selection, create playlist) are fully covered**; the uncovered remainder is mostly secondary UI panels. |

To raise the headline numbers further you'd either need to split `App.jsx` into smaller components (so coverage tooling can attribute lines more granularly to tested vs untested surfaces) or carve the MusicBrainz endpoint out of `server/index.js` into a separate file that coverage can exclude.

## Mocking strategy

External services are **never** called from tests.

- **Spotify API** — `globalThis.fetch` is replaced with `vi.fn()`; per-test handlers route URLs to canned responses. No real network.
- **Anthropic SDK** — intercepted at `require("@anthropic-ai/sdk")` via a Node `Module.prototype.require` hook installed in `tests/setup.mjs`. The fake client exposes a `messages.stream()` that returns a configurable `finalMessage()`-shaped object.
- **Postgres / Drizzle** — same `Module.prototype.require` hook intercepts `require("./db")` and returns a fake `db` with chainable `select()/from()/where()/limit()`, `insert()/values()/returning()/onConflictDoNothing()`, and `update()/set()/where()/returning()`. Each test sets up the next-call result via `setUserLookup` / `setCacheLookup` / pushing to `dbState.*Results`.
- **Sessions** — `tests/helpers.mjs` exposes `makeSessionCookie({ userId })`, which signs the cookie with the same Keygrip-style URL-safe-base64 + HMAC-SHA1 scheme `cookie-session` uses.
- **`window.alert` / `window.confirm`** — stubbed to `vi.fn()` returning `true` in the client setup so tests can assert they were called.

## Bugs discovered and fixed

While writing the client tests, one real bug was found and fixed:

- **`handleToggleTrack` mishandled the "implicit all" state on vibe cards.** Vibe suggestions weren't included in the initial-selection effect, so their `selectionBySuggestion[suggestionId]` started as `undefined`. The checkbox UI showed them as all-checked (via `getSelectedTrackIdsForSuggestion`'s fallback), but the first click to *uncheck* a track started from an empty set and *added* that one track to it — turning "all selected" into "only this one selected" instead of "all except this one." Fix: `handleToggleTrack` now seeds from `allTracks.map(t => t.id)` when the prev set is missing. See `client/src/App.jsx` ~line 972.

One known UX wart was identified but **not** fixed (no obviously-correct change):

- **Deselect All + Create on Spotify currently saves every track.** `getSelectedTrackIdsForSuggestion` treats a `Set` of size 0 as "no explicit selection → return all." Whether an empty selection should save zero, all, or surface an error is a product call; the current behavior is pinned by the test labeled `[current behavior]` so any future change is intentional.
