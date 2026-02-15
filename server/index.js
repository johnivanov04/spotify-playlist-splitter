// server/index.js
require("dotenv").config();

const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");
const cookieSession = require("cookie-session");
const querystring = require("querystring");
const crypto = require("crypto");

const app = express();
const PORT = process.env.PORT || 4000;

const {
  SPOTIFY_CLIENT_ID,
  SPOTIFY_CLIENT_SECRET,
  SPOTIFY_REDIRECT_URI,
  FRONTEND_URL,
  SESSION_SECRET,
  MUSICBRAINZ_USER_AGENT
} = process.env;

if (!SPOTIFY_CLIENT_ID || !SPOTIFY_CLIENT_SECRET || !SPOTIFY_REDIRECT_URI) {
  console.error("Missing Spotify env vars in .env");
  process.exit(1);
}

app.use(express.json({ limit: "2mb" }));

app.use(
  cookieSession({
    name: "session",
    keys: [SESSION_SECRET || "dev-secret"],
    maxAge: 24 * 60 * 60 * 1000
  })
);

const allowedOrigins = [
  FRONTEND_URL,
  "http://127.0.0.1:5173",
  "http://localhost:5173"
].filter(Boolean);

const corsOptions = {
  origin: allowedOrigins,
  credentials: true
};

app.use(cors(corsOptions));

// IMPORTANT FIX: Express/path-to-regexp can throw on "*" in some versions.
// Use a RegExp route instead of app.options("*", ...)
app.options(/.*/, cors(corsOptions));

const SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize";
const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";
const SPOTIFY_API_BASE = "https://api.spotify.com/v1";

const MB_BASE = "https://musicbrainz.org/ws/2";
const AB_BASE = "https://acousticbrainz.org/api/v1";

// ---- Helpers --------------------------------------------------

function generateRandomString(length) {
  return crypto.randomBytes(length).toString("hex");
}

function base64encode(str) {
  return Buffer.from(str).toString("base64");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 12_000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    return res;
  } finally {
    clearTimeout(t);
  }
}

async function fetchWithAuth(path, accessToken, options = {}) {
  const res = await fetchWithTimeout(`${SPOTIFY_API_BASE}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  if (!res.ok) {
    const text = await res.text();
    console.error("Spotify API error:", res.status, text);
    throw new Error(`Spotify API error ${res.status}`);
  }
  return res.json();
}

// ---- Auth routes ----------------------------------------------

app.get("/auth/login", (req, res) => {
  const state = generateRandomString(8);
  req.session.spotifyState = state;

  const scope = [
    "user-read-email",
    "playlist-read-private",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-library-read"
  ].join(" ");

  const params = querystring.stringify({
    response_type: "code",
    client_id: SPOTIFY_CLIENT_ID,
    scope,
    redirect_uri: SPOTIFY_REDIRECT_URI,
    state,
    show_dialog: "true"
  });

  res.redirect(`${SPOTIFY_AUTH_URL}?${params}`);
});

app.get("/auth/callback", async (req, res) => {
  const { code, state } = req.query;

  if (!state || state !== req.session.spotifyState) {
    return res.status(400).send("State mismatch");
  }
  req.session.spotifyState = null;

  const body = querystring.stringify({
    grant_type: "authorization_code",
    code,
    redirect_uri: SPOTIFY_REDIRECT_URI
  });

  try {
    const tokenRes = await fetchWithTimeout(SPOTIFY_TOKEN_URL, {
      method: "POST",
      headers: {
        Authorization:
          "Basic " +
          base64encode(`${SPOTIFY_CLIENT_ID}:${SPOTIFY_CLIENT_SECRET}`),
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body
    });

    const tokenData = await tokenRes.json();
    if (!tokenRes.ok) {
      console.error("Token exchange error:", tokenData);
      return res.status(400).send("Token exchange failed");
    }

    req.session.accessToken = tokenData.access_token;
    req.session.refreshToken = tokenData.refresh_token;
    req.session.expiresIn = tokenData.expires_in;
    req.session.obtainedAt = Date.now();

    res.redirect(FRONTEND_URL || "http://127.0.0.1:5173");
  } catch (err) {
    console.error("Error in auth callback:", err);
    res.status(500).send("Internal auth error");
  }
});

app.post("/auth/logout", (req, res) => {
  req.session = null;
  res.json({ ok: true });
});

function requireSpotifyAuth(req, res, next) {
  if (!req.session || !req.session.accessToken) {
    return res.status(401).json({ error: "Not authenticated" });
  }
  next();
}

// ---- Spotify API routes ---------------------------------------

app.get("/api/me", requireSpotifyAuth, async (req, res) => {
  try {
    const me = await fetchWithAuth("/me", req.session.accessToken);
    res.json(me);
  } catch (err) {
    console.error("Error fetching /me:", err);
    res.status(500).json({ error: "Failed to fetch user profile" });
  }
});

app.get("/api/playlists", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;

  try {
    let url = `${SPOTIFY_API_BASE}/me/playlists?limit=50`;
    const items = [];

    while (url) {
      const r = await fetchWithTimeout(url, {
        headers: { Authorization: `Bearer ${accessToken}` }
      });
      if (!r.ok) {
        const text = await r.text();
        console.error("Spotify playlists error:", r.status, text);
        throw new Error("Failed to fetch playlists");
      }
      const data = await r.json();
      items.push(...data.items);
      url = data.next;
    }

    const playlists = items.map((pl) => ({
      id: pl.id,
      name: pl.name,
      images: pl.images || [],
      tracksTotal: pl.tracks?.total ?? 0
    }));

    res.json({ playlists });
  } catch (err) {
    console.error("Error fetching playlists:", err);
    res.status(500).json({ error: "Failed to fetch playlists" });
  }
});

// - batch-fetch /tracks?ids=... to get external_ids.isrc
// - keep audio-features fetch (best-effort)
app.get("/api/playlists/:id/tracks", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;
  const playlistId = req.params.id;

  try {
    // 1) Fetch playlist tracks (paginated)
    let url = `${SPOTIFY_API_BASE}/playlists/${playlistId}/tracks?limit=100`;
    const playlistTracks = [];

    while (url) {
      const r = await fetchWithTimeout(url, {
        headers: { Authorization: `Bearer ${accessToken}` }
      });

      if (!r.ok) {
        const text = await r.text();
        console.error("Spotify playlist tracks error:", r.status, text);
        throw new Error("Failed to fetch playlist tracks");
      }

      const data = await r.json();
      playlistTracks.push(...(data.items || []));
      url = data.next;
    }

    // 2) Extract basic metadata
    const trackInfos = playlistTracks
      .filter((item) => item.track && item.track.id)
      .map((item) => {
        const t = item.track;
        const album = t.album || {};
        const artists = (t.artists || []).map((a) => a.name);
        const imageUrl =
          album.images && album.images.length ? album.images[0].url : null;

        let year = null;
        if (album.release_date) {
          year = parseInt(album.release_date.slice(0, 4), 10);
        }

        return {
          id: t.id,
          name: t.name,
          artists,
          album: album.name || "",
          year: year || null,
          spotifyUrl: t.external_urls?.spotify || "",
          imageUrl,
          popularity: t.popularity,
          previewUrl: t.preview_url || null,
          durationMs: typeof t.duration_ms === "number" ? t.duration_ms : null,
          isrc: null
        };
      });

    const ids = trackInfos.map((t) => t.id);

    // 3) Batch fetch track details to get ISRCs (Spotify: /tracks supports 50 ids)
    const isrcMap = {};
    const tracksChunkSize = 50;

    for (let i = 0; i < ids.length; i += tracksChunkSize) {
      const chunk = ids.slice(i, i + tracksChunkSize);
      if (!chunk.length) break;

      const r = await fetchWithTimeout(
        `${SPOTIFY_API_BASE}/tracks?ids=${chunk.join(",")}&market=from_token`,
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );

      if (!r.ok) {
        const text = await r.text();
        console.warn("Spotify /tracks lookup failed (continuing):", r.status, text);
        break;
      }

      const data = await r.json();
      const tracks = data.tracks || [];
      for (const tr of tracks) {
        if (!tr || !tr.id) continue;
        const isrc = tr.external_ids?.isrc || null;
        if (isrc) isrcMap[tr.id] = String(isrc).trim().toUpperCase();
      }
    }

    for (const t of trackInfos) {
      t.isrc = isrcMap[t.id] || null;
    }

    // 4) Best-effort audio features (100 ids)
    const featuresMap = {};
    const afChunkSize = 100;

    for (let i = 0; i < ids.length; i += afChunkSize) {
      const chunk = ids.slice(i, i + afChunkSize);
      if (!chunk.length) break;

      const featuresRes = await fetchWithTimeout(
        `${SPOTIFY_API_BASE}/audio-features?ids=${chunk.join(",")}`,
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );

      if (!featuresRes.ok) {
        const text = await featuresRes.text();
        console.error("Audio features error:", featuresRes.status, text);

        if (featuresRes.status === 401 || featuresRes.status === 403) {
          console.warn(
            "Spotify audio-features endpoint restricted. Continuing without energy/valence/tempo/danceability."
          );
          break;
        }
        throw new Error("Failed to fetch audio features");
      }

      const { audio_features } = await featuresRes.json();
      (audio_features || []).forEach((af) => {
        if (af && af.id) featuresMap[af.id] = af;
      });
    }

    // 5) Merge
    const tracks = trackInfos.map((t) => {
      const f = featuresMap[t.id] || {};
      return {
        ...t,
        energy: typeof f.energy === "number" ? f.energy : null,
        tempo: typeof f.tempo === "number" ? f.tempo : null,
        valence: typeof f.valence === "number" ? f.valence : null,
        danceability: typeof f.danceability === "number" ? f.danceability : null
      };
    });

    res.json({ tracks });
  } catch (err) {
    console.error("Error fetching playlist tracks + features:", err);
    res.status(500).json({ error: "Failed to fetch playlist tracks" });
  }
});

app.post("/api/playlists", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;
  const { name, description, trackIds } = req.body || {};

  if (!name || !Array.isArray(trackIds) || trackIds.length === 0) {
    return res.status(400).json({ error: "Missing name or trackIds" });
  }

  try {
    const me = await fetchWithAuth("/me", accessToken);
    const userId = me.id;

    const createRes = await fetchWithTimeout(
      `${SPOTIFY_API_BASE}/users/${encodeURIComponent(userId)}/playlists`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name,
          description: description || "",
          public: false
        })
      }
    );

    if (!createRes.ok) {
      const text = await createRes.text();
      console.error("Create playlist error:", createRes.status, text);
      throw new Error("Failed to create playlist");
    }

    const created = await createRes.json();
    const newPlaylistId = created.id;

    const uris = trackIds.map((id) => `spotify:track:${id}`);
    const chunkSize = 100;

    for (let i = 0; i < uris.length; i += chunkSize) {
      const chunk = uris.slice(i, i + chunkSize);
      const addRes = await fetchWithTimeout(
        `${SPOTIFY_API_BASE}/playlists/${newPlaylistId}/tracks`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${accessToken}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ uris: chunk })
        }
      );

      if (!addRes.ok) {
        const text = await addRes.text();
        console.error("Add tracks error:", addRes.status, text);
        throw new Error("Failed to add tracks to playlist");
      }
    }

    res.json({
      id: newPlaylistId,
      spotifyUrl: created.external_urls?.spotify || ""
    });
  } catch (err) {
    console.error("Error creating playlist:", err);
    res.status(500).json({ error: "Failed to create playlist" });
  }
});

app.post("/api/playlists/:id/remove-tracks", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;
  const playlistId = req.params.id;
  const { trackIds } = req.body;

  if (!Array.isArray(trackIds) || trackIds.length === 0) {
    return res.status(400).json({ error: "No trackIds provided" });
  }

  const tracksPayload = trackIds.map((id) => ({
    uri: `spotify:track:${id}`
  }));

  try {
    const spotifyRes = await fetchWithTimeout(
      `${SPOTIFY_API_BASE}/playlists/${playlistId}/tracks`,
      {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ tracks: tracksPayload })
      }
    );

    const text = await spotifyRes.text();
    if (!spotifyRes.ok) {
      return res
        .status(spotifyRes.status)
        .json({ error: "Spotify remove-tracks failed", details: text });
    }

    return res.json({ ok: true });
  } catch (err) {
    console.error("Error calling Spotify remove-tracks:", err);
    return res.status(500).json({ error: "Internal server error removing tracks" });
  }
});

// ---- MusicBrainz / AcousticBrainz enrichment -------------------

const MB_UA = MUSICBRAINZ_USER_AGENT || "PlaylistSplitter/0.1 (dev@example.com)";

const cacheIsrcToMbid = new Map(); // isrc -> mbid|null
const cacheRecording = new Map(); // mbid -> {tags, genres, title, artistCredit}
const cacheAbHigh = new Map(); // mbid -> json|null
const cacheAbLow = new Map(); // mbid -> json|null

let mbNextAllowedAt = 0;
async function mbGetJson(url) {
  // soft 1 req/sec
  const now = Date.now();
  if (now < mbNextAllowedAt) await sleep(mbNextAllowedAt - now);
  mbNextAllowedAt = Date.now() + 1100;

  const r = await fetchWithTimeout(url, {
    headers: { "User-Agent": MB_UA, Accept: "application/json" }
  });

  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`MusicBrainz ${r.status}: ${txt.slice(0, 200)}`);
  }
  return r.json();
}

async function abGetJson(url) {
  const r = await fetchWithTimeout(url, { headers: { Accept: "application/json" } }, 8_000);
  if (!r.ok) return null;
  return r.json();
}

function normalizeIsrc(isrc) {
  return String(isrc || "")
    .trim()
    .toUpperCase();
}

app.post("/api/brainz/enrich", requireSpotifyAuth, async (req, res) => {
  try {
    const {
      isrcs,
      includeTags = true,
      includeAcoustic = true,
      includeLowLevel = false,
      limit = 20
    } = req.body || {};

    if (!Array.isArray(isrcs) || isrcs.length === 0) {
      return res.status(400).json({ error: "isrcs must be a non-empty array" });
    }

    const capped = Math.max(1, Math.min(Number(limit) || 20, 60));
    const uniq = Array.from(new Set(isrcs.map(normalizeIsrc).filter(Boolean))).slice(0, capped);

    const byIsrc = {};

    for (const isrc of uniq) {
      let mbid = cacheIsrcToMbid.get(isrc);

      // 1) ISRC -> recording MBID
      if (mbid === undefined) {
        try {
          const data = await mbGetJson(`${MB_BASE}/isrc/${encodeURIComponent(isrc)}?fmt=json`);
          const recs =
            data.recordings || data.recording_list || data["recording-list"] || [];
          const first = Array.isArray(recs) && recs.length ? recs[0] : null;
          mbid = first?.id || null;
        } catch (e) {
          mbid = null;
        }
        cacheIsrcToMbid.set(isrc, mbid);
      }

      const out = { isrc, mbid: mbid || null };

      // 2) Recording tags/genres
      if (mbid && includeTags) {
        let rec = cacheRecording.get(mbid);
        if (!rec) {
          try {
            const rdata = await mbGetJson(
              `${MB_BASE}/recording/${encodeURIComponent(mbid)}?fmt=json&inc=tags+genres+artist-credits`
            );

            const tags = Array.isArray(rdata.tags)
              ? rdata.tags
                  .map((t) => ({
                    name: t.name,
                    count: typeof t.count === "number" ? t.count : null
                  }))
                  .filter((t) => t.name)
              : [];

            const genres = Array.isArray(rdata.genres)
              ? rdata.genres
                  .map((g) => ({
                    name: g.name,
                    count: typeof g.count === "number" ? g.count : null
                  }))
                  .filter((g) => g.name)
              : [];

            rec = {
              title: rdata.title || null,
              artistCredit: rdata["artist-credit"] || null,
              tags,
              genres
            };
            cacheRecording.set(mbid, rec);
          } catch (e) {
            rec = { title: null, artistCredit: null, tags: [], genres: [] };
            cacheRecording.set(mbid, rec);
          }
        }

        out.tags = rec.tags || [];
        out.genres = rec.genres || [];
        out.title = rec.title || null;
      }

      // 3) AcousticBrainz high-level / low-level (may be null)
      if (mbid && includeAcoustic) {
        if (!cacheAbHigh.has(mbid)) {
          const high = await abGetJson(`${AB_BASE}/${encodeURIComponent(mbid)}/high-level`);
          cacheAbHigh.set(mbid, high);
        }
        out.acousticHighLevel = cacheAbHigh.get(mbid) || null;

        if (includeLowLevel) {
          if (!cacheAbLow.has(mbid)) {
            const low = await abGetJson(`${AB_BASE}/${encodeURIComponent(mbid)}/low-level`);
            cacheAbLow.set(mbid, low);
          }
          out.acousticLowLevel = cacheAbLow.get(mbid) || null;
        } else {
          out.acousticLowLevel = null;
        }
      }

      byIsrc[isrc] = out;
    }

    res.json({ byIsrc });
  } catch (err) {
    console.error("Brainz enrich error:", err);
    res.status(500).json({ error: "Failed to enrich via MusicBrainz/AcousticBrainz" });
  }
});

// ---------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`Server listening on http://127.0.0.1:${PORT}`);
});
