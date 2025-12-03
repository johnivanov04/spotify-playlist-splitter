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
  SESSION_SECRET
} = process.env;

if (!SPOTIFY_CLIENT_ID || !SPOTIFY_CLIENT_SECRET || !SPOTIFY_REDIRECT_URI) {
  console.error("Missing Spotify env vars in .env");
  process.exit(1);
}

app.use(express.json());
app.use(
  cookieSession({
    name: "session",
    keys: [SESSION_SECRET || "dev-secret"],
    maxAge: 24 * 60 * 60 * 1000 // 1 day
  })
);

// CORS for local dev – frontend on 5173
app.use(
  cors({
    origin: FRONTEND_URL || "http://127.0.0.1:5173",
    credentials: true
  })
);

const SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize";
const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";
const SPOTIFY_API_BASE = "https://api.spotify.com/v1";

// ---- Helpers --------------------------------------------------

function generateRandomString(length) {
  return crypto.randomBytes(length).toString("hex");
}

function base64encode(str) {
  return Buffer.from(str).toString("base64");
}

async function fetchWithAuth(path, accessToken, options = {}) {
  const res = await fetch(`${SPOTIFY_API_BASE}${path}`, {
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

// 1. Start login: redirect to Spotify authorize page
app.get("/auth/login", (req, res) => {
  const state = generateRandomString(8);
  req.session.spotifyState = state;

  const scope = [
    "playlist-read-private",
    "playlist-modify-private"
  ].join(" ");

  const params = querystring.stringify({
    response_type: "code",
    client_id: SPOTIFY_CLIENT_ID,
    scope,
    redirect_uri: SPOTIFY_REDIRECT_URI,
    state
  });

  res.redirect(`${SPOTIFY_AUTH_URL}?${params}`);
});

// 2. Callback: exchange code for tokens
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
    const tokenRes = await fetch(SPOTIFY_TOKEN_URL, {
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

    // Store tokens in session
    req.session.accessToken = tokenData.access_token;
    req.session.refreshToken = tokenData.refresh_token;
    req.session.expiresIn = tokenData.expires_in;
    req.session.obtainedAt = Date.now();

    // Redirect back to frontend
    res.redirect(FRONTEND_URL || "http://127.0.0.1:5173");
  } catch (err) {
    console.error("Error in auth callback:", err);
    res.status(500).send("Internal auth error");
  }
});

// 3. Logout (clear session)
app.post("/auth/logout", (req, res) => {
  req.session = null;
  res.json({ ok: true });
});

// Middleware to require auth
function requireSpotifyAuth(req, res, next) {
  if (!req.session || !req.session.accessToken) {
    return res.status(401).json({ error: "Not authenticated" });
  }
  next();
}

// ---- API routes -----------------------------------------------

// 4. Get current user profile
app.get("/api/me", requireSpotifyAuth, async (req, res) => {
  try {
    const me = await fetchWithAuth("/me", req.session.accessToken);
    res.json(me);
  } catch (err) {
    console.error("Error fetching /me:", err);
    res.status(500).json({ error: "Failed to fetch user profile" });
  }
});

// 5. Get current user's playlists (aggregate all pages)
app.get("/api/playlists", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;

  try {
    let url = `${SPOTIFY_API_BASE}/me/playlists?limit=50`;
    const items = [];

    while (url) {
      const r = await fetch(url, {
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

// 6. Get tracks (+ audio features if allowed) for a playlist
app.get("/api/playlists/:id/tracks", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;
  const playlistId = req.params.id;

  try {
    // 1) Fetch playlist tracks (paginated)
    let url = `${SPOTIFY_API_BASE}/playlists/${playlistId}/tracks?limit=100`;
    const playlistTracks = [];

    while (url) {
      const r = await fetch(url, {
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

    // 2) Extract track ids + basic metadata
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
          popularity: t.popularity
        };
      });

    const ids = trackInfos.map((t) => t.id);

    // 3) Try to fetch audio features (but don't crash if Spotify blocks it)
    const featuresMap = {};
    const chunkSize = 100;

    for (let i = 0; i < ids.length; i += chunkSize) {
      const chunk = ids.slice(i, i + chunkSize);
      if (chunk.length === 0) break;

      const featuresRes = await fetch(
        `${SPOTIFY_API_BASE}/audio-features?ids=${chunk.join(",")}`,
        {
          headers: { Authorization: `Bearer ${accessToken}` }
        }
      );

      if (!featuresRes.ok) {
        const text = await featuresRes.text();
        console.error("Audio features error:", featuresRes.status, text);

        // Spotify recently restricted this endpoint for many new apps.
        // If we get a permission error, log it and continue WITHOUT mood features.
        if (featuresRes.status === 401 || featuresRes.status === 403) {
          console.warn(
            "Spotify audio-features endpoint is restricted for this app. " +
              "Continuing without energy/valence/tempo data."
          );
          break; // stop trying further chunks
        }

        // Other errors (500, etc.) we still treat as fatal:
        throw new Error("Failed to fetch audio features");
      }

      const { audio_features } = await featuresRes.json();
      (audio_features || []).forEach((af) => {
        if (af && af.id) {
          featuresMap[af.id] = af;
        }
      });
    }

    // 4) Merge metadata + features (if any)
    const tracks = trackInfos.map((t) => {
      const f = featuresMap[t.id] || {};
      return {
        ...t,
        energy: typeof f.energy === "number" ? f.energy : null,
        tempo: typeof f.tempo === "number" ? f.tempo : null,
        valence: typeof f.valence === "number" ? f.valence : null,
        danceability:
          typeof f.danceability === "number" ? f.danceability : null
      };
    });

    res.json({ tracks });
  } catch (err) {
    console.error("Error fetching playlist tracks + features:", err);
    res.status(500).json({ error: "Failed to fetch playlist tracks" });
  }
});

// 7. Create a new playlist with given tracks
app.post("/api/playlists", requireSpotifyAuth, async (req, res) => {
  const accessToken = req.session.accessToken;
  const { name, description, trackIds } = req.body || {};

  if (!name || !Array.isArray(trackIds) || trackIds.length === 0) {
    return res.status(400).json({ error: "Missing name or trackIds" });
  }

  try {
    // Get current user id
    const me = await fetchWithAuth("/me", accessToken);
    const userId = me.id;

    // Create playlist
    const createRes = await fetch(
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

    // Add tracks (Spotify expects track URIs)
    const uris = trackIds.map((id) => `spotify:track:${id}`);
    const chunkSize = 100;
    for (let i = 0; i < uris.length; i += chunkSize) {
      const chunk = uris.slice(i, i + chunkSize);
      const addRes = await fetch(
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

// ----------------------------------------------------------

app.listen(PORT, () => {
  console.log(`Server listening on http://127.0.0.1:${PORT}`);
});
