import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:4000";
const SAVED_SPLITS_KEY = "playlistSplitter.savedSplits";
const THRESHOLDS_KEY = "playlistSplitter.thresholds";

const DEFAULT_THRESHOLDS = {
  barelyPlayed: { maxPlays: 3, maxMinutes: 3 },
  tourists: { maxPlays: 2, maxMinutes: 2 },
  coreFavorites: { minPlays: 25, minMinutes: 30 },
};

const PRESET_THRESHOLDS = {
  balanced: DEFAULT_THRESHOLDS,
  aggressive: {
    barelyPlayed: { maxPlays: 5, maxMinutes: 5 },
    tourists: { maxPlays: 4, maxMinutes: 5 },
    coreFavorites: { minPlays: 40, minMinutes: 45 },
  },
  gentle: {
    barelyPlayed: { maxPlays: 2, maxMinutes: 2 },
    tourists: { maxPlays: 1, maxMinutes: 1 },
    coreFavorites: { minPlays: 20, minMinutes: 20 },
  },
};

function cloneThresholds(cfg) {
  return JSON.parse(JSON.stringify(cfg));
}

function buildSuggestions(tracks, usageMap, thresholds) {
  const suggestions = [];
  const minSize = 10; // minimum tracks for era/popularity suggestions

  const config = thresholds || DEFAULT_THRESHOLDS;
  const barelyCfg = config.barelyPlayed || DEFAULT_THRESHOLDS.barelyPlayed;
  const touristsCfg = config.tourists || DEFAULT_THRESHOLDS.tourists;
  const coreCfg = config.coreFavorites || DEFAULT_THRESHOLDS.coreFavorites;

  const barelyMaxPlays = barelyCfg.maxPlays;
  const barelyMaxMinutes = barelyCfg.maxMinutes;
  const touristsMaxPlays = touristsCfg.maxPlays;
  const touristsMaxMinutes = touristsCfg.maxMinutes;
  const coreMinPlays = coreCfg.minPlays;
  const coreMinMinutes = coreCfg.minMinutes;

  // ---------- ERA / YEAR SPLITS ----------
  const hasYear = tracks.filter((t) => t.year);
  const oldSchool = hasYear.filter((t) => t.year <= 2005);
  const mid = hasYear.filter((t) => t.year > 2005 && t.year <= 2015);
  const newer = hasYear.filter((t) => t.year > 2015);

  if (oldSchool.length >= minSize) {
    suggestions.push({
      id: "era-old-school",
      label: "Old School (≤ 2005)",
      description: "Tracks from earlier eras in your playlist.",
      ruleDescription: "year ≤ 2005",
      tracks: oldSchool,
    });
  }
  if (mid.length >= minSize) {
    suggestions.push({
      id: "era-mid",
      label: "2006–2015",
      description: "Tracks released between 2006 and 2015.",
      ruleDescription: "2006 ≤ year ≤ 2015",
      tracks: mid,
    });
  }
  if (newer.length >= minSize) {
    suggestions.push({
      id: "era-new",
      label: "Recent (2016+)",
      description: "More recent additions to your playlist.",
      ruleDescription: "year ≥ 2016",
      tracks: newer,
    });
  }

  // ---------- POPULARITY SPLITS ----------
  const hits = tracks.filter(
    (t) => typeof t.popularity === "number" && t.popularity >= 70
  );
  const deepCuts = tracks.filter(
    (t) => typeof t.popularity === "number" && t.popularity <= 40
  );

  if (hits.length >= minSize) {
    suggestions.push({
      id: "pop-hits",
      label: "Hits / Mainstream",
      description: "More popular tracks from your playlist.",
      ruleDescription: "popularity ≥ 70",
      tracks: hits,
    });
  }
  if (deepCuts.length >= minSize) {
    suggestions.push({
      id: "pop-deep",
      label: "Deeper Cuts",
      description: "Less popular tracks you might forget about.",
      ruleDescription: "popularity ≤ 40",
      tracks: deepCuts,
    });
  }

  // ---------- ADVANCED: USAGE-BASED SPLITS ----------
  if (usageMap) {
    // 1) Barely played / never seen
    const barelyPlayed = tracks.filter((t) => {
      const u = usageMap[t.id];
      if (!u) return true; // no history at all
      const plays = u.plays ?? 0;
      const totalMs = u.totalMs ?? 0;
      const totalMinutes = totalMs / 60_000;
      // "barely" if plays ≤ threshold OR total minutes ≤ threshold
      return (
        plays <= barelyMaxPlays || totalMinutes <= barelyMaxMinutes
      );
    });

    if (barelyPlayed.length >= 5) {
      suggestions.push({
        id: "usage-barely-played",
        label: "Barely played tracks",
        description:
          "Songs from this playlist that you’ve almost never listened to in your Spotify history.",
        ruleDescription: `plays ≤ ${barelyMaxPlays} or total listening time ≤ ${barelyMaxMinutes} minutes (from uploaded history)`,
        tracks: barelyPlayed,
      });
    }

    // 2) Old favorites not played recently
    const longAgoFavorites = tracks.filter((t) => {
      const u = usageMap[t.id];
      if (!u) return false;
      const plays = u.plays ?? 0;
      if (plays < 5 || !u.lastPlayed) return false;

      const last = new Date(u.lastPlayed);
      if (Number.isNaN(last.getTime())) return false;

      const now = new Date();
      const diffDays = (now - last) / (1000 * 60 * 60 * 24);
      return diffDays > 180; // > 6 months ago
    });

    if (longAgoFavorites.length >= 5) {
      suggestions.push({
        id: "usage-long-ago",
        label: "Old favorites (not played recently)",
        description:
          "Tracks you used to listen to a lot but haven’t played in over 6 months.",
        ruleDescription:
          "plays ≥ 5 and lastPlayed > 6 months ago (from uploaded history)",
        tracks: longAgoFavorites,
      });
    }

    // 3) Frequently skipped tracks
    const frequentlySkipped = tracks.filter((t) => {
      const u = usageMap[t.id];
      if (!u) return false;
      const plays = u.plays ?? 0;
      const skips = u.skips ?? 0;
      if (plays < 3 || skips < 2) return false;
      const skipRate = skips / plays;
      return skipRate >= 0.5;
    });

    if (frequentlySkipped.length >= 3) {
      suggestions.push({
        id: "usage-frequently-skipped",
        label: "Frequently skipped tracks",
        description:
          "Songs in this playlist that you skip a lot in your Spotify history.",
        ruleDescription:
          "skips ≥ 2 and skip rate ≥ 50% of plays (from uploaded history)",
        tracks: frequentlySkipped,
      });
    }

    // 4) Core favorites (high plays & time)
    const coreFavorites = tracks.filter((t) => {
      const u = usageMap[t.id];
      if (!u) return false;
      const plays = u.plays ?? 0;
      const totalMs = u.totalMs ?? 0;
      const totalMinutes = totalMs / 60_000;
      return (
        plays >= coreMinPlays && totalMinutes >= coreMinMinutes
      );
    });

    if (coreFavorites.length >= 5) {
      suggestions.push({
        id: "usage-core-favorites",
        label: "Core favorites",
        description:
          "The tracks you really live in – high plays and lots of listening time.",
        ruleDescription: `plays ≥ ${coreMinPlays} and total listening ≥ ${coreMinMinutes} minutes (from uploaded history)`,
        tracks: coreFavorites,
      });
    }

    // 5) Tourists / padding (low plays & time)
    const tourists = tracks.filter((t) => {
      const u = usageMap[t.id];
      if (!u) return true; // no history → basically a tourist
      const plays = u.plays ?? 0;
      const totalMs = u.totalMs ?? 0;
      const totalMinutes = totalMs / 60_000;
      return (
        plays <= touristsMaxPlays &&
        totalMinutes <= touristsMaxMinutes
      );
    });

    if (tourists.length >= 5) {
      suggestions.push({
        id: "usage-tourists",
        label: "Tourists / padding",
        description:
          "Tracks that rarely get listened to – good candidates for cleanup or a backup playlist.",
        ruleDescription: `plays ≤ ${touristsMaxPlays} and total listening ≤ ${touristsMaxMinutes} minutes (from uploaded history)`,
        tracks: tourists,
      });
    }
  }

  return suggestions;
}

function computePlaylistHealth(tracks, usageMap) {
  if (!usageMap || !tracks || tracks.length === 0) return null;

  const now = new Date();
  let neverPlayedCount = 0;
  let frequentlySkippedCount = 0;
  const playsList = [];
  const lastPlayAges = [];

  for (const t of tracks) {
    const u = usageMap[t.id];
    const plays = u?.plays ?? 0;
    const skips = u?.skips ?? 0;

    if (!u || plays === 0) {
      neverPlayedCount += 1;
    }

    if (u) {
      const skipRate = plays > 0 ? skips / plays : 0;
      if (plays >= 3 && skips >= 2 && skipRate >= 0.5) {
        frequentlySkippedCount += 1;
      }

      playsList.push(plays);

      if (u.lastPlayed) {
        const d = new Date(u.lastPlayed);
        if (!Number.isNaN(d.getTime())) {
          const ageDays = (now - d) / (1000 * 60 * 60 * 24);
          if (ageDays >= 0) lastPlayAges.push(ageDays);
        }
      }
    } else {
      playsList.push(0);
    }
  }

  const total = tracks.length || 1;
  const neverPlayedPct = (neverPlayedCount / total) * 100;
  const frequentlySkippedPct = (frequentlySkippedCount / total) * 100;

  playsList.sort((a, b) => a - b);
  let medianPlays = 0;
  if (playsList.length) {
    const mid = Math.floor(playsList.length / 2);
    medianPlays =
      playsList.length % 2 === 0
        ? (playsList[mid - 1] + playsList[mid]) / 2
        : playsList[mid];
  }

  const avgLastPlayAgeDays = lastPlayAges.length
    ? lastPlayAges.reduce((sum, v) => sum + v, 0) / lastPlayAges.length
    : null;

  return {
    neverPlayedPct: Math.round(neverPlayedPct),
    frequentlySkippedPct: Math.round(frequentlySkippedPct),
    medianPlays: Math.round(medianPlays),
    avgLastPlayAgeDays:
      avgLastPlayAgeDays !== null ? Math.round(avgLastPlayAgeDays) : null,
  };
}

async function fetchJson(path, options = {}) {
  const res = await fetch(path, {
    credentials: "include",
    ...options,
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

function App() {
  const [user, setUser] = useState(null);
  const [playlists, setPlaylists] = useState([]);
  const [loadingMe, setLoadingMe] = useState(true);
  const [loadingPlaylists, setLoadingPlaylists] = useState(false);
  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [loadingTracks, setLoadingTracks] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [expandedSuggestionId, setExpandedSuggestionId] = useState(null);
  const [selectionBySuggestion, setSelectionBySuggestion] = useState({});
  const [savedSplits, setSavedSplits] = useState([]);
  const [error, setError] = useState("");
  const [usageMap, setUsageMap] = useState(null); // { [trackId]: { plays, totalMs, lastPlayed, skips } }

  const [dismissedSuggestionIds, setDismissedSuggestionIds] = useState(
    () => new Set()
  );

  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  const [presetKey, setPresetKey] = useState("balanced");

  const health =
    usageMap && tracks.length ? computePlaylistHealth(tracks, usageMap) : null;

  const visibleSuggestions = suggestions.filter(
    (s) => !dismissedSuggestionIds.has(s.id)
  );

  // ---- Load saved splits from localStorage on first mount ----
  useEffect(() => {
    try {
      const raw = localStorage.getItem(SAVED_SPLITS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setSavedSplits(parsed);
        }
      }
    } catch (e) {
      console.warn("Failed to load saved splits", e);
    }
  }, []);

  // ---- Load thresholds from localStorage on first mount ----
  useEffect(() => {
    try {
      const raw = localStorage.getItem(THRESHOLDS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          setThresholds((prev) => ({
            ...prev,
            ...parsed,
          }));
          setPresetKey("custom");
        }
      }
    } catch (e) {
      console.warn("Failed to load thresholds", e);
    }
  }, []);

  // Persist saved splits whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(SAVED_SPLITS_KEY, JSON.stringify(savedSplits));
    } catch (e) {
      console.warn("Failed to persist saved splits", e);
    }
  }, [savedSplits]);

  // Persist thresholds whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(THRESHOLDS_KEY, JSON.stringify(thresholds));
    } catch (e) {
      console.warn("Failed to persist thresholds", e);
    }
  }, [thresholds]);

  // Try to fetch /api/me on load to see if we already have a session
  useEffect(() => {
    async function init() {
      try {
        setLoadingMe(true);
        setError("");
        const me = await fetchJson(`${API_BASE}/api/me`);
        setUser(me);
        setLoadingMe(false);

        setLoadingPlaylists(true);
        const data = await fetchJson(`${API_BASE}/api/playlists`);
        setPlaylists(data.playlists || []);
        setLoadingPlaylists(false);
      } catch (err) {
        setLoadingMe(false);
        setLoadingPlaylists(false);
      }
    }
    init();
  }, []);

  // Recompute suggestions when advanced usage data or thresholds are loaded
  useEffect(() => {
    if (!selectedPlaylist || !tracks.length || !usageMap) return;

    const nextSuggestions = buildSuggestions(tracks, usageMap, thresholds);
    setSuggestions(nextSuggestions);
    setDismissedSuggestionIds(new Set());

    const initialSelection = {};
    nextSuggestions.forEach((s) => {
      initialSelection[s.id] = new Set(s.tracks.map((t) => t.id));
    });
    setSelectionBySuggestion(initialSelection);
  }, [usageMap, selectedPlaylist, tracks, thresholds]);

  // ---------- Advanced history upload ----------
  async function handleHistoryFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    try {
      const allEntries = [];

      for (const file of files) {
        const text = await file.text();
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
          allEntries.push(...parsed);
        }
      }

      const map = {};

      for (const entry of allEntries) {
        const uri =
          entry.spotify_track_uri || entry.trackUri || entry.uri || null;

        if (!uri || !uri.startsWith("spotify:track:")) continue;
        const trackId = uri.split(":")[2];

        const ms =
          entry.msPlayed ??
          entry.ms_played ??
          entry.ms_played_in_interval ??
          0;

        const endTime = entry.endTime || entry.timestamp || entry.ts || null;

        if (!map[trackId]) {
          map[trackId] = {
            plays: 0,
            totalMs: 0,
            lastPlayed: null,
            skips: 0,
            skipMs: 0,
          };
        }

        map[trackId].plays += 1;
        map[trackId].totalMs += ms;

        if (entry.skipped === true) {
          map[trackId].skips += 1;
          map[trackId].skipMs += ms;
        }

        if (endTime) {
          const currentLast = map[trackId].lastPlayed;
          const newDate = new Date(endTime);
          if (
            !currentLast ||
            (newDate instanceof Date &&
              !Number.isNaN(newDate.getTime()) &&
              newDate > new Date(currentLast))
          ) {
            map[trackId].lastPlayed = endTime;
          }
        }
      }

      console.log("Unique tracks in usageMap:", Object.keys(map).length);
      setUsageMap(map);
      alert(
        "Advanced listening data loaded! We'll use it to suggest barely-played, tourists, and core favorites."
      );
    } catch (err) {
      console.error("Failed to parse history files:", err);
      alert(
        "Could not read those files. Make sure they are the StreamingHistory/Streaming_History JSON files from Spotify."
      );
    }
  }

  const handleLogin = () => {
    window.location.href = `${API_BASE}/auth/login`;
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } finally {
      setUser(null);
      setPlaylists([]);
      setSelectedPlaylist(null);
      setTracks([]);
      setSuggestions([]);
      setSelectionBySuggestion({});
      setExpandedSuggestionId(null);
      setError("");
    }
  };

  const handleSelectPlaylist = async (pl) => {
    setSelectedPlaylist(pl);
    setTracks([]);
    setSuggestions([]);
    setError("");
    setExpandedSuggestionId(null);
    setSelectionBySuggestion({});
    setDismissedSuggestionIds(new Set());

    try {
      setLoadingTracks(true);
      const data = await fetchJson(
        `${API_BASE}/api/playlists/${pl.id}/tracks`
      );
      const t = data.tracks || [];
      setTracks(t);
      const s = buildSuggestions(t, usageMap, thresholds);
      setSuggestions(s);

      const initialSelection = {};
      s.forEach((suggestion) => {
        initialSelection[suggestion.id] = new Set(
          suggestion.tracks.map((track) => track.id)
        );
      });
      setSelectionBySuggestion(initialSelection);
    } catch (err) {
      console.error(err);
      setError("Failed to load tracks for that playlist.");
    } finally {
      setLoadingTracks(false);
    }
  };

  const handleToggleTrack = (suggestionId, trackId) => {
    setSelectionBySuggestion((prev) => {
      const prevSet = prev[suggestionId] || new Set();
      const nextSet = new Set(prevSet);
      if (nextSet.has(trackId)) {
        nextSet.delete(trackId);
      } else {
        nextSet.add(trackId);
      }
      return { ...prev, [suggestionId]: nextSet };
    });
  };

  const handleDismissSuggestion = (suggestionId) => {
    setDismissedSuggestionIds((prev) => {
      const next = new Set(prev);
      next.add(suggestionId);
      return next;
    });
  };

  const handlePresetChange = (event) => {
    const key = event.target.value;
    setPresetKey(key);
    const presetCfg = PRESET_THRESHOLDS[key];
    if (presetCfg) {
      setThresholds(cloneThresholds(presetCfg));
    }
  };

  const handleThresholdChange = (category, field, rawValue) => {
    const value = Number(rawValue);
    if (Number.isNaN(value) || value < 0) return;
    setThresholds((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value,
      },
    }));
    setPresetKey("custom");
  };

  const getSelectedTrackIdsForSuggestion = (suggestion) => {
    const selectedSet = selectionBySuggestion[suggestion.id];
    if (!selectedSet || selectedSet.size === 0) {
      return suggestion.tracks.map((t) => t.id);
    }
    return suggestion.tracks
      .filter((t) => selectedSet.has(t.id))
      .map((t) => t.id);
  };

  const handleCreatePlaylist = async (suggestion) => {
    try {
      const trackIds = getSelectedTrackIdsForSuggestion(suggestion);

      if (!trackIds.length) {
        alert("No tracks selected for this split.");
        return;
      }

      const body = {
        name: suggestion.label,
        description: `Generated by Playlist Splitter from "${selectedPlaylist.name}"`,
        trackIds,
      };

      const res = await fetch(`${API_BASE}/api/playlists`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      alert(
        `Playlist created! ${data.spotifyUrl ? "Open: " + data.spotifyUrl : ""}`
      );
    } catch (err) {
      console.error(err);
      alert("Failed to create playlist on Spotify.");
    }
  };

  const handleRemoveTracksFromPlaylist = async (suggestion) => {
    if (!selectedPlaylist) return;

    const trackIds = getSelectedTrackIdsForSuggestion(suggestion);

    if (!trackIds.length) {
      alert("No tracks selected to remove.");
      return;
    }

    const confirm = window.confirm(
      `Remove ${trackIds.length} tracks from "${selectedPlaylist.name}" on Spotify? This can't be undone.`
    );
    if (!confirm) return;

    try {
      const res = await fetch(
        `${API_BASE}/api/playlists/${selectedPlaylist.id}/remove-tracks`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trackIds }),
        }
      );

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      alert("Tracks removed from your playlist on Spotify.");
      await handleSelectPlaylist(selectedPlaylist);
    } catch (err) {
      console.error(err);
      alert("Failed to remove tracks from playlist.");
    }
  };

  // ----- Favorite split (save/unsave) -----------------
  const handleToggleSaveSuggestion = (suggestion) => {
    if (!selectedPlaylist) return;
    const key = `${selectedPlaylist.id}::${suggestion.id}`;

    setSavedSplits((prev) => {
      const existing = prev.find((s) => s.key === key);
      if (existing) {
        return prev.filter((s) => s.key !== key);
      }

      const trackIds = getSelectedTrackIdsForSuggestion(suggestion);

      const newSplit = {
        key,
        playlistId: selectedPlaylist.id,
        playlistName: selectedPlaylist.name,
        suggestionId: suggestion.id,
        label: suggestion.label,
        ruleDescription: suggestion.ruleDescription,
        trackIds,
        createdAt: Date.now(),
      };
      return [...prev, newSplit];
    });
  };

  const handleRemoveSavedSplit = (key) => {
    setSavedSplits((prev) => prev.filter((s) => s.key !== key));
  };

  const handleCreateSavedSplit = async (split) => {
    try {
      if (!split.trackIds || split.trackIds.length === 0) {
        alert("This saved split has no tracks.");
        return;
      }

      const body = {
        name: split.label,
        description: `Saved split from "${split.playlistName}"`,
        trackIds: split.trackIds,
      };

      const res = await fetch(`${API_BASE}/api/playlists`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      alert(
        `Playlist created! ${data.spotifyUrl ? "Open: " + data.spotifyUrl : ""}`
      );
    } catch (err) {
      console.error(err);
      alert("Failed to create playlist on Spotify.");
    }
  };

  const loggedIn = !!user;

  return (
    <div className="app-root">
      {/* 1. Dashboard Header - Only show this when logged in */}
      {loggedIn && (
        <header className="header">
          <div>
            <h1>Playlist Splitter</h1>
            <p className="tagline">
              Take one big Spotify playlist and split it into smaller, vibe-based
              sub-playlists.
            </p>
          </div>
          <div className="header-right">
            <span className="user-label">
              Logged in as <strong>{user.display_name || user.id}</strong>
            </span>
            <button className="btn-secondary" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </header>
      )}

      {/* 2. Landing Page */}
      {!loggedIn && (
        <div className="landing-layout">
          <div className="landing-card">
            <h1 className="landing-title">Playlist Splitter</h1>
            <p className="landing-desc">
              Take one big Spotify playlist and split it into smaller,
              vibe-based sub-playlists automatically.
            </p>

            <button
              className="btn-primary btn-login-large"
              onClick={handleLogin}
            >
              LOG IN WITH SPOTIFY
            </button>
          </div>

          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon">✨</div>
              <div className="feature-text">AI-Powered Analysis</div>
            </div>
            <div className="feature-item">
              <div className="feature-icon">📂</div>
              <div className="feature-text">Smart Organization</div>
            </div>
            <div className="feature-item">
              <div className="feature-icon">🚀</div>
              <div className="feature-text">Instant Export</div>
            </div>
          </div>
        </div>
      )}

      {/* 3. Main Dashboard */}
      {loggedIn && (
        <main className="layout">
          <section className="sidebar">
            <h2>Your Playlists</h2>
            {loadingPlaylists && <p>Loading playlists…</p>}
            {!loadingPlaylists && playlists.length === 0 && (
              <p>No playlists found.</p>
            )}
            <ul className="playlist-list">
              {playlists.map((pl) => (
                <li
                  key={pl.id}
                  className={
                    selectedPlaylist?.id === pl.id
                      ? "playlist-item selected"
                      : "playlist-item"
                  }
                  onClick={() => handleSelectPlaylist(pl)}
                >
                  {pl.images && pl.images.length > 0 && (
                    <img
                      src={pl.images[0].url}
                      alt={pl.name}
                      className="playlist-cover"
                    />
                  )}
                  <div>
                    <div className="playlist-name">{pl.name}</div>
                    <div className="playlist-meta">{pl.tracksTotal} tracks</div>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="content">
            {/* Advanced history upload card */}
            <div className="card advanced-card">
              <div className="advanced-card-header">
                <h2>Advanced listening data (optional)</h2>
                {usageMap && (
                  <span className="pill pill-success">Data loaded</span>
                )}
              </div>
              <p>
                For deeper cleanup suggestions, you can upload your Spotify{" "}
                <strong>extended streaming history</strong> export. All processing
                happens in your browser – nothing is sent to our server.
              </p>
              <details className="advanced-details">
                <summary>How do I get this from Spotify?</summary>
                <ol className="advanced-steps">
                  <li>
                    Go to your Spotify account &gt; Privacy &gt; Download your
                    data.
                  </li>
                  <li>
                    Request your <em>extended streaming history</em>.
                  </li>
                  <li>
                    When Spotify emails the ZIP, unzip it on your computer.
                  </li>
                  <li>
                    In the <code>MyData</code> folder, select the files named
                    like <code>StreamingHistory_music_0.json</code>,{" "}
                    <code>StreamingHistory_music_1.json</code>, etc.
                  </li>
                  <li>Upload them here:</li>
                </ol>
              </details>

              <input
                type="file"
                multiple
                accept=".json,application/json"
                onChange={handleHistoryFilesSelected}
              />
            </div>

            {!selectedPlaylist && (
              <div className="card">
                <h2>Select a playlist</h2>
                <p>
                  Choose a playlist on the left to analyze it and see suggested
                  sub-playlists.
                </p>
              </div>
            )}

            {selectedPlaylist && (
              <>
                {/* Playlist header card */}
                <div className="card">
                  <h2>{selectedPlaylist.name}</h2>
                  {loadingTracks && <p>Analyzing your playlist…</p>}
                  {!loadingTracks && !!tracks.length && (
                    <>
                      <p>
                        Tracks analyzed: <strong>{tracks.length}</strong>
                      </p>
                      <p className="hint">
                        Suggestions below are based on era, energy/mood,
                        popularity, and your listening history.
                      </p>
                    </>
                  )}
                  {error && <p className="error">{error}</p>}
                </div>

                {/* Playlist health card */}
                {usageMap && tracks.length > 0 && health && (
                  <div className="card health-card">
                    <h3>Playlist health</h3>
                    <p className="health-summary">
                      This playlist:{" "}
                      <strong>{health.neverPlayedPct}%</strong> never played ·{" "}
                      <strong>{health.frequentlySkippedPct}%</strong> frequently
                      skipped · median plays{" "}
                      <strong>{health.medianPlays}</strong>
                      {health.avgLastPlayAgeDays !== null && (
                        <>
                          {" "}
                          · average last play{" "}
                          <strong>{health.avgLastPlayAgeDays} days</strong> ago
                        </>
                      )}
                    </p>
                  </div>
                )}

                {/* Threshold controls */}
                {usageMap && tracks.length > 0 && (
                  <div className="card thresholds-card">
                    <div className="thresholds-header">
                      <h3>Cleanup thresholds</h3>
                      <label className="thresholds-preset">
                        Preset:&nbsp;
                        <select
                          value={presetKey}
                          onChange={handlePresetChange}
                        >
                          <option value="balanced">Balanced</option>
                          <option value="aggressive">Aggressive cleanup</option>
                          <option value="gentle">Gentle cleanup</option>
                          <option value="custom">Custom</option>
                        </select>
                      </label>
                    </div>

                    <div className="thresholds-body">
                      <div className="threshold-row">
                        <div className="threshold-label">Barely played</div>
                        <div className="threshold-controls">
                          <label>
                            plays ≤{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.barelyPlayed.maxPlays}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "barelyPlayed",
                                  "maxPlays",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                          <label>
                            minutes ≤{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.barelyPlayed.maxMinutes}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "barelyPlayed",
                                  "maxMinutes",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                        </div>
                      </div>

                      <div className="threshold-row">
                        <div className="threshold-label">Tourists / padding</div>
                        <div className="threshold-controls">
                          <label>
                            plays ≤{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.tourists.maxPlays}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "tourists",
                                  "maxPlays",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                          <label>
                            minutes ≤{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.tourists.maxMinutes}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "tourists",
                                  "maxMinutes",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                        </div>
                      </div>

                      <div className="threshold-row">
                        <div className="threshold-label">Core favorites</div>
                        <div className="threshold-controls">
                          <label>
                            plays ≥{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.coreFavorites.minPlays}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "coreFavorites",
                                  "minPlays",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                          <label>
                            minutes ≥{" "}
                            <input
                              type="number"
                              min="0"
                              value={thresholds.coreFavorites.minMinutes}
                              onChange={(e) =>
                                handleThresholdChange(
                                  "coreFavorites",
                                  "minMinutes",
                                  e.target.value
                                )
                              }
                            />
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Scrollable suggestions + saved splits */}
                <div className="content-scroll">
                  {!loadingTracks &&
                    suggestions.length === 0 &&
                    tracks.length > 0 && (
                      <div className="card">
                        <h3>No strong groupings found</h3>
                        <p>
                          We couldn&apos;t find big enough clusters by year,
                          energy, or popularity. Try another playlist or we can
                          add adjustable thresholds later.
                        </p>
                      </div>
                    )}

                  {!loadingTracks && visibleSuggestions.length > 0 && (
                    <div className="suggestions-grid">
                      {visibleSuggestions.map((s) => {
                        const expanded = expandedSuggestionId === s.id;
                        const selectedSet = selectionBySuggestion[s.id];
                        const isSaved = savedSplits.some(
                          (split) =>
                            split.playlistId === selectedPlaylist.id &&
                            split.suggestionId === s.id
                        );

                        return (
                          <article key={s.id} className="card suggestion-card">
                            <div className="suggestion-header">
                              <div>
                                <h3>{s.label}</h3>
                                <p className="suggestion-sub">
                                  {s.description}
                                </p>
                              </div>
                              <div className="suggestion-header-right">
                                <button
                                  className={
                                    isSaved ? "star-btn starred" : "star-btn"
                                  }
                                  onClick={() => handleToggleSaveSuggestion(s)}
                                  title={
                                    isSaved
                                      ? "Unsave this split"
                                      : "Save this split"
                                  }
                                >
                                  {isSaved ? "★" : "☆"}
                                </button>
                                <span className="count-pill">
                                  {s.tracks.length} tracks
                                </span>
                                <button
                                  className="dismiss-btn"
                                  title="Hide this suggestion"
                                  onClick={() =>
                                    handleDismissSuggestion(s.id)
                                  }
                                >
                                  ×
                                </button>
                              </div>
                            </div>
                            <p className="rule-text">
                              Rule: <code>{s.ruleDescription}</code>
                            </p>
                            <div className="suggestion-actions">
                              <button
                                className="btn-secondary"
                                onClick={() =>
                                  setExpandedSuggestionId(
                                    expanded ? null : s.id
                                  )
                                }
                              >
                                {expanded ? "Hide tracks" : "View tracks"}
                              </button>

                              {s.id === "usage-frequently-skipped" ? (
                                <button
                                  className="btn-secondary"
                                  style={{
                                    borderColor: "#ff4d4f",
                                    color: "#ff4d4f",
                                  }}
                                  onClick={() =>
                                    handleRemoveTracksFromPlaylist(s)
                                  }
                                >
                                  Remove from playlist
                                </button>
                              ) : (
                                <button
                                  className="btn-primary"
                                  onClick={() => handleCreatePlaylist(s)}
                                >
                                  Create playlist on Spotify
                                </button>
                              )}
                            </div>

                            {expanded && (
                              <div className="tracks-list">
                                {s.tracks.map((t) => {
                                  const isSelected =
                                    !selectedSet || selectedSet.has(t.id);
                                  return (
                                    <div
                                      key={t.id}
                                      className={
                                        isSelected
                                          ? "track-row"
                                          : "track-row track-row--dim"
                                      }
                                    >
                                      <label className="track-checkbox">
                                        <input
                                          type="checkbox"
                                          checked={isSelected}
                                          onChange={() =>
                                            handleToggleTrack(s.id, t.id)
                                          }
                                        />
                                      </label>
                                      {t.imageUrl && (
                                        <img
                                          src={t.imageUrl}
                                          alt={t.name}
                                          className="track-cover"
                                          style={{
                                            width: 24,
                                            height: 24,
                                            borderRadius: 3,
                                            marginRight: "0.5rem",
                                          }}
                                        />
                                      )}
                                      <div className="track-main">
                                        <div className="track-title">
                                          {t.name}
                                        </div>
                                        <div className="track-artist">
                                          {t.artists?.join(", ")}
                                        </div>
                                        {t.spotifyUrl && (
                                          <a
                                            href={t.spotifyUrl}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="track-link"
                                          >
                                            Open in Spotify ↗
                                          </a>
                                        )}
                                      </div>
                                      <div className="track-meta">
                                        {t.year && (
                                          <span className="pill">{t.year}</span>
                                        )}
                                        {typeof t.energy === "number" && (
                                          <span className="pill">
                                            energy {t.energy.toFixed(2)}
                                          </span>
                                        )}
                                        {typeof t.popularity === "number" && (
                                          <span className="pill">
                                            pop {t.popularity}
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </article>
                        );
                      })}
                    </div>
                  )}

                  {savedSplits.length > 0 && (
                    <div className="card saved-splits-card">
                      <h2>Saved splits</h2>
                      <ul className="saved-splits-list">
                        {savedSplits.map((split) => (
                          <li key={split.key} className="saved-split-item">
                            <div>
                              <div className="saved-split-title">
                                {split.label}
                              </div>
                              <div className="saved-split-meta">
                                from{" "}
                                <span className="saved-split-playlist">
                                  {split.playlistName}
                                </span>{" "}
                                · {split.trackIds.length} tracks
                              </div>
                            </div>
                            <div className="saved-split-actions">
                              <button
                                className="btn-secondary"
                                onClick={() => handleCreateSavedSplit(split)}
                              >
                                Create on Spotify
                              </button>
                              <button
                                className="btn-secondary"
                                onClick={() =>
                                  handleRemoveSavedSplit(split.key)
                                }
                              >
                                Remove
                              </button>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
