import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:4000";
const SAVED_SPLITS_KEY = "playlistSplitter.savedSplits";

function buildSuggestions(tracks) {
  const suggestions = [];
  const minSize = 10;

  // Era buckets
  const hasYear = tracks.filter((t) => t.year);
  const oldSchool = hasYear.filter((t) => t.year <= 2005);
  const mid = hasYear.filter((t) => t.year > 2005 && t.year <= 2015);
  const newer = hasYear.filter((t) => t.year > 2015);

  if (oldSchool.length >= minSize) {
    suggestions.push({
      id: "era-old-school",
      label: "Old School Rap",
      description: "Tracks from before 2006",
      ruleDescription: "year <= 2005",
      tracks: oldSchool
    });
  }
  if (mid.length >= minSize) {
    suggestions.push({
      id: "era-mid",
      label: "Mid-Era Rap (2006–2015)",
      description: "Tracks released between 2006 and 2015",
      ruleDescription: "2006 <= year <= 2015",
      tracks: mid
    });
  }
  if (newer.length >= minSize) {
    suggestions.push({
      id: "era-new",
      label: "New Rap (2016+)",
      description: "More recent tracks",
      ruleDescription: "year >= 2016",
      tracks: newer
    });
  }

  // Energy / mood
  const hype = tracks.filter(
    (t) => t.energy !== null && t.energy >= 0.75 && t.tempo >= 110
  );
  const chill = tracks.filter(
    (t) => t.energy !== null && t.energy <= 0.5
  );
  const happy = tracks.filter(
    (t) => t.valence !== null && t.valence >= 0.7
  );
  const dark = tracks.filter(
    (t) => t.valence !== null && t.valence <= 0.4
  );

  if (hype.length >= minSize) {
    suggestions.push({
      id: "mood-hype",
      label: "Hype / Gym",
      description: "High-energy, high-tempo tracks",
      ruleDescription: "energy >= 0.75 && tempo >= 110",
      tracks: hype
    });
  }
  if (chill.length >= minSize) {
    suggestions.push({
      id: "mood-chill",
      label: "Chill / Late Night",
      description: "Lower-energy tracks, good for late nights",
      ruleDescription: "energy <= 0.5",
      tracks: chill
    });
  }
  if (happy.length >= minSize) {
    suggestions.push({
      id: "mood-happy",
      label: "Feel-Good",
      description: "Higher-valence (happier) tracks",
      ruleDescription: "valence >= 0.7",
      tracks: happy
    });
  }
  if (dark.length >= minSize) {
    suggestions.push({
      id: "mood-dark",
      label: "Darker / Moodier",
      description: "Lower-valence tracks with darker mood",
      ruleDescription: "valence <= 0.4",
      tracks: dark
    });
  }

  // Popularity
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
      description: "More popular tracks from your playlist",
      ruleDescription: "popularity >= 70",
      tracks: hits
    });
  }
  if (deepCuts.length >= minSize) {
    suggestions.push({
      id: "pop-deep",
      label: "Deeper Cuts",
      description: "Less popular tracks you might have overlooked",
      ruleDescription: "popularity <= 40",
      tracks: deepCuts
    });
  }

  return suggestions;
}

async function fetchJson(path, options = {}) {
  const res = await fetch(path, {
    credentials: "include",
    ...options
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
  // suggestionId -> Set of selected trackIds
  const [selectionBySuggestion, setSelectionBySuggestion] = useState({});
  // saved splits in localStorage
  const [savedSplits, setSavedSplits] = useState([]);
  const [error, setError] = useState("");

  // Load saved splits from localStorage on first mount
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

  // Persist saved splits whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(SAVED_SPLITS_KEY, JSON.stringify(savedSplits));
    } catch (e) {
      console.warn("Failed to persist saved splits", e);
    }
  }, [savedSplits]);

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

  const handleLogin = () => {
    window.location.href = `${API_BASE}/auth/login`;
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        credentials: "include"
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

    try {
      setLoadingTracks(true);
      const data = await fetchJson(
        `${API_BASE}/api/playlists/${pl.id}/tracks`
      );
      const t = data.tracks || [];
      setTracks(t);
      const s = buildSuggestions(t);
      setSuggestions(s);

      // initialize selection: all tracks in each suggestion are selected by default
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

  const getSelectedTrackIdsForSuggestion = (suggestion) => {
    const selectedSet = selectionBySuggestion[suggestion.id];
    if (!selectedSet || selectedSet.size === 0) {
      // fallback: treat as all tracks selected
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
        trackIds
      };

      const res = await fetch(`${API_BASE}/api/playlists`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
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

  // ----- Favorite split (save/unsave) -----------------

  const handleToggleSaveSuggestion = (suggestion) => {
    if (!selectedPlaylist) return;
    const key = `${selectedPlaylist.id}::${suggestion.id}`;

    setSavedSplits((prev) => {
      const existing = prev.find((s) => s.key === key);
      if (existing) {
        // un-save
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
        createdAt: Date.now()
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
        trackIds: split.trackIds
      };

      const res = await fetch(`${API_BASE}/api/playlists`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
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

      {/* 2. New Landing Page Layout */}
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

          {/* Features Grid for visual balance */}
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

      {/* 3. Main Dashboard (unchanged logic, just kept inside the check) */}
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
                    <div className="playlist-meta">
                      {pl.tracksTotal} tracks
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="content">
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
                <div className="card">
                  <h2>{selectedPlaylist.name}</h2>
                  {loadingTracks && <p>Analyzing your playlist…</p>}
                  {!loadingTracks && !!tracks.length && (
                    <>
                      <p>
                        Tracks analyzed: <strong>{tracks.length}</strong>
                      </p>
                      <p className="hint">
                        Suggestions below are based on era, energy/mood, and
                        popularity.
                      </p>
                    </>
                  )}
                  {error && <p className="error">{error}</p>}
                </div>

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

                {!loadingTracks && suggestions.length > 0 && (
                  <div className="suggestions-grid">
                    {suggestions.map((s) => {
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
                                onClick={() =>
                                  handleToggleSaveSuggestion(s)
                                }
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
                            <button
                              className="btn-primary"
                              onClick={() => handleCreatePlaylist(s)}
                            >
                              Create playlist on Spotify
                            </button>
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
                                        style={{ width: 24, height: 24, borderRadius: 3, marginRight: '0.5rem' }}
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
                                        <span className="pill">
                                          {t.year}
                                        </span>
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
              </>
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
                          onClick={() => handleRemoveSavedSplit(split.key)}
                        >
                          Remove
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
