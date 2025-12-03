import { useEffect, useState } from "react";
const API_BASE = "http://127.0.0.1:4000";

function buildSuggestions(tracks) {
  const suggestions = [];
  
  const minSize = 10; // minimum tracks to show a suggestion

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

  // Energy/mood
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
  const [error, setError] = useState("");

  // Try to fetch /api/me on load to see if we already have a session
  useEffect(() => {
    async function init() {
      try {
        setLoadingMe(true);
        setError("");
        const me = await fetchJson(`${API_BASE}/api/me`);
        setUser(me);
        setLoadingMe(false);

        // Fetch playlists
        setLoadingPlaylists(true);
        const data = await fetchJson(`${API_BASE}/api/playlists`);
        setPlaylists(data.playlists || []);
        setLoadingPlaylists(false);
      } catch (err) {
        // Not logged in or error – that’s fine
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
    }
  };

  const handleSelectPlaylist = async (pl) => {
    setSelectedPlaylist(pl);
    setTracks([]);
    setSuggestions([]);
    setError("");
    setExpandedSuggestionId(null);

    try {
      setLoadingTracks(true);
      const data = await fetchJson(
        `${API_BASE}/api/playlists/${pl.id}/tracks`
      );
      const t = data.tracks || [];
      setTracks(t);
      const s = buildSuggestions(t);
      setSuggestions(s);
    } catch (err) {
      console.error(err);
      setError("Failed to load tracks for that playlist.");
    } finally {
      setLoadingTracks(false);
    }
  };

  const handleCreatePlaylist = async (suggestion) => {
    try {
      const body = {
        name: suggestion.label,
        description: `Generated by Playlist Splitter from "${selectedPlaylist.name}"`,
        trackIds: suggestion.tracks.map((t) => t.id)
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
      <header className="header">
        <div>
          <h1>Playlist Splitter</h1>
          <p className="tagline">
            Take one big Spotify playlist and split it into smaller, vibe-based
            sub-playlists.
          </p>
        </div>
        <div className="header-right">
          {loggedIn && (
            <>
              <span className="user-label">
                Logged in as <strong>{user.display_name || user.id}</strong>
              </span>
              <button className="btn-secondary" onClick={handleLogout}>
                Log out
              </button>
            </>
          )}
        </div>
      </header>

      {!loggedIn && (
        <main className="main-panel">
          <section className="card">
            <h2>Connect your Spotify</h2>
            <p>
              Log in with your Spotify account, choose one of your playlists,
              and we&apos;ll suggest sub-playlists like:
              <br />
              <em>&quot;90s Rap&quot;, &quot;Hype/Gym&quot;, &quot;Late
              Night&quot;, &quot;Deeper Cuts&quot;</em>.
            </p>
            <button className="btn-primary" onClick={handleLogin}>
              Log in with Spotify
            </button>
          </section>
        </main>
      )}

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

                {!loadingTracks && suggestions.length === 0 && tracks.length > 0 && (
                  <div className="card">
                    <h3>No strong groupings found</h3>
                    <p>
                      We couldn&apos;t find big enough clusters by year,
                      energy, or popularity. Try another playlist or we can add
                      adjustable thresholds later.
                    </p>
                  </div>
                )}

                {!loadingTracks && suggestions.length > 0 && (
                  <div className="suggestions-grid">
                    {suggestions.map((s) => {
                      const expanded = expandedSuggestionId === s.id;
                      return (
                        <article key={s.id} className="card suggestion-card">
                          <div className="suggestion-header">
                            <div>
                              <h3>{s.label}</h3>
                              <p className="suggestion-sub">
                                {s.description}
                              </p>
                            </div>
                            <span className="count-pill">
                              {s.tracks.length} tracks
                            </span>
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
                              {s.tracks.map((t) => (
                                <div key={t.id} className="track-row">
                                  <div className="track-main">
                                    <div className="track-title">
                                      {t.name}
                                    </div>
                                    <div className="track-artist">
                                      {t.artists?.join(", ")}
                                    </div>
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
                              ))}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
